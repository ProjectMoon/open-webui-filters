"""
title: EXIF Filter
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.2.0
license: AGPL-3.0+
required_open_webui_version: 0.6.5
requirements: exifread
"""

import json
import os
import tempfile
from base64 import b64decode
from io import BytesIO
from typing import Any, Awaitable, Callable, Literal, Optional
from urllib.parse import urljoin

import requests
from exifread import process_file
from open_webui.utils.misc import (
    add_or_update_system_message,
    get_last_user_message,
    get_last_user_message_item,
)
from pydantic import BaseModel, Field


def get_or_none(tags: dict, *keys: str) -> Optional[str]:
    """
    Try to extract a value from a dict by trying keys in order, or
    return None if none of the keys were found.
    """
    for key in keys:
        if key in tags:
            return tags[key]

    return None

def all_are_none(*args) -> bool:
    for arg in args:
        if arg is not None:
            return False

    return True

def parse_nominatim_address(address) -> Optional[str]:
    """Parse address from Nominatim address object."""
    house_number = get_or_none(address, "house_number")
    street = get_or_none(address, "road")
    city = get_or_none(address, "city")
    state = get_or_none(address, "state")
    postal_code = get_or_none(address, "postcode")
    country = get_or_none(address, "country")

    # if all are none, that means we don't know the address at all.
    if all_are_none(house_number, street, city, state, postal_code, country):
        return None

    # Handle missing values to create complete-ish addresses, even if
    # we have missing data. We will get either a partly complete
    # address, or None if all the values are missing.x
    line1 = filter(None, [street, house_number])
    line2 = filter(None, [city, state])
    line3 = filter(None, [postal_code, country])
    line1 = " ".join(line1).strip()
    line2 = " ".join(line2).strip()
    line3 = ", ".join(line3).strip()
    full_address = filter(None, [line1, line2, line3])
    full_address = ", ".join(full_address).strip()
    return full_address if len(full_address) > 0 else None

class OsmCache:
    def __init__(self, filename="/tmp/osm-exif.json"):
        self.filename = filename
        self.data = {}

        # Load existing cache if it exists
        try:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            pass

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def get_or_set(self, key, func_to_call):
        """
        Retrieve the value from the cache for a given key. If the key is not found,
        call `func_to_call` to generate the value and store it in the cache.

        :param key: The key to look up or set in the cache
        :param func_to_call: A callable function that returns the value if key is missing
        :return: The cached or generated value
        """
        if key not in self.data:
            value = func_to_call()
            self.set(key, value)
        return self.data[key]

    def clear_cache(self):
        """
        Clear all entries from the cache.
        """
        self.data.clear()
        try:
            # Erase contents of the cache file.
            with open(self.filename, 'w'):
                pass
        except FileNotFoundError:
            pass

class OsmSearcher:
    def __init__(self, valves):
        self.valves = valves
        self.cache = OsmCache()

    def create_headers(self) -> Optional[dict]:
        if len(self.valves.user_agent) == 0 or len(self.valves.from_header) == 0:
            return None

        return {
            'User-Agent': self.valves.user_agent,
            'From': self.valves.from_header
        }

    async def nominatim_search(self, query, format="json", limit: int=1) -> Optional[dict]:
        cache_key = f"nominatim_search_{query}"
        cache = OsmCache()
        data = cache.get(cache_key)

        if data:
            print(f"[EXIF-OSM] Got nominatim search data for {query} from cache!")
            return data[:limit]

        print(f"[EXIF-OSM] Searching Nominatim for: {query}")

        url = urljoin(self.valves.nominatim_url, "search")
        params = {
            'q': query,
            'format': format,
            'addressdetails': 1,
            'limit': limit,
        }

        headers = self.create_headers()
        if not headers:
            print("[EXIF-OSM] WARNING: Headers not set. Returning GPS coordinates.")
            return f"{lat},{lon}"

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()

            if not data:
                raise ValueError(f"No results found for query '{query}'")

            print(f"Got result from Nominatim for: {query}")
            cache.set(cache_key, data)
            return data[:limit]
        else:
            print(response.text)
            return None

    async def reverse_geocode(self, lat, lon):
        try:
            nominatim_result = await self.nominatim_search(f"{lat},{lon}", limit=1)
        except ValueError:
            nominatim_result = []

        if not nominatim_result or len(nominatim_result) == 0:
            return f"{lat},{lon}"

        nominatim_result = nominatim_result[0]
        place_name = None

        if 'address' in nominatim_result:
            addr = parse_nominatim_address(nominatim_result['address'])
            if addr is not None:
                place_name = addr

        if place_name is None and 'display_name' in nominatim_result:
            place_name = ",".join(nominatim_result['display_name'].split(",")[:3])

        if place_name is None:
            print(f"[EXIF-OSM] WARNING: Could not find display name for coords: {lat},{lon}")
            place_name = f"{lat},{lon}"

        return place_name

def convert_to_decimal(tags, gps_tag, gps_ref_tag):
    if gps_tag not in tags or gps_ref_tag not in tags:
        return None

    values = tags[gps_tag].values
    ref = tags[gps_ref_tag].values[0]

    degrees = sum(
        values[i].numerator / values[i].denominator * (1/(60**i))
        for i in range(3)
    )

    return -degrees if (ref == 'W' and gps_tag == 'GPSLongitude') or \
                     (ref == 'S' and gps_tag == 'GPSLatitude') else degrees

def extract_gps(img_bytes):
    try:
        f = BytesIO(img_bytes)
        tags = process_file(f, strict=False)
        lat = convert_to_decimal(tags, 'GPS GPSLatitude', 'GPS GPSLatitudeRef')
        lon = convert_to_decimal(tags, 'GPS GPSLongitude', 'GPS GPSLongitudeRef')

        if lon is not None and lat is not None:
            return (round(lat, 4), round(lon, 4))
        else:
            return None
    except Exception as e:
        print(f"[EXIF-OSM] WARNING: Could not load image for GPS processing: {e}")
        return None

def exif_instructions(geocoding, user_image_count):
    if geocoding:
        return valid_instructions(geocoding, user_image_count)
    else:
        return invalid_instructions()

def valid_instructions(geocoding, user_image_count):
    lat = geocoding.get("lat", "unknown")
    lon = geocoding.get("lon", "unknown")
    place_name = geocoding.get("place", None)

    if place_name:
        place_inst = f"The location (accurate to radius of 5 to 10 meters) is: {place_name}"
    else:
        place_inst = "The name of the location could not be determined"

    count_inst = (f"There are {user_image_count} images from the user in this chat. "
                  f"The most recent image is image number {user_image_count}.")

    if place_name:
        osm_link = f"https://www.openstreetmap.org/#map=16/{lat}/{lon}"
        osm_inst = f"The location can be viewed on OpenStreetMap: {osm_link}"
    else:
        osm_inst = ""

    return (f"\n\nYou have access to GPS location information about the "
            f"most recent image in this chat. The image's GPS coordinates "
            f"are: {lat},{lon}. {place_inst}. {osm_inst}"
            f"\n\nThis applies to ONLY the most recent image in the chat. {count_inst}")

def invalid_instructions():
    return ("\n\nThe most recent image in this chat does not have any EXIF location "
            "data. If the user asks about the location of the most recent image, "
            "pleasantly and helpfully inform them that the image does not have "
            "EXIF location data, and thus you cannot determine its location. "
            "Make sure to otherwise answer the user's query.")

def count_user_images(messages):
    count = 0
    for message in reversed(messages):
        if message["role"] == "user":
            message_content = message.get("content")
            if message_content is not None and isinstance(message_content, list):
                has_images = any(
                    item.get("type") == "image_url" for item in message_content
                )

                if has_images:
                    count += 1

    return count

def get_most_recent_user_message_with_image(messages):
    for message in reversed(messages):
        if message["role"] == "user":
            message_content = message.get("content")
            if message_content is not None and isinstance(message_content, list):
                has_images = any(
                    item.get("type") == "image_url" for item in message_content
                )

                if has_images:
                    return message

    return None

class Filter:
    class Valves(BaseModel):
        user_agent: str = Field(
            default="", description="Unique user agent to identify your OSM API requests."
        )
        from_header: str = Field(
            default="", description="Email address to identify your OSM requests."
        )
        nominatim_url: str = Field(
            default="https://nominatim.openstreetmap.org/",
            description="URL of OSM Nominatim API for reverse geocoding (address lookup)."
        )

        pass

    def __init__(self):
        self.valves = self.Valves()

    async def process_image(self, image_data_url):
        base64_img = image_data_url.split(',', maxsplit=1)[1]
        img_bytes = b64decode(base64_img)
        coords = extract_gps(img_bytes)
        if coords:
            searcher = OsmSearcher(self.valves)
            geocoded_name = await searcher.reverse_geocode(coords[0], coords[1])
            return { "lat": coords[0], "lon": coords[1], "place": geocoded_name }
        else:
            return None

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages")
        if messages is None:
            return body

        user_message = get_most_recent_user_message_with_image(messages)
        if user_message is None:
            return body

        user_message_content = user_message.get("content")
        if user_message_content is not None and isinstance(user_message_content, list):
            first_image = next(
                (item for item in user_message_content if item.get("type") == "image_url"),
                {}
            )

            data_url = first_image.get('image_url', {}).get('url', None)

            if data_url and data_url.startswith("data:"):
                geocoding = await self.process_image(data_url)
                user_image_count = count_user_images(messages)
                instructions = exif_instructions(geocoding, user_image_count)
                add_or_update_system_message(instructions, messages)
                body["messages"] = messages

        return body
