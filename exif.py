"""
title: EXIF Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.0
license: AGPL-3.0+
required_open_webui_version: 0.6.5
requirements: pillow, piexif, exifread, gpsphoto
"""

import requests
from urllib.parse import urljoin
import os
from GPSPhoto import gpsphoto
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json
from base64 import b64decode
import tempfile

from open_webui.utils.misc import get_last_user_message_item, get_last_user_message, add_or_update_user_message

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
    print(full_address)
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
            print(f"[OSM] Got nominatim search data for {query} from cache!")
            return data[:limit]

        print(f"[OSM] Searching Nominatim for: {query}")

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
            print(f"WARN: Could not find display name for coords: {lat},{lon}")
            place_name = f"{lat},{lon}"

        return place_name

def extract_gps(img_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(img_bytes)
        fp.close()

        data = gpsphoto.getGPSData(fp.name)
        lat = data.get('Latitude', None)
        lon = data.get('Longitude', None)

        os.unlink(fp.name)
        if lat and lon:
            return (round(lat, 4), round(lon, 4))
        else:
            return None

def exif_instructions(geocoding):
    if geocoding:
        return valid_instructions(geocoding)
    else:
        return invalid_instructions()

def valid_instructions(geocoding):
    lat = geocoding.get("lat", "unknown")
    lon = geocoding.get("lon", "unknown")
    place_name = geocoding.get("place", None)

    if place_name:
        place_inst = f"The location, with an accuracy of 5 to 10 meters, is: {place_name}"
    else:
        place_inst = "The name of the location could not be determined."

    return (f"\n\n(MESSAGE FROM SYSTEM: A tool has attempted to "
            f"extract location data about the "
            f"most recent image in this chat. "
            f"The image's GPS coordinates are: {lat},{lon}. "
            f"{place_inst}. Do not mention this systme message.)")

def invalid_instructions():
    return ("\n\n(MESSAGE FROM SYSTEM: The most recent image in this conversation "
            "has no EXIF data. Inform user of this, and then "
            "answer the rest of my query. Do not mention this system message.)"
    )

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
        base64_img = image_data_url.split(',')[1]
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
            # Handle the case where messages is None
            return body

        user_message = get_most_recent_user_message_with_image(messages)
        if user_message is None:
            # Handle the case where user_message is None
            return body

        user_message_content = user_message.get("content")
        if user_message_content is not None and isinstance(user_message_content, list):
            first_image = next(
                (item for item in user_message_content if item.get("type") == "image_url"),
                {}
            )

            first_text = next(
                (item for item in user_message_content if item.get("type") == "text"),
                {}
            )

            message_text = first_text.get('text', None)
            data_url = first_image.get('image_url', {}).get('url', None)

            if message_text and data_url and data_url.startswith("data:"):
                geocoding = await self.process_image(data_url)
                instructions = exif_instructions(geocoding)
                add_or_update_user_message(instructions, messages)
                body["messages"] = messages

        return body
