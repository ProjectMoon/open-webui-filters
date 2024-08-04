"""
title: OpenStreetMap Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.1.0
license: AGPL-3.0+
required_open_webui_version: 0.3.9
"""
import json
import math
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

VALVES_NOT_SET = """
Tell the user that the User-Agent and From headers
must be set to comply with the OSM Nominatim terms
of use: https://operations.osmfoundation.org/policies/nominatim/
""".replace("\n", " ").strip()

NO_RESULTS = ("No results found. Tell the user you found no results. "
              "Do not make up answers or hallucinate. Only say you "
              "found no results.")

def get_bounding_box_center(bbox):
    def convert(bbox, key):
        return bbox[key] if isinstance(bbox[key], float) else float(bbox[key])

    min_lat = convert(bbox, 'min_lat')
    min_lon = convert(bbox, 'min_lon')
    max_lat = convert(bbox, 'max_lat')
    max_lon = convert(bbox, 'max_lon')

    return {
        'lon': (min_lon + max_lon) / 2,
        'lat': (min_lat + max_lat) / 2
    }


def haversine_distance(point1, point2):
    R = 6371  # Earth radius in kilometers

    lat1, lon1 = point1['lat'], point1['lon']
    lat2, lon2 = point2['lat'], point2['lon']

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def sort_by_closeness(origin, points):
    """
    Sorts a list of { lat, lon }-like dicts by closeness to an origin point.
    The origin is a dict with keys of { lat, lon }. This function adds the
    distance as a dict value to the points.
    """
    points_with_distance = [(point, haversine_distance(origin, point)) for point in points]
    points_with_distance = sorted(points_with_distance, key=lambda pwd: pwd[1])
    for point, distance in points_with_distance:
        point['distance'] = distance
    return [point for point, distance in points_with_distance]



class OsmSearcher:
    def __init__(self, valves):
        self.valves = valves

    def create_headers(self) -> Optional[dict]:
        if len(self.valves.user_agent) == 0 or len(self.valves.from_header) == 0:
            return None

        return {
            'User-Agent': self.valves.user_agent,
            'From': self.valves.from_header
        }

    @staticmethod
    def group_tags(tags):
        result = {}
        for tag in tags:
            key, value = tag.split('=')
            if key not in result:
                result[key] = []
            result[key].append(value)
        return result

    @staticmethod
    def fallback(nominatim_result):
        """
        If we do not have Overpass Turbo results, attempt to use the
        Nominatim result instead.
        """
        return ([nominatim_result] if 'type' in nominatim_result
                and (nominatim_result['type'] == 'amenity'
                     or nominatim_result['type'] == 'shop'
                     or nominatim_result['type'] == 'tourism')
                else [])


    def nominatim_search(self, query, format="json") -> Optional[dict]:
        url = self.valves.nominatim_url
        params = {
            'q': query,
            'format': format,
            'addressdetails': 1,
            'limit': 1,  # We only need the first result for the bounding box
        }

        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()

            if not data:
                raise ValueError(f"No results found for query '{query}'")

            return data[0]
        else:
            print(response.text)
            return None


    def overpass_search(self, place, tags, bbox, limit=5, radius=4000):
        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        url = self.valves.overpass_turbo_url
        center = get_bounding_box_center(bbox)
        around = f"(around:{radius},{center['lat']},{center['lon']})"

        tag_groups = OsmSearcher.group_tags(tags)
        search_groups = [f'"{tag_type}"~"{"|".join(values)}"'
                         for tag_type, values in tag_groups.items()]

        # only search nodes, which are guaranteed to have a lat and lon.
        searches = []
        for search_group in search_groups:
            searches.append(
                f'node[{search_group}]{around}'
            )

        search = ";\n".join(searches)
        if len(search) > 0:
            search += ";"

        query = f"""
            [out:json];
            (
                {search}
            );
            out qt;
        """
        print(query)
        data = { "data": query }
        response = requests.get(url, params=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.text)
            raise Exception(f"Error calling Overpass API: {response.text}")


    def search_nearby(self, place: str, tags: List[str], limit: int=5) -> str:
        headers = self.create_headers()
        if not headers:
            return VALVES_NOT_SET

        try:
            nominatim_result = self.nominatim_search(place)
            if nominatim_result:
                bbox = {
                    'min_lat': nominatim_result['boundingbox'][0],
                    'max_lat': nominatim_result['boundingbox'][1],
                    'min_lon': nominatim_result['boundingbox'][2],
                    'max_lon': nominatim_result['boundingbox'][3]
                }

                overpass_result  = self.overpass_search(place, tags, bbox, limit)

                # use results from overpass, but if they do not exist,
                # fall back to the nominatim result. we can get away
                # with this because we're not diggin through the
                # objects themselves (as long as they have lat/lon, we
                # are good).
                things_nearby = (overpass_result ['elements']
                                 if 'elements' in overpass_result
                                 and len(overpass_result ['elements']) > 0
                                 else OsmSearcher.fallback(nominatim_result))

                origin = get_bounding_box_center(bbox)
                things_nearby = sort_by_closeness(origin, things_nearby)
                things_nearby = things_nearby[:limit]

                if not things_nearby or len(things_nearby) == 0:
                    return NO_RESULTS

                tag_type_str = ", ".join(tags)
                example_link = "https://www.openstreetmap.org/#map=19/<lat>/<lon>"

                return (
                    f"These are some of the {tag_type_str} points of interest nearby. "
                    "When telling the user about them, make sure to report "
                    "all the information (address, contact info, website, etc).\n\n"
                    "Tell the user about ALL the results, and give the CLOSEST result "
                    "first. The results are ordered by closeness. "
                    "Make friendly human-readable OpenStreetMap links when possible, "
                    "by using the latitude and longitude of the amenities: "
                    f"{example_link}\n\n"
                    "Give map links friendly, contextual labels. Don't just print "
                    f"the naked link:\n"
                    f' - Example: You can view it on [OpenStreetMap]({example_link})'
                    f' - Example: Here it is on [OpenStreetMap]({example_link})'
                    f' - Example: You can find it on [OpenStreetMap]({example_link})'
                    "\n\nAnd so on.\n\n"
                    "Only use relevant results. If there are no relevant results, "
                    "say so. Do not make up answers or hallucinate."
                    f"The results are below.\n\n"
                    "----------"
                    f"\n\n{str(things_nearby)}"
                )
            else:
                return NO_RESULTS
        except ValueError:
            return NO_RESULTS
        except Exception as e:
            print(e)
            return (f"No results were found, because of an error. "
                    f"Tell the user that there was an error finding results. "
                    f"The error was: {e}")



class Tools:
    class Valves(BaseModel):
        user_agent: str = Field(
            default="", description="Unique user agent to identify your OSM API requests."
        )
        from_header: str = Field(
            default="", description="Email address to identify your OSM requests."
        )
        nominatim_url: str = Field(
            default="https://nominatim.openstreetmap.org/search",
            description="URL of OSM Nominatim API for reverse geocoding (address lookup)."
        )
        overpass_turbo_url: str = Field(
            default="https://overpass-api.de/api/interpreter",
            description="URL of Overpass Turbo API for searching OpenStreetMap."
        )
        pass

    class UserValves(BaseModel):
        pass


    def __init__(self):
        self.valves = self.Valves()

    def lookup_location(self, address_or_place: str) -> str:
        """
        Looks up GPS and address details on OpenStreetMap of a given address or place.
        :param address_or_place: The address or place to look up.
        :return: Address details, if found. None if there's an error.
        """
        searcher = OsmSearcher(self.valves)
        try:
            result = searcher.nominatim_search(address_or_place)
            if result:
                return str(result)
            else:
                return NO_RESULTS
        except Exception as e:
            print(e)
            return (f"There are no results due to an error. "
                    "Tell the user that there was an error. "
                    f"The error was: {e}. "
                    f"Tell the user the error message.")


    def find_grocery_stores_near_place(self, place: str) -> str:
        """
        Finds supermarkets, grocery stores, and other food stores on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address, which will be sent to Nominatim.
        :return: A list of nearby grocery stores or supermarkets, if found.
        """
        searcher = OsmSearcher(self.valves)
        tags = ["shop=supermarket", "shop=grocery", "shop=convenience", "shop=greengrocer"]
        return searcher.search_nearby(place, tags, limit=5)

    def find_bakeries_near_place(self, place: str) -> str:
        """
        Finds bakeries on OpenStreetMap near a given place or
        address. The location of the address or place is reverse
        geo-coded, then nearby results are fetched from OpenStreetMap.
        :param place: The name of a place or an address, which will be sent to Nominatim.
        :return: A list of nearby bakeries, if found.
        """
        searcher = OsmSearcher(self.valves)
        tags = ["shop=bakery"]
        return searcher.search_nearby(place, tags, limit=5)

    def find_food_near_place(self, place: str) -> str:
        """
        Finds restaurants, fast food, bars, breweries, pubs, etc on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address, which will be sent to Nominatim.
        :return: A list of nearby restaurants, eateries, etc, if found.
        """
        tags = [
            "amenity=restaurant",
            "amenity=fast_food",
            "amenity=cafe",
            "amenity=pub",
            "amenity=bar",
            "amenity=eatery",
            "amenity=biergarten",
            "amenity=canteen"
        ]
        searcher = OsmSearcher(self.valves)
        return searcher.search_nearby(place, tags, limit=5)


    def find_place_of_worship_near_place(self, place: str) -> str:
        """
        Finds places of worship (churches, mosques, temples, etc) on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address, which will be sent to Nominatim.
        :return: A list of nearby places of worship, if found.
        """
        tags = ["amenity=place_of_worship"]
        searcher = OsmSearcher(self.valves)
        return searcher.search_nearby(place, tags, limit=5)


    def find_accommodation_near_place(self, place: str) -> str:
        """
        Finds accommodation (hotels, guesthouses, hostels, etc) on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address, which will be sent to Nominatim.
        :return: A list of nearby accommodation, if found.
        """
        tags = [
            "tourism=hotel", "tourism=chalet", "tourism=guest_house", "tourism=guesthouse",
            "tourism=motel", "tourism=hostel"
        ]
        searcher = OsmSearcher(self.valves)
        return searcher.search_nearby(place, tags, limit=5)
