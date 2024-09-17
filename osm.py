"""
title: OpenStreetMap Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 0.5.0
license: AGPL-3.0+
required_open_webui_version: 0.3.21
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

NO_CONFUSION = ("**IMPORTANT!:** Check that the results match the location "
                "the user is talking about, by analyzing the conversation history. "
                "Sometimes there are places with the same "
                "names, but in different cities or countries. If the results are for "
                "a different city or country than the user is interested in, say so: "
                "tell the user that the results are for the wrong place, and tell them "
                "to be more specific in their query.")

# Give examples of OSM links to help prevent wonky generated links
# with correct GPS coords but incorrect URLs.
EXAMPLE_OSM_LINK = "https://www.openstreetmap.org/#map=19/<lat>/<lon>"
OSM_LINK_INSTRUCTIONS = (
    "Make friendly human-readable OpenStreetMap links when possible, "
    "by using the latitude and longitude of the amenities: "
    f"{EXAMPLE_OSM_LINK}\n\n"
)

def detailed_instructions(tag_type_str: str) -> str:
    """
    Produce detailed instructions for models good at following
    detailed instructions.
    """
    return (
        "# Detailed Search Result Instructions\n"
        f"These are some of the {tag_type_str} points of interest nearby. "
        "These are the results known to be closest to the requested location. "
        "When telling the user about them, make sure to report "
        "all the information (address, contact info, website, etc).\n\n"
        "Tell the user about ALL the results, and give the CLOSEST result "
        "first. The results are ordered by closeness. "
        f"{OSM_LINK_INSTRUCTIONS}"
        "Give map links friendly, contextual labels. Don't just print "
        f"the naked link:\n"
        f' - Example: You can view it on [OpenStreetMap]({EXAMPLE_OSM_LINK})\n'
        f' - Example: Here it is on [OpenStreetMap]({EXAMPLE_OSM_LINK})\n'
        f' - Example: You can find it on [OpenStreetMap]({EXAMPLE_OSM_LINK})\n'
        "\n\nAnd so on.\n\n"
        "Only use relevant results. If there are no relevant results, "
        "say so. Do not make up answers or hallucinate. "
        f"\n\n{NO_CONFUSION}\n\n"
        "The primary results are below. "
        "Remember that the CLOSEST result is first, and you should use "
        "that result first. "
        "Prioritize OSM **nodes** over **ways** and **relations**.\n\n"
        "The results (if present) are below, in Markdown format."
    )

def simple_instructions(tag_type_str: str) -> str:
    """
    Produce simpler markdown-oriented instructions for models that do
    better with that.
    """
    return (
        "# OpenStreetMap Result Instructions\n"
        f"These are some of the {tag_type_str} points of interest nearby. "
        "These are the results known to be closest to the requested location. "
        "For each result, report the following information: \n"
        " - Name\n"
        " - Address\n"
        " - OpenStreetMap Link (make it a human readable link like 'View on OpenStreetMap')\n"
        " - Contact information (address, phone, website, email, etc)\n\n"
        "Tell the user about ALL the results, and give the CLOSEST result "
        "first. The results are ordered by closeness. "
        "Only use relevant results. If there are no relevant results, "
        "say so. Do not make up answers or hallucinate. "
        "Make sure that your results are in the actual location the user is talking about, "
        "and not a place of the same name in a different country."
        "The primary results are below. "
        "Prioritize OSM **nodes** over **ways** and **relations**."
    )

def way_has_info(way):
    """
    Determine if an OSM way entry is useful to us. This means it
    has something more than just its main classification tag, and
    has at least a name.
    """
    return len(way['tags']) > 1 and any('name' in tag for tag in way['tags'])

def process_way_result(way) -> Optional[dict]:
    """
    Post-process an OSM Way dict to remove the geometry and node
    info, and calculate a single GPS coordinate from its bounding
    box.
    """
    if 'nodes' in way:
        del way['nodes']

    if 'geometry' in way:
        del way['geometry']

    if 'bounds' in way:
        way_center = get_bounding_box_center(way['bounds'])
        way['lat'] = way_center['lat']
        way['lon'] = way_center['lon']
        del way['bounds']
        return way

    return None

def get_bounding_box_center(bbox):
    def convert(bbox, key):
        return bbox[key] if isinstance(bbox[key], float) else float(bbox[key])

    min_lat = convert(bbox, 'minlat')
    min_lon = convert(bbox, 'minlon')
    max_lat = convert(bbox, 'maxlat')
    max_lon = convert(bbox, 'maxlon')

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

def parse_thing_address(tags: dict) -> Optional[str]:
    house_number = get_or_none(tags, "addr:housenumber", "addr:house_number")
    street = get_or_none(tags, "addr:street")
    city = get_or_none(tags, "addr:city")
    state = get_or_none(tags, "addr:state", "addr:province")
    postal_code = get_or_none(
        tags,
        "addr:postcode", "addr:post_code", "addr:postal_code",
        "addr:zipcode", "addr:zip_code"
    )

    # if all are none, that means we don't know the address at all.
    if all_are_none(house_number, street, city, state, postal_code):
        return None

    # Handle missing values to create complete-ish addresses, even if
    # we have missing data. We will get either a partly complete
    # address, or None if all the values are missing.
    line1 = filter(None, [street, house_number])
    line2 = filter(None, [city, state, postal_code])
    line1 = " ".join(line1).strip()
    line2 = " ".join(line2).strip()
    full_address = filter(None, [line1, line2])
    full_address = ", ".join(full_address).strip()
    return full_address if len(full_address) > 0 else None

def parse_and_validate_thing(thing: dict) -> Optional[dict]:
    """
    Parse an OSM result (node or post-processed way) and make it
    more friendly to work with. Helps remove ambiguity of the LLM
    interpreting the raw JSON data. If there is not enough data,
    discard the result.
    """
    tags: dict = thing['tags'] if 'tags' in thing else {}

    # Currently we define "enough data" as at least having lat, lon,
    # and name.
    if 'lat' not in thing or 'lon' not in thing or 'name' not in tags:
        return None

    friendly_thing = {}
    address: string = parse_thing_address(tags)
    distance: Optional[float] = thing['distance'] if 'distance' in thing else None
    name: str = tags['name'] if 'name' in tags else str(thing['id'])
    lat: Optional[float] = thing['lat'] if 'lat' in thing else None
    lon: Optional[float] = thing['lon'] if 'lon' in thing else None

    amenity_type: Optional[str] = (
        tags['amenity'] if 'amenity' in tags else None
    )

    friendly_thing['name'] = name if name else "unknown"
    friendly_thing['distance'] = "{:.3f}".format(distance) if distance else "unknown"
    friendly_thing['address'] = address if address else "unknown"
    friendly_thing['lat'] = lat if lat else "unknown"
    friendly_thing['lon'] = lon if lon else "unknown"
    friendly_thing['amenity_type'] = amenity_type if amenity_type else "unknown"
    return friendly_thing

def create_osm_link(lat, lon):
    return EXAMPLE_OSM_LINK.replace("<lat>", str(lat)).replace("<lon>", str(lon))

def convert_and_validate_results(
    original_location: str,
    things_nearby: List[dict]
) -> Optional[str]:
    """
    Converts the things_nearby JSON into Markdown-ish results to
    (hopefully) improve model understanding of the results. Intended
    to stop misinterpretation of GPS coordinates when creating map
    links. Also drops incomplete results.
    """
    entries = []
    for thing in things_nearby:
        # Convert to friendlier data, drop results without names etc.
        # No need to instruct LLM to generate map links if we do it
        # instead.
        friendly_thing = parse_and_validate_thing(thing)
        if not friendly_thing:
            continue

        map_link = create_osm_link(friendly_thing['lat'], friendly_thing['lon'])
        entry = (f"## {friendly_thing['name']}\n"
                 f" - Latitude: {friendly_thing['lat']}\n"
                 f" - Longitude: {friendly_thing['lon']}\n"
                 f" - Address: {friendly_thing['address']}\n"
                 f" - Amenity Type: {friendly_thing['amenity_type']}\n"
                 f" - Distance from Origin: {friendly_thing['distance']} km\n"
                 f" - OpenStreetMap link: {map_link}\n\n"
                 f"Raw JSON data:\n"
                 "```json\n"
                 f"{str(thing)}\n"
                 "```")

        entries.append(entry)

    if len(entries) == 0:
        return None

    result_text = "\n\n".join(entries)
    header = ("# Search Results\n"
              f"Ordered by closeness to {original_location}.")

    return f"{header}\n\n{result_text}"


class OsmSearcher:
    def __init__(self, valves: dict, user_valves: Optional[dict]):
        self.valves = valves
        self.user_valves = user_valves

    def create_headers(self) -> Optional[dict]:
        if len(self.valves.user_agent) == 0 or len(self.valves.from_header) == 0:
            return None

        return {
            'User-Agent': self.valves.user_agent,
            'From': self.valves.from_header
        }

    def use_detailed_interpretation_mode(self) -> bool:
        """Let user valve for instruction mode override the global setting."""
        print(str(self.user_valves))
        if self.user_valves:
            return self.user_valves.instruction_oriented_interpretation
        else:
            return self.valves.instruction_oriented_interpretation

    def get_result_instructions(self, tag_type_str: str) -> str:
        if self.use_detailed_interpretation_mode():
            return detailed_instructions(tag_type_str)
        else:
            return simple_instructions(tag_type_str)

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
                     or nominatim_result['type'] == 'leisure'
                     or nominatim_result['type'] == 'tourism')
                else [])


    def nominatim_search(self, query, format="json", limit: int=1) -> Optional[dict]:
        url = self.valves.nominatim_url
        params = {
            'q': query,
            'format': format,
            'addressdetails': 1,
            'limit': limit,
        }

        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()

            if not data:
                raise ValueError(f"No results found for query '{query}'")

            return data[:limit]
        else:
            print(response.text)
            return None


    def overpass_search(
            self, place, tags, bbox, limit=5, radius=4000
    ) -> (List[dict], List[dict]):
        """
        Return a list relevant of OSM nodes and ways. Some
        post-processing is done on ways in order to add coordinates to
        them.
        """
        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        url = self.valves.overpass_turbo_url
        center = get_bounding_box_center(bbox)
        around = f"(around:{radius},{center['lat']},{center['lon']})"

        tag_groups = OsmSearcher.group_tags(tags)
        search_groups = [f'"{tag_type}"~"{"|".join(values)}"'
                         for tag_type, values in tag_groups.items()]

        searches = []
        for search_group in search_groups:
            searches.append(
                f'nwr[{search_group}]{around}'
            )

        search = ";\n".join(searches)
        if len(search) > 0:
            search += ";"

        # "out geom;" is needed to get bounding box info of ways,
        # so we can calculate the coordinates.
        query = f"""
            [out:json];
            (
                {search}
            );
            out geom;
        """

        print(query)
        data = { "data": query }
        response = requests.get(url, params=data, headers=headers)
        if response.status_code == 200:
            # nodes have have exact GPS coordinates. we also include
            # useful way entries, post-processed to remove extra data
            # and add a centered calculation of their GPS coords. any
            # way that doesn't have enough info for us to use is
            # dropped.
            results = response.json()
            results = results['elements'] if 'elements' in results else []
            nodes = []
            ways = []

            for res in results:
                if 'type' not in res:
                    continue
                if res['type'] == 'node':
                    nodes.append(res)
                elif res['type'] == 'way' and way_has_info(res):
                    ways.append(process_way_result(res))

            return nodes, ways
        else:
            print(response.text)
            raise Exception(f"Error calling Overpass API: {response.text}")


    def search_nearby(self, place: str, tags: List[str], limit: int=5, radius: int=4000) -> str:
        headers = self.create_headers()
        if not headers:
            return VALVES_NOT_SET

        try:
            nominatim_result = self.nominatim_search(place, limit=1)
            if nominatim_result:
                nominatim_result = nominatim_result[0]
                bbox = {
                    'minlat': nominatim_result['boundingbox'][0],
                    'maxlat': nominatim_result['boundingbox'][1],
                    'minlon': nominatim_result['boundingbox'][2],
                    'maxlon': nominatim_result['boundingbox'][3]
                }

                nodes, ways  = self.overpass_search(place, tags, bbox, limit, radius)

                # use results from overpass, but if they do not exist,
                # fall back to the nominatim result. this may or may
                # not be a good idea.
                things_nearby = (nodes + ways
                                 if len(nodes) > 0 or len(ways) > 0
                                 else OsmSearcher.fallback(nominatim_result))

                origin = get_bounding_box_center(bbox)
                things_nearby = sort_by_closeness(origin, things_nearby)
                things_nearby = things_nearby[:limit] # drop down to requested limit
                search_results = convert_and_validate_results(place, things_nearby)

                if not things_nearby or len(things_nearby) == 0:
                    return NO_RESULTS

                tag_type_str = ", ".join(tags)

                # Only print the full result instructions if we
                # actually have something.
                if search_results:
                    result_instructions = self.get_result_instructions(tag_type_str)
                else:
                    result_instructions = ("No results found at all. "
                                           "Tell the user there are no results.")

                resp = (
                    f"{result_instructions}\n\n"
                    f"{search_results}"
                )

                print(resp)
                return resp
            else:
                return NO_RESULTS
        except ValueError:
            return NO_RESULTS
        except Exception as e:
            print(e)
            return (f"No results were found, because of an error. "
                    f"Tell the user that there was an error finding results. "
                    f"The error was: {e}")


def do_osm_search(valves, user_valves, place, tags, limit=5, radius=4000):
    searcher = OsmSearcher(valves, user_valves)
    return searcher.search_nearby(place, tags, limit=limit, radius=radius)

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
        instruction_oriented_interpretation: bool = Field(
            default=True,
            description=("Give detailed result interpretation instructions to the model. "
                         "Switch this off if results are inconsistent, wrong, or missing.")
        )
        pass

    class UserValves(BaseModel):
        instruction_oriented_interpretation: bool = Field(
            default=True,
            description=("Give detailed result interpretation instructions to the model. "
                         "Switch this off if results are inconsistent, wrong, or missing.")
        )
        pass


    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = None

    def lookup_location(self, address_or_place: str) -> str:
        """
        Looks up GPS and address details on OpenStreetMap of a given address or place.
        :param address_or_place: The address or place to look up.
        :return: Address details, if found. None if there's an error.
        """
        searcher = OsmSearcher(self.valves, self.user_valves)
        try:
            result = searcher.nominatim_search(address_or_place, limit=5)
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


    def find_grocery_stores_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds supermarkets, grocery stores, and other food stores on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby grocery stores or supermarkets, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=supermarket", "shop=grocery", "shop=convenience", "shop=greengrocer"]
        return do_osm_search(self.valves, user_valves, place, tags)


    def find_bakeries_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds bakeries on OpenStreetMap near a given place or
        address. The location of the address or place is reverse
        geo-coded, then nearby results are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby bakeries, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=bakery"]
        return do_osm_search(self.valves, user_valves, place, tags)

    def find_food_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds restaurants, fast food, bars, breweries, pubs, etc on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby restaurants, eateries, etc, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
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
        return do_osm_search(self.valves, user_valves, place, tags)


    def find_swimming_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds swimming pools, water parks, swimming areas, and other aquatic
        activities on OpenStreetMap near a given place or address. The location
        of the address or place is reverse geo-coded, then nearby results are fetched
        from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of swimming poools or places, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["leisure=swimming_pool", "leisure=swimming_area",
                "leisure=water_park", "tourism=theme_park"]
        return do_osm_search(self.valves, user_valves, place, tags, radius=10000)


    def find_recreation_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds playgrounds, theme parks, frisbee golf, ice skating, and other recreational
        activities on OpenStreetMap near a given place or address. The location
        of the address or place is reverse geo-coded, then nearby results are fetched
        from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of recreational places, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["leisure=horse_riding", "leisure=ice_rink", "leisure=park",
                "leisure=playground", "leisure=disc_golf_course",
                "leisure=amusement_arcade", "tourism=theme_park"]
        return do_osm_search(self.valves, user_valves, place, tags, limit=10, radius=10000)


    def find_place_of_worship_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds places of worship (churches, mosques, temples, etc) on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby places of worship, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=place_of_worship"]
        return do_osm_search(self.valves, user_valves, place, tags)


    def find_accommodation_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds accommodation (hotels, guesthouses, hostels, etc) on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby accommodation, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = [
            "tourism=hotel", "tourism=chalet", "tourism=guest_house", "tourism=guesthouse",
            "tourism=motel", "tourism=hostel"
        ]
        return do_osm_search(self.valves, user_valves, place, tags, limit=10, radius=10000)

    def find_alcohol_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds beer stores, liquor stores, and similar shops on OpenStreetMap
        near a given place or address. The location of the address or place is
        reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby alcohol shops, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=alcohol"]
        return do_osm_search(self.valves, user_valves, place, tags)

    def find_drugs_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds cannabis dispensaries, coffeeshops, smartshops, and similar stores on OpenStreetMap
        near a given place or address. The location of the address or place is
        reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby cannabis and smart shops, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=coffeeshop", "shop=cannabis", "shop=headshop", "shop=smartshop"]
        return do_osm_search(self.valves, user_valves, place, tags)

    def find_schools_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds schools (NOT universities) on OpenStreetMap near a given place or address.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby schools, if found.
        """
        tags = ["amenity=school"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return do_osm_search(self.valves, user_valves, place, tags, limit=10)

    def find_universities_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds universities and colleges on OpenStreetMap near a given place or address.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby schools, if found.
        """
        tags = ["amenity=university", "amenity=college"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return do_osm_search(self.valves, user_valves, place, tags, limit=10)

    def find_libraries_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds libraries on OpenStreetMap near a given place or address.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby libraries, if found.
        """
        tags = ["amenity=library"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return do_osm_search(self.valves, user_valves, place, tags)

    def find_public_transport_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds public transportation stops on OpenStreetMap near a given place or address.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby public transportation stops, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["highway=bus_stop", "amenity=bus_station",
                "railway=station", "railway=halt", "railway=tram_stop",
                "station=subway", "amenity=ferry_terminal",
                "public_transport=station"]
        return do_osm_search(self.valves, user_valves, place, tags, limit=10)

    def find_bike_rentals_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds bike rentals on OpenStreetMap near a given place or address.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby bike rentals, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=bicycle_rental", "amenity=bicycle_library", "service:bicycle:rental=yes"]
        return do_osm_search(self.valves, user_valves, place, tags)

    def find_car_rentals_near_place(self, __user__: dict, place: str) -> str:
        """
        Finds bike rentals on OpenStreetMap near a given place or address.
        :param place: The name of a place or an address. City and country must be specified, if known.
        :return: A list of nearby bike rentals, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=car_rental", "car:rental=yes", "rental=car", "car_rental=yes"]
        return do_osm_search(self.valves, user_valves, place, tags, radius=6000)
