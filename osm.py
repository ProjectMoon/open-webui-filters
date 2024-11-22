"""
title: OpenStreetMap Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 1.2.0
license: AGPL-3.0+
required_open_webui_version: 0.4.3
requirements: openrouteservice, markdown
"""
import itertools
import json
import math
import requests

import markdown
import openrouteservice
from openrouteservice.directions import directions as ors_directions

from urllib.parse import urljoin
from operator import itemgetter
from typing import List, Optional
from pydantic import BaseModel, Field

NOMINATIM_LOOKUP_TYPES = {
    "node": "N",
    "route": "R",
    "way": "W"
}

OLD_VALVE_SETTING = """ Tell the user that you cannot search
OpenStreetMap until the configuration is fixed. The Nominatim URL
valve setting needs to be updated. There has been a breaking change in
1.0 of the OpenStreetMap tool. The valve setting is currently set to:
`{OLD}`.

It shoule be set to the root URL of the Nominatim endpoint, for
example:

`https://nominatim.openstreetmap.org/`

Inform the user they need to fix this configuration setting.
""".replace("\n", " ").strip()

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

def chunk_list(input_list, chunk_size):
    it = iter(input_list)
    return list(
        itertools.zip_longest(*[iter(it)] * chunk_size, fillvalue=None)
    )

def to_lookup(thing) -> Optional[str]:
    lookup_type = NOMINATIM_LOOKUP_TYPES.get(thing['type'])
    if lookup_type is not None:
        return f"{lookup_type}{thing['id']}"

def specific_place_instructions() -> str:
    return (
        "# Result Instructions\n"
        "These are search results ordered by relevance for the "
        "address, place, landmark, or location the user is asking "
        "about. **IMPORTANT!:** Tell the user all relevant information, "
        "including address, contact information, and the OpenStreetMap link. "
        "Make the map link into a nice human-readable markdown link."
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
        "Tell the user about ALL the results, and give closer results "
        "first. Closer results are higher in the list. When telling the "
        "user the distance, use the TRAVEL DISTANCE. Do not say one "
        "distance is farther away than another. Just say what the "
        "distances are. "
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
        "Remember that the CLOSEST result is first, and you should use "
        "that result first.\n\n"
        "The results (if present) are below, in Markdown format.\n\n"
        "**ALWAYS SAY THE CLOSEST RESULT FIRST!**"
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
        "first. The results are ordered by closeness as the crow flies. "
        "When telling the user about distances, use the TRAVEL DISTANCE only. "
        "Only use relevant results. If there are no relevant results, "
        "say so. Do not make up answers or hallucinate. "
        "Make sure that your results are in the actual location the user is talking about, "
        "and not a place of the same name in a different country."
        "The search results are below."
    )

def merge_from_nominatim(thing, nominatim_result) -> Optional[dict]:
    """Merge information into object missing all or some of it."""
    if thing is None:
        return None

    if 'address' not in nominatim_result:
        return None

    nominatim_address = nominatim_result['address']

    # prioritize actual name, road name, then display name. display
    # name is often the full address, which is a bit much.
    nominatim_name = nominatim_result.get('name')
    nominatim_road = nominatim_address.get('road')
    nominatim_display_name = nominatim_result.get('display_name')
    thing_name = thing.get('name')

    if nominatim_name and not thing_name:
        thing['name'] = nominatim_name.strip()
    elif nominatim_road and not thing_name:
        thing['name'] = nominatim_road.strip()
    elif nominatim_display_name and not thing_name:
        thing['name'] = nominatim_display_name.strip()

    tags = thing.get('tags', {})

    for key in nominatim_address:
        obj_key = f"addr:{key}"
        if obj_key not in tags:
            tags[obj_key] = nominatim_address[key]

    thing['tags'] = tags
    return thing

def thing_is_useful(thing):
    """
    Determine if an OSM way entry is useful to us. This means it
    has something more than just its main classification tag, and
    (usually) has at least a name. Some exceptions are made for ways
    that do not have names.
    """
    tags = thing.get('tags', {})
    has_tags = len(tags) > 1
    has_useful_tags = (
        'leisure' in tags or
        'shop' in tags or
        'amenity' in tags or
        'car:rental' in tags or
        'rental' in tags or
        'car_rental' in tags or
        'service:bicycle:rental' in tags
    )
    return has_tags and has_useful_tags

def thing_has_info(thing):
    has_name = any('name' in tag for tag in thing['tags'])
    return thing_is_useful(thing) and has_name
    # is_exception = way['tags'].get('leisure', None) is not None
    # return has_tags and (has_name or is_exception)

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

def sort_by_closeness(origin, points, *keys: str):
    """
    Sorts a list of { lat, lon }-like dicts by closeness to an origin point.
    The origin is a dict with keys of { lat, lon }.
    """
    return sorted(points, key=itemgetter(*keys))

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

def friendly_shop_name(shop_type: str) -> str:
    """
    Make certain shop types more friendly for LLM interpretation.
    """
    if shop_type == "doityourself":
        return "hardware"
    else:
        return shop_type

def parse_thing_address(thing: dict) -> Optional[str]:
    """
    Parse address from either an Overpass result or Nominatim
    result.
    """
    if 'address' in thing:
        # nominatim result
        return parse_address_from_address_obj(thing['address'])
    else:
        return parse_address_from_tags(thing['tags'])

def parse_address_from_address_obj(address) -> Optional[str]:
    """Parse address from Nominatim address object."""
    house_number = get_or_none(address, "house_number")
    street = get_or_none(address, "road")
    city = get_or_none(address, "city")
    state = get_or_none(address, "state")
    postal_code = get_or_none(address, "postcode")

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

def parse_address_from_tags(tags: dict) -> Optional[str]:
    """Parse address from Overpass tags object."""
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

def parse_thing_amenity_type(thing: dict, tags: dict) -> Optional[dict]:
    """
    Extract amenity type or other identifying category from
    Nominatim or Overpass result object.
    """
    if 'amenity' in tags:
        return tags['amenity']

    if thing.get('class') == 'amenity' or thing.get('class') == 'shop':
        return thing.get('type')

    # fall back to tag categories, like shop=*
    if 'shop' in tags:
        return friendly_shop_name(tags['shop'])
    if 'leisure' in tags:
        return friendly_shop_name(tags['leisure'])

    return None

def parse_and_validate_thing(thing: dict) -> Optional[dict]:
    """
    Parse an OSM result (node or post-processed way) and make it
    more friendly to work with. Helps remove ambiguity of the LLM
    interpreting the raw JSON data. If there is not enough data,
    discard the result.
    """
    tags: dict = thing['tags'] if 'tags' in thing else {}

    # Currently we define "enough data" as at least having lat, lon,
    # and a name. nameless things are allowed if they are in a certain
    # class of POIs (leisure).
    has_name = 'name' in tags or 'name' in thing
    is_leisure = 'leisure' in tags or 'leisure' in thing
    if 'lat' not in thing or 'lon' not in thing:
        return None

    if not has_name and not is_leisure:
        return None

    friendly_thing = {}
    name: str = (tags['name'] if 'name' in tags
                 else thing['name'] if 'name' in thing
                 else str(thing['id']) if 'id' in thing
                 else str(thing['osm_id']) if 'osm_id' in thing
                 else "unknown")

    address: str = parse_thing_address(thing)
    distance: Optional[float] = thing.get('distance', None)
    nav_distance: Optional[float] = thing.get('nav_distance', None)

    lat: Optional[float] = thing.get('lat', None)
    lon: Optional[float] = thing.get('lon', None)
    amenity_type: Optional[str] = parse_thing_amenity_type(thing, tags)

    # use the navigation distance if it's present. but if not, set to
    # the haversine distance so that we at least get coherent results
    # for LLM.
    friendly_thing['distance'] = "{:.3f}".format(distance) if distance else "unknown"
    if nav_distance:
        friendly_thing['nav_distance'] = "{:.3f}".format(nav_distance) + " km"
    else:
        friendly_thing['nav_distance'] = f"a bit more than {friendly_thing['distance']}km"

    friendly_thing['name'] = name if name else "unknown"
    friendly_thing['address'] = address if address else "unknown"
    friendly_thing['lat'] = lat if lat else "unknown"
    friendly_thing['lon'] = lon if lon else "unknown"
    friendly_thing['amenity_type'] = amenity_type if amenity_type else "unknown"
    return friendly_thing

def create_osm_link(lat, lon):
    return EXAMPLE_OSM_LINK.replace("<lat>", str(lat)).replace("<lon>", str(lon))

def convert_and_validate_results(
    original_location: str,
    things_nearby: List[dict],
    sort_message: str="closeness",
    use_distance: bool=True
) -> Optional[str]:
    """
    Converts the things_nearby JSON into Markdown-ish results to
    (hopefully) improve model understanding of the results. Intended
    to stop misinterpretation of GPS coordinates when creating map
    links. Also drops incomplete results. Supports Overpass and
    Nominatim results.
    """
    entries = []
    for thing in things_nearby:
        # Convert to friendlier data, drop results without names etc.
        # No need to instruct LLM to generate map links if we do it
        # instead.
        friendly_thing = parse_and_validate_thing(thing)
        if not friendly_thing:
            continue

        distance = (f" - Haversine Distance from Origin: {friendly_thing['distance']} km\n"
                    if use_distance else "")
        travel_distance = (f" - Travel Distance from Origin: {friendly_thing['nav_distance']}\n"
                        if use_distance and 'nav_distance' in friendly_thing else "")
        map_link = create_osm_link(friendly_thing['lat'], friendly_thing['lon'])
        entry = (f"## {friendly_thing['name']}\n"
                 f" - Latitude: {friendly_thing['lat']}\n"
                 f" - Longitude: {friendly_thing['lon']}\n"
                 f" - Address: {friendly_thing['address']}\n"
                 f" - Amenity Type: {friendly_thing['amenity_type']}\n"
                 f"{distance}"
                 f"{travel_distance}"
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
              f"Ordered by {sort_message} to {original_location}.")

    return f"{header}\n\n{result_text}"

class OsmCache:
    def __init__(self, filename="/tmp/osm.json"):
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

class OrsRouter:
    def __init__(
            self, valves, user_valves: Optional[dict], event_emitter=None
    ):
        self.valves = valves
        self.event_emitter = event_emitter
        self.user_valves = user_valves

        if self.valves.ors_api_key is not None and self.valves.ors_api_key != "":
            if self.valves.ors_instance is not None:
                self._client = openrouteservice.Client(
                    base_url=self.valves.ors_instance,
                    key=self.valves.ors_api_key
                )
            else:
                self._client = openrouteservice.Client(key=self.valves.ors_api_key)
        else:
            self._client = None


    def calculate_distance(
            self, from_thing: dict, to_thing: dict
    ) -> Optional[float]:
        """
        Calculate navigation distance between A and B. Returns the
        distance calculated, if successful, or None if the distance
        could not be calculated, or if ORS is not configured.
        """
        if not self._client:
            return None

        # select profile based on distance for more accurate
        # measurements. very close haversine distances use the walking
        # profile, which should (usually?) essentially cover walking
        # and biking. further away = use car.
        if to_thing.get('distance', 9000) <= 1.5:
            profile = "foot-walking"
        else:
            profile = "driving-car"

        coords = ((from_thing['lon'], from_thing['lat']),
                  (to_thing['lon'], to_thing['lat']))

        resp = ors_directions(self._client, coords, profile=profile,
                              preference="fastest", units="km")
        routes = resp.get('routes', [])
        if len(routes) > 0:
            return routes[0].get('summary', {}).get('distance', None)
        else:
            return None


class OsmSearcher:
    def __init__(self, valves, user_valves: Optional[dict], event_emitter=None):
        self.valves = valves
        self.event_emitter = event_emitter
        self.user_valves = user_valves
        self._ors = OrsRouter(valves, user_valves, event_emitter)

    def create_headers(self) -> Optional[dict]:
        if len(self.valves.user_agent) == 0 or len(self.valves.from_header) == 0:
            return None

        return {
            'User-Agent': self.valves.user_agent,
            'From': self.valves.from_header
        }

    async def event_resolving(self, done: bool=False):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        if done:
            message = "OpenStreetMap: resolution complete."
        else:
            message = "OpenStreetMap: resolving..."

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": message,
                "done": done,
            },
        })

    async def event_fetching(self, done: bool=False, message="OpenStreetMap: fetching additional info"):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": message,
                "done": done,
            },
        })

    async def event_searching(
            self, category: str, place: str,
            status: str="in_progress", done: bool=False
    ):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": status,
                "description": f"OpenStreetMap: searching for {category} near {place}",
                "done": done,
            },
        })

    async def event_search_complete(self, category: str, place: str, num_results: int):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "complete",
                "description": f"OpenStreetMap: found {num_results} '{category}' results",
                "done": True,
            },
        })

    async def event_error(self, exception: Exception):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "error",
                "description": f"Error searching OpenStreetMap: {str(exception)}",
                "done": True,
            },
        })

    def calculate_navigation_distance(self, start, destination) -> float:
        """Calculate real distance from A to B, instead of Haversine."""
        return self._ors.calculate_distance(start, destination)

    def attempt_ors(self, origin, things_nearby) -> bool:
        """Update distances to use ORS navigable distances, if ORS enabled."""
        used_ors = False
        cache = OsmCache()
        for thing in things_nearby:
            cache_key = f"ors_{origin}_{thing['id']}"
            nav_distance = cache.get(cache_key)

            if nav_distance:
                print(f"Got nav distance for {thing['id']} from cache!")
            else:
                print(f"Checking ORS for {thing['id']}")
                nav_distance = self.calculate_navigation_distance(origin, thing)

            if nav_distance:
                used_ors = True
                cache.set(cache_key, nav_distance)
                thing['nav_distance'] = nav_distance

        return used_ors

    def calculate_haversine(self, origin, things_nearby):
        for thing in things_nearby:
            if 'distance' not in thing:
                thing['distance'] = haversine_distance(origin, thing)

    def use_detailed_interpretation_mode(self) -> bool:
        # Let user valve for instruction mode override the global
        # setting.
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


    async def nominatim_lookup_by_id(self, things, format="json"):
        await self.event_fetching(done=False)
        updated_things = [] # the things with merged info.

        # handle last chunk, which can have nones in order due to the
        # way chunking is done.
        things = [thing for thing in things if thing is not None]
        lookups = []

        for thing in things:
            if thing is None:
                continue
            lookup = to_lookup(thing)
            if lookup is not None:
                lookups.append(lookup)

        # Nominatim likes it if we cache our data.
        cache = OsmCache()
        lookups_to_remove = []
        for lookup_id in lookups:
            from_cache = cache.get(lookup_id)
            if from_cache is not None:
                updated_things.append(from_cache)
                lookups_to_remove.append(lookup_id)

        # only need to look up things we do not have cached.
        lookups = [id for id in lookups if id not in lookups_to_remove]

        if len(lookups) == 0:
            print("Got all Nominatim info from cache!")
            await self.event_fetching(done=True)
            return updated_things
        else:
            print(f"Looking up {len(lookups)} things from Nominatim")


        url = urljoin(self.valves.nominatim_url, "lookup")
        params = {
            'osm_ids': ",".join(lookups),
            'format': format
        }

        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()

            if not data:
                print("No results found for lookup")
                await self.event_fetching(done=True)
                return []

            addresses_by_id = {item['osm_id']: item for item in data}

            for thing in things:
                nominatim_result = addresses_by_id.get(thing['id'], {})
                if nominatim_result != {}:
                    updated = merge_from_nominatim(thing, nominatim_result)
                    if updated is not None:
                        lookup = to_lookup(thing)
                        cache.set(lookup, updated)
                        updated_things.append(updated)

            await self.event_fetching(done=True)
            return updated_things
        else:
            await self.event_error(Exception(response.text))
            print(response.text)
            return []


    async def nominatim_search(self, query, format="json", limit: int=1) -> Optional[dict]:
        await self.event_resolving(done=False)
        cache_key = f"nominatim_search_{query}"
        cache = OsmCache()
        data = cache.get(cache_key)

        if data:
            print(f"Got nominatim search data for {query} from cache!")
            await self.event_resolving(done=True)
            return data[:limit]

        print(f"Searching Nominatim for: {query}")

        url = urljoin(self.valves.nominatim_url, "search")
        params = {
            'q': query,
            'format': format,
            'addressdetails': 1,
            'limit': limit,
        }

        headers = self.create_headers()
        if not headers:
            await self.event_error("Headers not set")
            raise ValueError("Headers not set")

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            await self.event_resolving(done=True)
            data = response.json()

            if not data:
                raise ValueError(f"No results found for query '{query}'")

            print(f"Got result from Nominatim for: {query}")
            cache.set(cache_key, data)
            return data[:limit]
        else:
            await self.event_error(Exception(response.text))
            print(response.text)
            return None


    async def overpass_search(
            self, place, tags, bbox, limit=5, radius=4000
    ) -> (List[dict], List[dict]):
        """
        Return a list relevant of OSM nodes and ways. Some
        post-processing is done on ways in order to add coordinates to
        them.
        """
        print(f"Searching Overpass Turbo around origin {place}")
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
            things_missing_names = []

            for res in results:
                if 'type' not in res or not thing_is_useful(res):
                    continue
                if res['type'] == 'node':
                    if thing_has_info(res):
                        nodes.append(res)
                    else:
                        things_missing_names.append(res)
                elif res['type'] == 'way':
                    processed = process_way_result(res)
                    if processed is not None and thing_has_info(res):
                        ways.append(processed)
                    else:
                        if processed is not None:
                            things_missing_names.append(processed)

            # attempt to update ways that have no names/addresses.
            if len(things_missing_names) > 0:
                print(f"Updating {len(things_missing_names)} things with info")
                for way_chunk in chunk_list(things_missing_names, 20):
                    updated = await self.nominatim_lookup_by_id(way_chunk)
                    ways = ways + updated

            return nodes, ways
        else:
            print(response.text)
            raise Exception(f"Error calling Overpass API: {response.text}")

    async def get_things_nearby(self, nominatim_result, place, tags, bbox, limit, radius):
        nodes, ways = await self.overpass_search(place, tags, bbox, limit, radius)

        # use results from overpass, but if they do not exist,
        # fall back to the nominatim result. this may or may
        # not be a good idea.
        things_nearby = (nodes + ways
                         if len(nodes) > 0 or len(ways) > 0
                         else OsmSearcher.fallback(nominatim_result))

        # in order to not spam ORS, we first sort by haversine
        # distance and drop number of results to the limit. then, if
        # enabled, we calculate ORS distances. then we sort again.
        origin = get_bounding_box_center(bbox)
        self.calculate_haversine(origin, things_nearby)
        things_nearby = sort_by_closeness(origin, things_nearby, 'distance')
        things_nearby = things_nearby[:limit] # drop down to requested limit

        if self.attempt_ors(origin, things_nearby):
            things_nearby = sort_by_closeness(origin, things_nearby, 'nav_distance', 'distance')
        return things_nearby

    async def search_nearby(
            self, place: str, tags: List[str], limit: int=5, radius: int=4000,
            category: str="POIs"
    ) -> dict:
        headers = self.create_headers()
        if not headers:
            return { "place_display_name": place, "results": VALVES_NOT_SET }

        try:
            nominatim_result = await self.nominatim_search(place, limit=1)
            if not nominatim_result:
                await self.event_search_complete(category, place, 0)
                return { "place_display_name": place, "results": NO_RESULTS }

            nominatim_result = nominatim_result[0]

            # display friendlier searching message if possible
            if 'display_name' in nominatim_result:
                place_display_name = ",".join(nominatim_result['display_name'].split(",")[:3])
            elif 'address' in nominatim_result:
                addr = parse_thing_address(nominatim_result)
                if addr is not None:
                    place_display_name = ",".join(addr.split(",")[:3])
                else:
                    place_display_name = place
            else:
                print(f"WARN: Could not find display name for place: {place}")
                place_display_name = place

            await self.event_searching(category, place_display_name, done=False)

            bbox = {
                'minlat': nominatim_result['boundingbox'][0],
                'maxlat': nominatim_result['boundingbox'][1],
                'minlon': nominatim_result['boundingbox'][2],
                'maxlon': nominatim_result['boundingbox'][3]
            }

            things_nearby = await self.get_things_nearby(nominatim_result, place, tags,
                                                         bbox, limit, radius)

            if not things_nearby or len(things_nearby) == 0:
                await self.event_search_complete(category, place_display_name, 0)
                return { "place_display_name": place, "results": NO_RESULTS }

            tag_type_str = ", ".join(tags)

            # Only print the full result instructions if we
            # actually have something.
            search_results = convert_and_validate_results(place, things_nearby)
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
            await self.event_search_complete(category, place_display_name, len(things_nearby))
            return { "place_display_name": place_display_name, "results": resp }
        except ValueError:
            await self.event_search_complete(category, place_display_name, 0)
            return { "place_display_name": place_display_name, "results": NO_RESULTS }
        except Exception as e:
            print(e)
            await self.event_error(e)
            result = (f"No results were found, because of an error. "
                      f"Tell the user that there was an error finding results. "
                      f"The error was: {e}")
            return { "place_display_name": place_display_name, "results": result }


async def do_osm_search(
        valves, user_valves, place, tags,
        category="POIs", event_emitter=None, limit=5, radius=4000
):
    # handle breaking 1.0 change, in case of old Nominatim valve settings.
    if valves.nominatim_url.endswith("/search"):
        message = "Old Nominatim URL setting still in use!"
        print(f"[OSM] ERROR: {message}")
        if valves.status_indicators and event_emitter is not None:
            await event_emitter({
                "type": "status",
                "data": {
                    "status": "error",
                    "description": f"Error searching OpenStreetMap: {message}",
                    "done": True,
                },
            })
        return OLD_VALVE_SETTING.replace("{OLD}", valves.nominatim_url)

    print(f"[OSM] Searching for [{category}] ({tags[0]}, etc) near place: {place}")
    searcher = OsmSearcher(valves, user_valves, event_emitter)
    search = await searcher.search_nearby(place, tags, limit=limit, radius=radius, category=category)

    place_display_name = search["place_display_name"]
    results = search["results"]

    # send a citation about what we found.
    if valves.status_indicators and event_emitter is not None:
        # we generally assume that category is plural.
        await event_emitter({
            "type": "source",
            "data": {
                "document": [markdown.markdown(results)],
                "metadata": [{"source": "OpenStreetMap", "html": True }],
                "source": {"name": f"{category.title()} near {place_display_name}"},
            }
        })

    return results


class Tools:
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
        overpass_turbo_url: str = Field(
            default="https://overpass-api.de/api/interpreter",
            description="URL of Overpass Turbo API for searching OpenStreetMap."
        )
        instruction_oriented_interpretation: bool = Field(
            default=True,
            description=("Give detailed result interpretation instructions to the model. "
                         "Switch this off if results are inconsistent, wrong, or missing.")
        )
        ors_api_key: Optional[str] = Field(
            default=None,
            description=("Provide an Open Route Service API key to calculate "
                         "more accurate distances (leave default to disable).")
        )
        ors_instance: Optional[str] = Field(
            default=None,
            description="Use a custom ORS instance (leave default to use public ORS instance)."
        )
        status_indicators: bool = Field(
            default=True,
            description=("Emit status update events to the web UI.")
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

    async def find_address_for_coordinates(self, latitude: float, longitude: float, __event_emitter__) -> str:
        """
        Resolves GPS coordinates to a specific address or place.
        :param latitude: The latitude portion of the GPS coordinate.
        :param longitude: The longitude portion of the GPS coordinate.
        :return: Information about the address or place at the coordinates.
        """
        print(f"[OSM] Resolving [{latitude}, {longitude}] to address.")
        return await self.find_specific_place(f"{latitude}, {longitude}", __event_emitter__)

    async def find_store_or_place_near_coordinates(
            self, store_or_business_name: str, latitude: float, longitude: float, __event_emitter__
    ) -> str:
        """
        Finds specifically named stores, businesses, or landmarks near the given
        GPS coordinates. This can be used for general or specific things,
        like 'gas station' or 'grocery store', or the name of a specific
        chain (like 'Wal-Mart' or 'Albert Heijn'). This function is intended
        for finding specific chains or businesses near the given coordinates.
        Use this if the user asks about businesses or places nearby.
        :param store_or_business_name: Name of store or business to look for.
        :param latitude: The latitude portion of the GPS coordinate.
        :param longitude: The longitude portion of the GPS coordinate.
        :return: Information about the address or places near the coordinates.
        """
        print(f"Searching for '{store_or_business_name}' near {latitude},{longitude}")
        query = f"{store_or_business_name} {latitude},{longitude}"
        return await self.find_specific_place(query, __event_emitter__)

    async def find_specific_place(self, address_or_place: str, __event_emitter__) -> str:
        """
        Looks up details on OpenStreetMap of a specific address, landmark,
        place, named building, or location. Used for when the user asks where
        a specific unique entity (like a specific museum, or church, or shopping
        center) is.
        :param address_or_place: The address or place to look up.
        :return: Information about the place, if found.
        """
        print(f"[OSM] Searching for info on [{address_or_place}].")
        searcher = OsmSearcher(self.valves, self.user_valves, __event_emitter__)
        try:
            result = await searcher.nominatim_search(address_or_place, limit=5)
            if result:
                results_in_md = convert_and_validate_results(
                    address_or_place, result,
                    sort_message="importance", use_distance=False
                )

                resp = f"{specific_place_instructions()}\n\n{results_in_md}"
                print(resp)
                return resp
            else:
                return NO_RESULTS
        except Exception as e:
            print(e)
            return (f"There are no results due to an error. "
                    "Tell the user that there was an error. "
                    f"The error was: {e}. "
                    f"Tell the user the error message.")


    async def find_grocery_stores_near_place(
            self, place: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Finds supermarkets, grocery stores, and other food stores on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby grocery stores or supermarkets, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=supermarket", "shop=grocery", "shop=convenience", "shop=greengrocer"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="groceries",
                                   place=place, tags=tags, event_emitter=__event_emitter__)


    async def find_bakeries_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds bakeries on OpenStreetMap near a given place or
        address. The location of the address or place is reverse
        geo-coded, then nearby results are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby bakeries, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=bakery"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="bakeries",
                             place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_food_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds restaurants, fast food, bars, breweries, pubs, etc on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
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
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="restaurants and food",
                                   place=place, tags=tags, event_emitter=__event_emitter__)


    async def find_swimming_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds swimming pools, water parks, swimming areas, and other aquatic
        activities on OpenStreetMap near a given place or address. The location
        of the address or place is reverse geo-coded, then nearby results are fetched
        from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of swimming poools or places, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["leisure=swimming_pool", "leisure=swimming_area",
                "leisure=water_park", "tourism=theme_park"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="swimming",
                                   radius=10000, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_playgrounds_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds playgrounds and parks on OpenStreetMap near a given place, address, or coordinates.
        The location of the address or place is reverse geo-coded, then nearby results are fetched
        from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of recreational places, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["leisure=playground"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="playgrounds",
                                   limit=10, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_recreation_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds playgrounds, theme parks, parks, frisbee golf, ice skating, and other recreational
        activities on OpenStreetMap near a given place or address. The location
        of the address or place is reverse geo-coded, then nearby results are fetched
        from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of recreational places, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["leisure=horse_riding", "leisure=ice_rink", "leisure=disc_golf_course",
                "leisure=park", "leisure=amusement_arcade", "tourism=theme_park"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="recreational activities",
                                   limit=10, radius=10000, place=place, tags=tags, event_emitter=__event_emitter__)


    async def find_place_of_worship_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds places of worship (churches, mosques, temples, etc) on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby places of worship, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=place_of_worship"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="places of worship",
                                   place=place, tags=tags, event_emitter=__event_emitter__)


    async def find_accommodation_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds accommodation (hotels, guesthouses, hostels, etc) on
        OpenStreetMap near a given place or address. The location of the
        address or place is reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby accommodation, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = [
            "tourism=hotel", "tourism=chalet", "tourism=guest_house", "tourism=guesthouse",
            "tourism=motel", "tourism=hostel"
        ]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="accommodation",
                                   radius=10000, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_alcohol_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds beer stores, liquor stores, and similar shops on OpenStreetMap
        near a given place or address. The location of the address or place is
        reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby alcohol shops, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=alcohol"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="alcohol stores",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_drugs_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds cannabis dispensaries, coffeeshops, smartshops, and similar stores on OpenStreetMap
        near a given place or address. The location of the address or place is
        reverse geo-coded, then nearby results
        are fetched from OpenStreetMap.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby cannabis and smart shops, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=coffeeshop", "shop=cannabis", "shop=headshop", "shop=smartshop"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="cannabis and smartshops",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_schools_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds schools (NOT universities) on OpenStreetMap near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby schools, if found.
        """
        tags = ["amenity=school"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="schools",
                                   limit=10, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_universities_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds universities and colleges on OpenStreetMap near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby schools, if found.
        """
        tags = ["amenity=university", "amenity=college"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="universities",
                                   limit=10, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_libraries_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds libraries on OpenStreetMap near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby libraries, if found.
        """
        tags = ["amenity=library"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="libraries",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_public_transport_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds public transportation stops on OpenStreetMap near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby public transportation stops, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["highway=bus_stop", "amenity=bus_station",
                "railway=station", "railway=halt", "railway=tram_stop",
                "station=subway", "amenity=ferry_terminal",
                "public_transport=station"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="public transport",
                                   limit=10, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_bike_rentals_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds bike rentals on OpenStreetMap near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby bike rentals, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=bicycle_rental", "amenity=bicycle_library", "service:bicycle:rental=yes"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="bike rentals",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_car_rentals_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds bike rentals on OpenStreetMap near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby bike rentals, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=car_rental", "car:rental=yes", "rental=car", "car_rental=yes"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="car rentals",
                                   radius=6000, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_hardware_store_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds hardware stores, home improvement stores, and DIY stores
        near given a place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby hardware/DIY stores, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=doityourself", "shop=hardware", "shop=power_tools",
                "shop=groundskeeping", "shop=trade"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="hardware stores",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_electrical_store_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds electrical stores and lighting stores near a given place
        or address. These are stores that sell lighting and electrical
        equipment like wires, sockets, and so forth.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby electrical/lighting stores, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=lighting", "shop=electrical"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="electrical stores",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_electronics_store_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds consumer electronics stores near a given place or address.
        These stores sell computers, cell phones, video games, and so on.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby electronics stores, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["shop=electronics"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves,
                                   category="consumer electronics stores", place=place,
                                   tags=tags, event_emitter=__event_emitter__)

    async def find_doctor_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds doctors near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby electronics stores, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=clinic", "amenity=doctors", "healthcare=doctor"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="doctors",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_hospital_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds doctors near a given place or address.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby electronics stores, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["healthcare=hospital", "amenity=hospitals"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="hospitals",
                                   place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_pharmacy_near_place(self, __user__: dict, place: str, __event_emitter__) -> str:
        """
        Finds pharmacies and health shops near a given place or address
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :return: A list of nearby electronics stores, if found.
        """
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=pharmacy", "shop=chemist", "shop=supplements",
                "shop=health_food"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="pharmacies",
                                   radius=6000, place=place, tags=tags, event_emitter=__event_emitter__)

    # This function exists to help catch situations where the user is
    # too generic in their query, or is looking for something the tool
    # does not yet support. By having the model pick this function, we
    # can direct it to report its capabilities and tell the user how
    # to use it. It's not perfect, but it works sometimes.
    def find_other_things_near_place(
        self,
        __user__: dict,
        place: str,
        category: str
    ) -> str:
        """
        Find shops and other places not covered by a specific
        category available in the other functions. Use this if the
        user is asking for a type of store or place that other
        functions do not support.

        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param category: The category of place, shop, etc to look up.
        :return: A list of nearby shops.
        """
        print(f"Generic catch handler called with {category}")
        resp = (
            "# No Results Found\n"
            f"No results were found. There was an error. Finding {category} points of interest "
            "is not yet supported. Tell the user support will come eventually! "
            "Tell the user that you are only capable of finding specific "
            "categories of stores, amenities, and points of interest:\n"
            " - Car rentals and bike rentals\n"
            " - Public transport, libraries\n"
            " - Education institutions (schools and universities)\n"
            " - Grocery stores, supermarkets, convenience stores\n"
            " - Food and restaurants\n"
            " - Accommodation\n"
            " - Places of Worship\n"
            " - Hardware stores and home improvement centers\n"
            " - Electrical and lighting stores\n"
            " - Consumer electronics stores\n"
            " - Healthcare (doctors, hospitals, pharmacies, and health stores)\n"
            " - Various recreational and leisure activities\n\n"
            "Only mention things from the above list that you think the user "
            "will be interested in, given the conversation so far. Don't mention "
            "things not on the list. "
            "**IMPORTANT**: Tell the user to be specific in their "
            "query in their next message, so you can call the right function!")
        print(resp)
        return resp
