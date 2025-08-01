""""
title: OpenStreetMap Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 3.0.0
license: AGPL-3.0+
required_open_webui_version: 0.4.3
requirements: openrouteservice, pygments
"""
import itertools
import json
import math
import requests

from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter

import openrouteservice
from openrouteservice.directions import directions as ors_directions

from urllib.parse import urljoin
from operator import itemgetter
from typing import List, Optional
from pydantic import BaseModel, Field

# Yoinked from the OpenWebUI CSS
FONTS = ",".join([
    "-apple-system", "BlinkMacSystemFont", "Inter",
    "ui-sans-serif", "system-ui", "Segoe UI",
    "Roboto", "Ubuntu", "Cantarell", "Noto Sans",
    "sans-serif", "Helvetica Neue", "Arial",
    "\"Apple Color Emoji\"", "\"Segoe UI Emoji\"",
    "Segoe UI Symbol", "\"Noto Color Emoji\""
])

FONT_CSS = f"""
html {{ font-family: {FONTS}; }}

@media (prefers-color-scheme: dark) {{
  html {{
    --tw-text-opacity: 1;
    color: rgb(227 227 227 / var(--tw-text-opacity));
  }}
}}
"""

HIGHLIGHT_CSS = HtmlFormatter().get_style_defs('.highlight')

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

VALVES_NOT_SET = {
    "results": [],
    "instructions": (
        "Tell the user that the User-Agent and From headers"
        "must be set to comply with the OSM Nominatim terms"
        "of use: https://operations.osmfoundation.org/policies/nominatim/"
    ).replace("\n", " ").strip()
}

NO_RESULTS = {
    "results": [],
    "instructions": ("No results found. Tell the user you found no results. "
                     "Do not make up answers or hallucinate. Only say you "
                     "found no results.")
    }

NO_RESULTS_BAD_ADDRESS = {
    "results": [],
    "instructions":  ("No results found. Tell the user you found no results because "
                      "OpenStreetMap could not resolve the address. "
                      "Print the exact address or location you searched for. "
                      "Suggest to the user that they refine their "
                      "question, for example removing the apartment number, sub-unit, "
                      "etc. Example: If `123 Main Street, Apt 4` returned no results, "
                      "suggest that the user searc for `123 Main Street` instead. "
                      "Use the address the user searched for in your example.")
    }

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

def navigation_instructions(travel_type) -> str:
    return (
        "This is the navigation route that the user has requested. "
        f"These instructions are for travel by {travel_type}. "
        "Tell the user the total distance, "
        "and estimated travel time. "
        "If the user **specifically asked for it**, also tell "
        "them the route itself. When telling the route, you must tell "
        f"the user that it's a **{travel_type}** route."
    )

def detailed_instructions(tag_type_str: str) -> str:
    """
    Produce detailed instructions for models good at following
    detailed instructions.
    """
    return (
        f"These are some of the {tag_type_str} points of interest nearby. "
        "These are the results known to be closest to the requested location. "
        "When telling the user about them, make sure to report "
        "all the information (address, contact info, website, etc).\n\n"
        "Use this information to answer the user's query. Prefer closer results "
        "by TRAVEL DISTANCE first. Closer results are higher in the list. "
        "When telling the user the distance, use the TRAVEL DISTANCE. Do not say one "
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
        "**ALWAYS SAY THE CLOSEST RESULT FIRST!**"
    )

def simple_instructions(tag_type_str: str) -> str:
    """
    Produce simpler markdown-oriented instructions for models that do
    better with that.
    """
    return (
        f"These are some of the {tag_type_str} points of interest nearby. "
        "These are the results known to be closest to the requested location. "
        "For each result, report the following information: \n"
        " - Name\n"
        " - Address\n"
        " - OpenStreetMap Link (make it a human readable link like 'View on OpenStreetMap')\n"
        " - Contact information (address, phone, website, email, etc)\n\n"
        "Use the information provided to answer the user's question. "
        "The results are ordered by closeness as the crow flies. "
        "When telling the user about distances, use the TRAVEL DISTANCE only. "
        "Only use relevant results. If there are no relevant results, "
        "say so. Do not make up answers or hallucinate. "
        "Make sure that your results are in the actual location the user is talking about, "
        "and not a place of the same name in a different country."
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

def pretty_print_thing_json(thing):
    """Converts an OSM thing to nice JSON HTML."""
    formatted_json_str = json.dumps(thing, indent=2)
    lexer = JsonLexer()
    formatter = HtmlFormatter(style='colorful')
    return highlight(formatted_json_str, lexer, formatter)


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
        'service:bicycle:rental' in tags or
        'tourism' in tags
    )

    # there can be a lot of artwork in city centers. drop ones that
    # aren't as notable. we define notable by the thing having wiki
    # entries, or by being tagged as historical.
    if tags.get('tourism', '') == 'artwork':
        notable = (
            'wikipedia' in tags or
            'wikimedia_commons' in tags
        )
    else:
        notable = True

    return has_tags and has_useful_tags and notable

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

def sort_by_rank(things):
    """
    Calculate a rank for a list of things. More important ones are
    pushed towards the top. Currently only for tourism tags.
    """
    def rank_thing(thing: dict) -> int:
        tags = thing.get('tags', {})
        if not 'tourism' in tags:
            return 0

        rank = len([name for name in tags.keys()
                         if name.startswith("name")])
        rank += 5 if 'historic' in tags else 0
        rank += 5 if 'wikipedia' in tags else 0
        rank += 1 if 'wikimedia_commons' in tags else 0
        rank += 5 if tags.get('tourism', '') == 'museum' else 0
        rank += 5 if tags.get('tourism', '') == 'aquarium' else 0
        rank += 5 if tags.get('tourism', '') == 'zoo' else 0
        return rank

    return sorted(things, reverse=True, key=lambda t: (rank_thing(t), -t['distance']))

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
    opening_hours: Optional[str] = tags.get('opening_hours', None)

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
    friendly_thing['opening_hours'] = opening_hours if opening_hours else "not recorded"
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

        map_link = create_osm_link(friendly_thing['lat'], friendly_thing['lon'])
        hv_distance_json = (f"{friendly_thing['distance']} km" if use_distance
                            else "unavailable")
        trv_distance_json = (f"{friendly_thing['nav_distance']}"
                             if use_distance and 'nav_distance' in friendly_thing
                             else "unavailable")

        # remove distances from the raw OSM json.
        friendly_thing.pop('nav_distance', None)
        friendly_thing.pop('distance', None)

        entry_json = {
            "latitude": friendly_thing['lat'],
            "longitude": friendly_thing['lon'],
            "address": friendly_thing['address'],
            "amenity_type": friendly_thing['amenity_type'],
            "geographical_distance": hv_distance_json,
            "travel_distance": trv_distance_json,
            "openstreetmap_link": map_link,
            "raw_osm_json": thing
        }

        entries.append(entry_json)

    if len(entries) == 0:
        return None

    return entries

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
            self, valves, user_valves: Optional[dict], event_emitter=None,
    ):
        self.cache = OsmCache()
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


    def calculate_route(
            self, from_thing: dict, to_thing: dict
    ) -> Optional[dict]:
        """
        Calculate route between A and B. Returns the route,
        if successful, or None if the distance could not be
        calculated, or if ORS is not configured.
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

        # check cache first.
        cache_key = f"ors_route_{str(coords)}"
        cached_route = self.cache.get(cache_key)
        if cached_route:
            print("[OSM] Got route from cache!")
            return cached_route

        resp = ors_directions(self._client, coords, profile=profile,
                              preference="fastest", units="km")

        routes = resp.get('routes', [])
        if len(routes) > 0:
            self.cache.set(cache_key, routes[0])
            return routes[0]
        else:
            return None

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

        route = self.calculate_route(from_thing, to_thing)
        return route.get('summary', {}).get('distance', None) if route else None

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

    def create_result_document(self, thing) -> Optional[dict]:
        original_thing = thing
        thing = parse_and_validate_thing(thing)

        if not thing:
            return None

        if 'address' in original_thing:
            street = get_or_none(original_thing['address'], "road")
        else:
            street = get_or_none(original_thing['tags'], "addr:street")

        street_name = street if street is not None else ""
        source_name = f"{thing['name']} {street_name}"
        lat, lon = thing['lat'], thing['lon']
        osm_link = create_osm_link(lat, lon)
        json_data = pretty_print_thing_json(original_thing)
        addr = f"at {thing['address']}" if thing['address'] != 'unknown' else 'nearby'

        document = (f"<style>{HIGHLIGHT_CSS}</style>"
                    f"<style>{FONT_CSS}</style>"
                    f"<div>"
                    f"<p>{thing['name']} is located {addr}.</p>"
                    f"<ul>"
                    f"<li>"
                    f"  <strong>Opening Hours:</strong> {thing['opening_hours']}"
                    f"</li>"
                    f"</ul>"
                    f"<p>Raw JSON data:</p>"
                    f"{json_data}"
                    f"</div>")

        return { "source_name": source_name, "document": document, "osm_link": osm_link }

    async def emit_result_citation(self, thing):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        converted = self.create_result_document(thing)
        if not converted:
            return

        source_name = converted["source_name"]
        document = converted["document"]
        osm_link = converted["osm_link"]

        await self.event_emitter({
            "type": "source",
            "data": {
                "document": [document],
                "metadata": [{"source": source_name, "html": True }],
                "source": {"name": source_name, "url": osm_link},
            }
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
                print(f"[OSM] Got nav distance for {thing['id']} from cache!")
            else:
                print(f"[OSM] Checking ORS for {thing['id']}")
                try:
                    nav_distance = self.calculate_navigation_distance(origin, thing)
                except Exception as e:
                    print(f"[OSM] Error querying ORS: {e}")
                    print(f"[OSM] Falling back to regular distance due to ORS error!")
                    nav_distance = thing['distance']

            if nav_distance:
                used_ors = True
                cache.set(cache_key, nav_distance)
                thing['nav_distance'] = round(nav_distance, 3)

        return used_ors

    def calculate_haversine(self, origin, things_nearby):
        for thing in things_nearby:
            if 'distance' not in thing:
                thing['distance'] = round(haversine_distance(origin, thing), 3)

    def use_detailed_interpretation_mode(self) -> bool:
        # Let user valve for instruction mode override the global
        # setting.
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
            print("[OSM] Got all Nominatim info from cache!")
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
                print("[OSM] No results found for lookup")
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
            print(f"[OSM] Got nominatim search data for {query} from cache!")
            await self.event_resolving(done=True)
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

        # sort by importance + distance, drop to the liimt, then sort
        # by closeness.
        things_nearby = sort_by_rank(things_nearby)
        things_nearby = things_nearby[:limit] # drop down to requested limit
        things_nearby = sort_by_closeness(origin, things_nearby, 'distance')

        if self.attempt_ors(origin, things_nearby):
            sort_method = "travel distance"
            things_nearby = sort_by_closeness(origin, things_nearby, 'nav_distance', 'distance')
        else:
            sort_method = "haversine distance"

        return [things_nearby, sort_method]

    async def search_nearby(
            self, place: str, tags: List[str], limit: int=5, radius: int=4000,
            category: str="POIs"
    ) -> dict:
        headers = self.create_headers()
        if not headers:
            return { "place_display_name": place, "results": VALVES_NOT_SET }

        try:
            nominatim_result = await self.nominatim_search(place, limit=1)
        except ValueError:
            nominatim_result = []

        if not nominatim_result or len(nominatim_result) == 0:
            await self.event_search_complete(category, place, 0)
            return { "place_display_name": place, "results": NO_RESULTS_BAD_ADDRESS }

        try:
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

            print(f"[OSM] Searching for {category} near {place_display_name}")
            things_nearby, sort_method = await self.get_things_nearby(nominatim_result, place, tags,
                                                         bbox, limit, radius)

            if not things_nearby or len(things_nearby) == 0:
                await self.event_search_complete(category, place_display_name, 0)
                return { "place_display_name": place, "results": NO_RESULTS }

            print(f"[OSM] Found {len(things_nearby)} {category} results near {place_display_name}")

            tag_type_str = ", ".join(tags)

            # Only print the full result instructions if we
            # actually have something.
            search_results = convert_and_validate_results(place, things_nearby, sort_message=sort_method)
            if search_results:
                result_instructions = self.get_result_instructions(tag_type_str)
            else:
                result_instructions = "No results found at all. Tell the user there are no results."

            resp = {
                "instructions": result_instructions,
                "results": search_results if search_results else []
            }

            # emit citations for the actual results.
            await self.event_search_complete(category, place_display_name, len(things_nearby))
            for thing in things_nearby:
                await self.emit_result_citation(thing)

            return { "place_display_name": place_display_name, "results": resp, "things": things_nearby }
        except ValueError:
            await self.event_search_complete(category, place_display_name, 0)
            return { "place_display_name": place_display_name, "results": NO_RESULTS, "things": [] }
        except Exception as e:
            print(e)
            await self.event_error(e)
            instructions = (f"No results were found, because of an error. "
                            f"Tell the user that there was an error finding results.")

            result = { "instructions": instructions, "results": [], "error_message": f"{e}" }
            return { "place_display_name": place_display_name, "results": result, "things": [] }


async def do_osm_search(
        valves, user_valves, place, tags,
        category="POIs", event_emitter=None, limit=5, radius=4000,
        setting='urban', search_mode='OR'
):
    radius = radius * setting_to_multiplier(setting)
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

        return {
            "instructions": OLD_VALVE_SETTING.replace("{OLD}", valves.nominatim_url),
            "results": []
        }

    print(f"[OSM] Searching for [{category}] ({tags[0]}, etc) near place: {place} ({setting} setting)")
    searcher = OsmSearcher(valves, user_valves, event_emitter)
    search = await searcher.search_nearby(place, tags, limit=limit, radius=radius, category=category)
    return search["results"]

class OsmNavigator:
    def __init__(
        self, valves, user_valves: Optional[dict], event_emitter=None,
    ):
        self.valves = valves
        self.event_emitter = event_emitter
        self.user_valves = user_valves

    async def event_navigating(self, done: bool):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        if done:
            message = "OpenStreetMap: navigation complete"
        else:
            message = "OpenStreetMap: navigating..."

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": message,
                "done": done,
            },
        })

    async def event_error(self, exception: Exception):
        if not self.event_emitter or not self.valves.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "error",
                "description": f"Error navigating: {str(exception)}",
                "done": True,
            },
        })

    async def navigate(self, start_place: str, destination_place: str):
        await self.event_navigating(done=False)
        searcher = OsmSearcher(self.valves, self.user_valves, self.event_emitter)
        router = OrsRouter(self.valves, self.user_valves, self.event_emitter)

        try:
            start = await searcher.nominatim_search(start_place, limit=1)
            destination = await searcher.nominatim_search(destination_place, limit=1)

            if not start or not destination:
                await self.event_navigating(done=True)
                return NO_RESULTS

            start, destination = start[0], destination[0]
            route = router.calculate_route(start, destination)

            if not route:
                await self.event_navigating(done=True)
                return NO_RESULTS

            total_distance = round(route.get('summary', {}).get('distance', ''), 2)
            travel_time = round(route.get('summary', {}).get('duration', 0) / 60.0, 2)
            travel_type = "car" if total_distance > 1.5 else "walking/biking"

            def create_step_instruction(step):
                instruction = step['instruction']
                duration = round(step.get('duration', 0.0) / 60.0, 2)
                distance = round(step.get('distance', 0.0), 2)

                if duration <= 0.0 or distance <= 0.0:
                    return f"- {instruction}"

                if duration < 1.0:
                    duration = f"{round(duration * 60.0, 2)} sec"
                else:
                    duration = f"{duration} min"

                if distance < 1.0:
                    distance = f"{round(distance * 1000.0, 2)}m"
                else:
                    distance = f"{distance}km"

                return f"- {instruction} ({distance}, {duration})"

            nav_instructions = [
                create_step_instruction(step)
                for segment in route["segments"]
                for step in segment["steps"]
            ]

            result = {
                "instructions": navigation_instructions(travel_type),
                "travel_method": travel_type,
                "total_distance": f"{total_distance} km",
                "travel_time": f"{str(travel_time)} minutes",
                "route": nav_instructions
            }

            await self.event_navigating(done=True)
            return result
        except Exception as e:
            print(e)
            await self.event_error(e)
            message = (f"There are no results due to an error. "
                       "Tell the user that there was an error.")

            return {
                "instructions": message,
                "error_message": f"{e}",
                "travel_method": None,
                "total_distance": None,
                "travel_time": None,
                "route": None
            }

def normalize_setting(setting: Optional[str]) -> str:
    if setting:
        setting = setting.lower().replace('"', '').replace("'", '')
        if setting in ['urban', 'suburban', 'rural']:
            return setting
        else:
            print(f"[OSM] WARN: {setting} is not a valid urban setting. Defaulting to 'urban'.")
            return 'urban'

    return 'urban'

def setting_to_multiplier(setting: str) -> int:
    if setting == 'urban':
        return 1
    elif setting == 'suburban':
        return 2
    elif setting == 'rural':
        return 3.5

    return 1


def store_category_to_tags(store_type: str) -> List[str]:
    """
    Convert the specified type parameter for
    find_stores_near_place into the correct list of tags.
    """
    if store_type == "groceries":
        return ["shop=supermarket", "shop=grocery", "shop=greengrocer"]
    elif store_type == "convenience":
        return ["shop=convenience"]
    elif store_type == "alcohol":
        return ["shop=alcohol"]
    elif store_type == "drugs" or store_type == "cannabis":
        return ["shop=coffeeshop", "shop=cannabis", "shop=headshop", "shop=smartshop"]
    elif store_type == "electronics":
        return ["shop=electronics"]
    elif store_type == "electrical":
        return ["shop=lighting", "shop=electrical"]
    elif store_type == "hardware" or store_type == "diy":
        return ["shop=doityourself", "shop=hardware", "shop=power_tools", "shop=groundskeeping", "shop=trade"]
    elif store_type == "pharmacies":
        return ["amenity=pharmacy", "shop=chemist", "shop=supplements", "shop=health_food"]
    else:
        return []


def recreation_to_tags(recreation_type: str) -> List[str]:
    """
    Convert the specified type parameter for
    find_recreation_near_place into the correct list of tags.
    """
    if recreation_type == "swimming":
        return ["leisure=swimming_pool", "leisure=swimming_area",
                "leisure=water_park", "tourism=theme_park"]
    elif recreation_type == "playgrounds":
        return ["leisure=playground"]
    elif recreation_type == "amusement":
        return ["leisure=park", "leisure=amusement_arcade", "tourism=theme_park"]
    elif recreation_type == "sports":
        return ["leisure=horse_riding", "leisure=ice_rink", "leisure=disc_golf_course"]
    else:
        return []


def food_category_to_tags(food_type: str) -> List[str]:
    """
    Convert the specified type parameter for
    find_food_and_bakeries_near_place into the correct list of tags.
    """
    if food_type == "sit_down_restaurants":
        return ["amenity=restaurant", "amenity=eatery", "amenity=canteen"]
    elif food_type == "fast_food":
        return ["amenity=fast_food"]
    elif food_type == "cafe_or_bakery":
        return ["shop=bakery", "amenity=cafe"]
    elif food_type == "bars_and_pubs":
        return ["amenity=bar", "amenity=pub", "amenity=biergarten"]
    else:
        return []

def travel_category_to_tags(travel_type: str) -> List[str]:
    if travel_type == "tourist_attractions":
        return ["tourism=museum", "tourism=aquarium", "tourism=zoo",
                "tourism=attraction", "tourism=gallery", "tourism=artwork"]
    elif travel_type == "accommodation":
        return ["tourism=hotel", "tourism=chalet", "tourism=guest_house",
                "tourism=guesthouse", "tourism=motel", "tourism=hostel"]
    elif travel_type == "bike_rentals":
        return ["amenity=bicycle_rental", "amenity=bicycle_library", "service:bicycle:rental=yes"]
    elif travel_type == "car_rentals":
        return ["amenity=car_rental", "car:rental=yes", "rental=car", "car_rental=yes"]
    elif travel_type == "public_transport":
        return ["highway=bus_stop", "amenity=bus_station", "railway=station", "railway=halt",
                "railway=tram_stop", "station=subway", "amenity=ferry_terminal", "public_transport=station"]

# For certain things, we might need two-step searching: one call to
# return list of possible tags, and another to return the actual
# results. This will prevent the model from hallucinating. Maybe also
# use it to control limits. Urban = higher limits for stuff like
# restaurants.

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

    async def find_specific_named_business_or_place_near_coordinates(
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
                results = convert_and_validate_results(
                    address_or_place, result,
                    sort_message="importance", use_distance=False
                )


                resp = {
                    "instructions": specific_place_instructions(),
                    "results": results if results else []
                }

                return resp
            else:
                return NO_RESULTS
        except Exception as e:
            print(e)
            return (f"There are no results due to an error. "
                    "Tell the user that there was an error. "
                    f"The error was: {e}. "
                    f"Tell the user the error message.")


    async def navigate_between_places(
            self,
            start_address_or_place: str,
            destination_address_or_place: str,
            __event_emitter__
    ) -> str:
        """
        Retrieve a navigation route and associated information between two places.
        :param start_address_or_place: The address, place, or coordinates to start from.
        :param destination_address_or_place: The destination address, place, or coordinates to go.
        :return: The navigation route and associated info, if found.
        """
        print(f"[OSM] Navigating from [{start_address_or_place}] to [{destination_address_or_place}].")
        navigator = OsmNavigator(self.valves, self.user_valves, __event_emitter__)
        return await navigator.navigate(start_address_or_place, destination_address_or_place)

    async def find_stores_by_category_near_place(
            self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Finds stores of a specific type on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of store to search for. Must be one of "groceries", "convenience", "alcohol", "drugs", "cannabis", "electronics", "electrical", "hardware", or "diy".
        """
        allowed_categories = [
            "groceries", "convenience", "alcohol", "drugs",
            "cannabis",  "electronics", "electrical", "hardware",
            "diy"
        ]

        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = store_category_to_tags(category)

        if not tags:
            return {
                "results": [],
                "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
                "error_message": f"{category} is not a valid category. Must be one of: {', '.join(allowed_categories)}"
            }

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_recreation_by_category_near_place(
            self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Finds recreation facilities of a specific type on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        Note: amusement category can be arcades, theme parks, and similar.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of recreation to search for. Must be one of "swimming", "playgrounds", "amusement", or "sports".
        """
        allowed_categories = ["swimming", "playgrounds", "amusement", "sports"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = recreation_to_tags(category)

        if not tags:
            return {
                "results": [],
                "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
                "error_message": f"{category} is not a valid category. Must be one of: {', '.join(allowed_categories)}"
            }

        radius = 4000
        limit = 5

        if category == "swimming":
            radius = 10000
        elif category == "amusement" or category == "sports":
            radius = 10000
            limit = 10
        elif category == "playgrounds":
            limit = 10

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   radius=radius, limit=limit, setting=setting, place=place, tags=tags,
                                   event_emitter=__event_emitter__)

    async def find_eateries_by_category_near_place(
            self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Finds places to eat or drink on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        If it is unclear what category of eatery the user wants, ask for clarification.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of eateries to search for. Must be one of "sit_down_restaurants", "fast_food", "cafe_or_bakery", "bars_and_pubs".
        """
        allowed_categories = [ "sit_down_restaurants", "fast_food", "cafe_or_bakery", "bars_and_pubs"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = food_category_to_tags(category)

        if not tags:
            return {
                "results": [],
                "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
                "error_message": f"{category} is not a valid category. Must be one of: {', '.join(allowed_categories)}"
            }

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   limit=10, setting=setting, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_travel_info_by_category_near_place(
        self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Find tourist attractions, accommodation, public transport, bike rentals, or car rentals on OpenStreetMap.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        If it is unclear what category of eatery the user wants, ask for clarification.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of travel info to search for. Must be one of "tourist_attractions", "accommodation", "bike_rentals", "car_rentals", "public_transport".
        """
        allowed_categories = ["tourist_attractions", "accommodation", "bike_rentals", "car_rentals", "public_transport"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = travel_category_to_tags(category)

        if not tags:
            return {
                "results": [],
                "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
                "error_message": f"{category} is not a valid category. Must be one of: {', '.join(allowed_categories)}"
            }

        radius = 4000
        limit = 5

        if category == "car_rentals":
            radius = 6000
        elif category == "tourist_attractions":
            radius = 10000
            limit = 10
        elif category == "accommodation":
            radius = 10000
        elif category == "public_transport":
            limit = 10

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   limit=limit, radius=radius, setting=setting, place=place, tags=tags,
                                   event_emitter=__event_emitter__)

    async def find_place_of_worship_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds places of worship (churches, mosques, temples, etc) on OpenStreetMap near a
        given place or address. For setting, specify if the place is an urban area,
        a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby places of worship, if found.
        """
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=place_of_worship"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="places of worship",
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_schools_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds schools (NOT universities) on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby schools, if found.
        """
        setting = normalize_setting(setting)
        tags = ["amenity=school"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="schools",
                                   setting=setting, limit=10, place=place, tags=tags,
                                   event_emitter=__event_emitter__)

    async def find_universities_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds universities and colleges on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby schools, if found.
        """
        setting = normalize_setting(setting)
        tags = ["amenity=university", "amenity=college"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="universities",
                                   setting=setting, limit=10, place=place, tags=tags,
                                   event_emitter=__event_emitter__)

    async def find_libraries_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds libraries on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby libraries, if found.
        """
        setting = normalize_setting(setting)
        tags = ["amenity=library"]
        user_valves = __user__["valves"] if "valves" in __user__ else None
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="libraries",
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_doctor_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds doctors near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby electronics stores, if found.
        """
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=clinic", "amenity=doctors", "healthcare=doctor"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="doctors",
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_hospital_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds doctors near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby electronics stores, if found.
        """
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["healthcare=hospital", "amenity=hospitals"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="hospitals",
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__)

    async def find_fuel_near_place(self, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
        """
        Finds gas stations, petrol stations, and fuel stations near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby fueling stations, if found.
        """
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags = ["amenity=fuel"]
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="gas stations",
                                   setting=setting, radius=10000, place=place, tags=tags,
                                   event_emitter=__event_emitter__)

    async def find_ev_fast_chargers_near_place_with_type(
            self, __user__: dict,
            place: str,
            charger_type: str,
            setting: str,
            __event_emitter__
    ) -> str:
        """
        Finds EV (electric vehicle) DC fast chargers near a given place or address (22 kW+).
        Does NOT find regular/slow chargers (3 kW - 11 kW) that use AC.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        The charger_type parameter can be used to constrain the search to specific charger types.
        By default, search for all charger types unless user asks for specific types.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param charger_type: Must be "chademo", "chademo3", "chaoji", "ccs2", "ccs1", "gb/t", "nacs", or "all" (default).
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :return: A list of nearby fueling stations, if found.
        """
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        charger_type = charger_type.lower().replace('"', "").replace("'", "")

        # possible search constraints, mapped to corresponding osm
        # socket types.
        param_charger_types = {
            "chademo": "chademo",
            "chademo3": "chaoji",
            "chaoji": "chaoji",
            "ccs2": "type2_combo",
            "ccs1": "type1_combo",
            "nacs": "nacs",
            "gb/t": "gb_dc"
        }

        # socket types used on OSM.
        osm_socket_types = [
            # Various DC fast charging standards
            "chademo", # chademo
            "chaoji", # chademo3
            "type2_combo", # CCS2
            "type1_combo", # CCS1
            "gb_dc", # chinese standard
            "nacs", # north american standard

            # not supposed to be used, but can show up.
            "tesla_supercharger",
            "tesla_supercharger_ccs",
            "ccs"
        ]

        # normally, search for any of the possible chargers.
        tags = [f"socket:{charger}=\\.*" for charger in osm_socket_types]

        # or, constrain search?
        if charger_type != "all":
            if charger_type not in param_charger_types.keys():
                return {
                    "results": [],
                    "instructions": "There was an error searching due to an invalid charger type.",
                    "error_message": f"{charger_type} is not a valid charger type."
                }
            osm_socket_type = param_charger_types[charger_type]
            tags = [ f"socket:{osm_socket_type}=\\.*" ]

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="EV chargers",
                                   setting=setting, radius=10000, place=place, tags=tags,
                                   event_emitter=__event_emitter__)

    # This function exists to help catch situations where the user is
    # too generic in their query, or is looking for something the tool
    # does not yet support. By having the model pick this function, we
    # can direct it to report its capabilities and tell the user how
    # to use it. It's not perfect, but it works sometimes.
    def find_other_things_near_place(
        self,
        __user__: dict,
        place: str,
        setting: str,
        category: str
    ) -> str:
        """
        Find shops and other places not covered by a specific
        category available in the other functions. Use this if the
        user is asking for a type of store or place that other
        functions do not support. For setting, specify if the place is
        an urban area, a suburb, or a rural location.

        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: The category of place, shop, etc to look up.
        :return: A list of nearby shops.
        """
        print(f"[OSM] Generic catch handler called with {category}")
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
            " - Fuel & EV charging (gas stations, DC fast chargers)\n"
            " - Various recreational and leisure activities\n\n"
            "Only mention things from the above list that you think the user "
            "will be interested in, given the conversation so far. Don't mention "
            "things not on the list. "
            "**IMPORTANT**: Tell the user to be specific in their "
            "query in their next message, so you can call the right function!")
        return resp
