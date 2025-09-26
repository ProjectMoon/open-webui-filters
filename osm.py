"""
title: OpenStreetMap Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 3.2.2
license: AGPL-3.0+
required_open_webui_version: 0.6.30
requirements: openrouteservice, pygments
"""
import itertools
import json
import math
import requests
import uuid

from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter

import openrouteservice
from openrouteservice.directions import directions as ors_directions

from urllib.parse import urljoin
from operator import itemgetter
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

### temp todo list
# 1. Put instruction generation into a class.
# 2. Refactor nominatim displayname/address thing into separate method.
# 3. Separate result post-processing, fallback, ranking into clearer methods.
# 4. Separate Nominatim searching and Overpass searching into two separate classes?
# 5. remove direct use of valves in OsmNavigator.

#####################################################
# Citation CSS
#####################################################

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

#####################################################
# Useful Constants
#####################################################

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
    "When necessary, make friendly human-readable OpenStreetMap links "
    "by using the latitude and longitude of the amenities: "
    f"{EXAMPLE_OSM_LINK}\n\n"
)

CITATION_INSTRUCTIONS = [
    (
        "Always use inline citations in the format "
        "[id], using the citation_id of each source."
    ),
    (
        "The citation ID must be in the form of [<citation_id>]. "
        "For example, if the citation_id ID is 123456789, print [123456789]."
    ),
    "Do NOT print [id:123456789]. That will not work. ",
    "Always use inline citations for information relating to a result. "
]

#####################################################
# Global Utils
#####################################################

def chunk_list(input_list, chunk_size):
    it = iter(input_list)
    return list(
        itertools.zip_longest(*[iter(it)] * chunk_size, fillvalue=None)
    )

def to_lookup(thing) -> Optional[str]:
    lookup_type = NOMINATIM_LOOKUP_TYPES.get(thing['type'])
    if lookup_type is not None:
        return f"{lookup_type}{thing['id']}"

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

#####################################################
# Instruction Generation
#####################################################

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

def list_instructions(tag_type: str, used_rel: bool) -> List[str]:
    """
    Produce detailed instructions in a structured manner for
    models that support it.
    """
    instructions = {
        "result_accuracy": [],
        "information_reporting": [],
        "links": [],
        "citations": []
    }

    # basic instructions
    if used_rel:
        instructions["result_accuracy"].append(
            "You **MUST** Begin your reply by explaining to the user "
            "that we could not search the entire area, "
            "and therefore we did not get all results."
        )
        instructions["result_accuracy"].append(
            "Inform the user that more accurate results can be found by "
            "using more specific search terms like a street or specific landmark."
        )
    else:
        instructions["result_accuracy"].append("These are the results known to be closest to the requested location.")

    instructions["result_accuracy"].append("When saying the location name, use the resolved_location field.")

    # how to report information
    instructions["information_reporting"].append(
        "When telling the user about the results, make sure to report "
        "all information relevant to the user's query (address, contact info, website, etc)."
    )

    instructions["information_reporting"].append(
        "Do not report information that is irrelvant to the user's query."
    )

    instructions["information_reporting"].append(
        "Prefer closer results by TRAVEL DISTANCE first. "
        "Closer results are higher in the list."
    )

    instructions["information_reporting"].append(
        "When telling the user the distance, use the TRAVEL DISTANCE. Do not say one "
        "distance is farther away than another. Just say what the distances are. "
    )

    instructions["information_reporting"].append(
        "Only use relevant results. If there are no relevant results, "
        "say so. Do not make up answers or hallucinate. "
    )

    instructions["information_reporting"].append(NO_CONFUSION)
    instructions["information_reporting"].append(
        "Remember that the CLOSEST result is first, and you should use "
        "that result first."
    )

    # links and citations
    instructions["links"].append(OSM_LINK_INSTRUCTIONS)
    instructions["links"].append(
        "Give map links friendly, contextual labels. "
        "Don't just print the naked link. "
        f"Example: `You can view it on [OpenStreetMap]({EXAMPLE_OSM_LINK})`"
    )

    instructions["citations"].extend(CITATION_INSTRUCTIONS)


    return instructions


def simple_instructions(tag_type_str: str, used_rel: bool) -> str:
    """
    Produce simpler markdown-oriented instructions for models that do
    better with that.
    """
    if used_rel:
        rel_inst = (
            "**Mention that the central point of the town or city that the user is searching in "
            "was used, and that results may not cover the whole area.**"
        )
    else:
        rel_inst = ""

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
        f"{rel_inst}"
    )

class OsmEventEmitter:
    def __init__(self, event_emitter, status_indicators=True):
        self.event_emitter = event_emitter
        self.status_indicators = status_indicators

    async def navigating(self, done: bool):
        if not self.status_indicators:
            return

        if done:
            message = "Navigation complete"
        else:
            message = "Navigating..."

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "in_progress",
                "description": message,
                "done": done,
            },
        })

    async def navigation_error(self, exception: Exception):
        if not self.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "error",
                "description": f"Error navigating: {str(exception)}",
                "done": True,
            },
        })

    async def resolving(self, done: bool=False, message: Optional[str]=None, items=[]):
        if not self.status_indicators:
            return

        items = [
            {
                "title": get_or_none(item, "display_name"),
                "link": OsmUtils.create_osm_link(item.get('lat', -1), item.get('lon', -1))
            }
            for item in items
        ]

        if done:
            message = "Resolution complete"
        else:
            message = f" location: {message}" if message is not None else "..."
            message = f"Resolving{message}"

        await self.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "status": "in_progress",
                "items": items if items else None,
                "description": message,
                "done": done,
            },
        })

    async def searching(
        self, category: str, place: str,
        status: str="in_progress", done: bool=False,
        tags=[]
    ):
        if not self.status_indicators:
            return

        query = f"{category.capitalize()} POIs near {place}"

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": status,
                "action": "web_search_queries_generated",
                "queries": [query],
                "description": f"Searching for {category} near {place}",
                "done": done,
            },
        })

    async def search_complete(self, category: str, place: str, items=[]):
        if not self.status_indicators:
            return

        num_results = len(items)
        citation_items = []

        for item in items:
            name = get_or_none(item.get('tags', {}), "name", "brand")
            addr = get_or_none(item.get('tags', {}), "addr:street", "addr:city")
            addr = f"({addr})" if addr else ""
            title = f"{name} {addr}".strip()
            osm_link = OsmUtils.create_osm_link(item.get('lat', -1), item.get('lon', -1))

            citation_items.append({
                "title": title,
                "link": osm_link
            })

        await self.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "status": "in_progress",
                "items": citation_items if citation_items else None,
                "description": f"Found {num_results} results for {category}",
                "done": True,
            },
        })

        await self.event_emitter({
            "type": "status",
            "data": {
                "action": "sources_retrieved",
                "status": "in_progress",
                "count": num_results,
                "description": f"Found {num_results} results for {category}",
                "done": True,
            },
        })

    async def emit_result_citation(self, thing):
        if not self.status_indicators:
            return

        converted = OsmUtils.create_citation_document(thing)
        if not converted:
            return

        source_name = converted["source_name"]
        document = converted["document"]
        osm_link = converted["osm_link"]
        website = converted["website"]

        await self.event_emitter({
            "type": "source",
            "data": {
                "document": [document],
                "metadata": [{"source": source_name, "html": True }],
                "source": {"name": website, "url": website},
            }
        })

    async def error(self, exception: Exception):
        if not self.status_indicators:
            return

        await self.event_emitter({
            "type": "status",
            "data": {
                "status": "error",
                "description": f"Error searching OpenStreetMap: {str(exception)}",
                "done": True,
            },
        })


class OsmUtils:
    """
    Utility functions that are organized as static methods. AKA
    dumping ground for spaghetti code.
    """

    @staticmethod
    def create_citation_document(thing) -> Optional[dict]:
        if not thing:
            return None

        if 'address' in thing:
            street = get_or_none(thing['address'], "road")
        else:
            street = get_or_none(thing['tags'], "addr:street")

        id = thing.get('id', None)
        street_name = street if street is not None else ""
        source_name = f"{thing['name']} {street_name}"
        lat, lon = thing['lat'], thing['lon']
        addr = f"at {thing['address']}" if thing['address'] != 'unknown' else 'nearby'

        osm_link = OsmUtils.create_osm_link(lat, lon)
        json_data = OsmUtils.pretty_print_thing_json(thing)
        website = thing.get('website', osm_link)

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

        return {
            "id": id,
            "source_name": source_name,
            "document": document,
            "osm_link": osm_link,
            "website": website
        }

    @staticmethod
    def create_osm_link(lat, lon):
        return EXAMPLE_OSM_LINK.replace("<lat>", str(lat)).replace("<lon>", str(lon))

    @staticmethod
    def pretty_print_thing_json(thing):
        """Converts an OSM thing to nice JSON HTML."""
        formatted_json_str = json.dumps(thing, indent=2)
        lexer = JsonLexer()
        formatter = HtmlFormatter(style='colorful')
        return highlight(formatted_json_str, lexer, formatter)

    @staticmethod
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

    @staticmethod
    def thing_has_info(thing):
        has_name = any('name' in tag for tag in thing['tags'])
        return OsmUtils.thing_is_useful(thing) and has_name

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def add_haversine_distance_to_things(origin, things_nearby):
        for thing in things_nearby:
            if 'distance' not in thing:
                thing['distance'] = round(OsmUtils.haversine_distance(origin, thing), 3)


class OsmParser:
    """All result parsing-related functionality in one place."""

    def __init__(self, original_location: str, things_nearby: List[dict]):
        self.original_location = original_location
        self.things_nearby = things_nearby

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def parse_thing_address(thing: dict) -> Optional[str]:
        """
        Parse address from either an Overpass result or Nominatim
        result.
        """
        if 'address' in thing:
            # nominatim result
            return OsmParser.parse_address_from_address_obj(thing['address'])
        else:
            return OsmParser.parse_address_from_tags(thing['tags'])

    @staticmethod
    def friendly_shop_name(shop_type: str) -> str:
        """
        Make certain shop types more friendly for LLM interpretation.
        """
        if shop_type == "doityourself":
            return "hardware"
        else:
            return shop_type

    @staticmethod
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
            return OsmParser.friendly_shop_name(tags['shop'])
        if 'leisure' in tags:
            return OsmParser.friendly_shop_name(tags['leisure'])

        return None

    @staticmethod
    def parse_and_validate_thing(thing: dict) -> Optional[dict]:
        """
        Parse an OSM result (node or post-processed way) and make it
        more friendly to work with. Helps remove ambiguity of the LLM
        interpreting the raw JSON data. If there is not enough data,
        discard the result.
        """
        tags: dict = thing.get('tags', {})

        # Currently we define "enough data" as at least having lat, lon,
        # and a name. nameless things are allowed if they are in a certain
        # class of POIs (leisure).
        has_name = 'name' in tags or 'name' in thing
        is_leisure = 'leisure' in tags or 'leisure' in thing
        if 'lat' not in thing or 'lon' not in thing:
            return None

        if not has_name and not is_leisure:
            return None

        lat: float = thing['lat']
        lon: float = thing['lon']

        friendly_thing = {}
        name: str = (tags['name'] if 'name' in tags
                     else thing['name'] if 'name' in thing
                     else str(thing['id']) if 'id' in thing
                     else str(thing['osm_id']) if 'osm_id' in thing
                     else "unknown")

        osm_link = OsmUtils.create_osm_link(lat, lon)
        amenity_type: Optional[str] = OsmParser.parse_thing_amenity_type(thing, tags)
        address: str = OsmParser.parse_thing_address(thing)
        distance: Optional[float] = thing.get('distance', None)
        nav_distance: Optional[float] = thing.get('nav_distance', None)
        opening_hours: Optional[str] = tags.get('opening_hours', None)
        website: Optional[str] = get_or_none(
            tags, "operator:website", "website", "brand:website"
        )

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
        friendly_thing['website'] = website if website else osm_link
        return friendly_thing

    def convert_and_validate_results(
        self,
        sort_message: str="closeness",
        use_distance: bool=True
    ) -> Optional[str]:
        """
        Converts search results into something friendlier for LLM
        understanding. Also drops incomplete results. Supports
        Overpass and Nominatim results.
        """
        original_location = self.original_location
        things_nearby = self.things_nearby

        entries = []
        for thing in things_nearby:
            # Convert to friendlier data, drop results without names etc.
            # No need to instruct LLM to generate map links if we do it
            # instead.
            friendly_thing = OsmParser.parse_and_validate_thing(thing)
            if not friendly_thing:
                continue

            map_link = OsmUtils.create_osm_link(friendly_thing['lat'], friendly_thing['lon'])

            hv_distance_json = (f"{friendly_thing['distance']} km"
                                if use_distance
                                else "unavailable")

            trv_distance_json = (f"{friendly_thing['nav_distance']}"
                                 if use_distance and 'nav_distance' in friendly_thing
                                 else "unavailable")

            # prepare friendly thing for final result entry. some
            # values are on the "raw" JSON from earlier in the search
            # process.
            thing.pop('nav_distance', None)
            thing.pop('distance', None)
            citation_id = thing.pop('citation_id', None)

            friendly_thing.update({
                "geographical_distance": hv_distance_json,
                "travel_distance": trv_distance_json,
                "openstreetmap_link": map_link,
                "citation_id": citation_id,
                "raw_osm_json": thing
            })

            entries.append(friendly_thing)

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
        self, valves, user_valves, events: OsmEventEmitter
    ):
        self.cache = OsmCache()
        self.events = events

        if valves.ors_api_key is not None and valves.ors_api_key != "":
            if valves.ors_instance is not None:
                self._client = openrouteservice.Client(
                    base_url=valves.ors_instance,
                    key=valves.ors_api_key
                )
            else:
                self._client = openrouteservice.Client(key=valves.ors_api_key)
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
    def __init__(self, valves, user_valves, event_emitter):
        self.events = event_emitter

        # config settings
        self.nominatim_url = valves.nominatim_url
        self.overpass_turbo_url = valves.overpass_turbo_url
        self.user_agent = valves.user_agent
        self.from_header = valves.from_header
        self.detailed_instructions = (user_valves.instruction_oriented_interpretation
                                      if user_valves else valves.instruction_oriented_interpretation)

        # dependents
        self._ors = OrsRouter(valves, user_valves, event_emitter)

    @staticmethod
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

    @staticmethod
    def sort_by_closeness(origin, points, *keys: str):
        """
        Sorts a list of { lat, lon }-like dicts by closeness to an origin point.
        The origin is a dict with keys of { lat, lon }.
        """
        return sorted(points, key=itemgetter(*keys))

    @staticmethod
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
            way_center = OsmUtils.get_bounding_box_center(way['bounds'])
            way['lat'] = way_center['lat']
            way['lon'] = way_center['lon']
            del way['bounds']
            return way

        return None

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

    @staticmethod
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

    def create_headers(self) -> Optional[dict]:
        if len(self.user_agent) == 0 or len(self.from_header) == 0:
            return None

        return {
            'User-Agent': self.user_agent,
            'From': self.from_header
        }

    def calculate_navigation_distance(self, start, destination) -> float:
        """Calculate real distance from A to B, instead of Haversine."""
        return self._ors.calculate_distance(start, destination)

    def add_nav_distance_to_things(self, origin, things_nearby: List[dict]) -> bool:
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
                    print("[OSM] Falling back to regular distance due to ORS error!")
                    nav_distance = thing['distance']

            if nav_distance:
                used_ors = True
                cache.set(cache_key, nav_distance)
                thing['nav_distance'] = round(nav_distance, 3)

        return used_ors

    def get_result_instructions(self, tag_type_str: str, used_rel: bool) -> str:
        if self.detailed_instructions:
            return list_instructions(tag_type_str, used_rel)
        else:
            return simple_instructions(tag_type_str, used_rel)

    def calculate_bounding_box(self, nominatim_result):
        """
        Determine bounding box and/or point to use for starting
        search.
        """
        rel_types = ["village", "town", "city", "suburb"]

        # relations can have a lat/lon defined on them. useful for or
        # other settlements. only do this if the nominatim result is a
        # relation that defines an entire settled area.
        # see also OSM admin_level
        rel_osm_type = nominatim_result.get("osm_type", "node") == "relation"
        rel_type = nominatim_result.get("type", "n/a") == "administrative"
        rel_class = nominatim_result.get("class", None) == "boundary"
        rel_addr_type = nominatim_result.get("addresstype", None) in rel_types
        use_rel_bbox = rel_osm_type and rel_type and rel_class and rel_addr_type

        if use_rel_bbox:
            rel_lat = nominatim_result.get("lat", None)
            rel_lon = nominatim_result.get("lon", None)
            if rel_lat and rel_lon:
                print("[OSM] Requested search area is settled urban zone. Using lat/lon of OSM relation.")
                return True, {
                    'minlat': rel_lat,
                    'maxlat': rel_lat,
                    'minlon': rel_lon,
                    'maxlon': rel_lon
                }

        # fall back to nominatim bounding box for everything else.
        return False, {
            'minlat': nominatim_result['boundingbox'][0],
            'maxlat': nominatim_result['boundingbox'][1],
            'minlon': nominatim_result['boundingbox'][2],
            'maxlon': nominatim_result['boundingbox'][3]
        }

    async def enrich_from_nominatim(self, things, format="json"):
        updated_things = [] # the things with merged info.

        # handle last chunk, which can have nones due to the way
        # chunking is done.
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
            return updated_things
        else:
            print(f"Looking up {len(lookups)} things from Nominatim")

        url = urljoin(self.nominatim_url, "lookup")
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
                return []

            addresses_by_id = {item['osm_id']: item for item in data}

            for thing in things:
                nominatim_result = addresses_by_id.get(thing['id'], {})
                if nominatim_result != {}:
                    updated = OsmSearcher.merge_from_nominatim(thing, nominatim_result)
                    if updated is not None:
                        lookup = to_lookup(thing)
                        cache.set(lookup, updated)
                        updated_things.append(updated)

            return updated_things
        else:
            await self.events.error(Exception(response.text))
            print(response.text)
            return []


    async def nominatim_search(self, query, format="json", limit: int=1) -> Optional[dict]:
        await self.events.resolving(done=False, message=query)
        cache_key = f"nominatim_search_{query}"
        cache = OsmCache()
        data = cache.get(cache_key)

        if data:
            print(f"[OSM] Got nominatim search data for {query} from cache!")
            await self.events.resolving(done=True, message=query, items=data[:limit])
            return data[:limit]

        print(f"[OSM] Searching Nominatim for: {query}")

        url = urljoin(self.nominatim_url, "search")
        params = {
            'q': query,
            'format': format,
            'addressdetails': 1,
            'limit': limit,
        }

        headers = self.create_headers()
        if not headers:
            await self.events.error("Headers not set")
            raise ValueError("Headers not set")

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()

            if not data:
                raise ValueError(f"No results found for query '{query}'")

            await self.events.resolving(done=True, message=query, items=data[:limit])

            print(f"Got result from Nominatim for: {query}")
            cache.set(cache_key, data[:limit])
            return data[:limit]
        else:
            await self.events.error(Exception(response.text))
            print(response.text)
            return None

    async def search_osm(
            self, place, tags, bbox, limit=5, radius=4000, not_tag_groups={}
    ) -> (List[dict], List[dict]):
        """
        Return a list relevant of OSM nodes and ways. Some
        post-processing is done on ways in order to add coordinates to
        them. Optionally specify not_tag_groups, which is a dict of
        tag to values, all of which will be excluded from the search.
        """
        print(f"[OSM] Searching Overpass Turbo around origin {place}")
        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        center = OsmUtils.get_bounding_box_center(bbox)
        around = f"(around:{radius},{center['lat']},{center['lon']})"

        tag_groups = OsmSearcher.group_tags(tags)
        search_groups = [f'"{tag_type}"~"{"|".join(values)}"'
                         for tag_type, values in tag_groups.items()]

        not_search_groups = [f'"{tag_type}"!~"{"|".join(values)}"'
                                 for tag_type, values in not_tag_groups.items()]

        not_search_query = "".join([f"[{query}]" for query in not_search_groups])

        searches = []
        for search_group in search_groups:
            searches.append(
                f'nwr[{search_group}]{not_search_query}{around}'
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
        response = requests.get(self.overpass_turbo_url, params=data, headers=headers)
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
                if 'type' not in res or not OsmUtils.thing_is_useful(res):
                    continue
                if res['type'] == 'node':
                    if OsmUtils.thing_has_info(res):
                        nodes.append(res)
                    else:
                        things_missing_names.append(res)
                elif res['type'] == 'way':
                    processed = OsmSearcher.process_way_result(res)
                    if processed is not None and OsmUtils.thing_has_info(res):
                        ways.append(processed)
                    else:
                        if processed is not None:
                            things_missing_names.append(processed)

            # attempt to update ways that have no names/addresses.
            if len(things_missing_names) > 0:
                print(f"Updating {len(things_missing_names)} things with info")
                for way_chunk in chunk_list(things_missing_names, 20):
                    updated = await self.enrich_from_nominatim(way_chunk)
                    ways = ways + updated

            return nodes, ways
        else:
            print(response.text)
            raise Exception(f"Error calling Overpass API: {response.text}")


    async def search_and_rank(
            self, nominatim_result, place, tags,
            bbox, limit, radius, not_tag_groups):
        """
        Search OpenStreetMap and sort results according to
        distance and relevance to the query.
        """
        # initial OSM search
        nodes, ways = await self.search_osm(
            place, tags, bbox, limit, radius, not_tag_groups
        )

        # use results from the initial search, but if they do not
        # exist, fall back to the nominatim result. this may or may
        # not be a good idea.
        things_nearby = (nodes + ways
                         if len(nodes) > 0 or len(ways) > 0
                         else OsmSearcher.fallback(nominatim_result))

        # add distance to things and drop number of results to the
        # limit. then, if enabled, we calculate ORS distances. then we
        # sort again.
        origin = OsmUtils.get_bounding_box_center(bbox)
        OsmUtils.add_haversine_distance_to_things(origin, things_nearby)

        # sort by importance + distance, drop to the requested limit,
        # then sort by closeness.
        things_nearby = OsmSearcher.sort_by_rank(things_nearby)
        things_nearby = things_nearby[:limit] # drop down to requested limit
        things_nearby = OsmSearcher.sort_by_closeness(origin, things_nearby, 'distance')

        if self.add_nav_distance_to_things(origin, things_nearby):
            sort_method = "travel distance"
            things_nearby = OsmSearcher.sort_by_closeness(origin, things_nearby, 'nav_distance', 'distance')
        else:
            sort_method = "haversine distance"

        # post-process to add sequential citation/source IDs, because
        # web ui doesn't like arbitrary source IDs.
        for id, thing in enumerate(things_nearby):
            thing["citation_id"] = str(id + 1)

        return [things_nearby, sort_method]


    async def search_nearby(
            self, place: str, tags: List[str], limit: int=5, radius: int=4000,
            category: str="POIs", not_tag_groups={}
    ) -> dict:
        """
        Main entrypoint into the searching logic. Checks header
        validity, and then uses Nominatim to figure out WHERE to
        search. From that spot, search for POIs using Overpass Turbo
        (with data enrichment from Nominatim if necessary), rank them
        by distance and relevance, and then final result list plus
        some metadata.
        """
        headers = self.create_headers()
        if not headers:
            return { "place_display_name": place, "results": VALVES_NOT_SET }

        try:
            nominatim_result = await self.nominatim_search(place, limit=1)
        except ValueError:
            nominatim_result = []

        if not nominatim_result or len(nominatim_result) == 0:
            await self.events.search_complete(category, place, [])
            return { "place_display_name": place, "results": NO_RESULTS_BAD_ADDRESS }

        try:
            nominatim_result = nominatim_result[0]

            # display friendlier searching message if possible
            if 'display_name' in nominatim_result:
                place_display_name = ",".join(nominatim_result['display_name'].split(",")[:3])
            elif 'address' in nominatim_result:
                addr = OsmParser.parse_thing_address(nominatim_result)
                if addr is not None:
                    place_display_name = ",".join(addr.split(",")[:3])
                else:
                    place_display_name = place
            else:
                print(f"WARN: Could not find display name for place: {place}")
                place_display_name = place

            await self.events.searching(category, place_display_name, done=False, tags=tags)

            used_rel, bbox = self.calculate_bounding_box(nominatim_result)

            print(f"[OSM] Searching for {category} near {place_display_name}")
            things_nearby, sort_method = await self.search_and_rank(
                nominatim_result, place, tags,
                bbox, limit, radius, not_tag_groups
            )

            if not things_nearby or len(things_nearby) == 0:
                await self.events.search_complete(category, place_display_name, [])
                return { "place_display_name": place, "results": NO_RESULTS }

            print(f"[OSM] Found {len(things_nearby)} {category} results near {place_display_name}")

            tag_type_str = ", ".join(tags)

            # Only print the full result instructions if we
            # actually have something.
            parser = OsmParser(place, things_nearby)
            search_results = parser.convert_and_validate_results(sort_message=sort_method)
            if search_results:
                result_instructions = self.get_result_instructions(tag_type_str, used_rel)
            else:
                result_instructions = "No results found at all. Tell the user there are no results."

            resp = {
                "instructions": result_instructions,
                "results": search_results if search_results else []
            }

            # emit citations for the actual results.
            await self.events.search_complete(category, place_display_name, things_nearby)
            for thing in search_results:
                await self.events.emit_result_citation(thing)

            return { "place_display_name": place_display_name, "results": resp, "things": things_nearby }
        except ValueError:
            await self.events.search_complete(category, place_display_name, [])
            return { "place_display_name": place_display_name, "results": NO_RESULTS, "things": [] }
        except Exception as e:
            print(e)
            await self.events.error(e)
            instructions = (f"No results were found, because of an error. "
                            f"Tell the user that there was an error finding results.")

            result = { "instructions": instructions, "results": [], "error_message": f"{e}" }
            return { "place_display_name": place_display_name, "results": result, "things": [] }


async def do_osm_search(
        valves, user_valves, place, tags,
        category="POIs", event_emitter=None, limit=5, radius=4000,
        setting='urban', not_tag_groups={}
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

        return {
            "instructions": OLD_VALVE_SETTING.replace("{OLD}", valves.nominatim_url),
            "results": []
        }

    print(f"[OSM] Searching for [{category}] ({tags[0]}, etc) near place: {place} ({setting} setting)")
    radius = radius * setting_to_multiplier(setting)
    events = OsmEventEmitter(event_emitter)
    searcher = OsmSearcher(valves, user_valves, events)
    search = await searcher.search_nearby(place, tags, limit=limit, radius=radius,
                                          category=category, not_tag_groups=not_tag_groups)

    # move place display name to top level results returned to client.
    place_display_name = search.get("place_display_name", None)
    search.get("results", {}).update(resolved_location=place_display_name)
    return search["results"]

class OsmNavigator:
    def __init__(
        self, valves, user_valves: Optional[dict], events: OsmEventEmitter,
    ):
        self.valves = valves
        self.user_valves = user_valves
        self.events = events

    async def navigate(self, start_place: str, destination_place: str):
        await self.events.navigating(done=False)
        searcher = OsmSearcher(self.valves, self.user_valves, self.events)
        router = OrsRouter(self.valves, self.user_valves, self.events)

        try:
            start = await searcher.nominatim_search(start_place, limit=1)
            destination = await searcher.nominatim_search(destination_place, limit=1)

            if not start or not destination:
                await self.events.navigating(done=True)
                return NO_RESULTS

            start, destination = start[0], destination[0]
            route = router.calculate_route(start, destination)

            if not route:
                await self.evnts.event_navigating(done=True)
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

            await self.events.navigating(done=True)
            return result
        except Exception as e:
            print(e)
            await self.events.navigation_error(e)
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


def store_category_to_tags(store_type: str) -> Tuple[List[str], dict]:
    """
    Convert the specified type parameter for
    find_stores_near_place into the correct list of tags.
    """
    if store_type == "groceries":
        return ["shop=supermarket", "shop=grocery", "shop=greengrocer"], {}
    elif store_type == "convenience":
        return ["shop=convenience"], {}
    elif store_type == "alcohol":
        return ["shop=alcohol"], {}
    elif store_type == "drugs" or store_type == "cannabis":
        return ["shop=coffeeshop", "shop=cannabis", "shop=headshop", "shop=smartshop"], {}
    elif store_type == "electronics":
        return ["shop=electronics"], {}
    elif store_type == "electrical":
        return ["shop=lighting", "shop=electrical"], {}
    elif store_type == "hardware" or store_type == "diy":
        return ["shop=doityourself", "shop=hardware", "shop=power_tools", "shop=groundskeeping", "shop=trade"], {}
    elif store_type == "pharmacies":
        return ["amenity=pharmacy", "shop=chemist", "shop=supplements", "shop=health_food"], {}
    else:
        return [], {}


def recreation_to_tags(recreation_type: str) -> Tuple[List[str], dict]:
    """
    Convert the specified type parameter for
    find_recreation_near_place into the correct list of tags.
    """
    if recreation_type == "swimming":
        return ["leisure=swimming_pool", "leisure=swimming_area",
                "leisure=water_park", "tourism=theme_park"], {}
    elif recreation_type == "playgrounds":
        return ["leisure=playground"], {}
    elif recreation_type == "amusement":
        return ["leisure=park", "leisure=amusement_arcade", "tourism=theme_park"], {}
    elif recreation_type == "sports":
        return ["leisure=horse_riding", "leisure=ice_rink", "leisure=disc_golf_course"], {}
    else:
        return [], {}


def food_category_to_tags(food_type: str) -> Tuple[List[str], dict]:
    """
    Convert the specified type parameter for
    find_food_and_bakeries_near_place into the correct list of tags.
    """
    if food_type == "sit_down_restaurants":
        return ["amenity=restaurant", "amenity=eatery", "amenity=canteen"], {}
    elif food_type == "fast_food":
        return ["amenity=fast_food"], {}
    elif food_type == "cafe_or_bakery":
        # cannabis stores are sometimes tagged with amenity=cafe.
        return ["shop=bakery", "amenity=cafe"], { "shop": [ "cannabis" ] }
    elif food_type == "bars_and_pubs":
        # the search for amenity=pub can also wind up finding
        # public_bookcases. we don't want that.
        return ["amenity=bar", "amenity=pub", "amenity=biergarten"], { "amenity": [ "public_bookcase" ] }
    else:
        return [], {}

def travel_category_to_tags(travel_type: str) -> Tuple[List[str], dict]:
    if travel_type == "tourist_attractions":
        return ["tourism=museum", "tourism=aquarium", "tourism=zoo",
                "tourism=attraction", "tourism=gallery", "tourism=artwork"], {}
    elif travel_type == "accommodation":
        return ["tourism=hotel", "tourism=chalet", "tourism=guest_house",
                "tourism=guesthouse", "tourism=motel", "tourism=hostel"], {}
    elif travel_type == "bike_rentals":
        return ["amenity=bicycle_rental", "amenity=bicycle_library", "service:bicycle:rental=yes"], {}
    elif travel_type == "car_rentals":
        return ["amenity=car_rental", "car:rental=yes", "rental=car", "car_rental=yes"], {}
    elif travel_type == "public_transport":
        return ["highway=bus_stop", "amenity=bus_station", "railway=station", "railway=halt",
                "railway=tram_stop", "station=subway", "amenity=ferry_terminal", "public_transport=station"], {}
    else:
        return [], {}

def healthcare_category_to_tags(healthcare_type: str) -> Tuple[List[str], dict]:
    if healthcare_type == "doctor":
        return ["amenity=clinic", "amenity=doctors", "healthcare=doctor"], {}
    elif healthcare_type == "hospital":
        return ["healthcare=hospital", "amenity=hospitals"], {}
    elif healthcare_type == "pharmacy":
        return ["amenity=pharmacy", "shop=chemist"], {}
    else:
        return []

def education_category_to_tags(education_type: str) -> Tuple[List[str], dict]:
    if education_type == "schools":
        return ["amenity=school"], {}
    elif education_type == "universities_and_colleges":
        return ["amenity=university", "amenity=college"], {}
    elif education_type == "libraries":
        return ["amenity=library"], {}
    else:
        return [], {}

def validate_category(category, allowed_categories):
    if category not in allowed_categories:
        return {
            "results": [],
            "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
            "error_message": f"{category} is not a valid category. Must be one of: {', '.join(allowed_categories)}"
        }

    return None


def validate_tags(category, tags, allowed_categories):
    category_validation_error = validate_category(category, allowed_categories)
    if category_validation_error:
        return category_validation_error

    if not tags:
        return {
            "results": [],
            "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
            "error_message": f"{category} translated to no searchable OSM tags."
        }

    return None


# EV charger mappings, moved out of original function for readability.
EV_CHARGER_TOOL_INPUT_CHARGER_TYPES = {
    "chademo": "chademo",
    "chademo3": "chaoji",
    "chaoji": "chaoji",
    "ccs2": "type2_combo",
    "ccs1": "type1_combo",
    "nacs": "nacs",
    "gb/t": "gb_dc"
}

# socket types used on OSM.
EV_CHARGER_OSM_SOCKET_TYPES = [
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

    async def find_specific_named_business_or_landmark_near_coordinates(
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

    async def resolve_coordinates(self, latitude: float, longitude: float, __event_emitter__) -> str:
        """
        Resolves GPS coordinates to a human-readable location such
        as an address or place name. Use this to convert GPS
        coordinates before searching if you need to figure out urban,
        suburban, or rural setting.
        :param latitude: The latitude portion of the GPS coordinate.
        :param longitude: The longitude portion of the GPS coordinate.
        :return: Reverse geocoded location or address information.
        """
        print(f"Reverse geocoding {latitude},{longitude}")
        return await self.find_specific_place(f"{latitude},{longitude}", __event_emitter__)

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
        events = OsmEventEmitter(__event_emitter__)
        searcher = OsmSearcher(self.valves, self.user_valves, events)

        try:
            result = await searcher.nominatim_search(address_or_place, limit=5)
            if result:
                parser = OsmParser(address_or_place, result)
                results = parser.convert_and_validate_results(
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
        events = OsmEventEmitter(__event_emitter__)
        navigator = OsmNavigator(self.valves, self.user_valves, events)
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
        tags, not_tag_groups = store_category_to_tags(category)

        validation_error = validate_tags(category, tags, allowed_categories)
        if validation_error:
            return validation_error

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__,
                                   not_tag_groups=not_tag_groups)

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
        tags, not_tag_groups = recreation_to_tags(category)

        validation_error = validate_tags(category, tags, allowed_categories)
        if validation_error:
            return validation_error

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
                                   event_emitter=__event_emitter__, not_tag_groups=not_tag_groups)

    async def find_eateries_by_category_near_place(
            self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Finds places to eat or drink on OpenStreetMap near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        If it is unclear what category the user wants, ask for clarification.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of eateries to search for. Must be one of "sit_down_restaurants", "fast_food", "cafe_or_bakery", "bars_and_pubs".
        """
        allowed_categories = [ "sit_down_restaurants", "fast_food", "cafe_or_bakery", "bars_and_pubs"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags, not_tag_groups = food_category_to_tags(category)

        validation_error = validate_tags(category, tags, allowed_categories)
        if validation_error:
            return validation_error

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   limit=10, setting=setting, place=place, tags=tags, event_emitter=__event_emitter__,
                                   not_tag_groups=not_tag_groups)

    async def find_travel_info_by_category_near_place(
        self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Find tourist attractions, accommodation, public transport, bike rentals, or car rentals on OpenStreetMap.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        If it is unclear what category the user wants, ask for clarification.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of travel info to search for. Must be one of "tourist_attractions", "accommodation", "bike_rentals", "car_rentals", "public_transport".
        """
        allowed_categories = ["tourist_attractions", "accommodation", "bike_rentals", "car_rentals", "public_transport"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags, not_tag_groups = travel_category_to_tags(category)

        validation_error = validate_tags(category, tags, allowed_categories)
        if validation_error:
            return validation_error

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
                                   event_emitter=__event_emitter__, not_tag_groups=not_tag_groups)

    async def find_healthcare_by_category_near_place(
        self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Find healthcare, doctors, hospitals, and pharmacies on OpenStreetMap.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        If it is unclear what category the user wants, ask for clarification.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of healthcare to search for. Must be one of "doctor", "hospital", "pharmacy".
        """
        allowed_categories = ["doctor", "hospital", "pharmacy"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags, not_tag_groups = healthcare_category_to_tags(category)

        validation_error = validate_tags(category, tags, allowed_categories)
        if validation_error:
            return validation_error

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__,
                                   not_tag_groups=not_tag_groups)

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

    async def find_education_by_category_near_place(
        self, place: str, category: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Find education institutions and resources on OpenStreetMap.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        Note: category "schools" searches only for primary and secondary schools, NOT universities or colleges.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category of education to search for. Must be one of "schools", "universities_and_colleges", "libraries".
        """
        allowed_categories = ["schools", "universities_and_colleges", "libraries"]
        setting = normalize_setting(setting)
        user_valves = __user__["valves"] if "valves" in __user__ else None
        tags, not_tag_groups = education_category_to_tags(category)

        validation_error = validate_tags(category, tags, allowed_categories)
        if validation_error:
            return validation_error

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category.replace("_", " "),
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__,
                                   not_tag_groups=not_tag_groups)

    async def find_fuel_or_charging_by_category_near_place(
            self, place: str, category: str, fuel_type: str, ev_charger_type: str, setting: str, __user__: dict, __event_emitter__
    ) -> str:
        """
        Finds gas stations, petrol stations, fuel stations, or EV fast chargers near a given place or address.
        For setting, specify if the place is an urban area, a suburb, or a rural location.
        Does not find slow (regular) EV chargers.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: must be "urban", "suburban", or "rural". Controls search radius.
        :param category: Category to search for. Must be one of "gas_or_petrol", "ev_fast_charging".
        :param fuel_type: Must be one of "petrol", "diesel", or "all" (default). Should be null if searching for EV chargers.
        :param ev_charger_type: Must be one of "chademo", "chademo3", "chaoji", "ccs2", "ccs1", "gb/t", "nacs", or "all" (default). Should be null if searching for gas.
        :return: A list of nearby fueling stations, if found.
        """
        # subfunction to find fuel
        async def _find_fuel_near_place(valves, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
            setting = normalize_setting(setting)
            user_valves = __user__["valves"] if "valves" in __user__ else None
            tags = ["amenity=fuel"]
            return await do_osm_search(valves=valves, user_valves=user_valves, category="gas stations",
                                       setting=setting, radius=10000, place=place, tags=tags,
                                       event_emitter=__event_emitter__)

        # subfunction to find EV fast chargers.
        async def _find_ev_fast_chargers_near_place_with_type(
                valves, __user__: dict, place: str, charger_type: str, setting: str, __event_emitter__
        ) -> str:
            setting = normalize_setting(setting)
            user_valves = __user__["valves"] if "valves" in __user__ else None
            charger_type = charger_type.lower().replace('"', "").replace("'", "")

            # normally, search for any of the possible chargers.
            tags = [f"socket:{charger}=\\.*" for charger in EV_CHARGER_OSM_SOCKET_TYPES]

            # or, constrain search?
            if charger_type != "all":
                if charger_type not in EV_CHARGER_TOOL_INPUT_CHARGER_TYPES.keys():
                    return {
                        "results": [],
                        "instructions": "There was an error searching due to an invalid charger type.",
                        "error_message": f"{charger_type} is not a valid charger type."
                    }

                osm_socket_type = EV_CHARGER_TOOL_INPUT_CHARGER_TYPES[charger_type]
                tags = [ f"socket:{osm_socket_type}=\\.*" ]

            return await do_osm_search(valves=valves, user_valves=user_valves, category="EV chargers",
                                       setting=setting, radius=10000, place=place, tags=tags,
                                       event_emitter=__event_emitter__)


        # Main part of fuel search function
        allowed_categories = ["gas_or_petrol", "ev_fast_charging"]
        validation_error = validate_category(category, allowed_categories)

        if validation_error:
            return validation_error

        if category == "gas_or_petrol":
            print(f"[OSM] WARN: Currently ignoring fuel type parameter (was '{fuel_type}')")
            return await _find_fuel_near_place(self.valves, __user__, place, setting, __event_emitter__)
        elif category == "ev_fast_charging":
            return await _find_ev_fast_chargers_near_place_with_type(
                self.valves, __user__, place, ev_charger_type, setting, __event_emitter__
            )


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
