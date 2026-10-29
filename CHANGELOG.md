# EXIF Filter

**0.2.0:**
 - Drop requirement on GPSPhoto. Use only exifread.
 - Extract time of image creation from EXIF.

**0.1.0:**
 - Initial release.

# OpenStreetMap Tool

**5.0.0:**
 - New tool `show_route_map`: shows a route on a map, with the route line,
   start (A) and destination (B), distance and travel time, and the
   turn-by-turn directions. Clicking a direction highlights that part of
   the route. Buttons on the map step to the previous or next direction,
   or back to the whole route. `navigate_between_places` returns a `route_id` (the route is
   kept for 7 days); `show_route_map` only takes that ID, so the model
   must navigate first, and the map shows exactly the route it reported.
   Drawn from the route geometry ORS already returns: no extra requests.
 - **Breaking:** `show_map` is renamed to `show_poi_map`, so it pairs with
   `show_route_map` and isn't mistaken for a general map tool.
 - **Breaking:** fewer tools, 8 instead of 16 (about 30% less tool schema
   in every request). Update system prompts that name old tools.
   - `find_place` replaces `find_specific_place`, `find_address_for_coordinates`,
     `resolve_coordinates` and
     `find_specific_named_business_or_landmark_near_coordinates`: one
     lookup for addresses, landmarks, named businesses and coordinates,
     with an optional `near` place or coordinates.
   - `find_nearby` replaces the store, recreation, eatery, travel,
     healthcare and education tools and `find_other_things_near_place`:
     one `what` value from a flat list of 28 kinds of places
     (`shop_groceries`, `food_cafe_or_bakery`, `health_dentist`, ...),
     plus `other` with a free-text `what_else`. `other` now runs a
     Nominatim text search ("bicycle shop near Utrecht") instead of
     returning "not supported".
   - Merged duplicates: `drugs` and `cannabis` (same tags) are
     `shop_cannabis`; `hardware` and `diy` are `shop_hardware_or_diy`.
   - Fuel/EV charging and places of worship keep their own tools.
 - `setting` (urban/suburban/rural) is optional in all search tools. Left
   empty, it comes from the road density around the place (the same
   street stats as the locality info, cached): dense is urban, moderate is
   suburban, sparse is rural. Urban if there's no street data.
 - Fix Nominatim results cached per query text only: a 1-result lookup
   cached first capped later 5-result lookups of the same text at 1 result.
 - Overpass searches use `out center` instead of `out geom`: areas (ways
   and relations) come back as one center point instead of their full
   outline, which the tool threw away anyway. Same coordinates, up to ~3×
   less data for area searches (parks: 1.2 MB → 0.4 MB, ~1 s faster).
 - Areas mapped as relations (large parks, campuses, buildings with
   courtyards) are now included in search results. They were silently
   dropped before. Their Nominatim lookup type was also wrong (`route`
   instead of `relation`).
 - Nominatim cache keys ignore case and extra spaces, so "Station
   Haarlem" and "station  haarlem" share an entry. The "Resolving
   location" status only shows for real lookups, not cache hits.
 - Fix nonsense distances in place lookups (`"distance": "unknown"`,
   `"nav_distance": "a bit more than unknownkm"`). Results without a
   distance now leave those fields out, and only say "unavailable".

**4.2.0:**
 - Navigation takes an optional travel mode: `car`, `bike` or `walk`. The
   model should only set it when the user asks for a specific way of
   traveling.
 - Without a travel mode, the mode always comes from the locality (see
   `find_locality_info`) and the straight-line distance. This applies to
   navigation (locality of both ends) and to distances for nearby results
   (locality of the search location, checked once per search):
   - Up to 1.5 km: walking if the ORS instance has it, otherwise bike, or
     car if the area isn't bike-friendly.
   - 1.5 to 15 km: bike and car travel times are compared. Bike wins if it
     takes at most 2× as long as the car in bike-friendly areas, 1.2× if
     either end isn't, and 1.5× when there's no street data.
   - Over 15 km: car.
   - A mode is available if the configured ORS instance has it, or else
     the public fallback (if on). With fallback on and a custom instance
     without walking, short trips (and nearby distances under 1.5 km) are
     walked via the public ORS.
 - The tool asks the ORS instance which profiles it serves (`/v2/status`)
   and caches the answer per ORS URL for an hour, in the cache database
   (Open WebUI reloads tool code, so an in-memory cache didn't last).
   Instances without a
   status endpoint, such as the public ORS, are assumed to serve all three.
   If a requested mode isn't available, the model is told which ones are.
 - Route cache keys include the travel mode, so a cached car route is no
   longer returned for a bike request.
 - Navigation between two places no longer always uses the car profile.
   The result says why a mode was chosen and, when bike and car were
   compared, includes the other mode's distance and time. If only one mode
   has a route, that one is used. If neither does, the model is told why,
   from the ORS error: area not covered, request refused (no access or
   usage limit), route too long, or service not responding. ORS rate
   limits (429) are reported right away, instead of the client retrying
   for up to 60 s.
 - New valves `fallback_to_public_instances` (**off by default**) and
   `ors_fallback_api_key`. With a custom Overpass or ORS instance, a request
   that errors or returns nothing (e.g. outside the instance's map area)
   is retried on the public instance. For Overpass, an empty result only
   falls back if the custom instance has no roads within 1 km of the
   search location (checked once per ~1 km area, cached 30 days): inside
   its map, "nothing found" is a real answer and the public instance
   isn't contacted. ORS fallback uses `ors_fallback_api_key`, or
   `ors_api_key` if that's empty. Per travel mode: the custom instance if
   it has the mode, else the public ORS (e.g. walking). Self-hosted
   instances are assumed to have gaps.
 - New tool `find_locality_info`: what a place is like for getting around
   (country, city/town/village/hamlet, road density, one-way streets,
   cycling infrastructure, calm streets within 1.5 km), from Nominatim and
   one Overpass query that measures lengths. Street stats are cached for
   90 days on a ~100 m grid. Places the Overpass instance has no roads for
   are reported as having no street data, and not cached.
 - Bike-friendliness is yes, partly or no, from two measurements and no
   country rules. Calibrated on 17 places in NL, DK, DE, BE, FR, the UK
   and the US:
   - Cycling infrastructure, as a share of road length: dedicated
     cycleways, cycle tracks along roads, and `path`/`footway` with
     `bicycle=designated` (the usual German tagging), plus half of paths
     shared with pedestrians (`foot=designated`). Painted lanes don't
     count: US suburbs have as many as Dutch towns.
   - Calm streets: the share of streets with a mapped speed limit that are
     30 km/h (20 mph) or less, or living or cycle streets. Only judged
     when speed limits are mapped on at least half of the roads, since
     unmapped limits follow a legal default that differs by country.
   - Yes: infrastructure ≥ 30%, or ≥ 15% with calm streets ≥ 30%, or calm
     streets alone ≥ 60%. Partly: infrastructure ≥ 15% without evidence of
     calm streets. No: otherwise. Partly counts as in-between when bike
     and car are compared.
   - Results: the Dutch places, Copenhagen, Münster, Gent, Paris, Boulder
     and Portland are bike-friendly; Germantown MD, Manhattan, Milton
     Keynes and Davis are partly; Gilbert AZ and Katy TX are not. Davis is
     a known miss: its strength is painted lanes.
 - Travel distances for nearby results use the ORS matrix endpoint: one
   request per travel mode instead of one route per result (two when bike
   and car are compared). Destinations
   outside the ORS map are dropped and the request retried, instead of
   failing the whole batch. Results without a route keep their
   straight-line distance, which is no longer cached as a travel distance.
 - Overpass requests (any instance) are sent one at a time across tool
   calls and processes, using a lock file (`osm-overpass.lock` in the
   temp directory).
 - Overpass Retry-After is honored. A 429, or a 503/504 with Retry-After,
   stores when the host may be asked again in the cache, since the
   tool keeps no state between calls. Waits up to 15 s are slept off and
   the request retried once. Longer waits fail immediately without a
   request, or fall back to the public instance. A 429 without
   Retry-After waits 10 s.
 - Log lines show which Nominatim, Overpass and ORS host each request goes to.
 - The cache is now an SQLite database in Open WebUI's data directory
   (`DATA_DIR/cache/osm_tool/cache.sqlite3`), so it survives restarts.
   `DATA_DIR` comes from Open WebUI itself, so it also works outside
   Docker (pip installs: `backend/data`). Without Open WebUI it falls back
   to the `DATA_DIR` environment variable, then the temp directory. The
   old `/tmp/osm.json` is no longer used and isn't migrated.
   - Safe for simultaneous tool calls and processes. The old JSON file
     lost entries: each cache object wrote its own snapshot over the
     whole file.
   - Entries expire: routes and travel distances after 30 days, Nominatim
     results after 90 days, street stats after 90 days, Overpass coverage
     checks after 30 days, Retry-After deadlines when the wait is over.
     Expired entries are deleted. The old cache kept everything forever
     and rewrote the whole file on every write.
 - The `show_poi_map` map starts zoomed to fit all results again: Open WebUI
   sizes the embed after the page loads, so the map now refits on resize
   until you move it or pick a result.

**4.1.9:**
 - Fix map not rendering (stuck in a scrollable box) when the model passes
   `distance` as a string like `"0.486 km"`: `show_map` now coerces
   lat/lon/distance to numbers, and the map UI no longer aborts on a bad
   distance. Results without usable coordinates are reported back to the
   model instead of breaking the map.
 - `show_map` gives the model its own result message: the map is already
   shown above the reply, so give a sentence or two of useful commentary
   instead of announcing it (no `[Map displayed]` placeholders), and no
   images, markdown image syntax, or map links (models were inventing
   `![map](https://www.openstreetmap.org/#map=...)` images).
 - Escape `</` in embedded POI data so it can't close the map's script tag.
 - Stricter citation format instructions for all search results: only the
   bare ID in brackets (`[3]`), with examples of wrong forms such as
   `[citation 3]`.

**4.1.3:**
 - Fix map vertical sizing issues in newer versions of openwebui.

**4.1.2:*
 - ???

**4.1.1:**
 - Fix map UI responsive rendering by using media queries.

**4.1.0:**
 - New POI type: gyms.
 - Add more category-specific information to search results.
 - Always surface access, brand, and operator tags in search results,
   if present.
 - Search for computer stores, radiotechnics, and mobile phone stores
   when searching for electronics stores.

**4.0.1:**
Improvements:
 - Limit map POI list height to 400px, with auto-scrolling.
 - Add distance to map POI list.
 - Add a valve setting for custom tile server.

Bug fixes:
 - Properly zoom in on map results when selecting from the POI list.
 - Return phone number for search results that have it.
 - Return POI-specific information for EV chargers, restaurants.
 - Better POI descriptions on the generated map.

**4.0.0:**
Major Changes:
 - **UI Integration:** The tool is now able to generate an interactive
   map with results on it.
 - **Search Enhancements:** Ability to search for places of worship
   for specific religions, and denominations of religions.
 - **Increased Accuracy:** Much more consistent tool calls due to
   completely accurate JSON schema.
 - **Refactoring:** First step in internal refactoring to increase
   code maintainability.

Additional Changes:
 - No longer return raw OSM data in search results by default to save
   on token use.
 - Add website, opening hours to search result citations.
 - Tool can now find POIs that must fulfill multiple conditions. Used
   for place of worship search, but has future applications.

**3.2.1:**
 - Report resolved location name in tool results to improve UX when
   the tool is given raw coordinates.
 - Do not return public bookcases when searching for bars.

**3.2.0:**
 - Greatly improved source citations and status events, making use of
   new features in Open WebUI 0.6.30+.
 - Change detailed mode instructions to a structured set of data.
 - Better handling of generic city-wide queries (e.g. "What
   restaurants are in NYC?") by centering on OSM-defined lat/lon.

**3.1.3:**
 - Provide a specific function to guide LLMs in converting GPS
   coordinates to human-readable place names. Useful for `map` prompt
   input type.

**3.1.2:**
 - Minor improvements for GPT-OSS 20b when searching for EV chargers.

**3.1.1:**
 - Tool will not search for cannabis stores when looking for cafes and
   bakeries.

**3.1.0:**
 - Drastically reduce number of functions to save on LLM context
   window size.
 - There is a smaller number of functions, that take categories of
   things to search for.

**3.0.0:**
 - New feature: adjust search radius based on location being urban,
   suburban, or rural. Rural has a much larger search radius than
   urban.
   - Relies on the LLM's understanding of how to categorize the place.
   - You can also ask it to specifically search for urban, suburban,
     rural search radius.
 - New POI category: gas stations.
 - New POI category: EV fast chargers. The LLM can look for any type
   of charger, or specific types.
 - Return replies from functions as JSON instead of raw Markdown.
 - Change various labels (e.g. nav distance -> travel distance) in
   replies to give LLMs better understanding.

**2.2.2:**
 - Inform model of distance sorting method used to attempt to improve
   reasoning model performance.

**2.2.1:**
 - Round distances to 3 decimal places.

**2.2.0:**
 - Report distances and travel time for each step in a navigation
   route.

**2.1.0:**
 - New feature: navigation. LLMs can provide navigation directions and
   answer questions about distance between two places. Works best with
   Qwen and Mistral. Llama3 seems to require a very specific request
   to use the navigation tool.
 - New POI category: tourist attractions. Uses new a new ranking
   system to try and surface more prominent attractions.
 - Fall back to Haversine distance if OpenRouteService cannot
   calculate a navigation distance.
 - Cache ORS route information.

**1.3.1:**
 - Handle bad or unclear addresses by having the model tell you.
 - Fix a swallowed exception that resulted in no results, but no
   notification of finding no results.

**1.3.0:**
 - Emit one citation per result, with nicely formatted text.

**1.2.0:**
 - Update citations for compatibility with 0.4.3.
 - Very basic prettification of citation by converting the Markdown to
   HTML.
 - Show friendlier name for citation when searching for POIs (usually
   address instead of GPS coordinates).

**1.1.3:**
 - Better friendly display names when an address is available.

**1.1.2:**
 - Fix misleading Nominatim resolution complete event that was
   communicating as still resolving.

**1.1.1:**
 - Send resolution complete event when pulling Nominatim info from
   cache.
 - Send error event when headers not set for Nominatim search.

**1.1.0:**
 - Fix bad check in fallback name assembly behavior.
 - Slightly alter functions to improve specific place search.

**1.0.0:**
 - Breaking change: Nominatim URL must now be set to the root domain.
   The model will warn you about this.
 - Caching: Almost all data that the tool fetches is now cached in a
   JSON file (`/tmp/osm.json`). This reduces load on public OSM
   services.
   - Future updates will allow more control over cache behavior.
   - The only thing that is not cached at all is the Overpass Turbo
     search, that actually finds POIs.
 - Handling of unnamed points of interest: many smaller POIs like
   neighborhood playgrounds do not have names. The tool will now look
   up addresses for theses POIs using Nominatim.
 - Citations: The tool will now send citations when the indicators
   setting is turned on. The citation contains the results of the
   search given to the LLM.

**0.9.0:**
 - Integrate OpenRouteService to allow calculation of more accurate
   distances based on the distance of travel rather than "as the crow
   flies" distance.
 - If OpenRouteService is not enabled, the original method will be
   used for calculating distance.
 - ORS is not enabled by default.
 - Properly extract amenity type of leisure locations (playgrounds,
   parks, etc) in search results.

**0.8.0:**
 - Added ability to find specific stores and businesses near
   coordinates. Helps LLM answer questions about arbitrary businesses
   that it might not find with the pre-defined search functions.
 - Pull shop type from Nominatim results when class=shop. Improves
   accuracy of searching for specific places by telling LLM the type
   of shop.

**0.7.0:**
 - Emit event to web UI, showing what the tool is doing.
 - This can be toggled, as a valve setting (not a user valve).

**0.6.3:**
 - Add tool function for resolving GPS coordinates to address.
 - Improve handling of questions like "Where is X near me?"
 - Add basic logging (will be toggleable in a future version).

**0.6.2:**
 - Override `doityourself` store to `hardware`.

**0.6.1:**
 - Small adjustment to single location result instructions.

**0.6.0:**
 - Dramatically improved single location lookup (e.g. asking "where is
   Landmark/City?")
 - Add a catch-all handler that the model can pick if the user wants
   to searh for something not added to the tool. The function will
   force the LLM to tell the user what the OSM tool can find.
 - Added more findable POIs: Hardware stores/home improvement centers,
   electrical and lighting stores, consumer electronics stores, and
   healthcare (doctors' offices, hospitals, health stores).

**0.5.1:**
 - Remove instruction to prioritize nodes over ways in search results.

**0.5.0:**
 - Support Way results. This makes searching much more accurate and
   useful. Many map features are marked as "ways" (shapes on the map)
   rather than specific points.
 - Drop support for "secondary results," and instead return Ways that
   have enough information to be useful.

**0.4.0:**
 - Complete rewrite of search result handling to prevent incorrect OSM
   map links being generated, and bad info given.
 - New setting: Instruction Oriented Interpretation. Controls the
   result interpretation instructions given to the LLM. Should help
   certain models (Hermes Llama 3.1) give more consistent results.
 - The Instruction Oriented Interpretation setting is also a user
   valve that can be controlled on a per-chat basis.
 - Added ability to search for: public transit, schools/universities,
   libraries, bike rental locations, car rental locations.

**0.3.0:**
 - Add search for alcohol and cannabis/smart shops.
 - Increase hotel search radius to 10km.

**0.2.5:**
 - Better accuracy in turn by turn conversations: model encouraged to
   specify the name of the city and country when searching, so it's
   less likely to report results from a different country.

**0.2.4:**
 - Actually make use of the limit parameter when searching Nominatim.

**0.2.3:**
 - Allow leisure tags in nominatim fallback results.

**0.2.1:**
 - Only print secondary results if they exist.
 - Remove extraneous print logging.

**0.2.0:**
 - Include Ways as secondary information for improved search results.
 - Added searching for swimming and recreational areas.

**0.1.0:**
 - Initial release.

# Checkpoint Summarization Filter

**0.2.2:**
 - Fix imports for OpenWebUI refactoring.

**0.2.1:**
 - Fix `None - int` error when setting num_ctx in a chat, and then
   unsetting it.

**0.2.0:**
 - Update for newer versions of OpenWebUI (0.3.29+).

**0.1.0:**
 - Initial release.

# Narrative Memory Filter

**0.0.2:**
 - More reliable way of getting chat IDs.

**0.0.1:**
 - Initial proof-of-concept release.


# GPU Scaling Filter

**0.2.2:**
 - Fix imports for OpenWebUI refactoring.

**0.2.1:**
 - Fixes for internal OpenWebUI refactoring.

**0.2.0:**
 - Fix filter not working when using base models directly.
 - Do not fire the filter if model is not owned by ollama.

**0.1.0:**
 - Initial release.

# Output Sanitization Filter

**0.2.0:**
 - Prevent errors when an LLM reply is updated by the filter (i.e.
   return body from the outlet function).

**0.1.0:**
 - Initial Release

# Collapsible Thought Filter

**0.2.0:**
 - Fix issue with output disappearing.

**0.1.0:**
 - Initial release.

# Artificium Thinking Filter

**0.1.1:**
 - More reliable discovery of thinking and output sections.
 - Remove dead code.

**0.1.0:**
 - Initial release.

# Gemini Tool

**0.2.1:**
 - Basic support for making citations clickable.
 - Fix URL correction valve setting to be a toggle, not textbox.

**0.2.0:**
 - Emit events and citations.

**0.1.2:**
 - Check MIME type of response (and only handle it if it's gemtext).

**0.1.1:**
 - Do not correct URLs when following redirects.
 - Improve result handling.

**0.1.0:**
 - Handle redirects.

**0.0.1:**
 - Initial release with basic protocol handling.
