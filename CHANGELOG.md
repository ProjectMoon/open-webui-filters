# EXIF Filter

**0.2.0:**
 - Drop requirement on GPSPhoto. Use only exifread.
 - Extract time of image creation from EXIF.

**0.1.0:**
 - Initial release.

# OpenStreetMap Tool
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
