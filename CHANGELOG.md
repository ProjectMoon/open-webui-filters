# OpenStreetMap Tool

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

**0.1.0:**
 - Initial release.

# Narrative Memory Filter

**0.0.2:**
 - More reliable way of getting chat IDs.

**0.0.1:**
 - Initial proof-of-concept release.


# GPU Scaling Filter

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
