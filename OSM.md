# 🌍 OpenStreetMap Tool

The OpenStreetMap tool helps you find points of interest, get directions, and explore locations right from your chat interface. Whether you're looking for the nearest coffee shop, trying to navigate to a new address, or just curious about what's around you, this tool makes it easy to get the information you need from OpenStreetMap.

**Full Documentation (Gemini):** [gemini://agnos.is/projects/open-webui-filters/osm-tool](gemini://agnos.is/projects/open-webui-filters/osm-tool)

**Full Documentation (HTTP/Web):** [https://agnos.is/projects/open-webui-filters/osm-tool/](https://agnos.is/projects/open-webui-filters/osm-tool/)

**Changelog:** [located at https://git.agnos.is/](https://git.agnos.is/projectmoon/open-webui-filters/src/branch/master/CHANGELOG.md)

**License:** [AGPLv3 or later](https://www.gnu.org/licenses/agpl-3.0.en.html)

---

## 📌 Key Features

* 📍 **Find Points of Interest (POIs)** near locations, addresses, or GPS coordinates (e.g., bakeries, hospitals, schools, etc.)
* 🧭 **Reverse geocoding** to resolve coordinates into human-readable addresses
* 🗺️ **Navigation routes** via OpenRouteService (ORS) for walking, cycling, or driving directions
* ⏱️ **Travel time estimation** between locations using ORS distance calculations
* 🏙️ **Urban awareness** for accurate search radiuses.
* 💾 **Caching**  to reduce load on the public services and comply with OSM policies.
* 🔄 **Context-aware conversations** (e.g., "Find a café near me" → "How do I get there?")

---

## 📱 Supported Models

**Recommended**: Qwen 3 (14b+ parameters) with **native function calling** enabled for best performance.

### ✅ Regularly Tested Models (Mid-Range Quantization)

The tool is regularly tested with these models at quants ranging from `Q3_K_XL` to `Q5_K_M`, using llama.cpp.

- [🧠 Mistral Small 3.2 24b](https://ollama.com/library/mistral-small3.2)
- [🤖 Qwen 3 14b](https://ollama.com/library/qwen3:14b)
- [🧠 Qwen 3 30b-a3b Instruct 2507](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)

_No testing is currently done with ollama._

### ⚠️ Notes for Model Selection

- **Native function calling** allows the tool to work better (multiple function calls etc). Open WebUI's built-in function calling also works, but is limited to one function call per message.
- **Context Window**: Ensure models have sufficient context window size (a minimum of 8k is recommended).
- **Qwen 2.5 14b** and **Mistral Nemo** are older models that aren't officially recommended, but should work fine for most use cases.
- **Llama 3.1 8b** is not recommended. It may work with very explicit prompts.
- **Weaker models** may need the **"Instruction Oriented Interpretation"** setting disabled for consistent results.

---

## 🧩 How It Works

1. **Address Resolution**
   - Converts input (e.g., "123 Main St, Example City, USA") to GPS coordinates using Nominatim.
   - Requires valid **User-Agent** and **From Header** to avoid being blocked by Nominatim.

2. **Define Search Area**
   - Creates a geographic "box" around the resolved address to define the search area.
   - Uses the **center point** of the either the bounding box, or the OSM-defined center point of a town/city.

3. **POI Search**
   - Queries OpenStreetMap via Overpass Turbo API for points of interest within a radius of the center point.
   - Supports categories like:
     - 🛒 Groceries, restaurants, bakeries
     - 🏫 Schools, universities
     - 🚶‍♂️ Recreation (parks, playgrounds, sports facilities)
     - 🚇 Public transport (bus stops, train stations)
     - 🏥 Hospitals, clinics, doctors
   - One tool (`find_nearby`) covers all of these with a single `what`
     value, such as `shop_groceries` or `health_dentist`. For anything
     else, `what='other'` with a short description runs a general text
     search. Fuel/EV charging and places of worship have their own tools.
   - The search radius depends on how built-up the area is (urban,
     suburban or rural), worked out from the road density around the
     place unless the model specifies it.

4. **Navigation**
   - Uses **OpenRouteService (ORS)** to calculate routes (requires API key).
   - Can show the route on a map (`show_route_map`, after navigating),
     with turn-by-turn directions you can click to zoom to that part of
     the route.
   - Supports:
     - 🚶 Walking directions
     - 🚲 Cycling routes
     - 🚗 Driving navigation
   - Ask for a specific travel mode ("by bike", "walking"), or let the tool
     choose from the locality and distance: walking up to 1.5 km (via the
     public ORS if fallback is on and your ORS has no walking profile;
     otherwise bike or car, depending on the area);
     between 1.5 and 15 km it compares bike and car travel times, favoring
     the bike in places with lots of separated cycle paths; car beyond
     that. Distances for nearby results are chosen the same way.

5. **Locality info**
   - What a place is like for getting around: country, city/town/village,
     road density, one-way streets, and cycling infrastructure nearby.
   - Only modes the ORS instance serves are used. A self-hosted instance
     may have fewer profiles than the public one.

---

## 🛠️ Configuration Requirements

### ⚠️ Mandatory Settings
- **User-Agent**: A custom identifier for API requests (e.g., `"OSM-Tool/1.0 (https://example.com)"`)
- **From Header**: A valid email address (e.g., `"user@example.com"`) for API compliance

### 📡 API Endpoints
- **Nominatim API URL**: Default: `https://nominatim.openstreetmap.org`
  - Used for reverse geocoding (coordinates → address) and address resolution.
- **Overpass Turbo API URL**: Default: `https://overpass-turbo.eu`
  - Used for querying OpenStreetMap for POIs.

### 🔄 Advanced Options
- **Instruction Oriented Interpretation**:
  - **Enabled**: Provides detailed parsing instructions to the model (recommended).
  - **Disabled**: Simplifies output for weaker models.
- **Status Indicators**: Shows real-time tool activity in the UI.
- **ORS API Key**: Required for navigation and acccurate distance calculations.
  - Sign up at [OpenRouteService](https://openrouteservice.org/) and input the key in the tool settings.
- **ORS Instance**:
  - Default: Public ORS instance (`https://api.openrouteservice.org`).
  - Can be changed to a custom ORS instance if self-hosted.
- **Fallback to Public Instances** (**off by default**):
  - For self-hosted Overpass or ORS instances that only cover part of the
    world. When on, requests your instance can't answer (errors, or no data
    outside its map area) are retried on the public instance.
  - With it off, the tool log says so whenever a fallback would have been
    tried: `(public fallback is off: see fallback_to_public_instances)`.
- **ORS Fallback API Key**: key for the public ORS when falling back. If
  empty, the regular ORS API key is used.

---

## 📌 Usage Tips

### 📍 Best Practices
- Avoid city-wide searches - be specific when searching for locations.
- Avoid sub-unit details (e.g., apartment numbers, floor levels).

### 🗺️ Navigation Setup
Enabling OpenRouteService allows the tool to provide navigation instructions and more accurate distance calculations.
1. **Sign up** for [OpenRouteService](https://openrouteservice.org/).
2. **Generate an API token** in your ORS account.
3. **Input the API key** in the **ORS API Key** setting of the tool.
4. **Enable ORS integration** in the tool configuration.

### 📍 Real-Time Location Support
To enable location-based queries (e.g., "Find the nearest coffee shop"):
1. **Enable user location** in OpenWebUI settings (under "User Preferences" or "Privacy").
2. **Configure a model** with `{{USER_LOCATION}}` in its system prompt (e.g. `"The user's current location is {{USER_LOCATION}}."`).
3. OpenWebUI will automatically inject real-time GPS coordinates into the system prompt for every message.

---

## 📜 Nominatim Compliance

The tool will not function without valid **User-Agent** and **From Header** settings. [View Nominatim Terms of Use](https://operations.osmfoundation.org/policies/nominatim/)

**Rate Limiting**: The public Nominatim API server allows **1 request per second**. For production use, consider self-hosting Nominatim with caching.

---

## 🛠️ Troubleshooting

### 🔧 Common Issues
- **No Results Found**:
  - Verify the input address is valid and includes city/country.
  - Check the resolved location on OpenStreetMap by clicking on the search result.
  - Verify search radius (urban/suburban/rural) is correct. Explicitly tell the model to use the correct one if necessary.
- **Model Misbehavior**:
  - Use a stronger model.
  - Disable **Instruction Oriented Interpretation** for weaker models.
  - Explicitly specify which tool function for the model to use (e.g. "Use the find eateries tool").

---

## Recent Updates
**New in 3.2.x:**
 - Greatly improved source citations and status events, making use of new features in Open WebUI 0.6.30+.
 - Change detailed mode instructions to a structured set of data.
 - Better handling of generic city-wide queries (e.g. "What restaurants are in NYC?") by centering on OSM-defined lat/lon.
 - Report resolved location name in tool results.
 - Do not return public bookcases when searching for bars.

**New in 3.1.x:**
 - Drastically reduce number of functions to save on context window size.
 - Provide a specific function to guide LLMs in converting GPS coordinates to human-readable place names.

**New in 3.0.0:**
 - Urban awareness for urban, suburan, and rural areas.
 - New POI categories: gas stations and EV fast chargers.
 - Return replies from functions as JSON instead of markdown.
