"""
title: OpenStreetMap Tool
author: projectmoon
author_url: https://git.agnos.is/projectmoon/open-webui-filters
version: 5.0.0
license: AGPL-3.0+
required_open_webui_version: 0.6.31
requirements: openrouteservice, pygments
"""
import asyncio
import hashlib
import itertools
import json
import math
import os
import re
import requests
import sqlite3
import tempfile
import time

from contextlib import contextmanager

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape as html_escape

try:
    import fcntl
except ImportError:  # not on Linux/macOS: no cross-process lock
    fcntl = None

from fastapi.responses import HTMLResponse
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter

import openrouteservice
from openrouteservice.directions import directions as ors_directions
from openrouteservice.distance_matrix import distance_matrix as ors_distance_matrix
from openrouteservice.convert import decode_polyline

from urllib.parse import urljoin, urlparse
from operator import itemgetter
from typing import List, Optional, Tuple, Literal, TypeAlias
from pydantic import BaseModel, Field

#####################################################
# Map UI Integration
#####################################################

NWR: TypeAlias = Literal['node', 'way', 'relation']

POICategory: TypeAlias = Literal[
    'Groceries', 'Recreation', 'Restaurants & Bars', 'Travel & Tourism',
    'Healthcare', 'Education & Schools', 'Fuel & EV Charging',
    'Religious Locations'
]

class OsmPOI(BaseModel):
    """Represents a Point of Interest found during map search."""
    name: str = Field(description="Name of the Point of Interest.")
    address: Optional[str] = Field(description="Address of the Point of Interest.")
    description: str = Field(description=("Useful one sentence description of the Point of Interest, "
                                          "containing important or unique information. "
                                          "Do not put distance here."))
    distance: float = Field(description="Distance in km or miles from the search location.")
    distance_unit: Literal["km", "mi", "kilometers", "miles"] = Field(description="Unit of distance.")
    lat: float = Field(description="Latitude of the Point of Interest.")
    lon: float = Field(description="Longitude of the Point of Interest.")
    osm_id: str = Field(description="OpenStreetMap ID of the Point of Interest.")
    osm_type: NWR = Field(description="OpenStreetMap entity type of the Point of Interest.")
    pass


def _coerce_float(value) -> Optional[float]:
    """Best-effort float: models sometimes send "0.486 km" or "52,38"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        match = re.match(r"\s*(-?\d+(?:[.,]\d+)?)", value)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def normalize_map_pois(results) -> Tuple[List[dict], List[str]]:
    """
    Open WebUI passes tool arguments through without validating them
    against OsmPOI, so coerce the numeric fields the map UI does math on.
    Returns the cleaned POIs and a list of problems that make a POI unusable.
    """
    pois, problems = [], []
    for index, poi in enumerate(results or []):
        if isinstance(poi, BaseModel):
            poi = poi.model_dump()
        if not isinstance(poi, dict):
            problems.append(f"result {index + 1} is not an object")
            continue

        poi = dict(poi)
        lat, lon = _coerce_float(poi.get("lat")), _coerce_float(poi.get("lon"))
        if lat is None or lon is None:
            problems.append(f"result {index + 1} ({poi.get('name', '?')}) has no numeric lat/lon")
            continue
        poi["lat"], poi["lon"] = lat, lon

        raw_distance = poi.get("distance")
        poi["distance"] = _coerce_float(raw_distance)
        if not poi.get("distance_unit") and isinstance(raw_distance, str):
            unit = re.search(r"\b(km|mi|kilometers|miles)\b", raw_distance)
            poi["distance_unit"] = unit.group(1) if unit else None
        pois.append(poi)
    return pois, problems


# Sent to the model instead of Open WebUI's generic "Embedded UI result is
# active" message, to stop it from faking a map in its reply.
SHOW_MAP_CONTEXT = {
    "status": "success",
    "message": (
        "The interactive map is already displayed to the user, directly ABOVE "
        "your reply. Do not announce the map or point to where it is, and never "
        "write placeholders such as [Map displayed] or [map]. Instead, reply "
        "with one or two short, natural sentences of useful commentary, such as "
        "which result is closest or which stands out and why. Do NOT include "
        "images, markdown image syntax like ![...](...), map links, or "
        "map/tile URLs. Do NOT re-list the results or add a citation list."
    ),
}

# Styling shared by the POI and route maps.
MAP_STYLE = """
         * {
             margin: 0;
             padding: 0;
             box-sizing: border-box;
         }


         :root {
             --text: #263445;
             --muted: #526171;
             --surface: #fff;
             --hover: #f1f5f9;
             --border: #dce3eb;
             --accent: #1769aa;
         }

         body {
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
             color: var(--text);
             background: var(--surface);
             line-height: 1.5;
             padding: 12px;
         }

         .container { max-width: 1200px; margin: 0 auto; }
         .results {
             display: grid;
             grid-template-columns: minmax(0, 1fr) 300px;
             gap: 16px;
             align-items: start;
         }
         .map-container, .poi-list-container { min-width: 0; }
         h1 { font-size: 1.4rem; text-align: center; margin-bottom: 4px; }
         .subtitle { font-size: 0.85rem; color: var(--muted); text-align: center; margin-bottom: 12px; }
         h2 { font-size: 1.1rem; margin: 0 0 8px; font-weight: 600; }
         #map { height: 460px; border-radius: 8px; border: 1px solid var(--border); }
         #poi-list { list-style: none; max-height: 460px; overflow-y: auto; }
         #poi-list li { margin-bottom: 8px; }
         .poi-item {
             width: 100%;
             padding: 10px 12px;
             background: var(--surface);
             border-radius: 6px;
             cursor: pointer;
             border: 1px solid var(--border);
             color: var(--text);
             font: inherit;
             text-align: left;
             display: flex;
             flex-direction: column;
             overflow-wrap: anywhere;
         }
         .poi-item:hover { background: var(--hover); }
         .poi-item:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
         .poi-item.active { background: var(--accent); color: #fff; }
         .poi-title { font-weight: 600; }
         .poi-description, .poi-distance { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }
         .poi-item.active .poi-description, .poi-item.active .poi-distance { color: #fff; }
         .poi-marker { background: transparent; border: 0; }
         .poi-marker span {
             display: grid;
             place-items: center;
             width: 32px;
             height: 32px;
             border-radius: 50%;
             border: 2px solid #fff;
             background: #1769aa;
             color: #fff;
             font: bold 14px/1 'Segoe UI', sans-serif;
             box-shadow: 0 2px 6px #0006;
         }
         .poi-marker.active span { background: #a63b00; outline: 3px solid #fff; }
         .leaflet-popup-content-wrapper, .leaflet-popup-tip { background: var(--surface); color: var(--text); }
         .leaflet-popup-content a { color: var(--accent); }
         .empty-results { padding: 24px; color: var(--muted); }
         @media (max-width: 759px) {
             .results { grid-template-columns: minmax(0, 1fr); }
             #map { height: 340px; }
             #poi-list { max-height: none; }
         }
         @media (prefers-color-scheme: dark) {
             :root {
                 --text: #e5edf5;
                 --muted: #b0becd;
                 --surface: #18212d;
                 --hover: #253244;
                 --border: #3a495b;
                 --accent: #247abd;
             }
             .leaflet-popup-content a { color: #8bcaff; }
         }
"""

# Reports the page height to Open WebUI, which sizes the embed to fit.
MAP_HEIGHT_SCRIPT = """
         function getContentHeight() {
             const app = document.getElementById('app');
             if (!app) return 0;
             return Math.ceil(app.getBoundingClientRect().height);
         }

         let lastReportedHeight = 0;
         function notifyParentOfHeight() {
             const height = getContentHeight();
             if (!window.parent || height <= 0 || height === lastReportedHeight) return;
             lastReportedHeight = height;

             window.parent.postMessage(
                 { type: 'iframe:height', height: height + 24 },
                 '*'
             );
         }

         let heightReportPending = false;
         function scheduleHeightReport() {
             if (heightReportPending) return;
             heightReportPending = true;
             requestAnimationFrame(() => {
                 heightReportPending = false;
                 notifyParentOfHeight();
             });
         }
"""

MAP_UI = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="color-scheme" content="light dark">
        <title>Search Results</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
{MAP_STYLE}        </style>
    </head>
    <body>
        <div class="container" id="app">
            <h1>Map Results for {REPLACE_WITH_CATEGORY}</h1>
            <p class="subtitle">Data from OpenStreetMap</p>

            <div class="results">
                <div class="map-container">
                    <h2>Map</h2>
                    <div id="map"></div>
                </div>

                <div class="poi-list-container">
                    <h2>POI List</h2>
                    <ul id="poi-list"></ul>
                </div>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
         // POI data with coordinates and names
         // [ { name, coords, lat, lon, description } ]
         const poiData = {REPLACE_WITH_POI_RESULTS};

{MAP_HEIGHT_SCRIPT}

         // Function to create the map
         function createMap(poiData) {
             // Fit the initial view to the search results
             const map = L.map('map', { minZoom: 3, maxZoom: 18 });

             // Add OpenStreetMap tiles
             L.tileLayer('{REPLACE_WITH_TILE_SERVER}', {
                 attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
             }).addTo(map);

             // Store markers and popup references
             const markers = [];

             // Add markers to map and create list items dynamically
             poiData.forEach((poi, index) => {
                 const icon = L.divIcon({
                     className: 'poi-marker',
                     html: `<span>${index + 1}</span>`,
                     iconSize: [32, 32],
                     iconAnchor: [16, 16],
                     popupAnchor: [0, -18]
                 });
                 const marker = L.marker([poi.lat, poi.lon], { icon, title: poi.name }).addTo(map);
                 const osmLink = `https://www.openstreetmap.org/${poi.osm_type}/${poi.osm_id}`;
                 const header = `<h3><a target="_blank" href="${osmLink}">${poi.name}</a></h3>`;
                 const address = `<p>${poi.address ?? ''}</p>`;
                 const description = `<p>${poi.description}</p>`;
                 const coords = `<p>Coordinates: ${Number(poi.lat).toFixed(4)}, ${Number(poi.lon).toFixed(4)}</p>`;
                 // Models sometimes send distance as a string ("0.5 km");
                 // a bad value must not abort the whole map.
                 const distanceValue = Number.parseFloat(poi.distance);
                 const distance = Number.isFinite(distanceValue)
                     ? `${distanceValue.toFixed(2)} ${poi.distance_unit ?? ''}`.trim()
                     : '';
                 const distanceHtml = distance ? `<p>${distance}</p>` : '';

                 marker.bindPopup(`${header}${address}${description}${distanceHtml}${coords}`);
                 marker.on('popupopen', () => updateActiveListItem(index, markers, true));
                 marker.on('popupclose', scheduleHeightReport);
                 markers.push(marker);

                 const listItem = document.createElement('li');
                 const button = document.createElement('button');
                 button.type = 'button';
                 button.className = 'poi-item';
                 button.setAttribute('aria-pressed', 'false');
                 for (const [className, text] of [
                     ['poi-title', `${index + 1}. ${poi.name}`],
                     ['poi-description', poi.description],
                     ['poi-distance', distance]
                 ]) {
                     const span = document.createElement('span');
                     span.className = className;
                     span.textContent = text;
                     button.appendChild(span);
                 }
                 button.onclick = () => focusOnMarker(index, map, markers);
                 listItem.appendChild(button);
                 document.getElementById('poi-list').appendChild(listItem);
             });

             // OWUI sizes the iframe after the page loads, so the first fit
             // is done at the wrong size. Refit on resize until the user
             // moves the map or picks a result.
             let userMoved = false;
             function fitResults() {
                 if (poiData.length === 1) {
                     map.setView(markers[0].getLatLng(), 15);
                 } else {
                     map.fitBounds(L.latLngBounds(markers.map(marker => marker.getLatLng())), {
                         padding: [32, 32], maxZoom: 15
                     });
                 }
             }
             fitResults();
             map.on('dragstart popupopen', () => { userMoved = true; });
             map.getContainer().addEventListener('wheel', () => { userMoved = true; }, { passive: true });
             map.getContainer().querySelector('.leaflet-control-zoom')
                 ?.addEventListener('click', () => { userMoved = true; });
             function onResize() {
                 map.invalidateSize({ pan: false });
                 if (!userMoved) fitResults();
             }

             return { map, markers, onResize };
         }

         // Function to focus on a specific marker
         function focusOnMarker(index, map, markers) {
             const marker = markers[index];

             map.flyTo(marker.getLatLng(), 15);
             marker.openPopup();
             updateActiveListItem(index, markers);
             scheduleHeightReport();
         }

         // Function to update active list item
         function updateActiveListItem(index, markers, reveal = false) {
             const listItems = document.querySelectorAll('.poi-item');
             listItems.forEach((item, i) => {
                 const active = i === index;
                 item.classList.toggle('active', active);
                 item.setAttribute('aria-pressed', String(active));
                 markers[i].getElement()?.classList.toggle('active', active);
                 if (active && reveal) {
                     const list = document.getElementById('poi-list');
                     const itemBounds = item.getBoundingClientRect();
                     const listBounds = list.getBoundingClientRect();
                     if (itemBounds.top < listBounds.top) list.scrollTop -= listBounds.top - itemBounds.top;
                     else if (itemBounds.bottom > listBounds.bottom) list.scrollTop += itemBounds.bottom - listBounds.bottom;
                 }
             });
             scheduleHeightReport();
         }

         if (poiData.length > 0) {
             const { map, markers, onResize } = createMap(poiData);
             map.whenReady(() => {
                 updateActiveListItem(0, markers);
                 scheduleHeightReport();
             });
             const mapResizeObserver = new ResizeObserver(() => {
                 requestAnimationFrame(() => {
                     onResize();
                     scheduleHeightReport();
                 });
             });
             mapResizeObserver.observe(document.getElementById('map'));
         } else {
             document.querySelector('.results').replaceChildren();
             const empty = document.createElement('p');
             empty.className = 'empty-results';
             empty.textContent = 'No results to display.';
             document.querySelector('.results').appendChild(empty);
         }

         window.addEventListener('load', scheduleHeightReport);
         window.addEventListener('resize', scheduleHeightReport);
         const resizeObserver = new ResizeObserver(scheduleHeightReport);
         resizeObserver.observe(document.getElementById('app'));
        </script>
    </body>
</html>
"""

MAP_UI = (MAP_UI.replace("{MAP_STYLE}", MAP_STYLE)
          .replace("{MAP_HEIGHT_SCRIPT}", MAP_HEIGHT_SCRIPT))

ROUTE_MAP_UI = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="color-scheme" content="light dark">
        <title>Route</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
{MAP_STYLE}
         .route-marker span { width: 28px; height: 28px; font-size: 13px; }
         .route-marker.start span { background: #2e7d32; }
         .route-marker.end span { background: #a63b00; }
         .step-detail { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
         .poi-item.active .step-detail { color: #fff; }
         .route-steps-control a { font: bold 18px/26px 'Segoe UI', sans-serif; }
        </style>
    </head>
    <body>
        <div class="container" id="app">
            <h1>{REPLACE_WITH_TITLE}</h1>
            <p class="subtitle" id="route-summary"></p>

            <div class="results">
                <div class="map-container">
                    <h2>Map</h2>
                    <div id="map"></div>
                </div>

                <div class="poi-list-container">
                    <h2>Directions</h2>
                    <ul id="poi-list"></ul>
                </div>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
         // { mode, distance_km, duration_min, start: { name, lat, lon },
         //   end: { name, lat, lon }, coordinates: [[lat, lon], ...],
         //   steps: [{ instruction, distance_km, duration_s, from, to }] }
         // Step from/to are indexes into coordinates.
         const route = {REPLACE_WITH_ROUTE};

{MAP_HEIGHT_SCRIPT}

         function formatDistance(km) {
             return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`;
         }

         function formatDuration(minutes) {
             if (minutes < 1) return `${Math.max(1, Math.round(minutes * 60))} s`;
             if (minutes < 60) return `${Math.round(minutes)} min`;
             return `${Math.floor(minutes / 60)} h ${Math.round(minutes % 60)} min`;
         }

         function endpointMarker(point, label, className) {
             const icon = L.divIcon({
                 className: `poi-marker route-marker ${className}`,
                 html: `<span>${label}</span>`,
                 iconSize: [28, 28],
                 iconAnchor: [14, 14],
                 popupAnchor: [0, -16]
             });
             const marker = L.marker([point.lat, point.lon], { icon, title: point.name });
             const popup = document.createElement('div');
             const name = document.createElement('strong');
             name.textContent = point.name;
             popup.appendChild(name);
             marker.bindPopup(popup);
             return marker;
         }

         function createRouteMap(route) {
             const map = L.map('map', { minZoom: 3, maxZoom: 18 });
             L.tileLayer('{REPLACE_WITH_TILE_SERVER}', {
                 attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, routing by <a href="https://openrouteservice.org/">openrouteservice</a>'
             }).addTo(map);

             const line = L.polyline(route.coordinates, {
                 color: '#1769aa', weight: 5, opacity: 0.85
             }).addTo(map);
             endpointMarker(route.start, 'A', 'start').addTo(map);
             endpointMarker(route.end, 'B', 'end').addTo(map);

             let highlight = null;
             let current = -1;  // selected step, -1 for the whole route
             let userMoved = false;
             const buttons = [];

             function fitRoute(animate) {
                 const options = { padding: [32, 32], maxZoom: 17 };
                 if (animate) map.flyToBounds(line.getBounds(), options);
                 else map.fitBounds(line.getBounds(), options);
             }
             fitRoute(false);

             // Open WebUI sizes the embed only after the page reports its
             // height, so the first fit is for a tiny map. Fit the whole route
             // again on resizes, until the user pans, zooms, or picks a step.
             map.on('dragstart', () => { userMoved = true; });
             map.getContainer().addEventListener('wheel', () => { userMoved = true; }, { passive: true });
             map.getContainer().querySelector('.leaflet-control-zoom')
                 ?.addEventListener('click', () => { userMoved = true; });
             function onResize() {
                 map.invalidateSize({ pan: false });
                 if (current < 0 && !userMoved) fitRoute(false);
             }

             function selectStep(index) {
                 if (highlight) highlight.remove();
                 highlight = null;
                 current = index;

                 if (index < 0) {
                     userMoved = false;
                     fitRoute(true);
                 } else {
                     const step = route.steps[index];
                     const part = route.coordinates.slice(step.from, step.to + 1);
                     highlight = L.polyline(part.length > 1 ? part : [part[0], part[0]], {
                         color: '#a63b00', weight: 8, opacity: 0.9
                     }).addTo(map);
                     map.flyToBounds(highlight.getBounds(), { padding: [48, 48], maxZoom: 17 });
                 }

                 buttons.forEach((button, i) => {
                     button.classList.toggle('active', i === index);
                     button.setAttribute('aria-pressed', String(i === index));
                 });
                 if (index >= 0) {
                     const item = buttons[index];
                     const list = document.getElementById('poi-list');
                     const itemBounds = item.getBoundingClientRect();
                     const listBounds = list.getBoundingClientRect();
                     if (itemBounds.top < listBounds.top) list.scrollTop -= listBounds.top - itemBounds.top;
                     else if (itemBounds.bottom > listBounds.bottom) list.scrollTop += itemBounds.bottom - listBounds.bottom;
                 }
                 stepControl.update();
                 scheduleHeightReport();
             }

             // Previous step, whole route, next step: in the map panel.
             const StepControl = L.Control.extend({
                 options: { position: 'topleft' },
                 onAdd() {
                     const bar = L.DomUtil.create('div', 'leaflet-bar leaflet-control route-steps-control');
                     const make = (label, title, onClick) => {
                         const button = L.DomUtil.create('a', '', bar);
                         button.href = '#';
                         button.setAttribute('role', 'button');
                         button.title = title;
                         button.setAttribute('aria-label', title);
                         button.textContent = label;
                         L.DomEvent.on(button, 'click', (event) => {
                             L.DomEvent.preventDefault(event);
                             if (!button.classList.contains('leaflet-disabled')) onClick();
                         });
                         return button;
                     };
                     this.previous = make('‹', 'Previous step', () => selectStep(Math.max(current - 1, 0)));
                     this.reset = make('⤢', 'Show the whole route', () => selectStep(-1));
                     this.next = make('›', 'Next step', () => selectStep(Math.min(current + 1, route.steps.length - 1)));
                     L.DomEvent.disableClickPropagation(bar);
                     L.DomEvent.disableScrollPropagation(bar);
                     return bar;
                 },
                 update() {
                     this.previous.classList.toggle('leaflet-disabled', current <= 0);
                     this.reset.classList.toggle('leaflet-disabled', current < 0);
                     this.next.classList.toggle('leaflet-disabled', current >= route.steps.length - 1);
                 }
             });
             const stepControl = new StepControl().addTo(map);

             route.steps.forEach((step, index) => {
                 const listItem = document.createElement('li');
                 const button = document.createElement('button');
                 button.type = 'button';
                 button.className = 'poi-item';
                 button.setAttribute('aria-pressed', 'false');

                 const title = document.createElement('span');
                 title.className = 'poi-title';
                 title.textContent = `${index + 1}. ${step.instruction}`;
                 button.appendChild(title);

                 if (step.distance_km > 0) {
                     const detail = document.createElement('span');
                     detail.className = 'step-detail';
                     detail.textContent = `${formatDistance(step.distance_km)} · ${formatDuration(step.duration_s / 60)}`;
                     button.appendChild(detail);
                 }

                 button.onclick = () => selectStep(index);

                 buttons.push(button);
                 listItem.appendChild(button);
                 document.getElementById('poi-list').appendChild(listItem);
             });

             stepControl.update();
             return { map, onResize };
         }

         document.getElementById('route-summary').textContent =
             `${formatDistance(route.distance_km)} · ${formatDuration(route.duration_min)} by ${route.mode} · Data from OpenStreetMap`;

         if (route.coordinates.length > 0) {
             const { map, onResize } = createRouteMap(route);
             map.whenReady(scheduleHeightReport);
             const mapResizeObserver = new ResizeObserver(() => {
                 requestAnimationFrame(() => {
                     onResize();
                     scheduleHeightReport();
                 });
             });
             mapResizeObserver.observe(document.getElementById('map'));
         } else {
             document.querySelector('.results').replaceChildren();
             const empty = document.createElement('p');
             empty.className = 'empty-results';
             empty.textContent = 'No route to display.';
             document.querySelector('.results').appendChild(empty);
         }

         window.addEventListener('load', scheduleHeightReport);
         window.addEventListener('resize', scheduleHeightReport);
         const resizeObserver = new ResizeObserver(scheduleHeightReport);
         resizeObserver.observe(document.getElementById('app'));
        </script>
    </body>
</html>
"""

ROUTE_MAP_UI = (ROUTE_MAP_UI.replace("{MAP_STYLE}", MAP_STYLE)
                .replace("{MAP_HEIGHT_SCRIPT}", MAP_HEIGHT_SCRIPT))

# Sent to the model after show_route_map, to stop it from faking or
# announcing a map.
SHOW_ROUTE_MAP_CONTEXT = {
    "status": "success",
    "message": (
        "The route map, with the turn-by-turn directions, is already displayed "
        "to the user directly ABOVE your reply. Do not announce the map or point "
        "to where it is, and never write placeholders such as [Map displayed]. "
        "Reply with a short sentence or two at most. Do NOT include images, "
        "markdown image syntax, map links, or the directions again."
    ),
}

# How long a route found by navigation can be shown on a map.
ROUTE_MAP_TTL = 7 * 24 * 3600


def save_route_for_map(route: dict, mode: str, start: dict, destination: dict,
                       start_name: str, destination_name: str) -> str:
    """
    Keep a route found by navigation, so show_route_map can draw exactly
    that route. Returns the route_id the model passes to show_route_map.
    """
    key_data = json.dumps([mode, route.get('geometry'), start_name, destination_name])
    route_id = hashlib.sha1(key_data.encode()).hexdigest()[:12]
    OsmCache().set(f"route_map_{route_id}", {
        'route': route, 'mode': mode,
        'start': {k: start[k] for k in ('lat', 'lon', 'display_name') if k in start},
        'destination': {k: destination[k] for k in ('lat', 'lon', 'display_name') if k in destination},
        'start_name': start_name, 'destination_name': destination_name,
    }, ttl=ROUTE_MAP_TTL)
    return route_id


def route_map_html(route: dict, mode: str, start: dict, destination: dict,
                   start_name: str, destination_name: str, tile_server: str) -> str:
    """
    The route map page for an ORS route (see ROUTE_MAP_UI). The names are
    the places as asked for: Nominatim's display names can start with
    something like a platform number.
    """
    def endpoint(place: dict, name: str) -> dict:
        return {'name': name, 'lat': float(place['lat']), 'lon': float(place['lon'])}

    coordinates = decode_polyline(route['geometry'])['coordinates'] if route.get('geometry') else []
    steps = [
        {
            'instruction': step.get('instruction', ''),
            'distance_km': step.get('distance', 0.0),
            'duration_s': step.get('duration', 0.0),
            'from': step.get('way_points', [0, 0])[0],
            'to': step.get('way_points', [0, 0])[-1],
        }
        for segment in route.get('segments', [])
        for step in segment.get('steps', [])
    ]
    summary = route.get('summary', {})
    payload = {
        'mode': TRAVEL_MODE_NAMES[mode],
        'distance_km': summary.get('distance', 0.0),
        'duration_min': summary.get('duration', 0.0) / 60.0,
        'start': endpoint(start, start_name),
        'end': endpoint(destination, destination_name),
        'coordinates': [[round(lat, 5), round(lon, 5)] for lon, lat in coordinates],
        'steps': steps,
    }

    title = f"Route from {start_name} to {destination_name}"
    # Escape "</" so a name or instruction can't close the <script> early.
    route_json = json.dumps(payload).replace("</", "<\\/")
    return (ROUTE_MAP_UI
            .replace("{REPLACE_WITH_ROUTE}", route_json)
            .replace("{REPLACE_WITH_TITLE}", html_escape(title))
            .replace("{REPLACE_WITH_TILE_SERVER}", tile_server))

#####################################################
# Category Type Aliases
#####################################################

UrbanSetting: TypeAlias = Literal['urban', 'suburban', 'rural']

TravelMode: TypeAlias = Literal['car', 'bike', 'walk']

# ORS profile and display name for each travel mode.
ORS_PROFILES = {
    'car': 'driving-car',
    'bike': 'cycling-regular',
    'walk': 'foot-walking',
}

TRAVEL_MODE_NAMES = {'car': 'car', 'bike': 'bike', 'walk': 'walking'}

FuelCategory: TypeAlias = Literal['fossil_fuels',  'ev_fast_charging']
FuelTypeCategory: TypeAlias = Literal['petrol', 'diesel', 'all']
EVChargerCategory: TypeAlias = Literal[
    'chademo', 'chademo3', 'chaoji', 'ccs2',
    'ccs1', 'gb/t', 'nacs', 'all'
]

# Religions from the OSM wiki. Some make use of the denomination tag.
# Others do not. any_religion is a special "category" that forces the
# tool to search for any place of worship amenity.
ReligionCategory: TypeAlias = Literal[
    'any_religion', 'buddhist', 'christian', 'hindu', 'jain', 'jewish', 'muslim',
    'pagan', 'shinto', 'sikh', 'taoist', 'zoroastrian',
    'ancestor', 'animist', 'antoinist', 'bahai', 'benzhu',
    'caodaism', 'chinese_folk', 'confucian', 'shamanic', 'scientologist',
    'self-realization_fellowship', 'spiritualist', 'tenrikyo',
    'unitarian_universalist','vietnamese_folk', 'voodoo',
    'yazidi',
    'multifaith', 'humanist', 'atheist', 'null', 'laica'
]

DenominationCategory: TypeAlias = Literal[
    # Forcibly search for any denomination within the specified
    # religion.
    'any_denomination',

    # Buddhist
    "buddhist|mahayana", "buddhist|gelug", "buddhist|jishu", "buddhist|jodo_shinshu",
    "buddhist|jodo_shu", "buddhist|nichiren", "buddhist|nyingma", "buddhist|obaku",
    "buddhist|pure_land", "buddhist|rinzai", "buddhist|risshu", "buddhist|shingon_shu",
    "buddhist|soto", "buddhist|tiantai", "buddhist|tibetan","buddhist|vajrayana", "buddhist|won",
    "buddhist|yogacara", "buddhist|yuzu_nembutsu", "buddhist|zen", "buddhist|theravada",
    "buddhist|thai_mahanikaya", "buddhist|thai_thammayut",

    # Christian
    "christian|catholic", "christian|armenian_catholic", "christian|chaldean_catholic",
    "christian|coptic_catholic", "christian|eritrean_catholic", "christian|ethiopian_catholic",
    "christian|greek_catholic", "christian|hungarian_greek_catholic", "christian|maronite",
    "christian|polish_catholic", "christian|roman_catholic", "christian|romanian_catholic",
    "christian|syriac_catholic", "christian|syro-malabar_catholic", "christian|ukrainian_greek_catholic",
    "christian|orthodox", "christian|antiochian_orthodox", "christian|armenian_apostolic",
    "christian|bulgarian_orthodox", "christian|eritrean_orthodox", "christian|ethiopian_orthodox",
    "christian|coptic_orthodox", "christian|georgian_orthodox", "christian|greek_orthodox",
    "christian|macedonian_orthodox", "christian|old_believers", "christian|polish_orthodox",
    "christian|romanian_orthodox", "christian|russian_orthodox", "christian|serbian_orthodox",
    "christian|syriac_orthodox", "christian|ukrainian_orthodox", "christian|protestant",
    "christian|adventist", "christian|anabaptist", "christian|anglican", "christian|baptist",
    "christian|disciples_of_christ", "christian|episcopal", "christian|evangelical",
    "christian|evangelical_covenant", "christian|exclusive_brethren", "christian|lutheran",
    "christian|mennonite", "christian|methodist", "christian|moravian", "christian|pentecostal",
    "christian|presbyterian", "christian|quaker", "christian|reformed", "christian|uniting",
    "christian|polish_national_catholic", "christian|african_methodist_episcopal",
    "christian|african_methodist_episcopal_zion", "christian|alliance", "christian|apostolic_faith",
    "christian|assemblies_of_god", "christian|brethren_in_christ", "christian|calvinistic_methodist",
    "christian|catholic_apostolic", "christian|church_of_god_in_christ", "christian|church_of_scotland",
    "christian|churches_of_christ", "christian|czechoslovak_hussite", "christian|dutch_reformed",
    "christian|evangelical_free_church_of_america", "christian|evangelical_lutheran",
    "christian|evangelical_free_church_of_france", "christian|foursquare",
    "christian|free_church_of_scotland", "christian|living_waters_church", "christian|mission_covenant_church_of_sweden",
    "christian|mormon", "christian|nazarene", "christian|new_frontiers", "christian|orthodox_presbyterian_church",
    "christian|pkn", "christian|remonstrant", "christian|salvation_army", "christian|scottish_episcopal",
    "christian|seventh_day_adventist", "christian|strict_baptist", "christian|temple_society_australia",
    "christian|united", "christian|united_free_church_of_scotland", "christian|united_reformed",
    "christian|united_methodist", "christian|united_church_of_christ", "christian|welsh_baptist",
    "christian|welsh_independent", "christian|apostolic", "christian|assyrian",
    "christian|catholic_mariavite", "christian|bosnian_church", "christian|charismatic",
    "christian|christian_community", "christian|christ_scientist", "christian|church_of_christ",
    "christian|congregational", "christian|ecumenical", "christian|fsspx", "christian|harrist",
    "christian|iglesia_ni_cristo", "christian|jehovahs_witness", "christian|kimbanguist",
    "christian|la_luz_del_mundo", "christian|liberal_catholic", "christian|mariavite",
    "christian|messianic_jewish", "christian|new_apostolic", "christian|nondenominational",
    "christian|old_catholic", "christian|philippine_independent", "christian|shizmatic",
    "christian|simultaneum", "christian|spiritist",

    # Hindu
    "hindu|shaktism", "hindu|hare_krishna", "hindu|vaishnavism", "hindu|shaivism",
    "hindu|smartism", "hindu|ganpatya", "hindu|warkari", "hindu|dattatreya",

    # Islamic
    "muslim|ahmadiyya", "muslim|alevi", "muslim|bektashi", "muslim|ibadi",
    "muslim|ismaili", "muslim|shia", "muslim|sunni", "muslim|sufi",

    # Jain
    "jain|digambara", "jain|svetambara",

    # Jewish
    "jewish|unaffiliated", "jewish|ashkenazi", "jewish|buchari",
    "jewish|conservative", "jewish|hasidic", "jewish|kabbalistic",
    "jewish|karaite", "jewish|lubavitch", "jewish|mizrachi",
    "jewish|modern_orthodox", "jewish|neo_orthodox", "jewish|orthodox",
    "jewish|reconstructionist", "jewish|reform", "jewish|progressive",
    "jewish|liberal", "jewish|samaritan", "jewish|sephardi",
    "jewish|traditional", "jewish|unity",

    # Pagan
    "pagan|asatru", "pagan|baltic", "pagan|celtic", "pagan|greco-roman",
    "pagan|slavic", "pagan|wicca",

    # Sikh
    "sikh|khalsa", "sikh|akali_nihang", "sikh|giani_samparda", "sikh|nirmala_sect",
    "sikh|sewapanthi", "sikh|udasi", "sikh|ramgarhia", "sikh|radha_soami",
    "sikh|namdhari", "sikh|nirankari", "sikh|nanakpanthi", "sikh|ramraiyas",

    # Taoist
    "taoist|quanzhen", "taoist|zhengyi",

    # Zoroastrian
    "zoroastrian|irani", "zoroastrian|parsi",
]

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
    "relation": "R",
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

# Why ORS found no route, and what the model should tell the user.
ORS_FAILURE_INSTRUCTIONS = {
    'not_covered': ("No route could be found. Tell the user that the routing "
                    "service found no route, possibly because it doesn't cover "
                    "this area."),
    'refused': ("The routing service refused the request (no access, or its "
                "usage limit is reached). Tell the user that routing is "
                "unavailable right now."),
    'too_far': ("The route is longer than the routing service allows. Tell "
                "the user the trip is too long to route."),
    'unavailable': ("The routing service didn't respond. Tell the user that "
                    "routing is unavailable right now, and to try again later."),
    'error': ("The routing service returned an error. Tell the user that "
              "routing failed."),
}

def no_route(failure: Optional[str]) -> dict:
    instructions = ORS_FAILURE_INSTRUCTIONS.get(failure or 'not_covered',
                                                ORS_FAILURE_INSTRUCTIONS['error'])
    return {
        "results": [],
        "instructions": f"{instructions} Do not make up a route or travel times."
    }

def ors_failure_kind(error: Exception) -> str:
    """Group an ORS error into a key of ORS_FAILURE_INSTRUCTIONS."""
    if isinstance(error, openrouteservice.exceptions.ApiError):
        body = error.message if isinstance(error.message, dict) else {}
        details = body.get('error') if isinstance(body.get('error'), dict) else {}
        code = details.get('code')
        if error.status in (401, 403, 429):
            return 'refused'
        if code == 2004:
            return 'too_far'
        if error.status == 404 or code in (2009, 2010, 6010):
            return 'not_covered'
        if error.status >= 500:
            return 'unavailable'
        return 'error'
    if isinstance(error, (openrouteservice.exceptions.Timeout,
                          openrouteservice.exceptions.HTTPError,
                          requests.exceptions.RequestException)):
        return 'unavailable'
    return 'error'

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

NO_CITATION_LIST_PROHIBITION = (
    "**Do NOT** append a citation list, reference list, source list, or "
    "list of links at the end of your message."
)

# The interface automatically renders a source/citation entry below the
# message for every result, so a manually printed list of citations or
# links at the end of the message is always redundant.
NO_CITATION_LIST = (
    f"{NO_CITATION_LIST_PROHIBITION} The interface automatically displays "
    "the source of each result below your message, so a manual "
    "list is redundant. Use inline citations in the message body only."
)

CITATION_FORMAT = (
    "Cite with ONLY the bare citation_id inside square brackets, e.g. [3]. "
    "No words, prefixes, or other punctuation inside or around the brackets. "
    "WRONG: [citation 3], [id:3], [source 3], [Citation: 3], (3), [^3]. "
    "RIGHT: [3]."
)

CITATION_INSTRUCTIONS = [
    "Always use inline citations, using the citation_id of each source.",
    CITATION_FORMAT,
    "Always use inline citations for information relating to a result. ",
    NO_CITATION_LIST
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
        "Make the map link into a nice human-readable markdown link. "
        f"{NO_CITATION_LIST_PROHIBITION}"
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

PUBLIC_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass request pacing. The tool keeps no state between calls, so the
# Retry-After deadline per host lives in the cache file, and a lock file
# makes concurrent tool calls (and processes) send one request at a time.
OVERPASS_LOCK_FILE = os.path.join(tempfile.gettempdir(), "osm-overpass.lock")
OVERPASS_MAX_WAIT = 15        # seconds; longer waits fail (or fall back)
OVERPASS_DEFAULT_BACKOFF = 10 # seconds, for a 429 without Retry-After

class OverpassBusy(Exception):
    pass

class OverpassLock:
    def __enter__(self):
        self.file = open(OVERPASS_LOCK_FILE, 'w')
        if fcntl:
            fcntl.flock(self.file, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if fcntl:
            fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()

def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Retry-After in seconds: either a number of seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None

def overpass_request(url: str, query: str, headers: dict, timeout: int) -> requests.Response:
    """
    Send one Overpass query, one request at a time across tool calls,
    honoring Retry-After. A 429 (or a 503/504 with Retry-After) records
    when the host may be asked again. Short waits are slept off and the
    request is retried once; longer ones raise OverpassBusy.
    """
    host = api_host(url)
    key = f"overpass_retry_after_{host}"

    with OverpassLock():
        response = None
        for attempt in range(2):
            wait = (OsmCache().get(key) or 0) - time.time()
            if wait > OVERPASS_MAX_WAIT:
                raise OverpassBusy(f"{host} is rate limiting: retry in {round(wait)}s")
            if wait > 0:
                print(f"[OSM] Waiting {round(wait)}s before asking {host} (Retry-After)")
                time.sleep(wait)

            response = requests.get(url, params={"data": query}, headers=headers, timeout=timeout)
            retry_after = parse_retry_after(response.headers.get('Retry-After'))
            limited = (response.status_code == 429
                       or (response.status_code in (503, 504) and retry_after is not None))
            if not limited:
                return response

            delay = retry_after if retry_after is not None else OVERPASS_DEFAULT_BACKOFF
            OsmCache().set(key, time.time() + delay, ttl=delay)
            print(f"[OSM] {host} is rate limiting (HTTP {response.status_code}), "
                  f"retry after {round(delay)}s")

        return response

def api_host(url: str) -> str:
    """Host of an API URL, for logging which instance a request goes to."""
    return urlparse(url).netloc or url

def straight_line_distance(from_thing: dict, to_thing: dict) -> float:
    """Haversine distance in km. Nominatim results have string coordinates."""
    return OsmUtils.haversine_distance(
        {'lat': float(from_thing['lat']), 'lon': float(from_thing['lon'])},
        {'lat': float(to_thing['lat']), 'lon': float(to_thing['lon'])}
    )

def locality_instructions() -> str:
    return (
        "This describes the area around the requested place, within 1.5 km. "
        "Use it to answer the user's question about the place or about getting "
        "around there. The numbers are shares of road length mapped in "
        "OpenStreetMap: treat them as rough indications, not exact figures, "
        "and don't list raw numbers unless the user asks for them."
    )

def unavailable_travel_mode(travel_mode: str, available: List[str]) -> str:
    available_names = ", ".join(TRAVEL_MODE_NAMES[mode] for mode in available)
    return (
        f"Navigation by {TRAVEL_MODE_NAMES[travel_mode]} is not available. "
        "Tell the user that this travel method is not supported, and "
        f"that the available travel methods are: {available_names or 'none'}. "
        "Do not make up a route."
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
            " **Mention that the central point of the town or city that the user is searching in "
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
        f" Use inline citations for information relating to a result. "
        f"{CITATION_FORMAT} "
        f"{NO_CITATION_LIST}"
    )


#####################################################
# Event Emission
#####################################################

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

    async def search_complete(self, category: str, place: str, items=[], count=None):
        if not self.status_indicators:
            return

        num_results = len(items) if count is None else count
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
                "metadata": [{"source": f"{source_name} ({thing.get('citation_id')})", "html": True }],
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


#####################################################
# OSM-Related Utility Functions (AKA spaghetti land)
#####################################################

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


#####################################################
# OSM Result Parsing
#####################################################

# For each search category, we may want to add additional tags
# directly from the raw OSM data for a given search result. This is a
# map of search category as defined by the tool functions to OSM tag
# names. If a tag is in the list for the category, that tag will be
# included in search results' additional_info field (assuming it
# exists on the OSM entity).
#
# Tag values ending with a :* have a special meaning. The tool will
# find all "sub-tags" that begin with the prefix. For example,
# "socket:*" will find "socket:type2", "socket:type2:output",
# "socket:type1" and so on. This can work on "sub-sub-tags" too.
# For example, "socket:type2:*" would only look for tags that
# start with the prefix "socket:type2".
RAW_DATA_BY_CATEGORY = {
    # stores
    "groceries": { "organic", "fair_trade", "origin" },
    "convenience": { "organic", "fair_trade", "origin", "ice_cream" },
    "cannabis": { "cannabis:*" },
    "drugs": { "cannabis:*" },
    "electronics": { "electronics", "repair", "second_hand" },

    # food and drink
    "sit_down_restaurants": {"cuisine", "reservation"},
    "fast_food": {"cuisine"},
    "cafe_or_bakery": {"cuisine"},
    "bars_and_pubs": {"cocktails", "reservation"},

    # recreation
    "sports": { "sport", "surface", "lit" },
    "swimming": { "swimming_lanes", "outdoor", "indoor", "location",
                 "depth", "heated", "lifeguard", "sport:scuba_diving"},
    "playgrounds": { "surface", "min_age", "max_age", "fee", "playground=",
                     "playground:*", "stroller", "fenced"},

    # tourism
    "accommodation": { "stars", "internet_access", "rooms", "check_in", "check_out" },
    "tourist_attractions": { "museum", "outdoor", "zoo", "fee", "artwork_type",
                             "artist_name" },

    # education
    "schools": { "capacity", "grades", "isced:level", "min_age", "max_age", "religion",
                 "pedagogy", "language", "school:language" },

    # healthcare
    "pharmacies": { "dispensing", "drive_through" },

    # fuel etc
    "fossil_fuels": { "self_service", "full_service", "full_service:conditional", "car_wash",
                      "compressed_air", "toilets", "water_point", "sanitary_dump_station"  },
    "ev_fast_charging": {"fee", "socket:*", "payment:*", "authentication:*",
                         "motorcar", "truck", "ref" },

}

class OsmParser:
    """All result parsing-related functionality in one place."""

    def __init__(
        self, original_location: str, things_nearby: List[dict],
        category: Optional[str], include_raw_data=True
    ):
        self.original_location = original_location
        self.things_nearby = things_nearby
        self.include_raw_data = include_raw_data
        self.category = category

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
    def extract_raw_data(category: str, thing: dict) -> dict:
        tags_to_extract = RAW_DATA_BY_CATEGORY.get(category, None)
        additional_info = {}

        if not tags_to_extract:
            return additional_info

        # loop through all tags in thing. if regular tag to extract,
        # add to dict. if wildcard, do a beginswith check.
        additiona_info = {}
        tags: dict = thing.get('tags', {})

        for tag_name in tags_to_extract:
            if tag_name.endswith(':*'):
                tag_prefix = tag_name.split(':*')[0] + ":"
                for tag, value in tags.items():
                    if tag.startswith(tag_prefix):
                        additional_info[tag] = value
            else:
                for tag, value in tags.items():
                    if tag == tag_name:
                        additional_info[tag] = value

        return additional_info

    @staticmethod
    def parse_and_validate_thing(thing: dict, category: Optional[str]) -> Optional[dict]:
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
        distance: Optional[float] = thing.get('distance')
        nav_distance: Optional[float] = thing.get('nav_distance')
        opening_hours: Optional[str] = tags.get('opening_hours')
        operator: Optional[str] = tags.get('operator')
        access: Optional[str] = tags.get('access')
        brand: Optional[str] = tags.get('brand')
        website: Optional[str] = get_or_none(
            tags, "operator:website", "website", "brand:website"
        )

        # Distances only exist for searches around a place, not for direct
        # lookups (find_place); leave them out then. Without a travel
        # distance, the straight-line one is a lower bound for it.
        if distance is not None:
            friendly_thing['distance'] = "{:.3f}".format(distance)
            if nav_distance is not None:
                friendly_thing['nav_distance'] = "{:.3f}".format(nav_distance) + " km"
            else:
                friendly_thing['nav_distance'] = f"a bit more than {friendly_thing['distance']} km"

        if brand:
            friendly_thing['brand'] = brand

        if operator:
            friendly_thing['operator'] = operator

        if access:
            friendly_thing['access'] = access

        friendly_thing['osm_id'] = thing.get('id', 'unknown')
        friendly_thing['osm_type'] = thing.get('type', None)
        friendly_thing['name'] = name if name else "unknown"
        friendly_thing['address'] = address if address else "unknown"
        friendly_thing['lat'] = lat if lat else "unknown"
        friendly_thing['lon'] = lon if lon else "unknown"
        friendly_thing['amenity_type'] = amenity_type if amenity_type else "unknown"
        friendly_thing['opening_hours'] = opening_hours if opening_hours else "not recorded"
        friendly_thing['website'] = website if website else osm_link
        friendly_thing['phone_number'] = thing.get('phone', 'unknown')


        if category:
            friendly_thing['additional_info'] = OsmParser.extract_raw_data(category, thing)

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
            friendly_thing = OsmParser.parse_and_validate_thing(thing, self.category)
            if not friendly_thing:
                continue

            map_link = OsmUtils.create_osm_link(friendly_thing['lat'], friendly_thing['lon'])

            has_distance = use_distance and 'distance' in friendly_thing
            hv_distance_json = (f"{friendly_thing['distance']} km"
                                if has_distance
                                else "unavailable")

            trv_distance_json = (f"{friendly_thing['nav_distance']}"
                                 if has_distance
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
            })

            if self.include_raw_data:
                friendly_thing["raw_osm_json"] = thing

            entries.append(friendly_thing)

        if len(entries) == 0:
            return None

        return entries


#####################################################
# Caching of OSM Data
#####################################################

# How long cached data stays valid.
CACHE_TTL_ROUTES = 30 * 24 * 3600   # routes and travel distances (maps change)
CACHE_TTL_PLACES = 90 * 24 * 3600   # Nominatim search and lookup results

def osm_data_dir() -> str:
    """
    Persistent directory for the tool's data: Open WebUI's DATA_DIR (a
    volume in Docker, backend/data in a pip install), else the DATA_DIR
    environment variable, else the temp directory.
    """
    try:
        from open_webui.env import DATA_DIR
        base = str(DATA_DIR)
    except Exception:
        base = os.environ.get("DATA_DIR") or tempfile.gettempdir()

    path = os.path.join(base, "cache", "osm_tool")
    os.makedirs(path, exist_ok=True)
    return path

class OsmCache:
    """
    Persistent key/value cache in SQLite, in Open WebUI's data directory.
    Safe for concurrent tool calls and processes. Values are stored as
    JSON. Entries can expire (ttl in seconds); expired ones are removed.
    """
    FILENAME = "cache.sqlite3"

    def __init__(self, filename: Optional[str] = None):
        self.filename = filename or os.path.join(osm_data_dir(), self.FILENAME)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS cache ("
                       "key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS cache_expires_at ON cache (expires_at)")
            db.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.filename, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            yield db
            db.commit()
        finally:
            db.close()

    def get(self, key):
        """The cached value, or None if missing or expired."""
        with self._db() as db:
            row = db.execute("SELECT value FROM cache WHERE key = ? "
                             "AND (expires_at IS NULL OR expires_at > ?)",
                             (key, time.time())).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key, value, ttl: Optional[float] = None):
        expires_at = time.time() + ttl if ttl is not None else None
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                       (key, json.dumps(value), expires_at))

    def delete(self, key):
        with self._db() as db:
            db.execute("DELETE FROM cache WHERE key = ?", (key,))

    def get_or_set(self, key, func_to_call, ttl: Optional[float] = None):
        """
        Retrieve the value from the cache for a given key. If the key is not found,
        call `func_to_call` to generate the value and store it in the cache.

        :param key: The key to look up or set in the cache
        :param func_to_call: A callable function that returns the value if key is missing
        :param ttl: Seconds until the stored value expires (None: never)
        :return: The cached or generated value
        """
        value = self.get(key)
        if value is None:
            value = func_to_call()
            self.set(key, value, ttl)
        return value

    def clear_cache(self):
        """
        Clear all entries from the cache.
        """
        with self._db() as db:
            db.execute("DELETE FROM cache")


#####################################################
# Navigation Routing
#####################################################

class OrsRouter:
    PUBLIC_ORS_URL = "https://api.openrouteservice.org"

    # The public ORS has no status endpoint, but serves all of these.
    DEFAULT_PROFILES = set(ORS_PROFILES.values())

    # Haversine distance (km) up to which walking is preferred when
    # no travel mode is requested.
    WALKING_DISTANCE = 1.5

    # How long the profiles an ORS instance serves are cached. In the
    # cache database, not in memory: Open WebUI reloads tool code.
    PROFILE_CACHE_SECONDS = 3600

    def __init__(
        self, valves, user_valves, events: OsmEventEmitter
    ):
        self.cache = OsmCache()
        self.events = events

        # (base_url, api_key, client), tried in order. A custom instance
        # comes first; the public one is a fallback if enabled.
        self._instances = []
        if valves.ors_api_key is not None and valves.ors_api_key != "":
            if valves.ors_instance is not None:
                client = openrouteservice.Client(
                    base_url=valves.ors_instance,
                    key=valves.ors_api_key,
                    retry_over_query_limit=False
                )
                self._instances.append((valves.ors_instance, valves.ors_api_key, client))

                # The fallback key if set, else the regular one (which may
                # also be a real public ORS key).
                fallback_key = getattr(valves, 'ors_fallback_api_key', None) or valves.ors_api_key
                if getattr(valves, 'fallback_to_public_instances', False):
                    client = openrouteservice.Client(key=fallback_key, retry_over_query_limit=False)
                    self._instances.append((self.PUBLIC_ORS_URL, fallback_key, client))
            else:
                client = openrouteservice.Client(key=valves.ors_api_key, retry_over_query_limit=False)
                self._instances.append((self.PUBLIC_ORS_URL, valves.ors_api_key, client))

        self._client = self._instances[0][2] if self._instances else None

        custom_only = len(self._instances) == 1 and valves.ors_instance is not None
        self._fallback_hint = (" (public fallback is off: see fallback_to_public_instances)"
                               if custom_only else "")

    def _fetch_profiles(self, base_url: str, api_key: str) -> Tuple[set, bool]:
        """
        Ask an ORS instance which profiles it serves. Returns the
        profiles and whether the answer may be cached. Instances without
        a status endpoint (the public ORS) get the default profiles.
        """
        url = f"{base_url.rstrip('/')}/v2/status"
        print(f"[OSM] Getting ORS profiles from {api_host(url)}")
        try:
            resp = requests.get(url, headers={"Authorization": api_key}, timeout=10)
            if resp.status_code == 404:
                return set(self.DEFAULT_PROFILES), True

            resp.raise_for_status()
            profiles = resp.json().get('profiles', {})

            # ORS 8+ keys profiles by name. Older versions use
            # 'profile N' keys with a space-separated 'profiles' field.
            found = set(profiles.keys())
            for profile in profiles.values():
                if isinstance(profile, dict):
                    found.update(str(profile.get('profiles', '')).split())

            return found, True
        except Exception as e:
            print(f"[OSM] Could not get ORS profiles, assuming defaults: {e}")
            return set(self.DEFAULT_PROFILES), False

    def _profiles(self, base_url: str, api_key: str) -> set:
        cache_key = f"ors_profiles_{base_url}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return set(cached)

        profiles, cacheable = self._fetch_profiles(base_url, api_key)
        if cacheable:
            self.cache.set(cache_key, sorted(profiles), ttl=self.PROFILE_CACHE_SECONDS)
        return profiles

    def available_modes(self) -> List[str]:
        """Travel modes any configured ORS instance (incl. fallback) can route."""
        profiles = set()
        for base_url, api_key, _ in self._instances:
            profiles |= self._profiles(base_url, api_key)

        return [mode for mode, profile in ORS_PROFILES.items() if profile in profiles]

    # Straight-line distances (km) where bike and car travel times are
    # compared when no travel mode is requested. Longer trips go by car.
    COMPARE_MAX_DISTANCE = 15.0

    # Bike wins if it takes at most this many times as long as the car,
    # by whether the area is bike-friendly (None: no street data).
    BIKE_TIME_FACTOR = {True: 2.0, None: 1.5, False: 1.2}

    def is_available(self, mode: str) -> bool:
        """Whether any instance (incl. the public fallback) serves a mode."""
        return mode in self.available_modes()

    def auto_modes(self, distance: float, bike_friendly: Optional[bool]) -> List[str]:
        """
        Candidate modes when none was requested, from the straight-line
        distance (km) and the area's bike-friendliness (see OsmLocality).
        Two candidates (car and bike) mean: compare travel times with
        prefer_bike(). A mode counts as available if any instance serves
        it: the custom one if it has it, else the public fallback (if on).
        """
        available = self.available_modes()
        if distance <= self.WALKING_DISTANCE:
            if 'walk' in available:
                return ['walk']
            order = ['car', 'bike'] if bike_friendly is False else ['bike', 'car']
        elif distance <= self.COMPARE_MAX_DISTANCE and 'car' in available and 'bike' in available:
            return ['car', 'bike']
        else:
            order = ['car', 'bike']

        return [mode for mode in order if mode in available][:1]

    def auto_mode_reason(
        self, distance: float, mode: str, bike_friendly: Optional[bool]
    ) -> str:
        """Why auto_modes() picked a single mode, for the model to relay."""
        name = TRAVEL_MODE_NAMES[mode]
        if distance <= self.WALKING_DISTANCE:
            if mode == 'walk':
                return "short trip, within walking distance"
            area = {True: "the area is bike-friendly",
                    None: "the area is only partly bike-friendly (or there's no data)",
                    False: "the area isn't especially bike-friendly"}[bike_friendly]
            return f"short trip; walking routes aren't available and {area}, so {name}"
        if distance > self.COMPARE_MAX_DISTANCE:
            return f"long trip (over {round(self.COMPARE_MAX_DISTANCE)} km in a straight line), so {name}"
        return f"only {name} routes are available"

    def prefer_bike(
        self, bike_minutes: float, car_minutes: float, bike_friendly: Optional[bool]
    ) -> bool:
        return bike_minutes <= car_minutes * self.BIKE_TIME_FACTOR[bike_friendly]

    def calculate_route(
        self, from_thing: dict, to_thing: dict, mode: str
    ) -> Optional[Tuple[dict, str]]:
        """
        Calculate route between A and B by the given mode. Returns the
        route and the mode, if successful, or None if the route could not
        be calculated, the mode isn't available, or ORS is not configured.
        Each instance serving the mode is tried in turn, so a custom
        instance without data for the area falls back to the public one.
        Raises the last error if every instance failed with one.
        """
        if not self._client or not self.is_available(mode):
            return None

        coords = ((from_thing['lon'], from_thing['lat']),
                  (to_thing['lon'], to_thing['lat']))

        # check cache first.
        cache_key = f"ors_route_{mode}_{str(coords)}"
        cached_route = self.cache.get(cache_key)
        if cached_route:
            print("[OSM] Got route from cache!")
            return cached_route, mode

        profile = ORS_PROFILES[mode]
        error = None
        for base_url, api_key, client in self._instances:
            if profile not in self._profiles(base_url, api_key):
                continue

            print(f"[OSM] Routing via ORS ({api_host(base_url)}), profile {profile}")
            try:
                resp = ors_directions(client, coords, profile=profile,
                                      preference="fastest", units="km")
            except Exception as e:
                print(f"[OSM] ORS error from {api_host(base_url)}: {e}{self._fallback_hint}")
                error = e
                continue

            routes = resp.get('routes', [])
            if len(routes) > 0:
                self.cache.set(cache_key, routes[0], ttl=CACHE_TTL_ROUTES)
                return routes[0], mode
            print(f"[OSM] No route from {api_host(base_url)}{self._fallback_hint}")

        if error:
            raise error
        return None

    def calculate_distances(
        self, origin: dict, things: List[dict], bike_friendly: Optional[bool]
    ) -> dict:
        """
        Travel distances (km) from origin to each thing, keyed by thing id,
        with the mode for each thing chosen like for navigation
        (auto_modes, prefer_bike). Uses one ORS matrix request per mode
        instead of one route per thing. Things without any route are left
        out.
        """
        if not self._client or not things:
            return {}

        candidates = {}
        by_mode = {}
        for thing in things:
            distance = thing.get('distance')
            if distance is None:
                distance = straight_line_distance(origin, thing)
            candidates[thing['id']] = self.auto_modes(distance, bike_friendly)
            for mode in candidates[thing['id']]:
                by_mode.setdefault(mode, []).append(thing)

        found = {mode: self._matrix_for_mode(origin, group, mode)
                 for mode, group in by_mode.items()}

        distances = {}
        for thing in things:
            results = {mode: found[mode][thing['id']] for mode in candidates[thing['id']]
                       if thing['id'] in found.get(mode, {})}
            if 'car' in results and 'bike' in results:
                bike_minutes, car_minutes = results['bike'][1] / 60.0, results['car'][1] / 60.0
                mode = 'bike' if self.prefer_bike(bike_minutes, car_minutes, bike_friendly) else 'car'
            elif results:
                mode = next(iter(results))
            else:
                continue
            distances[thing['id']] = results[mode][0]

        return distances

    def _matrix_for_mode(self, origin: dict, things: List[dict], mode: str) -> dict:
        """
        {thing id: (distance km, duration s)} for one mode. Things an
        instance can't route are retried on the next one (fallback).
        """
        profile = ORS_PROFILES[mode]
        results = {}
        for base_url, api_key, client in self._instances:
            if not things:
                break
            if profile not in self._profiles(base_url, api_key):
                continue

            print(f"[OSM] Distance matrix via ORS ({api_host(base_url)}), "
                  f"profile {profile}, {len(things)} destinations")
            try:
                results.update(self._matrix_distances(client, origin, things, profile))
            except Exception as e:
                print(f"[OSM] ORS matrix error from {api_host(base_url)}: {e}{self._fallback_hint}")
                continue

            things = [thing for thing in things if thing['id'] not in results]

        return results

    @staticmethod
    def _matrix_distances(client, origin: dict, things: List[dict], profile: str) -> dict:
        """
        One matrix request from origin to things: {thing id: (distance
        km, duration s)}. ORS rejects the whole request if any destination
        is outside its map ("Destination point(s) [i, ...] out of bounds",
        0-based destination indices), so retry once without those.
        """
        for attempt in range(2):
            locations = [[float(origin['lon']), float(origin['lat'])]]
            locations += [[float(thing['lon']), float(thing['lat'])] for thing in things]
            try:
                resp = ors_distance_matrix(client, locations, profile=profile, sources=[0],
                                           destinations=list(range(1, len(locations))),
                                           metrics=['distance', 'duration'], units='km')
            except Exception as e:
                match = re.search(r"Destination point\(s\) \[([\d,\s]+)\]", str(e))
                if attempt > 0 or not match:
                    raise
                bad = {int(index) for index in match.group(1).split(',')}
                print(f"[OSM] {len(bad)} destination(s) outside the ORS map, retrying without them")
                things = [thing for index, thing in enumerate(things) if index not in bad]
                if not things:
                    return {}
                continue

            distances = (resp.get('distances') or [[]])[0]
            durations = (resp.get('durations') or [[]])[0]
            return {thing['id']: (distance, duration)
                    for thing, distance, duration in zip(things, distances, durations)
                    if distance is not None and duration is not None}

        return {}


#####################################################
# Main Search Logic
#####################################################

class OsmSearcher:
    def __init__(self, valves, user_valves, event_emitter):
        self.events = event_emitter
        self.valves = valves
        self.user_valves = user_valves

        # config settings
        self.nominatim_url = valves.nominatim_url
        self.overpass_turbo_url = valves.overpass_turbo_url
        self.fallback_to_public = getattr(valves, 'fallback_to_public_instances', False)
        self.user_agent = valves.user_agent
        self.from_header = valves.from_header
        self.include_raw_data = valves.include_raw_osm_data
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
        Post-process an OSM way or relation (an area, like a park or a
        building) into a single GPS coordinate: the center of its bounding
        box, which Overpass returns for "out center". Drops geometry, node,
        and member info. Also accepts "out geom" results (bounds).
        """
        for key in ('nodes', 'geometry', 'members'):
            way.pop(key, None)

        if 'center' in way:
            way['lat'] = way['center']['lat']
            way['lon'] = way['center']['lon']
            del way['center']
            return way

        if 'bounds' in way:
            way_center = OsmUtils.get_bounding_box_center(way['bounds'])
            way['lat'] = way_center['lat']
            way['lon'] = way_center['lon']
            del way['bounds']
            return way

        return None

    @staticmethod
    def flatten_tags(tag_query):
        """
        Make sure that the tags queried results in a completely
        flat list of OSM tags. That is, remove AND groups.
        """
        flat_tags = []
        for str_or_list in tag_query:
            if isinstance(str_or_list, str):
                flat_tags.append(str_or_list)
            else:
                flat_tags += [tag for tag in str_or_list]

        return flat_tags

    @staticmethod
    def group_tags(tags):
        """
        De-duplicate tags of the same type into a single Overpass
        Turbo search term. Also pluck out anything that needs to be
        ANDed together.
        """
        result = {}
        and_queries = []
        for tag in tags:
            # skip AND queries because they must be handled
            # separately.
            if not isinstance(tag, str):
                and_queries.append(tag)
                continue

            key, value = tag.split('=')
            if key not in result:
                result[key] = []
            result[key].append(value)

        return result, and_queries

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

    # Coverage checks are cached per ~1 km grid cell for this long.
    COVERAGE_CACHE_SECONDS = 30 * 24 * 3600

    def overpass_covers(self, url: str, center: dict, headers: dict) -> bool:
        """
        Whether an Overpass instance has map data around a point, judged
        by any roads within 1 km. Self-hosted instances often hold only a
        regional extract, so an empty search result can mean "outside my
        map" rather than "nothing there". On errors, assume it's covered.
        """
        lat, lon = float(center['lat']), float(center['lon'])
        cache_key = f"overpass_coverage_{api_host(url)}_{round(lat, 2)}_{round(lon, 2)}"
        cache = OsmCache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        query = f"[out:json][timeout:25];way[highway](around:1000,{lat},{lon});out count;"
        print(f"[OSM] Checking whether {api_host(url)} has data around {lat},{lon}")
        try:
            response = overpass_request(url, query, headers, timeout=30)
            response.raise_for_status()
            elements = response.json().get('elements') or [{}]
            covered = int(elements[0].get('tags', {}).get('ways', 0)) > 0
        except Exception as e:
            print(f"[OSM] Coverage check failed, assuming covered: {e}")
            return True

        cache.set(cache_key, covered, ttl=self.COVERAGE_CACHE_SECONDS)
        return covered

    def overpass_get(
        self, query: str, headers: dict, has_data=None, center: Optional[dict] = None
    ) -> requests.Response:
        """
        Run an Overpass query. With public fallback on and a custom
        instance, retry on the public instance if the custom one returns an
        error, or returns nothing for a place (center) outside its map
        data. An empty result inside its map is a real "nothing there".
        """
        if has_data is None:
            has_data = lambda result: len(result.get('elements', [])) > 0

        urls = [self.overpass_turbo_url]
        if self.fallback_to_public and self.overpass_turbo_url != PUBLIC_OVERPASS_URL:
            urls.append(PUBLIC_OVERPASS_URL)

        if len(urls) == 1 and self.overpass_turbo_url != PUBLIC_OVERPASS_URL:
            fallback_hint = " (public fallback is off: see fallback_to_public_instances)"
        else:
            fallback_hint = ""

        response, error = None, None
        for index, url in enumerate(urls):
            print(f"[OSM] Overpass query via {api_host(url)}")
            try:
                response = overpass_request(url, query, headers, timeout=90)
                if response.status_code == 200 and has_data(response.json()):
                    return response
                print(f"[OSM] No data from {api_host(url)} (HTTP {response.status_code})"
                      f"{fallback_hint}")
            except Exception as e:
                print(f"[OSM] Overpass error from {api_host(url)}: {e}{fallback_hint}")
                error = e
                continue

            is_last = index == len(urls) - 1
            if (not is_last and response.status_code == 200 and center is not None
                    and self.overpass_covers(url, center, headers)):
                print(f"[OSM] {api_host(url)} has map data here: no results is the answer")
                return response

        if response is None:
            raise error
        return response

    def create_headers(self) -> Optional[dict]:
        if len(self.user_agent) == 0 or len(self.from_header) == 0:
            return None

        return {
            'User-Agent': self.user_agent,
            'From': self.from_header
        }

    def add_nav_distance_to_things(self, origin, things_nearby: List[dict]) -> bool:
        """
        Update distances to use ORS navigable distances, if ORS enabled.
        The travel mode per thing is chosen like for navigation, from the
        locality of the search origin and the distance. Uncached distances
        are fetched in one matrix request per travel mode. Things without
        a route get their straight-line distance (not cached), since
        sorting needs a distance for every thing.
        """
        cache = OsmCache()
        cache_key = lambda thing: f"ors_{origin}_{thing['id']}"

        nav_distances = {}
        uncached = []
        for thing in things_nearby:
            cached = cache.get(cache_key(thing))
            if cached:
                print(f"[OSM] Got nav distance for {thing['id']} from cache!")
                nav_distances[thing['id']] = cached
            else:
                uncached.append(thing)

        if uncached:
            try:
                place = {'lat': origin['lat'], 'lon': origin['lon'],
                         'display_name': None, 'address': {}}
                locality = OsmLocality(self.valves, self.user_valves, self.events)
                bike_friendly = locality.describe(place)['bike_friendly']
                found = self._ors.calculate_distances(origin, uncached, bike_friendly)
            except Exception as e:
                print(f"[OSM] Error querying ORS: {e}")
                found = {}

            for thing in uncached:
                if thing['id'] in found:
                    cache.set(cache_key(thing), found[thing['id']], ttl=CACHE_TTL_ROUTES)
            nav_distances.update(found)

        if not nav_distances:
            return False

        for thing in things_nearby:
            if thing['id'] not in nav_distances:
                print(f"[OSM] No route to {thing['id']}, using straight-line distance")
            thing['nav_distance'] = round(nav_distances.get(thing['id'], thing['distance']), 3)

        return True

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
        rel_osm_type = nominatim_result.get("osm_type", None) == "relation"
        rel_type = nominatim_result.get("type", None) == "administrative"
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
        """
        Enrich search results missing data with extra info from
        Nominatim.
        """
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
            print(f"[OSM] Looking up {len(lookups)} things from Nominatim")

        url = urljoin(self.nominatim_url, "lookup")
        print(f"[OSM] Nominatim lookup via {api_host(url)}")
        params = {
            'osm_ids': ",".join(lookups),
            'format': format
        }

        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        response = await asyncio.to_thread(requests.get, url, params=params, headers=headers, timeout=30)
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
                        cache.set(lookup, updated, ttl=CACHE_TTL_PLACES)
                        updated_things.append(updated)

            return updated_things
        else:
            await self.events.error(Exception(response.text))
            print(response.text)
            return []


    async def nominatim_search(self, query, format="json", limit: int=1,
                               quiet: bool=False) -> Optional[dict]:
        normalized_query = " ".join(query.lower().split())
        cache_key = f"nominatim_search_{limit}_{normalized_query}"
        cache = OsmCache()
        data = cache.get(cache_key)

        if data:
            print(f"[OSM] Got nominatim search data for {query} from cache!")
            if not quiet:
                await self.events.resolving(done=True, message=query, items=data[:limit])
            return data[:limit]

        if not quiet:
            await self.events.resolving(done=False, message=query)

        url = urljoin(self.nominatim_url, "search")
        print(f"[OSM] Searching Nominatim ({api_host(url)}) for: {query}")
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

        response = await asyncio.to_thread(requests.get, url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()

            if not data:
                raise ValueError(f"No results found for query '{query}'")

            if not quiet:
                await self.events.resolving(done=True, message=query, items=data[:limit])

            print(f"[OSM] Got result from Nominatim for: {query}")
            cache.set(cache_key, data[:limit], ttl=CACHE_TTL_PLACES)
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
        Each item in the tag list is treated as an OR query. An entry
        in the tag list can be a string (basic search for "this tag"),
        or it can be a list of strings itself, which is treated as an
        AND query.
        """
        print(f"[OSM] Searching Overpass ({api_host(self.overpass_turbo_url)}) around origin {place}")
        headers = self.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        center = OsmUtils.get_bounding_box_center(bbox)
        around = f"(around:{radius},{center['lat']},{center['lon']})"

        tag_groups, and_queries = OsmSearcher.group_tags(tags)

        # tags is e.g. [amenity=restaurant, place=blah, ...].
        # and_queries is a list of those lists. These tags will be
        # ANDed in Overpass, i.e. the search must fulfill all
        # conditions. Normally, we OR instead.
        and_groups = ["".join(f"[{tag}]" for tag in tags) for tags in and_queries]


        search_groups = [f'"{tag_type}"~"{"|".join(values)}"'
                         for tag_type, values in tag_groups.items()]

        not_search_groups = [f'"{tag_type}"!~"{"|".join(values)}"'
                                 for tag_type, values in not_tag_groups.items()]

        not_search_query = "".join([f"[{query}]" for query in not_search_groups])

        searches = []
        for search_group in search_groups:
            searches.append(f'nwr[{search_group}]{not_search_query}{around}')

        for and_group in and_groups:
            searches.append(f'nwr{and_group}{not_search_query}{around}')

        search = ";\n".join(searches)
        if len(search) > 0:
            search += ";"

        # "out center" gives ways and relations (areas) one coordinate,
        # the center of their bounding box, without their full geometry.
        query = f"""
            [out:json];
            (
                {search}
            );
            out center;
        """

        print(query)
        response = await asyncio.to_thread(self.overpass_get, query, headers, center=center)
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
                elif res['type'] in ('way', 'relation'):
                    processed = OsmSearcher.process_way_result(res)
                    if processed is not None and OsmUtils.thing_has_info(res):
                        ways.append(processed)
                    else:
                        if processed is not None:
                            things_missing_names.append(processed)

            # attempt to update ways that have no names/addresses.
            if len(things_missing_names) > 0:
                print(f"[OSM] Updating {len(things_missing_names)} things with info")
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

        if await asyncio.to_thread(self.add_nav_distance_to_things, origin, things_nearby):
            sort_method = "travel distance"
            things_nearby = OsmSearcher.sort_by_closeness(origin, things_nearby, 'nav_distance', 'distance')
        else:
            sort_method = "haversine distance"

        return [things_nearby, sort_method]


    async def search_nearby(
            self, place: str, tags: List[str], category: str, limit: int=5,
            radius: int=4000, not_tag_groups={}, readable_category: str = "POIs"
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
            await self.events.search_complete(readable_category, place, [])
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
                print(f"[OSM] WARN: Could not find display name for place: {place}")
                place_display_name = place

            await self.events.searching(readable_category, place_display_name, done=False, tags=tags)

            used_rel, bbox = self.calculate_bounding_box(nominatim_result)

            print(f"[OSM] Searching for {readable_category} near {place_display_name}")
            things_nearby, sort_method = await self.search_and_rank(
                nominatim_result, place, tags,
                bbox, limit, radius, not_tag_groups
            )

            if not things_nearby or len(things_nearby) == 0:
                await self.events.search_complete(readable_category, place_display_name, [])
                return { "place_display_name": place, "results": NO_RESULTS }

            print(f"[OSM] Found {len(things_nearby)} {readable_category} results near {place_display_name}")

            tag_type_str = ", ".join(OsmSearcher.flatten_tags(tags))

            # Only print the full result instructions if we
            # actually have something.
            parser = OsmParser(place, things_nearby, category=category, include_raw_data=self.include_raw_data)
            search_results = parser.convert_and_validate_results(sort_message=sort_method)

            # post-process to add sequential citation/source IDs, because
            # web ui doesn't like arbitrary source IDs.
            for id, thing in enumerate(search_results or []):
                thing["citation_id"] = str(id + 1)

            if search_results:
                result_instructions = self.get_result_instructions(tag_type_str, used_rel)
            else:
                result_instructions = "No results found at all. Tell the user there are no results."

            resp = {
                "instructions": result_instructions,
                "results": search_results if search_results else []
            }

            # emit citations for the actual results.
            await self.events.search_complete(
                readable_category, place_display_name, things_nearby,
                count=len(search_results or [])
            )
            for thing in search_results or []:
                await self.events.emit_result_citation(thing)

            return { "place_display_name": place_display_name, "results": resp, "things": things_nearby }
        except ValueError:
            await self.events.search_complete(readable_category, place_display_name, [])
            return { "place_display_name": place_display_name, "results": NO_RESULTS, "things": [] }
        except Exception as e:
            print(e)
            await self.events.error(e)
            instructions = (f"No results were found, because of an error. "
                            f"Tell the user that there was an error finding results. "
                            f"**Do not make up results. There are no results available because of an error.**")

            result = { "instructions": instructions, "results": [], "error_message": f"{e}" }
            return { "place_display_name": place_display_name, "results": result, "things": [] }


async def do_osm_search(
        valves, user_valves, place, tags,
        category="POIs", event_emitter=None, limit=5, radius=4000,
        setting='urban', not_tag_groups={}, radius_multiplier=1,
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
    radius = radius * setting_to_multiplier(setting) * radius_multiplier
    events = OsmEventEmitter(event_emitter)
    readable_category=category.replace("_", " ")
    searcher = OsmSearcher(valves, user_valves, events)
    search = await searcher.search_nearby(place=place, tags=tags, limit=limit, radius=radius,
                                          category=category, not_tag_groups=not_tag_groups,
                                          readable_category=readable_category)

    # move place display name to top level results returned to client.
    place_display_name = search.get("place_display_name", None)
    search.get("results", {}).update(resolved_location=place_display_name)
    return search["results"]


#####################################################
# Locality Information
#####################################################

class OsmLocality:
    """
    What a place is like for getting around: country, settlement type,
    road density, one-way streets, cycling infrastructure, and calm
    streets. Street statistics are cached for a long time, since they
    rarely change.

    Bike-friendliness (see bike_friendliness) was calibrated on 17 places
    in the Netherlands, Denmark, Germany, Belgium, France, the UK and the
    US, measured by length within 1.5 km. It uses only these measurements,
    no country rules.
    """
    RADIUS = 1500
    CACHE_SECONDS = 90 * 24 * 3600
    ROADS = "primary|secondary|tertiary|unclassified|residential|living_street"

    # Bump when the stored street stats change shape, so old cache entries
    # are no longer read (they expire on their own).
    STATS_VERSION = 2

    # Bike-friendliness thresholds, as shares of road length.
    INFRA_YES = 0.30         # this much cycling infrastructure is enough on its own
    INFRA_SOME = 0.15        # this much needs calm streets as well
    CALM_WITH_INFRA = 0.30   # calm streets needed alongside INFRA_SOME
    CALM_ALONE = 0.60        # calm streets enough on their own
    SPEED_COVERAGE = 0.50    # speed limits must be mapped on this much to judge calm streets

    def __init__(self, valves, user_valves, events: OsmEventEmitter):
        self.searcher = OsmSearcher(valves, user_valves, events)
        self.cache = OsmCache()

    def _street_query(self, lat: float, lon: float) -> str:
        around = f"(around:{self.RADIUS},{lat},{lon})"
        side = '~"^cycleway(:both|:left|:right)?$"'
        calm_speed = '"^([1-9]|[12][0-9]|30)$|^([1-9]|1[0-9]|20) mph$"'

        def length(set_name: str, label: str) -> str:
            return f"(.{set_name};); make {label} n=count(ways),km=sum(length())/1000; out;"

        return f"""
            [out:json][timeout:60];
            way[highway~"^({self.ROADS})$"]{around}->.roads;
            way[highway=cycleway]{around}->.cycleways;
            way.cycleways[foot=designated][segregated!=yes]->.shared;
            (way.cycleways; - way.shared;)->.dedicated;
            way[highway~"^(footway|path)$"][bicycle=designated]{around}->.bike_paths;
            way.roads[{side}~"^(track|opposite_track)$"]->.tracks;
            way.roads[{side}~"^(lane|opposite_lane)$"]->.lanes;
            way.roads[oneway~"^(yes|-1)$"]->.oneway;
            way.roads[maxspeed~{calm_speed}]->.slow;
            way.roads[highway=living_street]->.living;
            way.roads[~"^(cyclestreet|bicycle_road)$"~"^yes$"]->.cyclestreets;
            (way.slow; way.living; way.cyclestreets;)->.calm;
            way.roads[maxspeed]->.speed_tagged;
            (way.speed_tagged; way.living; way.cyclestreets;)->.speed_known;
            {length('roads', 'roads')}
            {length('dedicated', 'dedicated')}
            {length('shared', 'shared')}
            {length('bike_paths', 'bike_paths')}
            {length('tracks', 'tracks')}
            {length('lanes', 'lanes')}
            {length('oneway', 'oneway')}
            {length('calm', 'calm')}
            {length('speed_known', 'speed_known')}
        """

    def street_stats(self, lat: float, lon: float) -> Optional[dict]:
        """
        Street statistics around a point, by length: road network,
        cycling infrastructure (dedicated cycleways, cycle tracks along
        roads, paths designated for bikes, paths shared with pedestrians),
        painted lanes, one-way roads, and calm streets (30 km/h or less,
        living streets, cycle streets).
        """
        # About a 100 m grid, so nearby lookups share an entry.
        cache_key = f"locality_v{self.STATS_VERSION}_{round(lat, 3)}_{round(lon, 3)}_{self.RADIUS}"
        cached = self.cache.get(cache_key)
        if cached:
            print(f"[OSM] Got street stats for {lat},{lon} from cache!")
            return cached

        headers = self.searcher.create_headers()
        if not headers:
            raise ValueError("Headers not set")

        print(f"[OSM] Getting street stats around {lat},{lon}")
        has_roads = lambda result: int(
            (result.get('elements') or [{}])[0].get('tags', {}).get('n', 0)) > 0
        response = self.searcher.overpass_get(self._street_query(lat, lon), headers, has_roads,
                                              center={'lat': lat, 'lon': lon})
        if response.status_code != 200:
            print(f"[OSM] Overpass error getting street stats: {response.text[:200]}")
            return None

        measured = {element['type']: element.get('tags', {})
                    for element in response.json().get('elements', [])}
        roads = int(measured.get('roads', {}).get('n', 0))
        road_km = float(measured.get('roads', {}).get('km', 0))
        if roads == 0 or road_km == 0:
            # Outside the Overpass instance's coverage (e.g. a regional
            # extract), not a place without roads. Don't cache it.
            print(f"[OSM] No roads found around {lat},{lon}: no street data there")
            return None

        share = lambda name: round(float(measured.get(name, {}).get('km', 0)) / road_km, 3)
        stats = {
            'roads': roads,
            'road_km': round(road_km, 1),
            'dedicated_share': share('dedicated'),
            'shared_share': share('shared'),
            'bike_path_share': share('bike_paths'),
            'track_share': share('tracks'),
            'lane_share': share('lanes'),
            'one_way_share': share('oneway'),
            'calm_share': share('calm'),
            'speed_known_share': share('speed_known'),
        }

        self.cache.set(cache_key, stats, ttl=self.CACHE_SECONDS)
        return stats

    @staticmethod
    def settlement(address: dict) -> Tuple[str, Optional[str]]:
        for kind in ('city', 'town', 'village', 'hamlet'):
            if kind in address:
                return kind, address[kind]
        return 'unknown', None

    @staticmethod
    def level(value: float, high: float, low: float, labels: Tuple[str, str, str]) -> str:
        if value >= high:
            return labels[0]
        return labels[1] if value >= low else labels[2]

    @staticmethod
    def infrastructure_share(stats: dict) -> float:
        """Cycling infrastructure as a share of road length; shared paths count half."""
        return (stats['dedicated_share'] + stats['track_share'] + stats['bike_path_share']
                + 0.5 * stats['shared_share'])

    @classmethod
    def calm_streets(cls, stats: dict) -> Optional[float]:
        """
        Share of streets with a known speed limit that are calm, or None if
        too few speed limits are mapped to tell. Unmapped limits are left
        out rather than guessed: the legal default differs by country.
        """
        if stats['speed_known_share'] < cls.SPEED_COVERAGE:
            return None
        return min(1.0, stats['calm_share'] / stats['speed_known_share'])

    @classmethod
    def bike_friendliness(cls, stats: dict) -> Tuple[str, Optional[bool]]:
        """
        'yes', 'some' or 'no', and the matching bike_friendly value
        (True, None, False) used to weigh bike against car.
        """
        infra = cls.infrastructure_share(stats)
        calm = cls.calm_streets(stats)
        if infra >= cls.INFRA_YES:
            verdict = 'yes'
        elif infra >= cls.INFRA_SOME:
            verdict = 'yes' if calm is not None and calm >= cls.CALM_WITH_INFRA else 'some'
        elif calm is not None and calm >= cls.CALM_ALONE:
            verdict = 'yes'
        else:
            verdict = 'no'
        return verdict, {'yes': True, 'some': None, 'no': False}[verdict]

    def describe(self, thing: dict) -> dict:
        """Locality info for a Nominatim result."""
        address = thing.get('address', {})
        country_code = address.get('country_code', '').lower()
        kind, name = OsmLocality.settlement(address)

        info = {
            'place': thing.get('display_name'),
            'country': address.get('country'),
            'country_code': country_code,
            'settlement_type': kind,
            'settlement_name': name,
        }

        try:
            stats = self.street_stats(float(thing['lat']), float(thing['lon']))
        except Exception as e:
            print(f"[OSM] Could not get street stats: {e}")
            stats = None

        if not stats:
            # No street data (outside the Overpass instance's area, or an
            # error): unknown, so neither assume nor rule out cycling.
            info['bike_friendliness'] = "unknown (no street data)"
            info['bike_friendly'] = None
            return info

        roads = stats['roads']
        infra = OsmLocality.infrastructure_share(stats)
        calm = OsmLocality.calm_streets(stats)
        verdict, bike_friendly = OsmLocality.bike_friendliness(stats)
        pct = lambda value: f"{round(value * 100)}%"
        infra_level = OsmLocality.level(infra, OsmLocality.INFRA_YES, OsmLocality.INFRA_SOME,
                                        ('extensive', 'moderate', 'little'))

        info.update({
            'road_network': (f"{OsmLocality.level(roads, 900, 450, ('dense', 'moderate', 'sparse'))} "
                             f"({roads} road segments, {stats['road_km']} km)"),
            'one_way_streets': (f"{OsmLocality.level(stats['one_way_share'], 0.5, 0.25, ('many', 'some', 'few'))} "
                                f"({pct(stats['one_way_share'])} of road length)"),
            'cycling_infrastructure': (
                f"{infra_level} (cycle paths and tracks: {pct(infra)} of road length, "
                f"of which paths shared with pedestrians {pct(stats['shared_share'])}; "
                f"painted cycle lanes on {pct(stats['lane_share'])} of roads)"
            ),
            'calm_streets': (
                f"{pct(calm)} of streets are 30 km/h (20 mph) or slower, living streets, or cycle streets"
                if calm is not None else
                f"unknown: speed limits are mapped on only {pct(stats['speed_known_share'])} of roads"
            ),
            'bike_friendliness': {
                'yes': "bike-friendly",
                'some': "partly: some cycling infrastructure, but no evidence of calm streets",
                'no': "not especially bike-friendly",
            }[verdict],
            'bike_friendly': bike_friendly,
        })
        return info

    async def for_place(self, place: str) -> Optional[dict]:
        result = await self.searcher.nominatim_search(place, limit=1)
        return await asyncio.to_thread(self.describe, result[0]) if result else None


#####################################################
# Navigation Logic
#####################################################

class OsmNavigator:
    def __init__(
        self, valves, user_valves: Optional[dict], events: OsmEventEmitter,
    ):
        self.valves = valves
        self.user_valves = user_valves
        self.events = events
        # Why the last route attempt failed (see try_route).
        self.failure = None

    @staticmethod
    def combined_bike_friendly(ends: List[dict]) -> Optional[bool]:
        """False if either end isn't bike-friendly, True if both are, else unknown."""
        values = [end['bike_friendly'] for end in ends]
        if False in values:
            return False
        return True if all(values) else None

    def try_route(self, router: OrsRouter, start: dict, destination: dict, mode: str):
        """
        The route and mode, or None. On failure, self.failure says why
        (a key of ORS_FAILURE_INSTRUCTIONS) for the message to the model.
        """
        try:
            result = router.calculate_route(start, destination, mode)
        except Exception as e:
            print(f"[OSM] No {mode} route: {e}")
            self.failure = ors_failure_kind(e)
            return None

        if not result:
            self.failure = 'not_covered'
        return result

    @staticmethod
    def route_summary(route: dict, mode: str) -> dict:
        summary = route.get('summary', {})
        return {
            "travel_method": TRAVEL_MODE_NAMES[mode],
            "total_distance": f"{round(summary.get('distance', 0), 2)} km",
            "travel_time": f"{round(summary.get('duration', 0) / 60.0, 2)} minutes",
        }

    @staticmethod
    def describe_end(end: dict) -> str:
        return f"{end['settlement_name'] or end['place']}: {end['bike_friendliness']}"

    def choose_route(
        self, router: OrsRouter, start: dict, destination: dict
    ) -> Tuple[Optional[Tuple[dict, str]], List[str], Optional[dict]]:
        """
        Route with no travel mode requested. The candidate modes come from
        the locality of both ends and the distance (OrsRouter.auto_modes).
        For car and bike, travel times are compared (prefer_bike). Returns
        the route and mode, the reasons for the choice, and a summary of
        the other mode, if one was compared.
        """
        distance = straight_line_distance(start, destination)
        locality = OsmLocality(self.valves, self.user_valves, self.events)
        ends = [locality.describe(start), locality.describe(destination)]
        bike_friendly = self.combined_bike_friendly(ends)

        # Both ends are often in the same place: say it once.
        reasons = list(dict.fromkeys(self.describe_end(end) for end in ends))
        reasons.append(f"{round(distance, 1)} km in a straight line")

        candidates = router.auto_modes(distance, bike_friendly)
        if len(candidates) < 2:
            if not candidates:
                return None, reasons, None
            reasons.append(router.auto_mode_reason(distance, candidates[0], bike_friendly))
            return self.try_route(router, start, destination, candidates[0]), reasons, None

        car = self.try_route(router, start, destination, 'car')
        bike = self.try_route(router, start, destination, 'bike')
        if not car or not bike:
            # One mode has no route (e.g. outside the map data): use the
            # other, and say so.
            found = car or bike
            if not found:
                return None, reasons, None
            missing = 'bike' if found is car else 'car'
            return found, reasons + [f"no {missing} route could be found for this trip"], None

        car_minutes = car[0].get('summary', {}).get('duration', 0) / 60.0
        bike_minutes = bike[0].get('summary', {}).get('duration', 0) / 60.0
        if router.prefer_bike(bike_minutes, car_minutes, bike_friendly):
            chosen, other = bike, car
        else:
            chosen, other = car, bike

        reasons.append(f"about {round(bike_minutes)} min by bike vs {round(car_minutes)} min by car")
        return chosen, reasons, self.route_summary(*other)


    async def navigate(
        self, start_place: str, destination_place: str,
        travel_mode: Optional[str] = None
    ):
        await self.events.navigating(done=False)
        searcher = OsmSearcher(self.valves, self.user_valves, self.events)
        router = OrsRouter(self.valves, self.user_valves, self.events)

        try:
            if travel_mode:
                available = await asyncio.to_thread(router.available_modes)
                if travel_mode not in available:
                    await self.events.navigating(done=True)
                    return unavailable_travel_mode(travel_mode, available)

            start = await searcher.nominatim_search(start_place, limit=1)
            destination = await searcher.nominatim_search(destination_place, limit=1)

            if not start or not destination:
                await self.events.navigating(done=True)
                return NO_RESULTS

            start, destination = start[0], destination[0]
            if travel_mode:
                result = await asyncio.to_thread(self.try_route, router, start, destination, travel_mode)
                reasons, alternative = [], None
            else:
                result, reasons, alternative = await asyncio.to_thread(
                    self.choose_route, router, start, destination)

            if not result:
                await self.events.navigating(done=True)
                return no_route(self.failure)

            route, mode = result
            total_distance = round(route.get('summary', {}).get('distance', ''), 2)
            travel_time = round(route.get('summary', {}).get('duration', 0) / 60.0, 2)
            travel_type = TRAVEL_MODE_NAMES[mode]

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

            if reasons:
                result["instructions"] += (
                    f" The user didn't ask for a travel method, so {travel_type} was "
                    "chosen automatically (see travel_method_reasons). Mention that."
                )
                result["travel_method_reasons"] = reasons
            if alternative:
                result["instructions"] += " Also mention the travel time of the alternative."
                result["alternative"] = alternative

            result["route_id"] = save_route_for_map(route, mode, start, destination,
                                                    start_place, destination_place)
            result["instructions"] += (
                " To show this route on a map, call show_route_map with this route_id."
            )

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


#####################################################
# Normalization & Validation Functions
#####################################################

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

# The various *_to_tags functions convert the LLM-requested thing to
# search for into a set of OSM tags. Most of the time, these are just
# simple lists of tags, where any OSM entity having at least one tag
# is returned. But the functions can also return ANDed tags (entities
# must have ALL specified tags), and tags that are rejected outright
# (entities must NOT have those tags).

def store_category_to_tags(store_type: str) -> Tuple[List[str], dict]:
    if store_type == "groceries":
        return ["shop=supermarket", "shop=grocery", "shop=greengrocer"], {}
    elif store_type == "convenience":
        return ["shop=convenience"], {}
    elif store_type == "alcohol":
        return ["shop=alcohol"], {}
    elif store_type == "drugs" or store_type == "cannabis":
        return ["shop=coffeeshop", "shop=cannabis", "shop=headshop", "shop=smartshop"], {}
    elif store_type == "electronics":
        return ["shop=electronics", "shop=computer", "shop=hifi", "shop=mobile_phone", "shop=radiotechnics"], {}
    elif store_type == "electrical":
        return ["shop=lighting", "shop=electrical"], {}
    elif store_type == "hardware" or store_type == "diy":
        return ["shop=doityourself", "shop=hardware", "shop=power_tools", "shop=groundskeeping", "shop=trade"], {}
    elif store_type == "pharmacies":
        return ["amenity=pharmacy", "shop=chemist", "shop=supplements", "shop=health_food"], {}
    else:
        return [], {}


def recreation_to_tags(recreation_type: str) -> Tuple[List[str], dict]:
    if recreation_type == "swimming":
        return ["leisure=swimming_pool", "leisure=swimming_area",
                "leisure=water_park", "tourism=theme_park"], {}
    elif recreation_type == "playgrounds":
        return ["leisure=playground"], {}
    elif recreation_type == "amusement":
        return ["leisure=park", "leisure=amusement_arcade", "tourism=theme_park"], {}
    elif recreation_type == "gyms":
        # gyms can either show up as leisure=fitness_centre, or a
        # combo of leisure=sports_centre and sport=fitness.
        return ["leisure=fitness_centre", "leisure=fitness_center",
                ["leisure=sports_centre", "sport=fitness" ]], {}
    elif recreation_type == "sports":
        # gyms are often tag with sports_center + leisure fitness. but
        # we want to search for those separately.
        return ["leisure=horse_riding", "leisure=ice_rink", "leisure=disc_golf_course",
                "leisure=pitch", "leisure=sports_center", "leisure=sports_centre",
                "leisure=sports_hall"], { "sport": [ "fitness" ] }
    else:
        return [], {}


def food_category_to_tags(food_type: str) -> Tuple[List[str], dict]:
    if food_type == "sit_down_restaurants":
        return ["amenity=restaurant", "amenity=eatery", "amenity=canteen"], {}
    elif food_type == "fast_food":
        return ["amenity=fast_food"], {}
    elif food_type == "cafe_or_bakery":
        # cannabis stores are sometimes tagged with amenity=cafe. do
        # not return them when searching for food to eat.
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
    elif healthcare_type == "dentist":
        return ["amenity=dentist", "healthcare=dentist", "healthcare:speciality=orthodontics"], {}
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


# What find_nearby can search for. Each value maps to the internal
# category name used by the rest of the tool (tag lookup, raw data,
# instructions), the function that turns it into OSM tags, and the base
# search radius (m) and result limit.
NearbyCategory: TypeAlias = Literal[
    'shop_groceries', 'shop_convenience', 'shop_alcohol', 'shop_cannabis',
    'shop_electronics', 'shop_electrical', 'shop_hardware_or_diy',
    'food_restaurants', 'food_fast_food', 'food_cafe_or_bakery', 'food_bars_and_pubs',
    'recreation_swimming', 'recreation_playgrounds', 'recreation_amusement',
    'recreation_sports', 'recreation_gyms',
    'travel_tourist_attractions', 'travel_accommodation', 'travel_bike_rentals',
    'travel_car_rentals', 'travel_public_transport',
    'health_doctor', 'health_dentist', 'health_hospital', 'health_pharmacy',
    'education_schools', 'education_universities_and_colleges', 'education_libraries',
    'other'
]

NEARBY_SEARCHES = {
    'shop_groceries': ('groceries', store_category_to_tags, 4000, 5),
    'shop_convenience': ('convenience', store_category_to_tags, 4000, 5),
    'shop_alcohol': ('alcohol', store_category_to_tags, 4000, 5),
    'shop_cannabis': ('cannabis', store_category_to_tags, 4000, 5),
    'shop_electronics': ('electronics', store_category_to_tags, 4000, 5),
    'shop_electrical': ('electrical', store_category_to_tags, 4000, 5),
    'shop_hardware_or_diy': ('hardware', store_category_to_tags, 4000, 5),
    'food_restaurants': ('sit_down_restaurants', food_category_to_tags, 4000, 10),
    'food_fast_food': ('fast_food', food_category_to_tags, 4000, 10),
    'food_cafe_or_bakery': ('cafe_or_bakery', food_category_to_tags, 4000, 10),
    'food_bars_and_pubs': ('bars_and_pubs', food_category_to_tags, 4000, 10),
    'recreation_swimming': ('swimming', recreation_to_tags, 10000, 5),
    'recreation_playgrounds': ('playgrounds', recreation_to_tags, 4000, 10),
    'recreation_amusement': ('amusement', recreation_to_tags, 10000, 10),
    'recreation_sports': ('sports', recreation_to_tags, 10000, 10),
    'recreation_gyms': ('gyms', recreation_to_tags, 4000, 5),
    'travel_tourist_attractions': ('tourist_attractions', travel_category_to_tags, 10000, 10),
    'travel_accommodation': ('accommodation', travel_category_to_tags, 10000, 5),
    'travel_bike_rentals': ('bike_rentals', travel_category_to_tags, 4000, 5),
    'travel_car_rentals': ('car_rentals', travel_category_to_tags, 6000, 5),
    'travel_public_transport': ('public_transport', travel_category_to_tags, 4000, 10),
    'health_doctor': ('doctor', healthcare_category_to_tags, 4000, 5),
    'health_dentist': ('dentist', healthcare_category_to_tags, 4000, 5),
    'health_hospital': ('hospital', healthcare_category_to_tags, 4000, 5),
    'health_pharmacy': ('pharmacy', healthcare_category_to_tags, 4000, 5),
    'education_schools': ('schools', education_category_to_tags, 4000, 5),
    'education_universities_and_colleges': ('universities_and_colleges', education_category_to_tags, 4000, 5),
    'education_libraries': ('libraries', education_category_to_tags, 4000, 5),
}

def invalid_value(name: str, value, allowed) -> dict:
    return {
        "results": [],
        "instructions": "There was an error. Attempt to correct the error, or inform the user (whichever is appropriate).",
        "error_message": f"{value} is not a valid value for {name}. Must be one of: {', '.join(allowed)}"
    }

def other_search_instructions(what: str, place: str) -> str:
    return (
        f"These are results of a general text search for '{what}' near {place}. "
        "This search is less precise than the category searches: results may "
        "be incomplete, or not match what the user asked for. Only mention "
        "results that match, and tell the user if none do. "
        f"{NO_CITATION_LIST_PROHIBITION}"
    )

async def resolve_setting(valves, user_valves, events, place: str,
                          setting: Optional[str]) -> str:
    """
    The urban setting for a search: the given one, or one derived from the
    road density around the place (see OsmLocality). Urban if unknown.
    """
    if setting:
        return normalize_setting(setting)

    try:
        locality = OsmLocality(valves, user_valves, events)
        result = await locality.searcher.nominatim_search(place, limit=1, quiet=True)
        stats = (await asyncio.to_thread(locality.street_stats,
                                         float(result[0]['lat']), float(result[0]['lon']))
                 if result else None)
        if stats:
            setting = OsmLocality.level(stats['roads'], 900, 450, ('urban', 'suburban', 'rural'))
            print(f"[OSM] Setting for {place}: {setting} ({stats['roads']} roads nearby)")
            return setting
    except Exception as e:
        print(f"[OSM] Could not determine the setting for {place}: {e}")

    return 'urban'


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


#####################################################
# Main Tool Entrypoint
#####################################################
#
# This is the part we've all been waiting for...
#

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
            default=PUBLIC_OVERPASS_URL,
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
        fallback_to_public_instances: bool = Field(
            default=False,
            description=("Off by default. When a custom Overpass or ORS instance returns an "
                         "error or no data (e.g. outside its map area), retry on the public instance. "
                         "ORS fallback uses ors_fallback_api_key, or ors_api_key if that's empty.")
        )
        ors_fallback_api_key: Optional[str] = Field(
            default=None,
            description=("API key for the public ORS when falling back from a custom ORS "
                         "instance. If empty, ors_api_key is used.")
        )
        status_indicators: bool = Field(
            default=True,
            description=("Emit status update events to the web UI.")
        )
        include_raw_osm_data: bool = Field(
            default=False,
            description="Include raw OSM data in search results or not."
        )
        tile_server: str = Field(
            default="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            description="OSM Tile Server"
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

    async def find_place(
            self, place_or_coordinates: str, near: Optional[str] = None,
            __event_emitter__=None
    ) -> str:
        """
        Looks up a specific place on OpenStreetMap: an address, landmark,
        named building or business, or GPS coordinates (what is at these
        coordinates). To find a named business or chain near somewhere
        (e.g. 'Albert Heijn' near given coordinates), pass its name and set
        `near`. To find kinds of places nearby (any supermarket, any
        dentist), use find_nearby instead.
        :param place_or_coordinates: An address, a place or business name, or GPS coordinates as 'lat,lon'.
        :param near: Optional place or 'lat,lon' coordinates to search around.
        :return: Information about the place, if found.
        """
        query = f"{place_or_coordinates} {near}" if near else place_or_coordinates
        print(f"[OSM] Searching for info on [{query}].")
        events = OsmEventEmitter(__event_emitter__)
        searcher = OsmSearcher(self.valves, self.user_valves, events)

        try:
            result = await searcher.nominatim_search(query, limit=5)
            if result:
                parser = OsmParser(query, result, category=None)
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
            travel_mode: Optional[TravelMode] = None,
            __event_emitter__=None
    ) -> str:
        """
        Retrieve a navigation route and associated information between two places.
        The result includes a route_id: pass it to show_route_map to show the
        route on a map.
        :param start_address_or_place: The address, place, or coordinates to start from.
        :param destination_address_or_place: The destination address, place, or coordinates to go.
        :param travel_mode: 'car', 'bike', or 'walk'. Only set this if the user asked for a specific way of traveling (driving, cycling, walking). Otherwise leave it empty, and a mode is chosen by distance.
        :return: The navigation route and associated info, if found.
        """
        print(f"[OSM] Navigating from [{start_address_or_place}] to [{destination_address_or_place}] ({travel_mode or 'auto'}).")
        events = OsmEventEmitter(__event_emitter__)
        navigator = OsmNavigator(self.valves, self.user_valves, events)
        return await navigator.navigate(
            start_address_or_place, destination_address_or_place, travel_mode
        )

    async def find_locality_info(self, place: str, __event_emitter__) -> str:
        """
        Describes what a place is like for getting around: the country,
        whether it's a city, town, village, or hamlet, how dense the road
        network is, how common one-way streets are, and how much cycling
        infrastructure there is nearby. Use this for questions about
        getting around a place, or whether walking, cycling, or driving
        makes sense there.
        :param place: The name of a place, an address, or GPS coordinates.
        :return: Information about the locality.
        """
        print(f"[OSM] Getting locality info for [{place}].")
        events = OsmEventEmitter(__event_emitter__)
        locality = OsmLocality(self.valves, self.user_valves, events)

        try:
            info = await locality.for_place(place)
            if not info:
                return NO_RESULTS

            return {"instructions": locality_instructions(), **info}
        except Exception as e:
            print(e)
            return (f"There are no results due to an error. "
                    "Tell the user that there was an error. "
                    f"The error was: {e}. "
                    f"Tell the user the error message.")

    async def find_nearby(
            self, place: str, what: NearbyCategory,
            what_else: Optional[str] = None,
            setting: Optional[UrbanSetting] = None,
            __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Finds things of a given kind on OpenStreetMap near a place or address:
        shops, food and drink, recreation, travel and tourism, healthcare, and
        education. Pick the closest matching `what`. For fuel, EV charging, and
        places of worship, use their own tools. For anything else, use
        what='other' and describe it in what_else.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param what: Kind of place to find. shop_cannabis also finds coffeeshops and smartshops. health_pharmacy is pharmacies and chemists (drugstores). food_restaurants is sit-down restaurants. recreation_amusement is arcades, theme parks, and parks. education_schools is primary and secondary schools only.
        :param what_else: Only for what='other': what to look for, in a few words (e.g. 'bicycle repair').
        :param setting: Optional: whether the place is urban, suburban, or rural (controls the search radius). Leave empty to determine it automatically.
        :return: A list of nearby places, if found.
        """
        user_valves = (__user__ or {}).get("valves")

        if what == 'other':
            return await self._find_other_near_place(place, what_else, __event_emitter__)

        search = NEARBY_SEARCHES.get(what)
        if not search:
            return invalid_value("what", what, list(NEARBY_SEARCHES) + ['other'])

        category, to_tags, radius, limit = search
        tags, not_tag_groups = to_tags(category)
        events = OsmEventEmitter(__event_emitter__)
        setting = await resolve_setting(self.valves, user_valves, events, place, setting)

        return await do_osm_search(valves=self.valves, user_valves=user_valves, category=category,
                                   radius=radius, limit=limit, setting=setting, place=place, tags=tags,
                                   event_emitter=__event_emitter__, not_tag_groups=not_tag_groups)

    async def _find_other_near_place(
            self, place: str, what: Optional[str], __event_emitter__
    ) -> str:
        """
        Free-text Nominatim search ('<what> near <place>') for kinds of
        places without their own category. Nominatim understands many OSM
        place types this way ('bicycle shop', 'post office', ...).
        """
        if not what:
            return {
                "results": [],
                "instructions": ("There was an error: what='other' needs what_else, a short "
                                 "description of what to look for. Call the tool again with it."),
            }

        query = f"{what} near {place}"
        print(f"[OSM] General search for [{query}].")
        events = OsmEventEmitter(__event_emitter__)
        searcher = OsmSearcher(self.valves, self.user_valves, events)

        try:
            result = await searcher.nominatim_search(query, limit=10)
        except ValueError:
            return NO_RESULTS
        except Exception as e:
            print(e)
            return (f"There are no results due to an error. "
                    "Tell the user that there was an error. "
                    f"The error was: {e}. "
                    f"Tell the user the error message.")

        if not result:
            return NO_RESULTS

        parser = OsmParser(query, result, category=None)
        results = parser.convert_and_validate_results(sort_message="importance", use_distance=False)
        return {
            "instructions": other_search_instructions(what, place),
            "results": results if results else []
        }

    async def find_place_of_worship_near_place(
            self, place: str,
            religion: ReligionCategory,
            denomination: DenominationCategory,
            is_common_for_area: bool,
            setting: Optional[UrbanSetting] = None,
            __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Finds places of worship (churches, mosques, temples, etc) on OpenStreetMap near a
        given place or address. If searching for a denomination, the denomination must be part of the
        given religion. If NOT searching for any specific denomination, use the special 'any_denomination'
        value. If a religion is specified as not common in the requested area, the search radius
        will be greatly increased. A religion is common in the requested area if it traditionally has
        a lot of adherents or a visible social presence in the area.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param religion: Narrow the search to a particular religion. Use "any_religion" to search for all places of worship.
        :param denomination: Narrow search to denomination of particular religion. Denomination must match religion.
        :param is_common_for_area: Whether or not the religion or denomination is common in the requested area.
        :param setting: Optional: whether the place is urban, suburban, or rural (controls the search radius). Leave empty to determine it automatically.
        :return: A list of nearby places of worship, if found.
        """
        user_valves = (__user__ or {}).get("valves")
        events = OsmEventEmitter(__event_emitter__)
        setting = await resolve_setting(self.valves, user_valves, events, place, setting)

        # by default, look for all places of worship.
        tags = ["amenity=place_of_worship"]

        # searching for a specific religion?
        if religion != "any_religion" and religion is not None:
            tags = [["amenity=place_of_worship", f"religion={religion}"]]


        # or even a specific religion + denomination?
        if denomination != 'any_denomination' and denomination is not None:
            split = denomination.split("|")
            if len(split) != 2 or '|' not in denomination:
                raise ValueError("Denomination must be specified as 'religion|denomination'")

            denom_religion = split[0]
            search_by_denom = split[1]

            if denom_religion != religion:
                raise ValueError(f'Denomination specified. religion parameter must be: {denom_religion}')

            # now we can actually search by religion + denom
            tags = [["amenity=place_of_worship", f"religion={religion}", f"denomination={search_by_denom}"]]

        radius_multiplier = 1 if is_common_for_area else 2.5
        return await do_osm_search(valves=self.valves, user_valves=user_valves, category="places of worship",
                                   setting=setting, place=place, tags=tags, event_emitter=__event_emitter__,
                                   radius_multiplier=radius_multiplier)

    async def find_fuel_or_charging_by_category_near_place(
            self, place: str, category: FuelCategory, fuel_type: FuelTypeCategory,
            ev_charger_type: EVChargerCategory, setting: Optional[UrbanSetting] = None,
            __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Finds gas stations, petrol stations, fuel stations, or EV fast chargers near a given place or address.
        Does not find slow (regular) EV chargers. EV fast charger note: 'chademo3' is not the same as 'chademo'.
        'chademo3' is also known as 'chaoji'.
        :param place: The name of a place, an address, or GPS coordinates. City and country must be specified, if known.
        :param setting: Optional: whether the place is urban, suburban, or rural (controls the search radius). Leave empty to determine it automatically.
        :param category: Category to search for: either fossil fuels or EV fast chargers.
        :param fuel_type: Type of fossil fuel to search for.
        :param ev_charger_type: Type of EV charger to search for.
        :return: A list of nearby fueling stations, if found.
        """
        # subfunction to find fuel
        async def _find_fossil_fuel_near_place(valves, __user__: dict, place: str, setting: str, __event_emitter__) -> str:
            setting = normalize_setting(setting)
            user_valves = __user__["valves"] if "valves" in __user__ else None
            tags = ["amenity=fuel"]
            return await do_osm_search(valves=valves, user_valves=user_valves, category="fossil_fuels",
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

            return await do_osm_search(valves=valves, user_valves=user_valves, category="ev_fast_charging",
                                       setting=setting, radius=10000, place=place, tags=tags,
                                       event_emitter=__event_emitter__)


        # Main part of fuel search function
        allowed_categories = ["fossil_fuels", "ev_fast_charging"]
        validation_error = validate_category(category, allowed_categories)

        if validation_error:
            return validation_error

        __user__ = __user__ or {}
        events = OsmEventEmitter(__event_emitter__)
        setting = await resolve_setting(self.valves, __user__.get("valves"), events, place, setting)

        if category == "fossil_fuels":
            print(f"[OSM] WARN: Currently ignoring fuel type parameter (was '{fuel_type}')")
            return await _find_fossil_fuel_near_place(self.valves, __user__, place, setting, __event_emitter__)
        elif category == "ev_fast_charging":
            return await _find_ev_fast_chargers_near_place_with_type(
                self.valves, __user__, place, ev_charger_type, setting, __event_emitter__
            )


    def show_route_map(self, route_id: str):
        """
        Shows a route on a map, with the turn-by-turn directions. You MUST call
        navigate_between_places first: pass the route_id from its result. This
        tool does not find routes itself. Use it when the user wants to see the
        route, or when a map helps more than the directions in text.
        :param route_id: The route_id returned by navigate_between_places.
        :return: HTML for displaying the route map.
        """
        saved = OsmCache().get(f"route_map_{str(route_id).strip()}")
        if not saved:
            return {
                "status": "error",
                "message": ("No route with this route_id. Call navigate_between_places "
                            "first, then call show_route_map with the route_id from "
                            "its result."),
            }

        page = route_map_html(saved['route'], saved['mode'], saved['start'], saved['destination'],
                              saved['start_name'], saved['destination_name'],
                              self.valves.tile_server)
        return HTMLResponse(content=page, headers={"Content-Disposition": "inline"}), SHOW_ROUTE_MAP_CONTEXT

    def show_poi_map(self, category: POICategory, results: List[OsmPOI]):
        """
        Shows a map for search results. Takes a list of Points of Interest that
        were returned in a previous search function call (find_nearby, or the fuel/
        charging and place of worship tools). For routes, use show_route_map instead.
        You **must** call another
        function first to get information required for this function! Make sure to
        pass all necessary results into this function! When specifying descriptions,
        make sure they are short and useful, mentioning unique or important info
        about each result. **The map is rendered by the client above your reply;
        do not announce it or write placeholders like [Map displayed].** Do not re-list the
        results, and do not append a citation list or list of links — the map
        and the sources are displayed automatically. Never put images, markdown
        image syntax, or map links in your reply. Pass lat, lon, and distance as
        plain numbers (distance 0.49, not "0.49 km").
        :param category: Results category searched for.
        :param results: Point of Interest (POI) results from a previous function call.
        :return: HTML for displaying a search results map.
        """
        pois, problems = normalize_map_pois(results)
        if not pois:
            return {
                "status": "error",
                "message": ("No map was shown: none of the results could be placed on "
                            "a map. Pass results exactly as returned by the search "
                            "function, with numeric lat, lon, and distance. "
                            f"Problems: {'; '.join(problems) or 'no results given'}"),
            }

        headers = {"Content-Disposition": "inline"}
        # Escape "</" so a name or description can't close the <script> early.
        results_json = json.dumps(pois).replace("</", "<\\/")

        ui_html = (MAP_UI
                   .replace("{REPLACE_WITH_POI_RESULTS}", results_json)
                   .replace("{REPLACE_WITH_CATEGORY}", category)
                   .replace("{REPLACE_WITH_TILE_SERVER}", self.valves.tile_server))

        context = dict(SHOW_MAP_CONTEXT)
        if problems:
            context["skipped"] = problems
        return HTMLResponse(content=ui_html, headers=headers), context
