"""Registry of everything a storyboard may reference. Claude picks only from here.

Adding a new camera move / material / style = implement it in the Blender scripts
and register its name + one-line description here.
"""
from __future__ import annotations

TERRAIN_STYLES = {
    "satellite": "Satellite imagery draped on the terrain",
    "stylized": "Stylized map: 3D forests, water, roads, snow from OSM",
    "hybrid": "Satellite as the base color + 3D trees and water",
    "lowpoly": "Planned, not implemented: renders as satellite",
}

CAMERAS = {
    "chase": "Behind and above the avatar, smoothly damped",
    "drone": "High and to the side, slow parallax",
    "orbit": "Orbit around the point of interest",
    "flyover": "Planned, not implemented: shot as chase",
    "reveal": "Planned, not implemented: shot as chase",
    "overview": "Top-down view of the whole route",
}

MATERIAL_PRESETS = {
    "default": "Regular terrain per the style",
    "volcano_snowcap": "Planned, not implemented: dark basalt, snow by altitude and slope, crater",
    "volcano_active": "Planned, not implemented: basalt, lava fields, glowing crater",
    "glacier": "Planned, not implemented: crevassed ice, moraine",
    "alpine_peak": "Planned, not implemented: rock, scree, snow above the snow line",
    "mountain_lake": "Planned, not implemented: turquoise water, rocky shore",
    "desert": "Planned, not implemented: sand, rocks, haze",
    "coast": "Planned, not implemented: beach, surf, shallows",
}

EFFECTS = {
    "steam_plume": "Steam column over the crater",
    "fumaroles": "Fumarole field: a dozen small steam jets around the point (Unzen Jigoku)",
    "clouds_low": "Planned, not implemented",
    "sun_rays": "Planned, not implemented",
    "snowfall": "Planned, not implemented",
}

TEXT_STYLES = {
    "bold_minimal": "Large numbers, thin captions, no boxes",
    "badge": "Numbers on translucent rounded boxes",
    "retro_map": "Serif font, paper box",
}

GAP_MODES = {
    "join": "Join over the ground: the dot rides across the gap without a cut",
    "straight": "Dashed straight line (arc over the terrain)",
    "road": "By road (OSRM): the dot follows the bus/car route, light trail",
    "skip": "Don't show the transfer, cut straight away",
    "ferry": "Ferry: straight white line over the water (usually within_day: true)",
}

DETAIL_LEVELS = {
    "normal": "Global 30 m DEM",
    "high_res_dem": "Local high-resolution DEM around the point",
    "google_3d_tiles": "Planned, not implemented: same as normal",
}

AVATARS = {
    "marker": "Glowing dot in the route color",
    "puck": "Flat puck in the route color with a white ring (default)",
    "procedural:grizl": "Procedural Canyon Grizl model (color configurable)",
    "procedural:grizl_rider": "Planned, not implemented: the Grizl without a rider",
    "file": "Planned, not implemented: your own .glb (avatar.path); renders as marker",
}

TRAIL_STYLES = {
    "tube": "3D tube behind the dot",
    "ribbon": "Flat ribbon on the ground",
    "band": "Flat band with slight thickness and a dark outline, like a line on a map (default)",
}

AHEAD_MODES = {
    "hidden": "Route ahead hidden, the trail is drawn behind the dot",
    "faint": "Route ahead shown translucent",
}

PLATFORMS = {
    "instagram_story": "One Instagram story: up to 60 s",
    "instagram_reels": "Reels: up to 3 min (60–90 s is best)",
    "none": "No platform limit (up to 180 s)",
}
PLATFORM_MAX_S = {"instagram_story": 60.0, "instagram_reels": 180.0, "none": 180.0}

LABEL_KINDS = {
    "peak": "Peak: a line up from the summit and a label with the elevation",
    "town": "Town: dot and label",
    "waterfall": "Waterfall: blue drop",
    "lake": "Lake: blue oval",
    "wetland": "Wetland: green tussock with reeds",
    "viewpoint": "Viewpoint: triangle",
    "sight": "Sight, temple: star",
}

OPENINGS = {
    "sunrise": "Sunrise at the day's start: a low camera faces the rising sun, then the ride",
    "sunrise_beach": "Sunrise on the shore: camera 30 m up behind palms, beach, surf, houses, sun over the sea",
}

CLOSINGS = {
    "sunset": "Sunset at the day's finish: camera over the water faces the setting sun over shore and hill silhouettes",
}

ALL = {
    "terrain_styles": TERRAIN_STYLES,
    "cameras": CAMERAS,
    "material_presets": MATERIAL_PRESETS,
    "effects": EFFECTS,
    "text_styles": TEXT_STYLES,
    "platforms": PLATFORMS,
    "label_kinds": LABEL_KINDS,
    "openings": OPENINGS,
    "closings": CLOSINGS,
    "gap_modes": GAP_MODES,
    "trail_styles": TRAIL_STYLES,
    "ahead_modes": AHEAD_MODES,
    "detail_levels": DETAIL_LEVELS,
    "avatars": AVATARS,
}
