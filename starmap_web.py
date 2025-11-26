import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.path import Path
from skyfield.api import Star, load, wgs84
from skyfield.projections import build_stereographic_projection
from skyfield.data import hipparcos
from geopy.geocoders import Nominatim
import numpy as np
import os
import urllib.request
import re
import ssl
import io
import datetime

# --- CONFIG ---
st.set_page_config(page_title="GiftyHearts Web Studio", layout="wide", page_icon="✨")
ssl._create_default_https_context = ssl._create_unverified_context

CONST_URL = "https://raw.githubusercontent.com/Stellarium/stellarium/v0.22.0/skycultures/western_SnT/constellationship.fab"
CONST_FILE = "constellationship.fab"
NAMES_FILE = "constellation_names.eng.fab"


# --- CACHED DATA LOADING ---
@st.cache_resource
def load_astronomy_data():
    try:
        eph = load('de421.bsp')
    except:
        try:
            # Manual fallback if auto-download fails
            if os.path.exists('de421.bsp'):
                eph = load('de421.bsp')
            else:
                # Last resort: Try standard load which might trigger download
                eph = load('de421.bsp')
        except:
            st.error("Could not load Planet Data (de421.bsp).")
            return None, None

    try:
        with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
    except:
        if os.path.exists('hip_main.dat'):
            with load.open('hip_main.dat') as f:
                df = hipparcos.load_dataframe(f)
        else:
            st.error("Could not load Star Catalog (hip_main.dat).")
            return None, None

    return eph, df


# --- HELPER FUNCTIONS ---
def get_file(filename):
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/Stellarium/stellarium/v0.22.0/skycultures/western_SnT/" + filename,
                filename)
        except:
            return False
    return True


def parse_google_link(url):
    try:
        if "goo.gl" in url or "maps.app" in url:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req) as response:
                url = response.geturl()
        match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
        if match: return float(match.group(1)), float(match.group(2))
    except:
        pass
    return None, None


def parse_constellation_lines(df_stars):
    if not get_file(CONST_FILE): return [], {}
    edges = []
    constellation_stars = {}
    try:
        with open(CONST_FILE, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2: continue
                abbr = parts[0]
                star_ids = [int(s) for s in parts[2:]]
                if abbr not in constellation_stars: constellation_stars[abbr] = []
                constellation_stars[abbr].extend(star_ids)
                for i in range(0, len(star_ids), 2):
                    if i + 1 < len(star_ids):
                        h1, h2 = star_ids[i], star_ids[i + 1]
                        if h1 in df_stars.index and h2 in df_stars.index:
                            edges.append((h1, h2))
    except:
        pass
    return edges, constellation_stars


def parse_constellation_names():
    if not get_file(NAMES_FILE): return {}
    names = {}
    try:
        with open(NAMES_FILE, 'r') as f:
            for line in f:
                parts = line.split('"')
                if len(parts) >= 2:
                    names[parts[0].strip()] = parts[1]
    except:
        pass
    return names


def get_clipping_patch(ax, shape):
    if shape == "Heart":
        t = np.linspace(0, 2 * np.pi, 100)
        x = (16 * np.sin(t) ** 3) / 16.5
        y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16.5 + 0.05
        return Polygon(list(zip(x, y)), transform=ax.transData)
    elif shape == "Square":
        return Rectangle((-0.95, -0.95), 1.9, 1.9, transform=ax.transData)
    return Circle((0, 0), radius=1.0, transform=ax.transData)


def get_safe_path(shape):
    if shape == "Heart":
        t = np.linspace(0, 2 * np.pi, 200)
        x = (16 * np.sin(t) ** 3) / 16.5 * 0.85
        y = ((13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16.5 + 0.05) * 0.85
        return Path(list(zip(x, y)))
    elif shape == "Square":
        return Path([(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8), (-0.8, -0.8)])
    return Path(list(zip(np.cos(np.linspace(0, 2 * np.pi, 100)) * 0.9, np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.9)))


# --- UI LAYOUT ---
st.title("🌌 GiftyHearts Creator Studio")
st.markdown("Professional Star Map Generator (Web Version)")

col_controls, col_preview = st.columns([1, 2])

with col_controls:
    with st.expander("📍 1. Location", expanded=True):
        loc_mode = st.radio("Input Mode:", ["Search City", "Google Link", "Manual Coords"], horizontal=True)

        lat, lon = 0.0, 0.0
        location_label = "Unknown"

        if loc_mode == "Search City":
            city_query = st.text_input("Enter City Name:", "Shahapur, Thane")
            if city_query:
                try:
                    geolocator = Nominatim(user_agent="gh_web")
                    loc = geolocator.geocode(city_query)
                    if loc:
                        lat, lon = loc.latitude, loc.longitude
                        location_label = loc.address.split(",")[0]
                        st.success(f"Found: {loc.address}")
                    else:
                        st.error("City not found.")
                except:
                    st.warning("Network error searching city.")

        elif loc_mode == "Google Link":
            link = st.text_input("Paste Google Maps Link:")
            if link:
                lat, lon = parse_google_link(link)
                if lat:
                    location_label = "Custom_Location"
                    st.success(f"Coordinates: {lat:.4f}, {lon:.4f}")
                else:
                    st.error("Invalid Link")

        elif loc_mode == "Manual Coords":
            lat = st.number_input("Latitude", value=19.0760, format="%.4f")
            lon = st.number_input("Longitude", value=72.8777, format="%.4f")
            location_label = "Manual_Coords"

    with st.expander("📅 2. Date & Time", expanded=True):
        col_d1, col_d2 = st.columns(2)

        date_input = col_d1.date_input(
            "Date",
            value=datetime.date(2004, 10, 11),
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date(2100, 12, 31)
        )

        time_input = col_d2.time_input(
            "Time",
            value=datetime.time(22, 00),
            step=60
        )

    with st.expander("🎨 3. Design & Layers", expanded=True):
        shape = st.selectbox("Shape", ["Circle", "Heart", "Square"])
        star_color = st.color_picker("Star Color", "#ffffff")

        col_l1, col_l2 = st.columns(2)
        show_stars = col_l1.checkbox("Show Stars", True)
        show_lines = col_l1.checkbox("Constellations", True)
        show_names = col_l1.checkbox("Const. Names", True)
        show_grid = col_l1.checkbox("Grid", False)

        show_planets = col_l2.checkbox("Planets", True)
        show_pnames = col_l2.checkbox("Planet Names", True)
        show_border = col_l2.checkbox("Border", True)

        st.markdown("---")
        transparent = st.checkbox("Transparent Background", False)

        quality_mode = st.selectbox("Export Quality",
                                    ["Standard (2400px)", "High (4800px) - A3", "Ultra (7200px) - Poster"])

    btn_generate = st.button("🚀 GENERATE MAP", type="primary", use_container_width=True)

# --- GENERATION LOGIC ---
if btn_generate:
    with col_preview:
        if lat == 0 and lon == 0:
            st.warning("Please select a location first!")
        else:
            with st.spinner("Calculating the cosmos..."):
                # Load Data
                eph, df = load_astronomy_data()
                if eph is None or df is None:
                    st.stop()

                # Filter & Calc
                ts = load.timescale()
                t = ts.utc(date_input.year, date_input.month, date_input.day, time_input.hour, time_input.minute)

                df_vis = df[df['magnitude'] <= 7.0]
                bright_stars = Star.from_dataframe(df_vis)

                loc_obj = eph['earth'] + wgs84.latlon(lat, lon)
                pos = loc_obj.at(t).from_altaz(alt_degrees=90, az_degrees=0)
                proj = build_stereographic_projection(pos)
                x, y = proj(loc_obj.at(t).observe(bright_stars))

                # Plot
                fig, ax = plt.subplots(figsize=(10, 10))
                bg_col = 'none' if transparent else 'black'
                fig.patch.set_facecolor(bg_col)
                ax.set_facecolor(bg_col)

                # Shape & Clip
                clip_patch = get_clipping_patch(ax, shape)
                safe_path = get_safe_path(shape)

                if show_border:
                    border = get_clipping_patch(ax, shape)
                    border.set_facecolor('none')
                    border.set_edgecolor(star_color)
                    border.set_linewidth(2.5)
                    ax.add_patch(border)

                # Prepare constellation data (Needed for lines OR names)
                edges, constellation_stars = parse_constellation_lines(df_vis)
                hip_idx = {hip: i for i, hip in enumerate(df_vis.index)}

                if show_lines and edges:
                    lines = []
                    for h1, h2 in edges:
                        if h1 in hip_idx and h2 in hip_idx:
                            lines.append([(x[hip_idx[h1]], y[hip_idx[h1]]), (x[hip_idx[h2]], y[hip_idx[h2]])])
                    lc = LineCollection(lines, colors=star_color, linewidths=0.3, alpha=0.4)
                    lc.set_clip_path(clip_patch);
                    ax.add_collection(lc)

                if show_stars:
                    mask = (x ** 2 + y ** 2) < 1.2
                    xv, yv = x[mask], y[mask]
                    mv = df_vis['magnitude'][mask]

                    # Dust & Bright Stars
                    dust = mv > 4.5
                    ax.scatter(xv[dust], yv[dust], s=0.05, color=star_color, alpha=0.3, lw=0).set_clip_path(clip_patch)
                    bright = mv <= 4.5
                    ax.scatter(xv[bright], yv[bright], s=(5.0 - mv[bright]) ** 2.5, color=star_color, alpha=0.2,
                               lw=0).set_clip_path(clip_patch)
                    ax.scatter(xv[bright], yv[bright], s=0.8, color=star_color, alpha=1.0, lw=0).set_clip_path(
                        clip_patch)

                if show_planets:
                    planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter barycenter', 'saturn barycenter']
                    pl_data = []
                    for p_name in planets:
                        p_pos = loc_obj.at(t).observe(eph[p_name])
                        alt, _, _ = p_pos.apparent().altaz()
                        if alt.degrees > 0:
                            px, py = proj(p_pos)
                            s_size = 12.0 if p_name == 'moon' else 4.0
                            ax.scatter(px, py, s=s_size, color=star_color, zorder=3).set_clip_path(clip_patch)
                            if p_name == 'moon':
                                ax.scatter(px, py, s=40, color=star_color, alpha=0.3, lw=0).set_clip_path(clip_patch)

                            if show_pnames and safe_path.contains_point((px, py)):
                                clean_name = p_name.replace(" barycenter", "").title()
                                pl_data.append({'x': px, 'y': py, 'name': clean_name})

                    # Stack Labels
                    pl_data.sort(key=lambda p: p['y'], reverse=True)
                    for i, p in enumerate(pl_data):
                        lbl_y = p['y']
                        if i > 0 and (pl_data[i - 1]['final_y'] - lbl_y < 0.04):
                            lbl_y = pl_data[i - 1]['final_y'] - 0.04
                        p['final_y'] = lbl_y
                        ax.text(p['x'] + 0.03, lbl_y, p['name'], color=star_color, fontsize=8, ha='left', va='center')
                        if abs(lbl_y - p['y']) > 0.01:
                            ax.plot([p['x'], p['x'] + 0.025], [p['y'], lbl_y], color=star_color, lw=0.5, alpha=0.5)

                # --- FIX: CONSTELLATION NAMES ---
                if show_names:
                    full_names = parse_constellation_names()
                    for abbr, star_list in constellation_stars.items():
                        # Check if we have stars for this constellation
                        valid_stars = [s for s in star_list if s in hip_idx]
                        if not valid_stars: continue

                        # Calculate Center
                        x_sum = sum(x[hip_idx[s]] for s in valid_stars)
                        y_sum = sum(y[hip_idx[s]] for s in valid_stars)
                        cx, cy = x_sum / len(valid_stars), y_sum / len(valid_stars)

                        if safe_path.contains_point((cx, cy)):
                            label_text = full_names.get(abbr, abbr)
                            ax.text(cx, cy, label_text, color=star_color, fontsize=5, ha='center', alpha=0.7)

                ax.set_xlim(-1.1, 1.1);
                ax.set_ylim(-1.1, 1.1);
                ax.axis('off')

                # Display
                st.pyplot(fig)

                # Download
                q_map = {"Standard (2400px)": 300, "High (4800px) - A3": 600, "Ultra (7200px) - Poster": 900}
                dpi_val = q_map.get(quality_mode, 600)

                fn = f"map_{location_label}_{date_input}.png"
                buf = io.BytesIO()
                fig.savefig(buf, format="png", facecolor=bg_col, transparent=transparent, dpi=dpi_val,
                            bbox_inches='tight', pad_inches=0)
                st.download_button(f"⬇️ Download Image ({dpi_val} DPI)", data=buf.getvalue(), file_name=fn,
                                   mime="image/png")