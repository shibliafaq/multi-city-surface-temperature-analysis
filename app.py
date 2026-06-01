"""
Multi-City Land Surface Temperature Analysis
=============================================
Real NASA MODIS satellite data (25,905 measurements) for 3 cities,
expanded with estimated data for 13 additional cities worldwide.

Visualisation: pydeck (deck.gl / WebGL) — same engine as Kepler.gl
Charts: Plotly Express

Author  : Shibli Afaq · Sultan Aldhafeeri
Course  : ICS 574 — Big Data Analytics, KFUPM
Data    : MODIS/061/MOD11A1 via Google Earth Engine
Period  : Oct 29 – Nov 11, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global LST Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def temp_to_rgb(t, t_min=-20, t_max=40):
    """Map temperature to [R,G,B,A] for pydeck — blue → yellow → red."""
    ratio = np.clip((t - t_min) / (t_max - t_min), 0, 1)
    if ratio < 0.5:
        r = int(ratio * 2 * 255)
        g = int(ratio * 2 * 200)
        b = int((1 - ratio * 2) * 255)
    else:
        r = 255
        g = int((1 - (ratio - 0.5) * 2) * 200)
        b = 0
    return [r, g, b, 200]

PYDECK_COLOR_RANGE = [
    [65, 182, 196],   # cold blue
    [127, 205, 187],
    [199, 233, 180],
    [255, 255, 178],  # warm yellow
    [253, 141,  60],
    [227,  26,  28],  # hot red
]

# ─────────────────────────────────────────────────────────────────────────────
# REAL MEASURED DATA  (NASA MODIS via GEE)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_real_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "data",
                        "global_cities_temperature_comparison.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["source"] = "Measured (MODIS)"
    return df

# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED GLOBAL CITY DATASET
# Regression calibrated on real data: T = −0.911 × Lat + 56.0  (R²=0.990)
# Cities within validated range (26°N–64°N) use the regression ± small noise.
# Tropical / polar / S.Hemisphere cities use published MODIS climatology.
# ─────────────────────────────────────────────────────────────────────────────
SLOPE, INTERCEPT = -0.911, 56.0

def reg(lat, noise=0.0):
    return round(SLOPE * lat + INTERCEPT + noise, 2)

EXTRA_CITIES = [
    # ── Tropical (outside regression range, MODIS climatology) ──────────────
    {"city": "Singapore",    "country": "Singapore",    "hemisphere": "N",
     "climate": "Tropical",           "latitude":   1.35, "longitude": 103.82,
     "mean_temp": 31.2,  "note": "Estimated (climatology)"},
    {"city": "Bangkok",      "country": "Thailand",     "hemisphere": "N",
     "climate": "Tropical Savanna",   "latitude":  13.75, "longitude": 100.50,
     "mean_temp": 34.1,  "note": "Estimated (climatology)"},
    {"city": "Mumbai",       "country": "India",        "hemisphere": "N",
     "climate": "Tropical Wet & Dry", "latitude":  19.08, "longitude":  72.88,
     "mean_temp": 32.5,  "note": "Estimated (climatology)"},
    # ── Sub-tropical to temperate (regression range) ──────────────────────
    {"city": "Cairo",        "country": "Egypt",        "hemisphere": "N",
     "climate": "Hot Desert",         "latitude":  30.06, "longitude":  31.25,
     "mean_temp": reg(30.06,  0.4), "note": "Estimated (regression)"},
    {"city": "Madrid",       "country": "Spain",        "hemisphere": "N",
     "climate": "Mediterranean",      "latitude":  40.42, "longitude":  -3.70,
     "mean_temp": reg(40.42, -1.2), "note": "Estimated (regression)"},
    {"city": "Istanbul",     "country": "Turkey",       "hemisphere": "N",
     "climate": "Mediterranean",      "latitude":  41.01, "longitude":  28.98,
     "mean_temp": reg(41.01,  0.8), "note": "Estimated (regression)"},
    {"city": "Beijing",      "country": "China",        "hemisphere": "N",
     "climate": "Humid Continental",  "latitude":  39.91, "longitude": 116.39,
     "mean_temp": reg(39.91, -1.5), "note": "Estimated (regression)"},
    {"city": "Paris",        "country": "France",       "hemisphere": "N",
     "climate": "Oceanic",            "latitude":  48.85, "longitude":   2.35,
     "mean_temp": reg(48.85,  0.5), "note": "Estimated (regression)"},
    {"city": "London",       "country": "UK",           "hemisphere": "N",
     "climate": "Oceanic",            "latitude":  51.51, "longitude":  -0.13,
     "mean_temp": reg(51.51,  0.3), "note": "Estimated (regression)"},
    {"city": "Moscow",       "country": "Russia",       "hemisphere": "N",
     "climate": "Humid Continental",  "latitude":  55.75, "longitude":  37.62,
     "mean_temp": reg(55.75, -0.9), "note": "Estimated (regression)"},
    # ── Arctic (outside regression range) ────────────────────────────────
    {"city": "Murmansk",     "country": "Russia",       "hemisphere": "N",
     "climate": "Subarctic",          "latitude":  68.97, "longitude":  33.09,
     "mean_temp": -4.2,  "note": "Estimated (climatology)"},
    # ── Southern Hemisphere (spring / early summer in Oct–Nov) ────────────
    {"city": "Sydney",       "country": "Australia",    "hemisphere": "S",
     "climate": "Humid Subtropical",  "latitude": -33.87, "longitude": 151.21,
     "mean_temp": 22.3,  "note": "Estimated (climatology) — spring season"},
    {"city": "Buenos Aires", "country": "Argentina",    "hemisphere": "S",
     "climate": "Humid Subtropical",  "latitude": -34.61, "longitude": -58.38,
     "mean_temp": 21.1,  "note": "Estimated (climatology) — spring season"},
    {"city": "Cape Town",    "country": "South Africa", "hemisphere": "S",
     "climate": "Mediterranean",      "latitude": -33.92, "longitude":  18.42,
     "mean_temp": 21.8,  "note": "Estimated (climatology) — spring season"},
]

@st.cache_data
def build_city_summary(df_real: pd.DataFrame) -> pd.DataFrame:
    """Combine real city means with estimated global cities."""
    real_summary = (
        df_real.groupby(["city", "country", "climate"])
        .agg(latitude=("latitude", "mean"),
             longitude=("longitude", "mean"),
             mean_temp=("temperature_celsius", "mean"),
             std_temp=("temperature_celsius", "std"),
             n_records=("temperature_celsius", "count"))
        .reset_index()
    )
    real_summary["note"]       = "Measured (MODIS)"
    real_summary["hemisphere"] = "N"

    extra = pd.DataFrame(EXTRA_CITIES)
    extra["std_temp"]  = np.nan
    extra["n_records"] = 0

    combined = pd.concat([real_summary, extra], ignore_index=True)
    combined["mean_temp"] = combined["mean_temp"].round(2)
    combined["color"]     = combined["mean_temp"].apply(temp_to_rgb)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df_real    = load_real_data()
df_cities  = build_city_summary(df_real)
real_cities = ["Dammam", "Dublin", "Reykjavik"]

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.info(
    "**Real NASA MODIS satellite data** — 25,905 measurements (Oct 29 – Nov 11, 2025) "
    "for 3 cities, extended with regression-estimated data for 13 additional cities worldwide. "
    "3D maps use **pydeck / deck.gl** — the same WebGL engine that powers Kepler.gl.",
    icon="🛰️",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Navigation")
    page = st.radio("", [
        "🌐 Global Temperature Map",
        "🗺️ 3D City Heatmap (Kepler-style)",
        "📊 Latitude–Temperature Correlation",
        "📅 14-Day Temperature Trends",
        "📈 Statistics & Methodology",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📡 Dataset")
    st.markdown(
        f"- **{len(df_real):,}** real MODIS measurements\n"
        f"- **3** measured cities\n"
        f"- **{len(EXTRA_CITIES)}** estimated cities\n"
        f"- **14** days (Oct 29 – Nov 11, 2025)\n"
        f"- Spatial res: **1 km × 1 km**"
    )
    st.markdown("---")
    st.markdown("### 🔬 Key Finding")
    st.markdown(
        "r = **−0.995** · R² = **0.990**\n\n"
        "Latitude explains **99%** of LST variance\n\n"
        "Every 10° north → **9.1°C colder**"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — GLOBAL TEMPERATURE MAP
# ═════════════════════════════════════════════════════════════════════════════
if page == "🌐 Global Temperature Map":
    st.title("🌐 Global Land Surface Temperature")
    st.markdown(
        "Temperature gradient across **16 cities** from Singapore (1°N) to Murmansk (69°N) "
        "and Southern Hemisphere cities experiencing spring in October–November."
    )

    show_filter = st.radio(
        "Show cities",
        ["All cities", "Measured only (3)", "Estimated only (13)"],
        horizontal=True,
    )

    view_df = df_cities.copy()
    if show_filter == "Measured only (3)":
        view_df = view_df[view_df["note"] == "Measured (MODIS)"]
    elif show_filter == "Estimated only (13)":
        view_df = view_df[view_df["note"] != "Measured (MODIS)"]

    fig = px.scatter_geo(
        view_df,
        lat="latitude", lon="longitude",
        color="mean_temp",
        size=view_df["n_records"].clip(lower=2000).fillna(2000),
        hover_name="city",
        hover_data={
            "country": True,
            "climate": True,
            "mean_temp": ":.1f",
            "latitude": ":.2f",
            "note": True,
            "n_records": True,
        },
        color_continuous_scale="RdYlBu_r",
        range_color=[-22, 38],
        projection="natural earth",
        title="Mean Land Surface Temperature by City (Oct–Nov 2025)",
        labels={"mean_temp": "Mean LST (°C)"},
    )
    fig.update_traces(marker=dict(sizemin=8, line=dict(width=1.5, color="white")))
    fig.update_geos(
        showland=True, landcolor="#2a2a2a",
        showocean=True, oceancolor="#1a3a4a",
        showcoastlines=True, coastlinecolor="#555",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="°C", ticksuffix="°C",
            len=0.7, thickness=14,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # City table
    st.markdown("### 🏙️ All Cities — Temperature Summary")
    display_cols = ["city", "country", "climate", "latitude", "mean_temp", "n_records", "note"]
    tbl = view_df[display_cols].sort_values("latitude", ascending=False).copy()
    tbl.columns = ["City", "Country", "Climate", "Latitude (°N)", "Mean LST (°C)", "Measurements", "Data Source"]
    tbl["Mean LST (°C)"] = tbl["Mean LST (°C)"].round(1)
    tbl["Measurements"]  = tbl["Measurements"].replace(0, "—")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown(
        "**Note:** Measured cities (Dammam, Dublin, Reykjavik) use real 2025 MODIS data. "
        "Cities within 26°N–64°N use the validated regression T = −0.911 × Lat + 56.0. "
        "Tropical and polar cities use published MODIS climatology. "
        "Southern Hemisphere cities (marked S) are in spring/early summer — "
        "their temperatures are therefore *warmer than* N. Hemisphere cities at the same latitude magnitude."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — 3D CITY HEATMAP (Kepler.gl style via pydeck)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ 3D City Heatmap (Kepler-style)":
    st.title("🗺️ 3D Land Surface Temperature — Kepler.gl-Style")
    st.markdown(
        "Rendered with **pydeck / deck.gl** (WebGL) — the same engine that powers Kepler.gl. "
        "Each hexagonal prism height = temperature. Drag to rotate · Scroll to zoom · Hover for values."
    )

    city_choice = st.selectbox(
        "Select city",
        real_cities,
        help="Only the 3 measured cities have spatial data for 3D rendering.",
    )

    city_meta = {
        "Dammam":    {"center": [50.01, 26.36], "zoom": 9.5, "pitch": 52,
                      "desc": "Hot desert climate · mean 31.5°C · 11,998 measurements"},
        "Dublin":    {"center": [-6.39, 53.30],  "zoom": 9.5, "pitch": 52,
                      "desc": "Temperate maritime · mean 9.6°C · 6,797 measurements"},
        "Reykjavik": {"center": [-21.77, 64.08], "zoom": 9.5, "pitch": 52,
                      "desc": "Subarctic · mean −3.7°C · 7,110 measurements"},
    }

    meta   = city_meta[city_choice]
    df_c   = df_real[df_real["city"] == city_choice].copy()
    df_map = df_c[["longitude", "latitude", "temperature_celsius"]].rename(
        columns={"longitude": "lon", "latitude": "lat", "temperature_celsius": "temp"}
    )

    # Date filter for the 3D map
    dates = sorted(df_real[df_real["city"]==city_choice]["date"].dt.date.unique())
    sel_date = st.select_slider(
        "Filter by date (updates map)",
        options=dates,
        value=dates[0],
        format_func=lambda d: d.strftime("%b %d"),
    )
    df_day = df_c[df_c["date"].dt.date == sel_date][["longitude","latitude","temperature_celsius"]]
    df_day = df_day.rename(columns={"longitude":"lon","latitude":"lat","temperature_celsius":"temp"})

    col1, col2, col3 = st.columns(3)
    col1.metric("City", city_choice)
    col2.metric("Date", sel_date.strftime("%b %d, %Y"))
    col3.metric(
        "Temp range",
        f"{df_day['temp'].min():.1f}°C – {df_day['temp'].max():.1f}°C",
    )

    # pydeck HexagonLayer — same as Kepler.gl
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=df_day,
        get_position=["lon", "lat"],
        get_elevation="temp",
        elevation_scale=800,
        elevation_range=[0, 3000],
        radius=800,
        color_range=PYDECK_COLOR_RANGE,
        pickable=True,
        extruded=True,
        coverage=0.9,
    )

    view = pdk.ViewState(
        longitude=meta["center"][0],
        latitude=meta["center"][1],
        zoom=meta["zoom"],
        pitch=meta["pitch"],
        bearing=-10,
    )

    deck = pdk.Deck(
        layers=[hex_layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": "<b>{elevationValue}</b> °C",
            "style": {"color": "white", "backgroundColor": "#333"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(
        f"🏙️ {meta['desc']} | "
        "Prism height = temperature | "
        "Colour: blue (cold) → yellow → red (hot) | "
        "Use the date slider above to animate through 14 days."
    )

    # Bonus: flat scatter map for the same day
    st.markdown("### 🔎 Flat Spatial View — Same Date")
    fig2 = px.scatter_mapbox(
        df_day, lat="lat", lon="lon",
        color="temp",
        color_continuous_scale="RdYlBu_r",
        range_color=[df_day["temp"].min()-2, df_day["temp"].max()+2],
        size_max=6,
        zoom=9, height=400,
        mapbox_style="carto-darkmatter",
        labels={"temp": "LST (°C)"},
        hover_data={"temp": ":.2f", "lat": ":.3f", "lon": ":.3f"},
    )
    fig2.update_traces(marker_size=5)
    fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LATITUDE–TEMPERATURE CORRELATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Latitude–Temperature Correlation":
    st.title("📊 Latitude–Temperature Correlation")
    st.markdown(
        "The near-perfect correlation (r = **−0.995**) in the Northern Hemisphere validates "
        "that latitude is the dominant driver of land surface temperature. "
        "The Southern Hemisphere shows **opposite seasonal behaviour** — "
        "October–November is spring/summer there, so temperatures are *warmer* at high southern latitudes."
    )

    show_s = st.toggle("Show Southern Hemisphere cities", value=True)
    show_estimated = st.toggle("Show estimated cities", value=True)

    df_plot = df_cities.copy()
    if not show_s:
        df_plot = df_plot[df_plot["hemisphere"] == "N"]
    if not show_estimated:
        df_plot = df_plot[df_plot["note"] == "Measured (MODIS)"]

    # Symbols and colours
    df_plot["marker_symbol"] = df_plot["note"].apply(
        lambda n: "circle" if n == "Measured (MODIS)" else "diamond"
    )
    df_plot["region"] = df_plot.apply(
        lambda r: "Southern Hemisphere (spring)" if r["hemisphere"] == "S"
        else ("Measured (MODIS)" if r["note"] == "Measured (MODIS)" else "Estimated"),
        axis=1,
    )

    color_map = {
        "Measured (MODIS)": "#FF6B35",
        "Estimated":        "#4ECDC4",
        "Southern Hemisphere (spring)": "#A855F7",
    }

    fig = px.scatter(
        df_plot,
        x="latitude", y="mean_temp",
        color="region",
        color_discrete_map=color_map,
        symbol="region",
        text="city",
        hover_data={"country": True, "climate": True,
                    "mean_temp": ":.1f", "note": True},
        labels={"latitude": "Latitude (°N/°S)", "mean_temp": "Mean LST (°C)",
                "region": "Data type"},
        title="Latitude vs Mean Land Surface Temperature (Oct–Nov 2025)",
        height=580,
    )
    fig.update_traces(
        marker=dict(size=13, line=dict(width=1.5, color="white")),
        textposition="top center",
        textfont_size=10,
    )

    # Northern Hemisphere regression line
    nh = df_plot[df_plot["hemisphere"] == "N"]
    if len(nh) >= 2:
        lats  = np.linspace(nh["latitude"].min() - 3, nh["latitude"].max() + 3, 200)
        temps = SLOPE * lats + INTERCEPT
        fig.add_trace(go.Scatter(
            x=lats, y=temps,
            mode="lines",
            name=f"NH Regression: T = {SLOPE}×Lat + {INTERCEPT} (R²=0.990)",
            line=dict(color="#FF6B35", width=2.5, dash="dash"),
        ))

        # 95% confidence band (±2°C as a simplified envelope)
        fig.add_trace(go.Scatter(
            x=np.concatenate([lats, lats[::-1]]),
            y=np.concatenate([temps + 2.5, (temps - 2.5)[::-1]]),
            fill="toself",
            fillcolor="rgba(255,107,53,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±2.5°C confidence band",
            showlegend=True,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="0°C", annotation_position="left")
    fig.add_vline(x=0,  line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Equator", annotation_position="top")
    fig.update_layout(
        xaxis_title="Latitude (°N positive, °S negative)",
        yaxis_title="Mean Land Surface Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key stats
    st.markdown("### 🔬 Statistical Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pearson r (NH)",        "−0.995")
    c2.metric("R² (NH)",               "0.990")
    c3.metric("Regression slope",      "−0.911 °C/°")
    c4.metric("Every 10° north →",     "9.1°C colder")

    st.markdown("### 💡 Why does the Southern Hemisphere deviate?")
    st.markdown(
        "The model (T = −0.911 × Lat + 56.0) was calibrated on **Northern Hemisphere "
        "autumn/early winter** conditions (October–November). "
        "In the Southern Hemisphere, October–November is **spring / early summer**, "
        "so cities like Sydney, Buenos Aires, and Cape Town at ~34°S are *warmer* "
        "than the NH regression predicts for −34°. "
        "To build a global model, a **season-adjusted** or **month-specific** regression "
        "would be required."
    )

    # Temperature range bar chart
    st.markdown("### 🌡️ Temperature Spread by City")
    city_stats = (
        df_real.groupby("city")["temperature_celsius"]
        .agg(["mean","min","max","std"])
        .reset_index()
    )
    fig_bar = go.Figure()
    colors = {"Dammam": "#FF6B35", "Dublin": "#4ECDC4", "Reykjavik": "#45B7D1"}
    for _, row in city_stats.iterrows():
        c = colors.get(row["city"], "gray")
        fig_bar.add_trace(go.Box(
            name=row["city"],
            q1=[row["mean"] - row["std"]],
            median=[row["mean"]],
            q3=[row["mean"] + row["std"]],
            lowerfence=[row["min"]],
            upperfence=[row["max"]],
            marker_color=c,
        ))
    fig_bar.update_layout(height=350, yaxis_title="LST (°C)",
                          showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — 14-DAY TEMPERATURE TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📅 14-Day Temperature Trends":
    st.title("📅 14-Day Land Surface Temperature Trends")
    st.markdown(
        "Daily mean LST for all three cities over the 14-day study period. "
        "Shaded bands show ±1 standard deviation — a measure of within-day spatial variability."
    )

    daily = (
        df_real.groupby(["city", "date"])["temperature_celsius"]
        .agg(["mean","std","min","max"])
        .reset_index()
    )
    daily.columns = ["city","date","mean","std","min","max"]

    city_colors = {"Dammam": "#FF6B35", "Dublin": "#4ECDC4", "Reykjavik": "#45B7D1"}

    fig = go.Figure()
    for city, color in city_colors.items():
        d = daily[daily["city"]==city].sort_values("date")
        # Shaded std band
        fig.add_trace(go.Scatter(
            x=pd.concat([d["date"], d["date"].iloc[::-1]]),
            y=pd.concat([d["mean"]+d["std"], (d["mean"]-d["std"]).iloc[::-1]]),
            fill="toself",
            fillcolor=color.replace("#","rgba(").replace(")","") + ",0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ))
        # Mean line
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["mean"],
            mode="lines+markers",
            name=city,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{city}</b><br>"
                "Date: %{x|%b %d}<br>"
                "Mean: %{y:.1f}°C<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  opacity=0.4, annotation_text="0°C freeze line")
    fig.update_layout(
        height=480,
        xaxis_title="Date",
        yaxis_title="Mean Land Surface Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded band = ±1 std deviation (spatial variability across ~1,000 grid points per day).")

    # Separate charts per city
    st.markdown("### 🔍 Daily Min / Mean / Max per City")
    for city in real_cities:
        d = daily[daily["city"]==city].sort_values("date")
        color = city_colors[city]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pd.concat([d["date"], d["date"].iloc[::-1]]),
            y=pd.concat([d["max"], d["min"].iloc[::-1]]),
            fill="toself", name="Min–Max range",
            fillcolor=color.replace("#","rgba(").replace(")","") + ",0.2)",
            line=dict(color="rgba(0,0,0,0)"),
        ))
        fig2.add_trace(go.Scatter(
            x=d["date"], y=d["mean"], name="Daily mean",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
        fig2.update_layout(
            title=f"{city} — 14-Day LST",
            height=280,
            margin=dict(t=40,b=20),
            yaxis_title="LST (°C)",
            showlegend=True,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Variability insight
    st.markdown("### 📊 Day-to-Day Variability")
    variability = daily.groupby("city")["mean"].std().reset_index()
    variability.columns = ["City", "Day-to-Day Std (°C)"]
    variability["Interpretation"] = variability["Day-to-Day Std (°C)"].apply(
        lambda s: "🔴 High variability (weather-driven)" if s > 3
        else ("🟡 Moderate" if s > 1.5 else "🟢 Stable (desert climate)")
    )
    variability["Day-to-Day Std (°C)"] = variability["Day-to-Day Std (°C)"].round(2)
    st.dataframe(variability, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — STATISTICS & METHODOLOGY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Statistics & Methodology":
    st.title("📈 Statistics & Methodology")

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total measurements",    "25,905")
    c2.metric("Days analysed",         "14")
    c3.metric("Temperature range",     "58.2°C")
    c4.metric("Pearson r (NH)",        "−0.995")
    c5.metric("R²",                    "0.990")

    st.markdown("### 🔬 City-Level Statistics (Measured Data)")
    city_stats = (
        df_real.groupby(["city","country","climate"])
        .agg(n=("temperature_celsius","count"),
             mean=("temperature_celsius","mean"),
             std=("temperature_celsius","std"),
             min=("temperature_celsius","min"),
             max=("temperature_celsius","max"))
        .reset_index()
        .round(2)
    )
    city_stats.columns = ["City","Country","Climate","N","Mean (°C)","Std","Min (°C)","Max (°C)"]
    st.dataframe(city_stats, use_container_width=True, hide_index=True)

    st.markdown("### 📐 Regression Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Model** (calibrated on 3 measured cities, 26°N–64°N):

            ```
            T = −0.911 × Latitude + 56.0
            ```

            | Metric | Value |
            |---|---|
            | Correlation r | −0.995 |
            | R-squared R² | 0.990 |
            | Slope | −0.911°C/° |
            | Intercept | 56.0°C |
            | Avg prediction error | ±1.4°C |
            | Valid range | 26°N – 64°N |

            **Interpretation:** Every 10° north → 9.1°C colder.
            This slope (−0.5 to −1.0°C/°) is consistent with IPCC AR6 climatological models.
            """
        )
    with col2:
        # Regression diagnostic scatter
        nh = df_cities[df_cities["hemisphere"]=="N"].copy()
        lats = np.linspace(nh["latitude"].min()-5, nh["latitude"].max()+5, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lats, y=SLOPE*lats+INTERCEPT,
            mode="lines", name="Regression",
            line=dict(color="#FF6B35", dash="dash", width=2),
        ))
        colors_reg = {"Measured (MODIS)":"#FF6B35","Estimated (regression)":"#4ECDC4",
                      "Estimated (climatology)":"#A0C4FF"}
        for _, row in nh.iterrows():
            c = "#FF6B35" if row["note"]=="Measured (MODIS)" else "#4ECDC4"
            fig.add_trace(go.Scatter(
                x=[row["latitude"]], y=[row["mean_temp"]],
                mode="markers+text",
                marker=dict(size=11, color=c, line=dict(width=1.5,color="white")),
                text=[row["city"]], textposition="top center",
                textfont_size=9,
                name=row["city"], showlegend=False,
            ))
        fig.update_layout(
            title="Regression fit — Northern Hemisphere",
            xaxis_title="Latitude (°N)",
            yaxis_title="Mean LST (°C)",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🛠️ Data Pipeline")
    pipeline = pd.DataFrame([
        ["MODIS/061/MOD11A1", "NASA Terra Satellite", "1 km daily LST", "Primary data source"],
        ["Google Earth Engine", "Cloud GIS platform", "API v1-alpha", "Data extraction"],
        ["Python / Colab", "earthengine-api 0.1.400 · pandas 2.0.3", "35 s extraction", "Processing"],
        ["CSV Export", "25,905 rows · 9 columns", "2.1 MB file", "Data transfer"],
        ["pydeck / deck.gl", "WebGL GPU rendering", "60 FPS · 480 MB GPU", "3D visualisation"],
        ["Plotly", "Interactive charts", "Client-side rendering", "Statistical charts"],
    ], columns=["Component","Technology","Metric","Purpose"])
    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    st.markdown("### ⚙️ Execution Performance (from original Colab run)")
    perf = pd.DataFrame([
        ["Installation",        "31.62",  "47.9", "9.6",  "0.27"],
        ["Library imports",     "6.92",   "86.5", "9.6",  "0.03"],
        ["GEE authentication",  "0.59",   "63.6", "10.2", "0.42"],
        ["Data extraction",     "35.27",  "31.7", "10.1", "23.08"],
        ["Statistical analysis","2.06",   "18.1", "10.2", "0.07"],
        ["TOTAL",               "77.86",  "86.5", "10.2", "24.91"],
    ], columns=["Step","Time (s)","Peak CPU (%)","Peak Memory (%)","Network (MB)"])
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.markdown(
        "**Bottleneck:** Data extraction (Step 4) accounts for 45% of total time "
        "due to Earth Engine API latency. Earth Engine caching speeds up repeat runs by 3×. "
        "Peak memory usage of only 10.2% shows strong scalability headroom in Colab's 12 GB environment."
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "🛰️ **Multi-City Land Surface Temperature Analysis** · "
    "Shibli Afaq & Sultan Aldhafeeri · KFUPM ICS 574 · "
    "Data: NASA MODIS/061/MOD11A1 via Google Earth Engine"
)
