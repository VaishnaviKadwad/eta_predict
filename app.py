from pathlib import Path
import os

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RailCast | Dynamic ETA Intelligence",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "demo_data.csv"
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
RESIDUAL_MODEL_PATH = BASE_DIR / "residual_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
RESIDUAL_FEATURES_PATH = BASE_DIR / "residual_features.pkl"

ASSETS_DIR = BASE_DIR / "assets"
BANNER_PATH = ASSETS_DIR / "train_banner.png"

COORDINATES_PATH = BASE_DIR / "station_coords.csv"


# ============================================================
# LOAD DATA + MODELS
# ============================================================

try:
    demo_data = pd.read_csv(DATA_PATH)

    xgb_model = joblib.load(MODEL_PATH)
    residual_model = joblib.load(RESIDUAL_MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    residual_features = joblib.load(RESIDUAL_FEATURES_PATH)

except Exception as e:
    st.error(f"Unable to load RailCast files: {e}")
    st.stop()


# ============================================================
# MODEL REFERENCE
# ============================================================

XGB_MAE = 7.88


# ============================================================
# OPTIONAL STATION COORDINATES
# ============================================================

station_coords = pd.DataFrame()

if COORDINATES_PATH.exists():

    try:
        station_coords = pd.read_csv(COORDINATES_PATH)

    except Exception:
        station_coords = pd.DataFrame()


# ============================================================
# THEME-FRIENDLY CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ---------------- BRAND ---------------- */

.brand {
    font-size: 40px;
    font-weight: 850;
    letter-spacing: -1.5px;
}

.brand-blue {
    color: #1261C9;
}

.brand-red {
    color: #D62828;
}

.tagline {
    font-size: 15px;
    opacity: 0.65;
    margin-top: -8px;
}

/* ---------------- SECTION TITLES ---------------- */

.section-title {
    font-size: 27px;
    font-weight: 800;
}

/* ---------------- TRAIN ---------------- */

.train-heading {
    font-size: 31px;
    font-weight: 850;
}

.route {
    font-size: 20px;
    font-weight: 650;
    opacity: 0.75;
}

.arrow {
    color: #D62828;
    padding: 0 9px;
}

/* ---------------- STATUS ---------------- */

.running {
    display: inline-block;
    padding: 5px 12px;
    margin-left: 9px;
    border-radius: 20px;
    background: #DCFCE7;
    color: #15803D;
    font-size: 12px;
    font-weight: 750;
}

/* ---------------- INFO PANEL ---------------- */

.info-panel {
    border-radius: 15px;
    padding: 18px;
    border: 1px solid rgba(100,100,100,0.15);
    background: rgba(100,100,100,0.035);
}

/* ---------------- SMALL LABEL ---------------- */

.small-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 750;
    opacity: 0.60;
}

/* ---------------- SIDEBAR ---------------- */

.sidebar-title {
    font-size: 27px;
    font-weight: 850;
}

/* ---------------- METRICS ---------------- */

[data-testid="stMetric"] {
    border-radius: 15px;
    border: 1px solid rgba(100,100,100,0.15);
    padding: 13px;
    background: rgba(100,100,100,0.035);
}

/* ---------------- BUTTONS ---------------- */

.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    opacity: 0.45;
    font-size: 12px;
    padding-top: 25px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

readable_names = {
    "rainfall_mm": "rainfall",
    "visibility_m": "visibility / fog conditions",
    "temperature_c": "temperature",
    "congestion_score": "downstream congestion",
    "historical_section_time": "historical section running time",
    "distance_to_next": "distance to next station",
    "station_sequence": "position along the route",
    "hour": "time of day",
    "day_of_week": "day of week",
    "delay_minutes": "current delay"
}


def explain_row(row):

    try:

        available = [
            f for f in residual_features
            if f in row.index
        ]

        if not available:
            return (
                "current operating conditions",
                "influencing the forecast"
            )

        values = (
            row[available]
            .to_frame()
            .T
            .apply(pd.to_numeric)
        )

        matrix = xgb.DMatrix(
            values,
            feature_names=available
        )

        contributions = (
            residual_model
            .get_booster()
            .predict(
                matrix,
                pred_contribs=True
            )[0][:-1]
        )

        idx = abs(contributions).argmax()

        feature = available[idx]

        if contributions[idx] >= 0:
            direction = "increasing the delay"
        else:
            direction = "reducing the delay"

        return (
            readable_names.get(
                feature,
                feature
            ),
            direction
        )

    except Exception:

        return (
            "current operating conditions",
            "influencing the forecast"
        )


def delay_risk(delay):

    if delay >= 15:
        return "🔴 HIGH"

    if delay >= 5:
        return "🟠 MEDIUM"

    return "🟢 LOW"


def build_schematic_map(train_rows, current_station):

    stations = []

    for _, row in train_rows.iterrows():

        a = str(row["station"])
        b = str(row["next_station"])

        if a not in stations:
            stations.append(a)

        if b not in stations:
            stations.append(b)

    if len(stations) < 2:
        return None

    x = list(range(len(stations)))
    y = [0] * len(stations)

    labels = []

    for station in stations:

        if station == str(current_station):
            labels.append("🚆 " + station)
        else:
            labels.append(station)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=13),
            line=dict(width=5),
            hovertemplate="%{text}<extra></extra>"
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    return fig


def build_real_map(train_rows, current_station):

    required = {
        "station",
        "latitude",
        "longitude"
    }

    if station_coords.empty:
        return None

    if not required.issubset(
        station_coords.columns
    ):
        return None

    route = []

    for _, row in train_rows.iterrows():

        a = str(row["station"])
        b = str(row["next_station"])

        if a not in route:
            route.append(a)

        if b not in route:
            route.append(b)

    coords = station_coords.copy()

    coords["station"] = (
        coords["station"]
        .astype(str)
    )

    coords = coords[
        coords["station"].isin(route)
    ].copy()

    if len(coords) < 2:
        return None

    order = {
        name: i
        for i, name in enumerate(route)
    }

    coords["_order"] = (
        coords["station"]
        .map(order)
    )

    coords = (
        coords
        .dropna(subset=["_order"])
        .sort_values("_order")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattermap(
            lat=coords["latitude"],
            lon=coords["longitude"],
            mode="lines+markers",
            text=coords["station"],
            line=dict(width=4),
            marker=dict(size=8),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Route"
        )
    )

    current = coords[
        coords["station"]
        == str(current_station)
    ]

    if not current.empty:

        fig.add_trace(
            go.Scattermap(
                lat=current["latitude"],
                lon=current["longitude"],
                mode="markers+text",
                text=[
                    "🚆 " + str(current_station)
                ],
                textposition="top center",
                marker=dict(size=18),
                name="Train"
            )
        )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(
                lat=coords["latitude"].mean(),
                lon=coords["longitude"].mean()
            ),
            zoom=5
        ),
        height=450,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        showlegend=False
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🚆 RailCast</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Dynamic ETA & Delay Intelligence"
    )

    st.divider()

    st.markdown("### DASHBOARDS")

    role = st.radio(
        "Dashboard",
        [
            "👤 Passenger",
            "🏢 Station Operator",
            "🚦 Traffic Controller",
            "🛠️ Maintenance Operator"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### MODEL")

    st.metric(
        "Validation MAE",
        f"{XGB_MAE:.2f} min"
    )

    st.caption(
        "XGBoost prediction engine"
    )

    st.caption(
        "Historical + operating-condition features"
    )

    st.divider()

    st.markdown("### ABOUT")

    st.caption(
        "Predictive railway decision-support prototype."
    )

    st.caption(
        "Operational actions remain with authorised railway staff."
    )


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        '<div class="brand">'
        '<span class="brand-blue">Rail</span>'
        '<span class="brand-red">Cast</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">'
        'Predict the arrival. Understand the delay.'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:

    now = pd.Timestamp.now()

    st.write(
        f"📅 **{now.strftime('%d %b %Y')}**"
    )

    st.write(
        f"🕐 **{now.strftime('%I:%M %p')}**"
    )


# ============================================================
# JOURNEY SELECTION
# ============================================================

st.markdown("## 🚆 Select Journey")

left, right = st.columns(
    [1, 2]
)

with left:

    train_options = (
        demo_data["train"]
        .astype(str)
        .unique()
        .tolist()
    )

    selected_train = st.selectbox(
        "Train",
        train_options
    )


train_rows = demo_data[
    demo_data["train"].astype(str)
    == str(selected_train)
].reset_index(drop=True)


with right:

    row_index = st.selectbox(
        "Journey point",
        train_rows.index.tolist(),
        format_func=lambda i:
            f"{train_rows.loc[i, 'station']} "
            f"→ "
            f"{train_rows.loc[i, 'next_station']} "
            f"({train_rows.loc[i, 'date']})"
    )


current_row = train_rows.loc[
    row_index
].copy()

current_station = str(
    current_row["station"]
)

next_station = str(
    current_row["next_station"]
)


# ============================================================
# TRAIN BANNER
# ============================================================

if BANNER_PATH.exists():

    st.image(
        str(BANNER_PATH),
        width="stretch"
    )

else:

    st.info(
        "🚆 RailCast train banner not found. "
        "Place the image at assets/train_banner.png"
    )


# ============================================================
# TRAIN INFORMATION
# ============================================================

train_left, train_right = st.columns(
    [3, 1]
)

with train_left:

    st.markdown(
        f'<div class="train-heading">'
        f'🚆 Train {selected_train}'
        f'<span class="running">● Running</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="route">'
        f'{current_station}'
        f'<span class="arrow">→</span>'
        f'{next_station}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Journey date: {current_row['date']}"
    )


with train_right:

    st.metric(
        "Current station",
        current_station
    )


# ============================================================
# BASELINE PREDICTION
# ============================================================

try:

    X_original = (
        pd.DataFrame(
            [current_row[feature_columns]]
        )
        .apply(pd.to_numeric)
    )

    normal_prediction = float(
        xgb_model.predict(X_original)[0]
    )

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ============================================================
# DISRUPTION SIMULATOR
# ============================================================

st.divider()

st.markdown(
    "## ⚡ What-if Disruption Simulator"
)

st.caption(
    "Test how an unexpected operating event could change the forecast."
)

disruption = st.selectbox(
    "Operating event",
    [
        "None",
        "Fog",
        "Heavy Rain / Storm",
        "Speed Restriction",
        "Signal Halt / Unscheduled Stoppage",
        "Track Congestion Spike",
        "Unscheduled Maintenance Block"
    ]
)


sim_row = current_row.copy()

sim_weather = str(
    current_row[
        "weather_condition_passenger"
    ]
)

disruption_note = ""


if disruption == "Fog":

    sim_row["visibility_m"] = 150

    sim_row["temperature_c"] = (
        sim_row["temperature_c"] - 3
    )

    sim_weather = "Foggy"

    disruption_note = (
        "Visibility reduced to 150m."
    )


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    sim_weather = "Heavy Rain / Storm"

    disruption_note = (
        "Rainfall increased to 80mm."
    )


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"] * 1.5
    )

    sim_weather = "Speed Restriction"

    disruption_note = (
        "Section running time increased by 50%."
    )


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 25
    )

    sim_weather = "Signal Halt"

    disruption_note = (
        "Additional 25-minute signal halt simulated."
    )


elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    sim_weather = "Congestion Spike"

    disruption_note = (
        "Downstream congestion increased to 0.95."
    )


elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 45
    )

    sim_weather = "Maintenance Block"

    disruption_note = (
        "Additional 45-minute maintenance delay simulated."
    )


# ============================================================
# DISRUPTION PREDICTION
# ============================================================

try:

    X_sim = (
        pd.DataFrame(
            [sim_row[feature_columns]]
        )
        .apply(pd.to_numeric)
    )

    disruption_prediction = float(
        xgb_model.predict(X_sim)[0]
    )

except Exception as e:

    st.error(
        f"Disruption prediction failed: {e}"
    )

    st.stop()


impact = (
    disruption_prediction
    - normal_prediction
)


# ============================================================
# FORECAST
# ============================================================

st.markdown("## ⏱️ Dynamic Forecast")

if disruption == "None":

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "CURRENT DELAY",
            f"{current_row['delay_minutes']:.0f} min"
        )

    with c2:

        st.metric(
            "RAILCAST FORECAST",
            f"{normal_prediction:.1f} min"
        )

    with c3:

        st.metric(
            "ESTIMATED RANGE",
            f"{normal_prediction - XGB_MAE:.0f}"
            f" → "
            f"{normal_prediction + XGB_MAE:.0f} min"
        )

    with c4:

        st.metric(
            "DELAY RISK",
            delay_risk(normal_prediction)
        )

else:

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "CURRENT DELAY",
            f"{current_row['delay_minutes']:.0f} min"
        )

    with c2:

        st.metric(
            "RAILCAST FORECAST",
            f"{normal_prediction:.1f} min"
        )

    with c3:

        st.metric(
            "DISRUPTION FORECAST",
            f"{disruption_prediction:.1f} min",
            delta=f"{impact:+.1f} min"
        )

    with c4:

        st.metric(
            "ESTIMATED RANGE",
            f"{disruption_prediction - XGB_MAE:.0f}"
            f" → "
            f"{disruption_prediction + XGB_MAE:.0f} min"
        )

    with c5:

        st.metric(
            "DELAY RISK",
            delay_risk(disruption_prediction)
        )

    st.warning(
        f"⚡ {disruption_note}"
    )


# ============================================================
# PASSENGER DASHBOARD
# ============================================================

if role == "👤 Passenger":

    st.divider()

    map_col, impact_col = st.columns(
        [1.6, 1]
    )

    # MAP
    with map_col:

        st.markdown("### 🗺️ Journey Map")

        real_map = build_real_map(
            train_rows,
            current_station
        )

        if real_map is not None:

            st.plotly_chart(
                real_map,
                use_container_width=True,
                theme="streamlit"
            )

        else:

            schematic = build_schematic_map(
                train_rows,
                current_station
            )

            if schematic is not None:

                st.plotly_chart(
                    schematic,
                    use_container_width=True,
                    theme="streamlit"
                )

                st.caption(
                    "Interactive geographic map will use real station "
                    "coordinates when station_coords.csv is added."
                )


    # IMPACT
    with impact_col:

        st.markdown(
            "### ⚡ Disruption Impact"
        )

        st.metric(
            "Normal forecast",
            f"{normal_prediction:.1f} min"
        )

        if disruption == "None":

            st.success(
                "No disruption currently simulated."
            )

        else:

            st.metric(
                "Current scenario",
                f"{disruption_prediction:.1f} min",
                delta=f"{impact:+.1f} min"
            )


    # EXPLANATION
    st.divider()

    exp_left, exp_right = st.columns(
        [1.4, 1]
    )

    top_feature, direction = explain_row(
        sim_row
    )

    with exp_left:

        st.markdown(
            "### 💡 Why is RailCast predicting this delay?"
        )

        st.write(
            f"RailCast forecasts approximately "
            f"**{normal_prediction:.1f} minutes of delay** "
            f"at this journey point."
        )

        st.info(
            f"📍 **Main influence:** {top_feature}\n\n"
            f"This factor is currently "
            f"**{direction}**."
        )

        st.markdown(
            "#### 🔎 What this means"
        )

        st.write(
            "RailCast combines the train's current delay, "
            "historical section behaviour and operating "
            "conditions to estimate the upcoming delay."
        )


    with exp_right:

        st.markdown(
            "### 🌦️ Operating Conditions"
        )

        a, b = st.columns(2)

        with a:

            st.metric(
                "Weather",
                sim_weather
            )

            st.metric(
                "Visibility",
                f"{sim_row['visibility_m']:.0f} m"
            )

            st.metric(
                "Rainfall",
                f"{sim_row['rainfall_mm']:.1f} mm"
            )

        with b:

            st.metric(
                "Temperature",
                f"{sim_row['temperature_c']:.1f} °C"
            )

            st.metric(
                "Congestion",
                f"{sim_row['congestion_score']:.2f}"
            )

            st.metric(
                "Section time",
                f"{sim_row['historical_section_time']:.1f} min"
            )


# ============================================================
# STATION OPERATOR
# ============================================================

elif role == "🏢 Station Operator":

    st.divider()

    st.markdown(
        "## 🏢 Station Operations Dashboard"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Current delay",
            f"{current_row['delay_minutes']:.0f} min"
        )

    with b:

        st.metric(
            "Expected delay",
            f"{normal_prediction:.1f} min"
        )

    with c:

        st.metric(
            "Congestion",
            f"{sim_row['congestion_score']:.2f}"
        )

    with d:

        st.metric(
            "Risk",
            delay_risk(normal_prediction)
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### 📢 Passenger Communication"
        )

        if normal_prediction >= 15:

            st.error(
                "High delay expected."
            )

            st.info(
                "Prepare a delay announcement and "
                "monitor passenger crowding."
            )

        elif normal_prediction >= 5:

            st.warning(
                "Moderate delay expected."
            )

            st.info(
                "Prepare a passenger delay update."
            )

        else:

            st.success(
                "Minor delay expected."
            )

            st.info(
                "No major passenger intervention required."
            )


    with right:

        st.markdown(
            "### 🚉 Station Conditions"
        )

        st.write(
            f"🌦️ Weather: **{sim_weather}**"
        )

        st.write(
            f"👁️ Visibility: "
            f"**{sim_row['visibility_m']:.0f} m**"
        )

        st.write(
            f"🌧️ Rainfall: "
            f"**{sim_row['rainfall_mm']:.1f} mm**"
        )

        st.write(
            f"🚦 Congestion: "
            f"**{sim_row['congestion_score']:.2f}**"
        )


# ============================================================
# TRAFFIC CONTROLLER
# ============================================================

elif role == "🚦 Traffic Controller":

    st.divider()

    st.markdown(
        "## 🚦 Traffic Control Intelligence"
    )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Delay Risk",
            delay_risk(disruption_prediction)
        )

    with b:

        st.metric(
            "Forecast",
            f"{disruption_prediction:.1f} min"
        )

    with c:

        st.metric(
            "Scenario Impact",
            f"{impact:+.1f} min"
        )

    st.divider()

    graph_col, intelligence_col = st.columns(
        [1.5, 1]
    )

    with graph_col:

        st.markdown(
            "### 📊 Forecast Impact"
        )

        labels = ["Normal"]

        values = [normal_prediction]

        if disruption != "None":

            labels.append(disruption)
            values.append(disruption_prediction)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                text=[
                    f"{v:.1f} min"
                    for v in values
                ],
                textposition="outside"
            )
        )

        fig.update_layout(
            height=350,
            yaxis_title="Delay (minutes)",
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=30
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme="streamlit"
        )


    with intelligence_col:

        st.markdown(
            "### ⚠️ Operational Intelligence"
        )

        st.metric(
            "Current delay",
            f"{current_row['delay_minutes']:.0f} min"
        )

        st.metric(
            "Congestion",
            f"{sim_row['congestion_score']:.2f}"
        )

        st.metric(
            "Section time",
            f"{sim_row['historical_section_time']:.1f} min"
        )

        st.info(
            "Monitor downstream sections and coordinate "
            "intervention through authorised railway operations."
        )


# ============================================================
# MAINTENANCE OPERATOR
# ============================================================

elif role == "🛠️ Maintenance Operator":

    st.divider()

    st.markdown(
        "## 🛠️ Infrastructure & Maintenance Dashboard"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Visibility",
            f"{sim_row['visibility_m']:.0f} m"
        )

    with b:

        st.metric(
            "Rainfall",
            f"{sim_row['rainfall_mm']:.1f} mm"
        )

    with c:

        st.metric(
            "Congestion",
            f"{sim_row['congestion_score']:.2f}"
        )

    with d:

        st.metric(
            "Section Time",
            f"{sim_row['historical_section_time']:.1f} min"
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### 🌦️ Infrastructure Conditions"
        )

        st.write(
            f"Weather: **{sim_weather}**"
        )

        st.write(
            f"Visibility: "
            f"**{sim_row['visibility_m']:.0f} m**"
        )

        st.write(
            f"Rainfall: "
            f"**{sim_row['rainfall_mm']:.1f} mm**"
        )

        st.write(
            f"Congestion: "
            f"**{sim_row['congestion_score']:.2f}**"
        )

        st.write(
            f"Historical section time: "
            f"**{sim_row['historical_section_time']:.1f} min**"
        )


    with right:

        st.markdown(
            "### ⚠️ Predicted Impact"
        )

        st.metric(
            "Normal",
            f"{normal_prediction:.1f} min"
        )

        st.metric(
            "Current scenario",
            f"{disruption_prediction:.1f} min",
            delta=(
                f"{impact:+.1f} min"
                if disruption != "None"
                else None
            )
        )

        if disruption != "None":

            st.warning(
                disruption_note
            )

        else:

            st.info(
                "Select an infrastructure event "
                "to simulate its effect."
            )


# ============================================================
# PASSENGER FEEDBACK
# ============================================================

st.divider()

st.markdown(
    "### 💬 Passenger Feedback"
)

feedback_left, feedback_right = st.columns(
    [2, 1]
)

with feedback_left:

    actual_delay_input = st.number_input(
        "Actual delay experienced (minutes)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=1.0
    )


with feedback_right:

    st.write("")

    if st.button(
        "Submit actual delay",
        type="primary",
        use_container_width=True
    ):

        feedback_path = (
            BASE_DIR /
            "passenger_feedback.csv"
        )

        feedback_row = pd.DataFrame(
            [{
                "train": selected_train,
                "station": current_station,
                "next_station": next_station,
                "predicted_delay": normal_prediction,
                "actual_delay": actual_delay_input,
                "submitted_at": pd.Timestamp.now()
            }]
        )

        feedback_row.to_csv(
            feedback_path,
            mode="a",
            header=not feedback_path.exists(),
            index=False
        )

        st.success(
            "Actual delay recorded successfully."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🚆 RailCast • Dynamic ETA & Delay Intelligence • SIH 2026'
    '</div>',
    unsafe_allow_html=True
)
