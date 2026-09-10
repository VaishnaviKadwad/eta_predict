import os
from pathlib import Path

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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

BANNER_PATH = ASSETS_DIR / "train_banner.png"

MODEL_PATH = BASE_DIR / "xgb_model.pkl"
RESIDUAL_MODEL_PATH = BASE_DIR / "residual_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
RESIDUAL_FEATURES_PATH = BASE_DIR / "residual_features.pkl"
DATA_PATH = BASE_DIR / "demo_data.csv"

COORDINATES_PATH = BASE_DIR / "station_coords.csv"


# ============================================================
# LOAD MODEL + DATA
# ============================================================

try:
    xgb_model = joblib.load(MODEL_PATH)
    residual_model = joblib.load(RESIDUAL_MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    residual_features = joblib.load(RESIDUAL_FEATURES_PATH)

    demo_data = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error(f"Could not load RailCast project files: {e}")
    st.stop()


# ============================================================
# MODEL REFERENCE
# ============================================================

XGB_MAE = 7.88


# ============================================================
# OPTIONAL STATION COORDINATES
# ============================================================

if COORDINATES_PATH.exists():

    try:
        station_coords = pd.read_csv(COORDINATES_PATH)
    except Exception:
        station_coords = pd.DataFrame()

else:
    station_coords = pd.DataFrame()


# ============================================================
# EXPLANATION NAMES
# ============================================================

readable_names = {

    "rainfall_mm":
        "rainfall",

    "visibility_m":
        "visibility / fog conditions",

    "temperature_c":
        "temperature",

    "congestion_score":
        "downstream congestion",

    "historical_section_time":
        "historical section running time",

    "distance_to_next":
        "distance to next station",

    "station_sequence":
        "position along the route",

    "hour":
        "time of day",

    "day_of_week":
        "day of week",

    "delay_minutes":
        "current delay",
}


# ============================================================
# THEME-FRIENDLY CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- BRAND ---------- */

    .brand {
        font-size: 36px;
        font-weight: 850;
        letter-spacing: -1.5px;
        line-height: 1;
    }

    .brand-blue {
        color: #1261c9;
    }

    .brand-red {
        color: #d62828;
    }

    .tagline {
        font-size: 14px;
        opacity: 0.68;
        margin-top: 8px;
    }


    /* ---------- SIDEBAR ---------- */

    .side-logo {
        font-size: 28px;
        font-weight: 850;
        letter-spacing: -1px;
    }

    .side-subtitle {
        font-size: 12px;
        opacity: 0.65;
        margin-top: -3px;
    }


    /* ---------- TRAIN HEADER ---------- */

    .train-name {
        font-size: 27px;
        font-weight: 850;
        line-height: 1.2;
    }

    .route {
        font-size: 18px;
        font-weight: 600;
        opacity: 0.75;
    }

    .route-arrow {
        color: #d62828;
        padding: 0 8px;
        font-weight: 900;
    }


    /* ---------- STATUS ---------- */

    .running {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: #d9f7df;
        color: #16753a;
        font-size: 12px;
        font-weight: 750;
        margin-left: 7px;
        vertical-align: middle;
    }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 15px;
        padding: 17px;
        min-height: 126px;
        background: rgba(128,128,128,0.035);
    }

    .metric-card-blue {
        border-top: 4px solid #3182ce;
    }

    .metric-card-green {
        border-top: 4px solid #25a05a;
    }

    .metric-card-orange {
        border-top: 4px solid #e67e22;
    }

    .metric-card-purple {
        border-top: 4px solid #805ad5;
    }

    .metric-card-red {
        border-top: 4px solid #d62828;
    }

    .metric-label {
        font-size: 12px;
        opacity: 0.68;
        font-weight: 600;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 850;
        margin-top: 7px;
    }

    .metric-sub {
        font-size: 12px;
        opacity: 0.62;
        margin-top: 6px;
    }


    /* ---------- PANEL ---------- */

    .panel {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 15px;
        padding: 18px;
        background: rgba(128,128,128,0.025);
    }


    /* ---------- ROLE BADGES ---------- */

    .role-badge {
        display: inline-block;
        padding: 6px 13px;
        border-radius: 20px;
        background: rgba(18,97,201,0.10);
        color: #1261c9;
        font-weight: 700;
        font-size: 12px;
    }


    /* ---------- ALERT ---------- */

    .recommendation {
        border-left: 4px solid #1261c9;
        padding: 12px 15px;
        border-radius: 8px;
        background: rgba(18,97,201,0.07);
        margin-top: 10px;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        opacity: 0.5;
        font-size: 12px;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def explain_row(feature_row):

    try:

        available_features = [
            feature
            for feature in residual_features
            if feature in feature_row.index
        ]

        if not available_features:
            return (
                "current operating conditions",
                "influencing the forecast"
            )

        row_df = (
            feature_row[available_features]
            .to_frame()
            .T
            .apply(pd.to_numeric)
        )

        dmat = xgb.DMatrix(
            row_df,
            feature_names=available_features
        )

        contribs = (
            residual_model
            .get_booster()
            .predict(
                dmat,
                pred_contribs=True
            )[0][:-1]
        )

        top_index = abs(contribs).argmax()

        top_feature = available_features[top_index]

        if contribs[top_index] > 0:
            direction = "increasing the delay"
        else:
            direction = "reducing the delay"

        return (
            readable_names.get(
                top_feature,
                top_feature
            ),
            direction
        )

    except Exception:

        return (
            "current operating conditions",
            "influencing the forecast"
        )


def risk_level(delay):

    if delay >= 15:
        return "HIGH"

    elif delay >= 5:
        return "MEDIUM"

    return "LOW"


def risk_message(delay):

    if delay >= 15:
        return "High downstream delay risk detected."

    elif delay >= 5:
        return "Moderate delay risk. Continue monitoring."

    return "No major delay risk detected."


def make_route_map(
    train_rows,
    current_station
):

    if station_coords.empty:
        return None

    required = {
        "station",
        "latitude",
        "longitude"
    }

    if not required.issubset(
        station_coords.columns
    ):
        return None

    stations = []

    for station in train_rows["station"]:
        if station not in stations:
            stations.append(station)

    for station in train_rows["next_station"]:
        if station not in stations:
            stations.append(station)

    coords = station_coords[
        station_coords["station"]
        .astype(str)
        .isin(
            [str(x) for x in stations]
        )
    ].copy()

    if len(coords) < 2:
        return None

    # Preserve journey order
    order = {
        str(station): i
        for i, station in enumerate(stations)
    }

    coords["_order"] = (
        coords["station"]
        .astype(str)
        .map(order)
    )

    coords = (
        coords
        .dropna(subset=["_order"])
        .sort_values("_order")
    )

    fig = go.Figure()

    # Route
    fig.add_trace(
        go.Scattermap(
            lat=coords["latitude"],
            lon=coords["longitude"],
            mode="lines+markers",
            text=coords["station"],
            hovertemplate=(
                "<b>%{text}</b>"
                "<extra></extra>"
            ),
            line=dict(
                width=4
            ),
            marker=dict(
                size=8
            ),
            name="Route"
        )
    )

    # Current train
    current = coords[
        coords["station"]
        .astype(str)
        == str(current_station)
    ]

    if not current.empty:

        fig.add_trace(
            go.Scattermap(
                lat=current["latitude"],
                lon=current["longitude"],
                mode="markers+text",
                text=[
                    "🚆 " +
                    str(current_station)
                ],
                textposition="top center",
                marker=dict(
                    size=18
                ),
                name="Current Train"
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

        height=470,

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
        """
        <div class="side-logo">
            🚆 RailCast
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-subtitle">
            Dynamic ETA & Delay Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### DASHBOARDS")

    role = st.radio(
        "Select operational view",
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
        "Historical data + operating conditions"
    )

    st.divider()

    st.markdown("### ABOUT")

    st.caption(
        "RailCast is a decision-support "
        "prototype for dynamic train ETA "
        "forecasting."
    )

    st.caption(
        "Operational actions remain with "
        "authorised railway staff."
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        """
        <div class="brand">
            🚆
            <span class="brand-blue">Rail</span><span class="brand-red">Cast</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="tagline">
            Dynamic ETA & Delay Intelligence
            &nbsp; • &nbsp;
            Predict the arrival. Understand the delay.
        </div>
        """,
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
# TRAIN SELECTION
# ============================================================

st.markdown("### 🚆 Select Journey")

select1, select2 = st.columns(
    [1, 2]
)

with select1:

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


with select2:

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

current_station = current_row[
    "station"
]

next_station = current_row[
    "next_station"
]


# ============================================================
# TRAIN BANNER
# ============================================================

if BANNER_PATH.exists():

    st.image(
        str(BANNER_PATH),
        use_container_width=True
    )


# ============================================================
# TRAIN SUMMARY
# ============================================================

summary_left, summary_right = st.columns(
    [2, 1]
)

with summary_left:

    st.markdown(
        f"""
        <div class="train-name">
            🚆 Train {selected_train}
            <span class="running">● Running</span>
        </div>

        <div class="route">
            {current_station}
            <span class="route-arrow">→</span>
            {next_station}
        </div>
        """,
        unsafe_allow_html=True
    )


with summary_right:

    st.markdown(
        """
        <div style="text-align:right;">
            <div class="section-label">
                JOURNEY DATE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="text-align:right;font-size:19px;font-weight:700;">
            📅 {current_row['date']}
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# NORMAL PREDICTION
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

st.markdown(
    "### ⚡ What-if Disruption Simulator"
)

disruption = st.selectbox(
    "Inject an operating event",

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

disruption_note = ""

sim_weather = current_row[
    "weather_condition_passenger"
]


if disruption == "Fog":

    sim_row["visibility_m"] = 150

    sim_row["temperature_c"] = (
        sim_row["temperature_c"] - 3
    )

    disruption_note = (
        "Visibility reduced to 150m, "
        "representing dense fog."
    )

    sim_weather = "Foggy"


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80mm, "
        "representing severe rainfall."
    )

    sim_weather = "Heavy Rain / Storm"


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"] * 1.5
    )

    disruption_note = (
        "Expected section running time "
        "increased by 50%."
    )

    sim_weather = "Speed Restriction"


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 25
    )

    disruption_note = (
        "Additional 25-minute signal halt "
        "introduced."
    )

    sim_weather = "Signal Halt"


elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    disruption_note = (
        "Downstream congestion increased "
        "to near maximum."
    )

    sim_weather = "Congestion Spike"


elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 45
    )

    disruption_note = (
        "Unscheduled maintenance block "
        "adds 45 minutes."
    )

    sim_weather = "Maintenance Block"


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
# MAIN FORECAST CARDS
# ============================================================

st.markdown(
    "### ⏱️ Dynamic Forecast"
)

c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.markdown(
        f"""
        <div class="metric-card metric-card-blue">

            <div class="metric-label">
                🕐 CURRENT DELAY
            </div>

            <div class="metric-value">
                {current_row['delay_minutes']:.0f} min
            </div>

            <div class="metric-sub">
                Recorded at current journey point
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-card metric-card-green">

            <div class="metric-label">
                🎯 RAILCAST FORECAST
            </div>

            <div class="metric-value">
                {normal_prediction:.1f} min
            </div>

            <div class="metric-sub">
                Predicted delay
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-card metric-card-orange">

            <div class="metric-label">
                ⚠️ DISRUPTION FORECAST
            </div>

            <div class="metric-value">
                {disruption_prediction:.1f} min
            </div>

            <div class="metric-sub">
                Current scenario
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-card metric-card-purple">

            <div class="metric-label">
                📊 EXPECTED RANGE
            </div>

            <div class="metric-value">
                {normal_prediction - XGB_MAE:.0f}
                → 
                {normal_prediction + XGB_MAE:.0f}
            </div>

            <div class="metric-sub">
                Based on validation MAE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c5:

    risk = risk_level(
        disruption_prediction
    )

    card_class = (
        "metric-card-red"
        if risk == "HIGH"
        else "metric-card-orange"
        if risk == "MEDIUM"
        else "metric-card-green"
    )

    st.markdown(
        f"""
        <div class="metric-card {card_class}">

            <div class="metric-label">
                🚦 DELAY RISK
            </div>

            <div class="metric-value">
                {risk}
            </div>

            <div class="metric-sub">
                {risk_message(disruption_prediction)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ROLE BADGE
# ============================================================

role_names = {
    "👤 Passenger": "Passenger Intelligence",
    "🏢 Station Operator": "Station Operations",
    "🚦 Traffic Controller": "Traffic Control",
    "🛠️ Maintenance Operator": "Infrastructure & Maintenance"
}

st.markdown(
    f"""
    <br>
    <span class="role-badge">
        {role_names[role]}
    </span>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PASSENGER DASHBOARD
# ============================================================

if role == "👤 Passenger":

    st.divider()

    map_col, action_col = st.columns(
        [1.6, 1]
    )

    # ---------------- MAP ----------------

    with map_col:

        st.markdown(
            "### 📍 Journey Map"
        )

        route_map = make_route_map(
            train_rows,
            current_station
        )

        if route_map is not None:

            st.plotly_chart(
                route_map,
                use_container_width=True,
                theme="streamlit"
            )

        else:

            # Fallback visual route using station sequence.
            route_display = train_rows[
                [
                    "station",
                    "next_station"
                ]
            ].copy()

            stations = []

            for _, row in route_display.iterrows():

                if row["station"] not in stations:
                    stations.append(row["station"])

                if row["next_station"] not in stations:
                    stations.append(row["next_station"])

            if len(stations) > 1:

                fig = go.Figure()

                x_values = list(
                    range(len(stations))
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=[0] * len(stations),
                        mode="lines+markers+text",
                        text=[
                            (
                                "🚆 "
                                if str(s) == str(current_station)
                                else ""
                            ) + str(s)
                            for s in stations
                        ],
                        textposition="top center",
                        marker=dict(
                            size=13
                        ),
                        line=dict(
                            width=5
                        ),
                        hovertemplate=(
                            "<b>%{text}</b>"
                            "<extra></extra>"
                        )
                    )
                )

                fig.update_layout(
                    height=250,
                    xaxis=dict(
                        visible=False
                    ),
                    yaxis=dict(
                        visible=False
                    ),
                    margin=dict(
                        l=20,
                        r=20,
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

                st.caption(
                    "Add real station coordinates in "
                    "`station_coords.csv` to enable "
                    "the geographic map."
                )


    # ---------------- PASSENGER ACTION ----------------

    with action_col:

        st.markdown(
            "### ⚡ Disruption Impact"
        )

        if disruption == "None":

            st.success(
                "No disruption currently simulated."
            )

            st.metric(
                "Current forecast",
                f"{normal_prediction:.1f} min"
            )

        else:

            st.warning(
                disruption_note
            )

            st.metric(
                "Normal forecast",
                f"{normal_prediction:.1f} min"
            )

            st.metric(
                "With disruption",
                f"{disruption_prediction:.1f} min",
                delta=f"{impact:+.1f} min"
            )

        st.markdown(
            "### 💡 Why this prediction?"
        )

        top_feature, direction = explain_row(
            sim_row
        )

        st.write(
            f"The forecast is mainly influenced by "
            f"**{top_feature}**, currently "
            f"**{direction}**."
        )

        st.info(
            f"🌦️ Current conditions: **{sim_weather}**"
        )


# ============================================================
# STATION OPERATOR
# ============================================================

elif role == "🏢 Station Operator":

    st.divider()

    st.markdown(
        "## 🏢 Station Operations"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Current delay",
            f"{current_row['delay_minutes']:.0f} min"
        )

    with b:

        st.metric(
            "Forecast delay",
            f"{normal_prediction:.1f} min"
        )

    with c:

        st.metric(
            "Congestion",
            f"{current_row['congestion_score']:.2f}"
        )

    with d:

        st.metric(
            "Route position",
            f"{current_row['station_sequence']:.0f}"
        )


    st.divider()

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.markdown(
            "### 📢 Passenger Communication"
        )

        if normal_prediction >= 15:

            st.error(
                "High delay expected."
            )

            recommendation = (
                "Prepare a passenger delay "
                "announcement and monitor platform crowding."
            )

        elif normal_prediction >= 5:

            st.warning(
                "Moderate delay expected."
            )

            recommendation = (
                "Prepare a brief delay update "
                "for passengers."
            )

        else:

            st.success(
                "Minor delay expected."
            )

            recommendation = (
                "No major passenger intervention "
                "required."
            )

        st.markdown(
            f"""
            <div class="recommendation">
                <b>Recommended action</b><br>
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )


    with right:

        st.markdown(
            "### 🌦️ Current Conditions"
        )

        o1, o2 = st.columns(2)

        with o1:

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

        with o2:

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
            risk_level(
                disruption_prediction
            )
        )

    with b:

        st.metric(
            "Predicted Delay",
            f"{disruption_prediction:.1f} min"
        )

    with c:

        st.metric(
            "Delay Impact",
            f"{impact:+.1f} min"
        )


    st.divider()

    left, right = st.columns(
        [1.3, 1]
    )


    with left:

        st.markdown(
            "### 📊 Forecast vs Disruption"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[
                    normal_prediction,
                    disruption_prediction
                ],
                y=[
                    "Normal",
                    disruption
                ],
                orientation="h",
                text=[
                    f"{normal_prediction:.1f} min",
                    f"{disruption_prediction:.1f} min"
                ],
                textposition="outside"
            )
        )

        fig.update_layout(
            height=300,
            xaxis_title="Predicted delay (minutes)",
            margin=dict(
                l=10,
                r=40,
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


    with right:

        st.markdown(
            "### ⚠️ Operational Intelligence"
        )

        st.metric(
            "Current delay",
            f"{current_row['delay_minutes']:.0f} min"
        )

        st.metric(
            "Congestion score",
            f"{sim_row['congestion_score']:.2f}"
        )

        st.metric(
            "Historical section time",
            f"{sim_row['historical_section_time']:.1f} min"
        )

        st.metric(
            "Route position",
            f"{current_row['station_sequence']:.0f}"
        )

        st.markdown(
            f"""
            <div class="recommendation">
                <b>Control-room recommendation</b><br>
                {risk_message(disruption_prediction)}
                Monitor downstream sections and
                coordinate with authorised operations staff.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MAINTENANCE OPERATOR
# ============================================================

else:

    st.divider()

    st.markdown(
        "## 🛠️ Infrastructure & Maintenance"
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
            "Section time",
            f"{sim_row['historical_section_time']:.1f} min"
        )


    st.divider()

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.markdown(
            "### 🔧 Infrastructure Conditions"
        )

        st.write(
            f"**Weather:** {sim_weather}"
        )

        st.write(
            f"**Visibility:** "
            f"{sim_row['visibility_m']:.0f} m"
        )

        st.write(
            f"**Rainfall:** "
            f"{sim_row['rainfall_mm']:.1f} mm"
        )

        st.write(
            f"**Congestion:** "
            f"{sim_row['congestion_score']:.2f}"
        )

        st.write(
            f"**Expected section time:** "
            f"{sim_row['historical_section_time']:.1f} min"
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
            "With selected event",
            f"{disruption_prediction:.1f} min",
            delta=f"{impact:+.1f} min"
        )

        if disruption != "None":

            st.warning(
                disruption_note
            )

        else:

            st.info(
                "Select an infrastructure-related "
                "event above to simulate its impact."
            )


# ============================================================
# WHY THIS PREDICTION
# ============================================================

st.divider()

why1, why2 = st.columns(
    [1.5, 1]
)

with why1:

    st.markdown(
        "### 💡 Why this prediction?"
    )

    top_feature, direction = explain_row(
        sim_row
    )

    st.write(
        f"RailCast identifies **{top_feature}** "
        f"as the strongest contributing factor "
        f"in this forecast."
    )

    st.write(
        f"Effect: **{direction}**."
    )


with why2:

    st.markdown(
        "### 📊 Prediction Reference"
    )

    st.metric(
        "Model validation MAE",
        f"{XGB_MAE:.2f} min"
    )

    st.caption(
        "Estimated range is displayed using "
        "the validation MAE as a simple uncertainty reference."
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

    actual_delay = st.number_input(
        "Actual delay experienced at this journey point",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=1.0
    )


with feedback_right:

    st.write("")
    st.write("")

    if st.button(
        "Submit actual delay",
        type="primary",
        use_container_width=True
    ):

        feedback_file = (
            BASE_DIR /
            "passenger_feedback.csv"
        )

        feedback_row = pd.DataFrame(
            [
                {
                    "train":
                        selected_train,

                    "station":
                        current_station,

                    "next_station":
                        next_station,

                    "predicted_delay":
                        normal_prediction,

                    "actual_delay":
                        actual_delay,

                    "submitted_at":
                        pd.Timestamp.now()
                }
            ]
        )

        feedback_row.to_csv(
            feedback_file,
            mode="a",
            header=not feedback_file.exists(),
            index=False
        )

        st.success(
            "Actual delay recorded successfully."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RailCast • Dynamic ETA & Delay Intelligence •
        SIH 2026 Decision-Support Prototype
    </div>
    """,
    unsafe_allow_html=True
)
