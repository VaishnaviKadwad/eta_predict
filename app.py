from pathlib import Path
import os

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.graph_objects as go


# ============================================================
# PAGE
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

DATA_PATH = BASE_DIR / "demo_data.csv"

MODEL_PATH = BASE_DIR / "xgb_model.pkl"
RESIDUAL_MODEL_PATH = BASE_DIR / "residual_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
RESIDUAL_FEATURES_PATH = BASE_DIR / "residual_features.pkl"

COORDINATES_PATH = BASE_DIR / "station_coords.csv"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    xgb_model = joblib.load(MODEL_PATH)
    residual_model = joblib.load(RESIDUAL_MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    residual_features = joblib.load(RESIDUAL_FEATURES_PATH)
    demo_data = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error(f"RailCast could not load the project files: {e}")
    st.stop()


# ============================================================
# MODEL RESULT
# ============================================================

XGB_MAE = 7.88


# ============================================================
# OPTIONAL COORDINATES
# ============================================================

if COORDINATES_PATH.exists():
    try:
        station_coords = pd.read_csv(COORDINATES_PATH)
    except Exception:
        station_coords = pd.DataFrame()
else:
    station_coords = pd.DataFrame()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}

/* Brand */

.brand-title {
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -1.5px;
}

.brand-blue {
    color: #1261C9;
}

.brand-red {
    color: #D62828;
}

.subtitle {
    font-size: 14px;
    opacity: 0.65;
    margin-top: -5px;
}

/* Small labels */

.overline {
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.1px;
    opacity: 0.60;
    text-transform: uppercase;
}

/* Train header */

.train-title {
    font-size: 30px;
    font-weight: 850;
}

.route-text {
    font-size: 19px;
    font-weight: 650;
    opacity: 0.75;
}

.route-arrow {
    color: #D62828;
    padding: 0 8px;
}

.status-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    background: #DDF6E4;
    color: #17753C;
    font-size: 12px;
    font-weight: 750;
    margin-left: 8px;
}

/* Section */

.section-note {
    font-size: 13px;
    opacity: 0.65;
}

/* Sidebar */

.sidebar-brand {
    font-size: 27px;
    font-weight: 850;
}

/* Buttons */

.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 700;
}

/* Metrics */

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px;
    padding: 12px;
    background: rgba(128,128,128,0.035);
}

/* Footer */

.footer {
    text-align: center;
    opacity: 0.45;
    font-size: 12px;
    padding-top: 20px;
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
    "delay_minutes": "current delay",
}


def explain_row(feature_row):
    try:
        available_features = [
            f for f in residual_features
            if f in feature_row.index
        ]

        if not available_features:
            return "current operating conditions", "influencing the forecast"

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

        contributions = (
            residual_model
            .get_booster()
            .predict(
                dmat,
                pred_contribs=True
            )[0][:-1]
        )

        top_idx = abs(contributions).argmax()
        top_feature = available_features[top_idx]

        if contributions[top_idx] > 0:
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


def get_risk(delay):
    if delay >= 15:
        return "HIGH"
    elif delay >= 5:
        return "MEDIUM"
    return "LOW"


def get_risk_color_text(delay):
    risk = get_risk(delay)

    if risk == "HIGH":
        return "🔴 HIGH"
    elif risk == "MEDIUM":
        return "🟠 MEDIUM"

    return "🟢 LOW"


def make_map(train_rows, current_station):

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

    route_stations = []

    for _, row in train_rows.iterrows():

        station = str(row["station"])
        next_station = str(row["next_station"])

        if station not in route_stations:
            route_stations.append(station)

        if next_station not in route_stations:
            route_stations.append(next_station)

    coords = station_coords.copy()

    coords["station"] = (
        coords["station"]
        .astype(str)
    )

    coords = coords[
        coords["station"].isin(
            route_stations
        )
    ].copy()

    if len(coords) < 2:
        return None

    order = {
        station: index
        for index, station
        in enumerate(route_stations)
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
                name="Current train"
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


def make_schematic_route(train_rows, current_station):

    stations = []

    for _, row in train_rows.iterrows():

        station = str(row["station"])
        next_station = str(row["next_station"])

        if station not in stations:
            stations.append(station)

        if next_station not in stations:
            stations.append(next_station)

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
            hovertemplate="<b>%{text}</b><extra></extra>"
        )
    )

    fig.update_layout(
        height=230,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False
        ),
        showlegend=False
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🚆 RailCast</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Dynamic ETA & Delay Intelligence"
    )

    st.divider()

    st.markdown("### DASHBOARDS")

    role = st.radio(
        "Operational view",
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
# HEADER
# ============================================================

left, right = st.columns(
    [4, 1]
)

with left:

    st.markdown(
        '<div class="brand-title">'
        '<span class="brand-blue">Rail</span>'
        '<span class="brand-red">Cast</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict the arrival. Understand the delay.'
        '</div>',
        unsafe_allow_html=True
    )

with right:

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

st.markdown("### 🚆 Journey")

c1, c2 = st.columns(
    [1, 2]
)

with c1:

    train_options = (
        demo_data["train"]
        .astype(str)
        .unique()
        .tolist()
    )

    selected_train = st.selectbox(
        "Select train",
        train_options
    )


train_rows = demo_data[
    demo_data["train"].astype(str)
    == str(selected_train)
].reset_index(drop=True)


with c2:

    row_index = st.selectbox(
        "Select journey point",
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


# ============================================================
# TRAIN SUMMARY
# ============================================================

st.markdown(
    f'<div class="train-title">'
    f'🚆 Train {selected_train}'
    f'<span class="status-pill">● Running</span>'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="route-text">'
    f'{current_station}'
    f'<span class="route-arrow">→</span>'
    f'{next_station}'
    f'</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Journey date: {current_row['date']}"
)


# ============================================================
# NORMAL MODEL PREDICTION
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
        f"Normal prediction failed: {e}"
    )
    st.stop()


# ============================================================
# DISRUPTION
# ============================================================

st.divider()

st.markdown(
    "### ⚡ What-if Disruption Simulator"
)

st.caption(
    "Inject a hypothetical operating event and immediately compare its predicted impact."
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

disruption_note = ""

sim_weather = str(
    current_row[
        "weather_condition_passenger"
    ]
)


if disruption == "Fog":

    sim_row["visibility_m"] = 150

    sim_row["temperature_c"] = (
        sim_row["temperature_c"] - 3
    )

    disruption_note = (
        "Visibility reduced to 150m "
        "to represent dense fog."
    )

    sim_weather = "Foggy"


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80mm "
        "to represent severe rainfall."
    )

    sim_weather = "Heavy Rain / Storm"


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"] * 1.5
    )

    disruption_note = (
        "Section running time increased by 50% "
        "to represent a temporary speed restriction."
    )

    sim_weather = "Speed Restriction"


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 25
    )

    disruption_note = (
        "Additional 25-minute signal halt introduced."
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
        "adding 45 minutes."
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
# MAIN FORECAST
# ============================================================

st.markdown("### ⏱️ Dynamic Forecast")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:

    st.metric(
        "CURRENT DELAY",
        f"{current_row['delay_minutes']:.0f} min"
    )

with m2:

    st.metric(
        "RAILCAST FORECAST",
        f"{normal_prediction:.1f} min"
    )

with m3:

    st.metric(
        "DISRUPTION FORECAST",
        f"{disruption_prediction:.1f} min"
    )

with m4:

    st.metric(
        "ESTIMATED RANGE",
        f"{normal_prediction - XGB_MAE:.0f}"
        f" → "
        f"{normal_prediction + XGB_MAE:.0f} min"
    )

with m5:

    st.metric(
        "DELAY RISK",
        get_risk_color_text(
            disruption_prediction
        )
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

        route_map = make_map(
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

            schematic = make_schematic_route(
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
                    "Geographic map will activate automatically "
                    "when station coordinates are added."
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

            st.success(
                "No disruption currently simulated."
            )


    st.divider()

    why_col, condition_col = st.columns(
        [1.3, 1]
    )

    with why_col:

        st.markdown(
            "### 💡 Why this prediction?"
        )

        top_feature, direction = explain_row(
            sim_row
        )

        st.write(
            f"RailCast identifies **{top_feature}** "
            f"as the strongest contributing factor."
        )

        st.write(
            f"Current effect: **{direction}**."
        )


    with condition_col:

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
            f"{sim_row['congestion_score']:.2f}"
        )

    with d:

        st.metric(
            "Route position",
            f"{current_row['station_sequence']:.0f}"
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

            recommendation = (
                "Prepare a delay announcement "
                "and monitor passenger crowding."
            )

        elif normal_prediction >= 5:

            st.warning(
                "Moderate delay expected."
            )

            recommendation = (
                "Prepare a passenger delay update."
            )

        else:

            st.success(
                "Minor delay expected."
            )

            recommendation = (
                "No major passenger intervention required."
            )

        st.info(
            f"Recommended action: {recommendation}"
        )


    with right:

        st.markdown(
            "### 🌦️ Station Conditions"
        )

        x, y = st.columns(2)

        with x:

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

        with y:

            st.metric(
                "Congestion",
                f"{sim_row['congestion_score']:.2f}"
            )

            st.metric(
                "Section time",
                f"{sim_row['historical_section_time']:.1f} min"
            )

            st.metric(
                "Route position",
                f"{sim_row['station_sequence']:.0f}"
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
            get_risk_color_text(
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
            "Disruption Impact",
            f"{impact:+.1f} min"
        )

    st.divider()

    graph_col, intelligence_col = st.columns(
        [1.4, 1]
    )

    with graph_col:

        st.markdown(
            "### 📊 Forecast Comparison"
        )

        labels = [
            "Normal",
            disruption
            if disruption != "None"
            else "Current"
        ]

        values = [
            normal_prediction,
            disruption_prediction
        ]

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
            height=340,
            yaxis_title="Predicted delay (minutes)",
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
            "Section running time",
            f"{sim_row['historical_section_time']:.1f} min"
        )

        st.metric(
            "Route position",
            f"{current_row['station_sequence']:.0f}"
        )

        if disruption != "None":

            st.warning(
                disruption_note
            )

        st.info(
            "Monitor downstream sections and "
            "coordinate intervention through authorised railway operations."
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

    left, right = st.columns(2)

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
            f"**Historical section time:** "
            f"{sim_row['historical_section_time']:.1f} min"
        )


    with right:

        st.markdown(
            "### ⚠️ Predicted Impact"
        )

        st.metric(
            "Normal forecast",
            f"{normal_prediction:.1f} min"
        )

        st.metric(
            "With selected event",
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
# MODEL EXPLANATION
# ============================================================

st.divider()

st.markdown(
    "### 💡 Prediction Explanation"
)

top_feature, direction = explain_row(
    sim_row
)

e1, e2, e3 = st.columns(3)

with e1:

    st.metric(
        "Top contributing factor",
        top_feature
    )

with e2:

    st.metric(
        "Effect",
        direction
    )

with e3:

    st.metric(
        "Validation MAE",
        f"{XGB_MAE:.2f} min"
    )


st.caption(
    "The displayed estimated range uses the model's validation MAE "
    "as a simple uncertainty reference; it is not a calibrated statistical confidence interval."
)


# ============================================================
# FEEDBACK
# ============================================================

st.divider()

st.markdown(
    "### 💬 Passenger Feedback"
)

feedback_col1, feedback_col2 = st.columns(
    [2, 1]
)

with feedback_col1:

    actual_delay_input = st.number_input(
        "Actual delay experienced (minutes)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=1.0
    )


with feedback_col2:

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

        feedback = pd.DataFrame(
            [{
                "train": selected_train,
                "station": current_station,
                "next_station": next_station,
                "predicted_delay": normal_prediction,
                "actual_delay": actual_delay_input,
                "submitted_at": pd.Timestamp.now()
            }]
        )

        feedback.to_csv(
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
    'RailCast • Dynamic ETA & Delay Intelligence • SIH 2026'
    '</div>',
    unsafe_allow_html=True
)
