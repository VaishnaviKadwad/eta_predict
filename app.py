import os
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RailCast | Dynamic ETA",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODELS + DATA
# ============================================================

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

try:
    xgb_model = joblib.load(BASE_DIR / "xgb_model.pkl")
    residual_model = joblib.load(BASE_DIR / "residual_model.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    residual_features = joblib.load(BASE_DIR / "residual_features.pkl")
    demo_data = pd.read_csv(BASE_DIR / "demo_data.csv")

except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# Your validated model MAE
XGB_MAE = 7.88


# ============================================================
# OPTIONAL TIMETABLE DATA
# ============================================================
#
# If timetable.csv exists, it should contain:
#
# train, station, scheduled_arrival_time, actual_arrival_time
#
# Example:
#
# 1023,PUNE,06:00,06:06
# 1023,SSV,06:20,06:24
#
# If it doesn't exist, the application still runs.
# ============================================================

TIMETABLE_FILE = BASE_DIR / "timetable.csv"

if TIMETABLE_FILE.exists():
    timetable = pd.read_csv(TIMETABLE_FILE)
else:
    timetable = pd.DataFrame()


# ============================================================
# OPTIONAL STATION COORDINATES
# ============================================================
#
# station_coords.csv:
#
# station,latitude,longitude
# PUNE,18.5284,73.8743
#
# ============================================================

COORD_FILE = BASE_DIR / "station_coords.csv"

if COORD_FILE.exists():
    station_coords = pd.read_csv(COORD_FILE)
else:
    station_coords = pd.DataFrame()


# ============================================================
# FEATURE NAMES FOR EXPLANATION
# ============================================================

readable_names = {
    "rainfall_mm": "rainfall",
    "visibility_m": "low visibility / fog",
    "temperature_c": "temperature",
    "congestion_score": "congestion on this stretch",
    "historical_section_time": "historical section running time",
    "distance_to_next": "distance to next station",
    "station_sequence": "position along the route",
    "hour": "time of day",
    "day_of_week": "day of week",
    "delay_minutes": "current delay",
}


# ============================================================
# CSS
# ============================================================
#
# Important:
# No fixed dark background.
# Colors are mainly accents and the surrounding Streamlit
# theme controls light/dark mode.
# ============================================================

st.markdown(
    """
    <style>

    /* Main width */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.18);
    }

    /* Logo */
    .rail-logo {
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .rail-subtitle {
        font-size: 13px;
        opacity: 0.7;
        margin-top: -5px;
        margin-bottom: 18px;
    }

    /* Header */
    .brand-title {
        font-size: 35px;
        font-weight: 850;
        letter-spacing: -1.5px;
        margin-bottom: -5px;
    }

    .brand-blue {
        color: #1261c9;
    }

    .brand-red {
        color: #d62828;
    }

    .brand-tagline {
        font-size: 14px;
        opacity: 0.72;
        margin-bottom: 8px;
    }

    /* Status badge */
    .status-running {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: #d9f7df;
        color: #16753a;
        font-weight: 700;
        font-size: 13px;
    }

    .status-warning {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: #fff0d1;
        color: #a55a00;
        font-weight: 700;
        font-size: 13px;
    }

    /* Small section labels */
    .section-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.65;
        font-weight: 700;
    }

    /* Big route */
    .route-title {
        font-size: 25px;
        font-weight: 800;
        margin-top: 4px;
    }

    .route-arrow {
        color: #d62828;
        padding: 0 8px;
    }

    /* Metric cards */
    .metric-card {
        border: 1px solid rgba(100,100,100,0.16);
        border-radius: 14px;
        padding: 17px;
        min-height: 130px;
        background: rgba(128,128,128,0.035);
    }

    .metric-label {
        font-size: 13px;
        opacity: 0.7;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-small {
        font-size: 12px;
        opacity: 0.65;
        margin-top: 7px;
    }

    /* Colored metric accents */
    .metric-blue {
        border-top: 4px solid #3182ce;
    }

    .metric-green {
        border-top: 4px solid #25a05a;
    }

    .metric-orange {
        border-top: 4px solid #e67e22;
    }

    .metric-purple {
        border-top: 4px solid #805ad5;
    }

    /* Dashboard panels */
    .panel-title {
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    /* Timeline */
    .timeline-row {
        padding: 8px 4px;
        border-bottom: 1px solid rgba(128,128,128,0.13);
        font-size: 13px;
    }

    .timeline-current {
        font-weight: 800;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        opacity: 0.55;
        font-size: 12px;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """Return first matching column from candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def explain_row(feature_row):
    try:
        available_features = [
            f for f in residual_features
            if f in feature_row.index
        ]

        if not available_features:
            return "operating conditions", "affecting the forecast"

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

        top_idx = abs(contribs).argmax()
        top_feature = available_features[top_idx]

        direction = (
            "increasing the delay"
            if contribs[top_idx] > 0
            else "reducing the delay"
        )

        return (
            readable_names.get(top_feature, top_feature),
            direction
        )

    except Exception:
        return "current operating conditions", "influencing the forecast"


def format_time(value):
    """Convert a time-like value to 12-hour display."""
    if pd.isna(value):
        return "—"

    try:
        parsed = pd.to_datetime(str(value))
        return parsed.strftime("%I:%M %p")
    except Exception:
        return str(value)


def get_timetable_value(train, station, column):
    """Find timetable value for train + station."""
    if timetable.empty:
        return None

    required = {
        "train",
        "station",
        column
    }

    if not required.issubset(timetable.columns):
        return None

    rows = timetable[
        (timetable["train"].astype(str) == str(train)) &
        (timetable["station"].astype(str) == str(station))
    ]

    if rows.empty:
        return None

    return rows.iloc[0][column]


def add_minutes_to_time(time_value, minutes):
    try:
        dt = pd.to_datetime(str(time_value))
        return dt + timedelta(minutes=float(minutes))
    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="rail-logo">🚆 RailCast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="rail-subtitle">'
        'Dynamic ETA & Delay Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    dashboard_view = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Passenger View",
            "Control Room / Officer"
        ],
        index=0
    )

    st.divider()

    st.markdown("### Model Reference")

    st.metric(
        "Validation MAE",
        f"{XGB_MAE:.2f} min"
    )

    st.caption(
        "XGBoost prediction engine"
    )

    st.caption(
        "Historical data + operating-condition features"
    )

    st.divider()

    st.markdown("### About")

    st.caption(
        "Decision-support prototype. "
        "Operational actions remain with authorised railway staff."
    )


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns([4, 1.5])

with header_left:

    st.markdown(
        """
        <div class="brand-title">
            🚆 <span class="brand-blue">Rail</span><span class="brand-red">Cast</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="brand-tagline">
            Dynamic ETA & Delay Intelligence
            &nbsp; • &nbsp;
            Predict the arrival. Understand the delay.
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:

    now = datetime.now()

    st.write(
        f"📅 {now.strftime('%a, %d %b %Y')}"
    )

    st.write(
        f"🕐 **{now.strftime('%I:%M %p')}**"
    )


# ============================================================
# TRAIN SELECTION
# ============================================================

train_options = demo_data["train"].astype(str).unique()

selected_train = st.selectbox(
    "Select Train",
    train_options
)

train_rows = demo_data[
    demo_data["train"].astype(str) == str(selected_train)
].reset_index(drop=True)

if train_rows.empty:
    st.warning("No data available for this train.")
    st.stop()


journey_options = train_rows.index.tolist()

row_index = st.selectbox(
    "Select Journey Point",
    journey_options,
    format_func=lambda i:
        f'{train_rows.loc[i, "station"]} → '
        f'{train_rows.loc[i, "next_station"]} '
        f'({train_rows.loc[i, "date"]})'
)

current_row = train_rows.loc[row_index].copy()

current_station = current_row["station"]
next_station = current_row["next_station"]


# ============================================================
# TRAIN HERO IMAGE
# ============================================================

banner_file = ASSETS_DIR / "train_banner.png"

if banner_file.exists():

    st.image(
        str(banner_file),
        use_container_width=True
    )

else:

    st.info(
        "Add your train banner as "
        "`assets/train_banner.png` to show the railway hero image."
    )


# ============================================================
# TRAIN INFORMATION
# ============================================================

info1, info2 = st.columns([2.2, 1])

with info1:

    st.markdown(
        '<div class="section-label">Selected Train</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="route-title">
            🚆 Train {selected_train}
            <span class="status-running">● Running</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="font-size:16px; opacity:0.75;">
            {current_station}
            <span class="route-arrow">→</span>
            {next_station}
        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        '<div class="section-label">Journey Date</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"### 📅 {current_row['date']}"
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

try:

    # Normal
    X_original = (
        pd.DataFrame(
            [current_row[feature_columns]]
        )
        .apply(pd.to_numeric)
    )

    original_predicted_delay = float(
        xgb_model.predict(X_original)[0]
    )

except Exception as e:

    st.error(
        f"Model prediction failed: {e}"
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

sim_weather_label = current_row.get(
    "weather_condition_passenger",
    "Current conditions"
)


if disruption == "Fog":

    sim_row["visibility_m"] = 150

    if "temperature_c" in sim_row:
        sim_row["temperature_c"] = (
            sim_row["temperature_c"] - 3
        )

    disruption_note = (
        "Visibility dropped to 150m, "
        "representing dense fog conditions."
    )

    sim_weather_label = "Foggy"


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80mm, "
        "representing storm-level rainfall."
    )

    sim_weather_label = "Heavy Rain / Storm"


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"] * 1.5
    )

    disruption_note = (
        "Expected section running time "
        "increased by 50%."
    )

    sim_weather_label = "Speed Restriction in effect"


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 25
    )

    disruption_note = (
        "Train held for an additional 25 minutes."
    )

    sim_weather_label = "Signal Halt in effect"


elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    disruption_note = (
        "Downstream congestion increased "
        "to near-maximum."
    )

    sim_weather_label = "Track Congestion Spike"


elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 45
    )

    disruption_note = (
        "Maintenance block adding 45 minutes "
        "to the current disruption scenario."
    )

    sim_weather_label = "Maintenance Block in effect"


# ============================================================
# SIMULATED PREDICTION
# ============================================================

try:

    X_input = (
        pd.DataFrame(
            [sim_row[feature_columns]]
        )
        .apply(pd.to_numeric)
    )

    predicted_delay = float(
        xgb_model.predict(X_input)[0]
    )

except Exception as e:

    st.error(
        f"Disruption prediction failed: {e}"
    )
    st.stop()


# ============================================================
# TIME CALCULATION
# ============================================================

scheduled_time = get_timetable_value(
    selected_train,
    current_station,
    "scheduled_arrival_time"
)

actual_time = get_timetable_value(
    selected_train,
    current_station,
    "actual_arrival_time"
)


# ============================================================
# ARRIVAL METRIC CARDS
# ============================================================

st.markdown("### ⏱️ Arrival Forecast")

m1, m2, m3, m4, m5 = st.columns(5)


with m1:

    scheduled_display = (
        format_time(scheduled_time)
        if scheduled_time is not None
        else "Not available"
    )

    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <div class="metric-label">🕐 Scheduled Arrival</div>
            <div class="metric-value">{scheduled_display}</div>
            <div class="metric-small">
                From timetable
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    predicted_display = "Not available"

    if scheduled_time is not None:

        predicted_time = add_minutes_to_time(
            scheduled_time,
            predicted_delay
        )

        if predicted_time is not None:
            predicted_display = predicted_time.strftime(
                "%I:%M %p"
            )

    st.markdown(
        f"""
        <div class="metric-card metric-green">
            <div class="metric-label">🎯 RailCast Predicted</div>
            <div class="metric-value">{predicted_display}</div>
            <div class="metric-small">
                +{predicted_delay:.1f} min delay
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    actual_display = (
        format_time(actual_time)
        if actual_time is not None
        else "Not available"
    )

    st.markdown(
        f"""
        <div class="metric-card metric-orange">
            <div class="metric-label">✓ Actual Arrival</div>
            <div class="metric-value">{actual_display}</div>
            <div class="metric-small">
                Recorded outcome
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    prediction_error = None

    if (
        scheduled_time is not None
        and actual_time is not None
    ):

        try:

            predicted_dt = add_minutes_to_time(
                scheduled_time,
                predicted_delay
            )

            actual_dt = pd.to_datetime(
                str(actual_time)
            )

            prediction_error = abs(
                (predicted_dt - actual_dt)
                .total_seconds()
            ) / 60

        except Exception:
            prediction_error = None


    error_display = (
        f"{prediction_error:.1f} min"
        if prediction_error is not None
        else "—"
    )

    st.markdown(
        f"""
        <div class="metric-card metric-purple">
            <div class="metric-label">📊 Prediction Error</div>
            <div class="metric-value">{error_display}</div>
            <div class="metric-small">
                Prediction vs actual
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m5:

    status_text = "ON TIME"

    if predicted_delay >= 15:
        status_text = "HIGH DELAY RISK"
    elif predicted_delay >= 5:
        status_text = "MINOR DELAY"

    st.markdown(
        f"""
        <div class="metric-card metric-green">
            <div class="metric-label">🚉 Current Status</div>
            <div class="metric-value"
                 style="font-size:18px;">
                 {status_text}
            </div>
            <div class="metric-small">
                Current: {current_station}
                → Next: {next_station}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ROUTE + TIMELINE + DISRUPTION
# ============================================================

map_col, timeline_col, disruption_col = st.columns(
    [1.55, 1.25, 1.1]
)


# ============================================================
# MAP
# ============================================================

with map_col:

    st.markdown(
        "### 📍 Route Map"
    )

    if (
        not station_coords.empty
        and
        {"station", "latitude", "longitude"}
        .issubset(station_coords.columns)
    ):

        route_stations = train_rows[
            [
                "station",
                "next_station",
                "station_sequence"
            ]
        ].copy()

        station_list = list(
            route_stations["station"]
        )

        station_list += list(
            route_stations["next_station"]
        )

        station_list = list(
            dict.fromkeys(station_list)
        )

        map_data = station_coords[
            station_coords["station"].astype(str).isin(
                [str(x) for x in station_list]
            )
        ].copy()

        map_data = map_data.sort_values(
            "station"
        )

        if len(map_data) >= 2:

            fig = go.Figure()

            fig.add_trace(
                go.Scattermap(
                    lat=map_data["latitude"],
                    lon=map_data["longitude"],
                    mode="lines+markers",
                    text=map_data["station"],
                    hovertemplate=(
                        "<b>%{text}</b>"
                        "<extra></extra>"
                    ),
                    line=dict(
                        width=4
                    ),
                    marker=dict(
                        size=9
                    ),
                    name="Route"
                )
            )

            current_map = map_data[
                map_data["station"].astype(str)
                == str(current_station)
            ]

            if not current_map.empty:

                fig.add_trace(
                    go.Scattermap(
                        lat=current_map["latitude"],
                        lon=current_map["longitude"],
                        mode="markers+text",
                        text=["🚆 " + str(current_station)],
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
                        lat=map_data["latitude"].mean(),
                        lon=map_data["longitude"].mean()
                    ),
                    zoom=5
                ),
                height=460,
                margin=dict(
                    l=0,
                    r=0,
                    t=0,
                    b=0
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme="streamlit"
            )

        else:

            st.info(
                "Add station coordinates to display the route map."
            )

    else:

        st.info(
            "Create `station_coords.csv` with "
            "`station,latitude,longitude` to enable the map."
        )


# ============================================================
# STATION TIMELINE
# ============================================================

with timeline_col:

    st.markdown(
        "### 🛤️ Station Timeline"
    )

    timeline_rows = train_rows.copy()

    timeline_rows = timeline_rows.sort_values(
        "station_sequence"
    )

    for _, row in timeline_rows.iterrows():

        station = row["station"]
        delay = float(row["delay_minutes"])

        scheduled = get_timetable_value(
            selected_train,
            station,
            "scheduled_arrival_time"
        )

        actual = get_timetable_value(
            selected_train,
            station,
            "actual_arrival_time"
        )

        scheduled_text = (
            format_time(scheduled)
            if scheduled is not None
            else "—"
        )

        actual_text = (
            format_time(actual)
            if actual is not None
            else "—"
        )

        is_current = (
            str(station)
            == str(current_station)
        )

        icon = "🚆" if is_current else "○"

        st.markdown(
            f"""
            <div class="timeline-row">
                <b>{icon} {station}</b>
                <br>
                <span style="opacity:.7;">
                    Scheduled: {scheduled_text}
                    &nbsp; | &nbsp;
                    Actual: {actual_text}
                    &nbsp; | &nbsp;
                    Delay: {delay:.0f}m
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DISRUPTION PANEL
# ============================================================

with disruption_col:

    st.markdown(
        "### ⚡ Disruption Impact"
    )

    if disruption == "None":

        st.info(
            "Select a disruption above to simulate "
            "its effect on the ETA."
        )

    else:

        st.warning(
            disruption_note
        )

        d1, d2 = st.columns(2)

        with d1:

            st.metric(
                "Current",
                f"+{original_predicted_delay:.1f} min"
            )

        with d2:

            st.metric(
                "With disruption",
                f"+{predicted_delay:.1f} min",
                delta=(
                    f"{predicted_delay - original_predicted_delay:+.1f}"
                )
            )

        impact = (
            predicted_delay
            - original_predicted_delay
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[
                    original_predicted_delay,
                    predicted_delay
                ],
                y=[
                    "Normal",
                    disruption
                ],
                orientation="h",
                text=[
                    f"{original_predicted_delay:.1f} min",
                    f"{predicted_delay:.1f} min"
                ],
                textposition="outside"
            )
        )

        fig.update_layout(
            title="Delay Impact",
            xaxis_title="Predicted delay (minutes)",
            height=230,
            margin=dict(
                l=10,
                r=30,
                t=45,
                b=30
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme="streamlit"
        )

        if impact > 0:

            st.error(
                f"The selected disruption increases "
                f"the forecast by {impact:.1f} minutes."
            )


# ============================================================
# LOWER SECTION
# ============================================================

st.divider()

lower1, lower2 = st.columns([1.1, 1.4])


# ============================================================
# OPERATING CONDITIONS
# ============================================================

with lower1:

    st.markdown(
        "### 🌦️ Operating Conditions"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Weather",
            sim_weather_label
        )

    with c2:

        st.metric(
            "Visibility",
            f"{sim_row['visibility_m']:.0f} m"
        )

    with c3:

        st.metric(
            "Rainfall",
            f"{sim_row['rainfall_mm']:.1f} mm"
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        st.metric(
            "Temperature",
            f"{sim_row['temperature_c']:.1f} °C"
        )

    with c5:

        st.metric(
            "Congestion",
            f"{sim_row['congestion_score']:.2f}"
        )

    with c6:

        st.metric(
            "Section Time",
            f"{sim_row['historical_section_time']:.1f} min"
        )


# ============================================================
# WHY THIS PREDICTION
# ============================================================

with lower2:

    st.markdown(
        "### 💡 Why this prediction?"
    )

    top_feature, direction = explain_row(
        sim_row
    )

    st.write(
        f"The forecast is mainly influenced by "
        f"**{top_feature}**, which is currently "
        f"**{direction}**."
    )

    st.caption(
        f"Estimated uncertainty: "
        f"{predicted_delay - XGB_MAE:.1f} to "
        f"{predicted_delay + XGB_MAE:.1f} minutes"
    )


# ============================================================
# PASSENGER FEEDBACK
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
        "Actual delay experienced at this station",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=1.0
    )

with feedback_col2:

    st.write("")
    st.write("")

    if st.button(
        "Submit actual delay",
        type="primary",
        use_container_width=True
    ):

        feedback_row = pd.DataFrame(
            [
                {
                    "train": selected_train,
                    "station": current_station,
                    "next_station": next_station,
                    "predicted_delay": predicted_delay,
                    "actual_delay": actual_delay_input,
                    "submitted_at": pd.Timestamp.now()
                }
            ]
        )

        feedback_file = (
            BASE_DIR / "passenger_feedback.csv"
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
# CONTROL ROOM ADDITIONAL VIEW
# ============================================================

if dashboard_view == "Control Room / Officer":

    st.divider()

    st.markdown(
        "## 🛠️ Control Room Intelligence"
    )

    risk = "LOW"

    if predicted_delay >= 15:
        risk = "HIGH"
    elif predicted_delay >= 5:
        risk = "MEDIUM"

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Delay Risk",
            risk
        )

    with r2:
        st.metric(
            "Current Delay",
            f"{current_row['delay_minutes']:.0f} min"
        )

    with r3:
        st.metric(
            "Route Position",
            f"{current_row['station_sequence']:.0f}"
        )

    if risk == "HIGH":

        st.error(
            "High-delay risk detected. "
            "Consider operational review by authorised staff."
        )

    elif risk == "MEDIUM":

        st.warning(
            "Moderate delay risk. "
            "Monitor downstream sections."
        )

    else:

        st.success(
            "No major delay risk detected."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        RailCast • Dynamic ETA & Delay Intelligence •
        Decision-support prototype
    </div>
    """,
    unsafe_allow_html=True
)
