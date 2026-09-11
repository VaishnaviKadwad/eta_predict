import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RailCast",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SECTION COLOURS
# ============================================================

st.markdown("""
<style>

/* Main coloured section headers */

.section-blue {
    background: linear-gradient(90deg, #1565C0, #42A5F5);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-purple {
    background: linear-gradient(90deg, #6A1B9A, #AB47BC);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-sky {
    background: linear-gradient(90deg, #0277BD, #29B6F6);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-green {
    background: linear-gradient(90deg, #2E7D32, #66BB6A);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-yellow {
    background: linear-gradient(90deg, #F9A825, #FDD835);
    color: #3E2723;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-orange {
    background: linear-gradient(90deg, #EF6C00, #FFA726);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-red {
    background: linear-gradient(90deg, #C62828, #EF5350);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-pink {
    background: linear-gradient(90deg, #AD1457, #EC407A);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-teal {
    background: linear-gradient(90deg, #00695C, #26A69A);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

.section-maintenance {
    background: linear-gradient(90deg, #5D4037, #8D6E63);
    color: white;
    padding: 13px 20px;
    border-radius: 12px;
    margin: 18px 0 10px 0;
    font-size: 22px;
    font-weight: 700;
}

/* Small subtitle styling */

.section-caption {
    color: #546E7A;
    font-size: 14px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "demo_data.csv"
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
RESIDUAL_MODEL_PATH = BASE_DIR / "residual_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
RESIDUAL_FEATURES_PATH = BASE_DIR / "residual_features.pkl"


# ============================================================
# FIND BANNER
# ============================================================

def find_banner():

    paths = [
        BASE_DIR / "assets" / "train_banner.png",
        BASE_DIR / "train_banner.png",
        Path.cwd() / "assets" / "train_banner.png",
        Path.cwd() / "train_banner.png",
    ]

    for path in paths:

        if path.exists() and path.is_file():
            return path

    try:

        for path in BASE_DIR.rglob("train_banner.png"):

            if path.is_file():
                return path

    except Exception:
        pass

    return None


BANNER_FILE = find_banner()


# ============================================================
# BANNER
# ============================================================

if BANNER_FILE is not None:

    try:

        banner = Image.open(BANNER_FILE)
        banner.load()

        if banner.mode not in ("RGB", "RGBA"):
            banner = banner.convert("RGB")

        st.image(
            banner,
            width="stretch"
        )

    except Exception as e:

        st.warning(
            f"Banner could not be displayed: {e}"
        )

        st.title("🚆 RailCast")

        st.caption(
            "Dynamic ETA Intelligence for Indian Railways"
        )

else:

    st.title("🚆 RailCast")

    st.info(
        "Dynamic ETA Intelligence for Indian Railways"
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        st.error(
            f"demo_data.csv not found at: {DATA_PATH}"
        )

        st.stop()

    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    residual_model = None
    feature_columns = None
    residual_features = None

    if RESIDUAL_MODEL_PATH.exists():

        residual_model = joblib.load(
            RESIDUAL_MODEL_PATH
        )

    if FEATURE_COLUMNS_PATH.exists():

        feature_columns = joblib.load(
            FEATURE_COLUMNS_PATH
        )

    if RESIDUAL_FEATURES_PATH.exists():

        residual_features = joblib.load(
            RESIDUAL_FEATURES_PATH
        )

    return (
        model,
        residual_model,
        feature_columns,
        residual_features
    )


try:

    (
        model,
        residual_model,
        feature_columns,
        residual_features
    ) = load_models()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_MAE = 7.88


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚆 RailCast")

    st.caption(
        "Dynamic ETA & Delay Intelligence"
    )

    st.divider()

    st.subheader("DASHBOARDS")

    role = st.radio(
        "Choose dashboard",
        [
            "👤 Passenger",
            "🏢 Station Operator",
            "🚦 Traffic Controller",
            "🛠️ Maintenance Operator"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("MODEL")

    st.metric(
        "Validation MAE",
        f"{MODEL_MAE:.2f} min"
    )

    st.caption(
        "XGBoost prediction engine"
    )

    st.caption(
        "Historical + operating-condition features"
    )


# ============================================================
# DATE / TIME — INDIA
# ============================================================

current_time = datetime.now(
    ZoneInfo("Asia/Kolkata")
)

date_col, time_col = st.columns([4, 1])

with time_col:

    st.write(
        f"📅 **{current_time.strftime('%d %b %Y')}**"
    )

    st.write(
        f"🕐 **{current_time.strftime('%I:%M %p')}**"
    )


# ============================================================
# SELECT JOURNEY
# ============================================================

st.markdown(
    '<div class="section-blue">🚆 Select Journey</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'Choose a train and journey point to generate a dynamic ETA forecast.'
    '</div>',
    unsafe_allow_html=True
)


trains = sorted(
    df["train"]
    .astype(str)
    .dropna()
    .unique()
    .tolist()
)


if not trains:

    st.error(
        "No train records found."
    )

    st.stop()


selected_train = st.selectbox(
    "Train",
    trains
)


# ============================================================
# TRAIN DATA
# ============================================================

train_df = df[
    df["train"].astype(str)
    == str(selected_train)
].copy()


if train_df.empty:

    st.warning(
        "No records found for this train."
    )

    st.stop()


# ============================================================
# JOURNEY OPTIONS
# ============================================================

train_df["journey_label"] = (
    train_df["station"].astype(str)
    + " → "
    + train_df["next_station"].astype(str)
    + " ("
    + train_df["date"].astype(str)
    + ")"
)


journey_options = (
    train_df["journey_label"]
    .drop_duplicates()
    .tolist()
)


selected_journey = st.selectbox(
    "Journey point",
    journey_options
)


selected_row = train_df[
    train_df["journey_label"]
    == selected_journey
].iloc[0].copy()


station = str(
    selected_row["station"]
)

next_station = str(
    selected_row["next_station"]
)


# ============================================================
# DISRUPTION SIMULATOR
# ============================================================

st.markdown(
    '<div class="section-orange">⚡ What-if Disruption Simulator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'Test how an unexpected operating event could change the forecast.'
    '</div>',
    unsafe_allow_html=True
)


disruption_options = [
    "None",
    "🌫️ Dense Fog",
    "🌧️ Heavy Rain",
    "🚧 Speed Restriction",
    "🚦 Signal Halt",
    "🚥 Track Congestion Spike",
    "🛤️ Maintenance Block"
]


disruption = st.selectbox(
    "Operating event",
    disruption_options
)


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(row, disruption=None):

    modified = row.copy()

    # --------------------------------------------------------
    # Disruption modifications
    # --------------------------------------------------------

    if disruption == "🌫️ Dense Fog":

        if "visibility_m" in modified:

            modified["visibility_m"] = 150

        if "temperature_c" in modified:

            modified["temperature_c"] = -3


    elif disruption == "🌧️ Heavy Rain":

        if "rainfall_mm" in modified:

            modified["rainfall_mm"] = 80


    elif disruption == "🚧 Speed Restriction":

        if "historical_section_time" in modified:

            modified["historical_section_time"] = (
                float(
                    modified[
                        "historical_section_time"
                    ]
                )
                * 1.5
            )


    elif disruption == "🚦 Signal Halt":

        if "delay_minutes" in modified:

            modified["delay_minutes"] = (
                float(
                    modified[
                        "delay_minutes"
                    ]
                )
                + 25
            )


    elif disruption == "🚥 Track Congestion Spike":

        if "congestion_score" in modified:

            modified["congestion_score"] = 0.95


    elif disruption == "🛤️ Maintenance Block":

        if "delay_minutes" in modified:

            modified["delay_minutes"] = (
                float(
                    modified[
                        "delay_minutes"
                    ]
                )
                + 45
            )


    # --------------------------------------------------------
    # Model columns
    # --------------------------------------------------------

    if feature_columns is not None:

        columns = list(feature_columns)

    else:

        columns = [
            "delay_minutes",
            "distance_to_next",
            "historical_section_time",
            "rainfall_mm",
            "temperature_c",
            "visibility_m",
            "congestion_score",
            "hour",
            "day_of_week",
            "station_sequence"
        ]


    values = {}


    for column in columns:

        if column in modified.index:

            value = modified[column]

            try:

                values[column] = float(value)

            except Exception:

                values[column] = 0.0

        else:

            values[column] = 0.0


    X = pd.DataFrame(
        [values],
        columns=columns
    )


    return X, modified


# ============================================================
# PREDICTION
# ============================================================

def predict_delay(row, disruption=None):

    X, modified = prepare_features(
        row,
        disruption
    )

    prediction = model.predict(X)

    value = float(
        np.asarray(prediction)
        .reshape(-1)[0]
    )

    return max(0.0, value), modified


# ============================================================
# NORMAL PREDICTION
# ============================================================

try:

    normal_prediction, normal_row = predict_delay(
        selected_row
    )

except Exception as e:

    st.error(
        f"Prediction error: {e}"
    )

    st.stop()


# ============================================================
# DISRUPTION PREDICTION
# ============================================================

disruption_prediction = None


if disruption != "None":

    try:

        disruption_prediction, disrupted_row = (
            predict_delay(
                selected_row,
                disruption
            )
        )

    except Exception as e:

        st.error(
            f"Disruption prediction error: {e}"
        )


# ============================================================
# BASIC VALUES
# ============================================================

current_delay = float(
    selected_row.get(
        "delay_minutes",
        0
    )
)

current_delay = max(
    0.0,
    current_delay
)


rainfall = float(
    selected_row.get(
        "rainfall_mm",
        0
    )
)


temperature = float(
    selected_row.get(
        "temperature_c",
        0
    )
)


visibility = float(
    selected_row.get(
        "visibility_m",
        0
    )
)


congestion = float(
    selected_row.get(
        "congestion_score",
        0
    )
)


section_time = float(
    selected_row.get(
        "historical_section_time",
        0
    )
)

# ============================================================
# ETA CALCULATION
# ============================================================

# Expected section time is the historical/base running time
# for the selected station-to-next-station section.
expected_section_time = max(
    0.0,
    section_time
)

# RailCast predicts the additional delay in minutes.
# Predicted section time = normal section time + predicted delay.
predicted_section_time = max(
    expected_section_time,
    expected_section_time + normal_prediction
)

# When a disruption is selected, calculate the corresponding
# predicted section time for the scenario as well.
disruption_predicted_section_time = None

if disruption_prediction is not None:
    disruption_predicted_section_time = max(
        expected_section_time,
        expected_section_time + disruption_prediction
    )


# ============================================================
# RISK
# ============================================================

def delay_risk(delay):

    if delay < 5:
        return "LOW"

    if delay < 15:
        return "MEDIUM"

    return "HIGH"


risk = delay_risk(
    normal_prediction
)


# ============================================================
# PASSENGER DASHBOARD
# ============================================================

if role == "👤 Passenger":

    st.markdown(
        '<div class="section-teal">🎫 Passenger View</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'A simple view of where your train is and what to expect next.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # CURRENT JOURNEY
    # ========================================================

    st.markdown(
        '<div class="section-blue">🚆 Current Journey Point</div>',
        unsafe_allow_html=True
    )

    journey_a, journey_b, journey_c = st.columns(
        [2, 2, 1]
    )


    with journey_a:

        st.metric(
            "From",
            station
        )


    with journey_b:

        st.metric(
            "Next Station",
            next_station
        )


    with journey_c:

        st.metric(
            "Train",
            selected_train
        )


    st.caption(
        f"Journey date: {selected_row['date']}"
    )


    # ========================================================
    # FORECAST
    # ========================================================

    st.markdown(
        '<div class="section-purple">🎯 Dynamic ETA Forecast</div>',
        unsafe_allow_html=True
    )


    if disruption == "None":

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Current Delay",
                f"{current_delay:.1f} min"
            )


        with col2:

            st.metric(
                "Expected Section Time",
                f"{expected_section_time:.1f} min"
            )


        with col3:

            st.metric(
                "Predicted Section Time",
                f"{predicted_section_time:.1f} min",
                delta=f"+{normal_prediction:.1f} min delay"
            )


        with col4:

            st.metric(
                "Delay Risk",
                risk
            )


        st.info(
            f"RailCast expects this section to take approximately "
            f"**{expected_section_time:.1f} minutes** under normal "
            f"historical running conditions. Based on the ML forecast, "
            f"the predicted section time is approximately "
            f"**{predicted_section_time:.1f} minutes**, including "
            f"**{normal_prediction:.1f} minutes** of predicted delay."
        )


    else:

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Current Delay",
                f"{current_delay:.1f} min"
            )


       with col2:

    st.metric(
        "Expected Section Time",
        f"{expected_section_time:.1f} min"
    )

with col3:

    st.metric(
        "Predicted Section Time",
        f"{disruption_predicted_section_time:.1f} min",
        delta=f"+{disruption_prediction:.1f} min delay"
    )


        with col4:

            st.metric(
                "Risk After Event",
                delay_risk(
                    disruption_prediction
                )
            )


        if disruption_predicted_section_time is not None:

            st.warning(
                f"Under **{disruption}**, the predicted section time "
                f"would increase to approximately "
                f"**{disruption_predicted_section_time:.1f} minutes**."
            )


    # ========================================================
    # PASSENGER MESSAGE
    # ========================================================

    st.markdown(
        '<div class="section-yellow">💡 What does this mean?</div>',
        unsafe_allow_html=True
    )


    if disruption == "None":

        if normal_prediction < 5:

            message = (
                "The train is currently expected to maintain "
                "a relatively low delay at this journey point."
            )

        elif normal_prediction < 15:

            message = (
                "A moderate delay is expected. Allow some "
                "additional time for onward travel."
            )

        else:

            message = (
                "A significant delay is forecast. Passengers "
                "with connections should allow extra time."
            )


        st.success(
            message
        )


    else:

        impact = (
            disruption_prediction
            - normal_prediction
        )


        if impact > 0:

            st.warning(
                f"⚠️ Under **{disruption}**, RailCast estimates "
                f"approximately **{impact:.1f} additional minutes** "
                f"of delay."
            )

        else:

            st.info(
                f"The selected scenario changes the forecast "
                f"by **{impact:+.1f} minutes**."
            )


    # ========================================================
    # JOURNEY MAP
    # ========================================================

    st.markdown(
        '<div class="section-sky">🗺️ Journey Map</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Current journey point and route progression.'
        '</div>',
        unsafe_allow_html=True
    )


    route_df = train_df[
        train_df["date"].astype(str)
        == str(selected_row["date"])
    ].copy()


    route_df = route_df.drop_duplicates(
        subset=[
            "station",
            "next_station"
        ]
    )


    route_stations = []


    for _, route_row in route_df.iterrows():

        first_station = str(
            route_row["station"]
        )

        second_station = str(
            route_row["next_station"]
        )


        if first_station not in route_stations:

            route_stations.append(
                first_station
            )


        if second_station not in route_stations:

            route_stations.append(
                second_station
            )


    if len(route_stations) < 2:

        route_stations = [
            station,
            next_station
        ]


    current_index = 0


    if station in route_stations:

        current_index = (
            route_stations.index(station)
        )


    x_values = list(
        range(len(route_stations))
    )


    y_values = [
        np.sin(i / 1.8) * 0.35
        for i in x_values
    ]


    fig = go.Figure()


    # Route

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line=dict(
                width=5
            ),
            marker=dict(
                size=11
            ),
            text=route_stations,
            hovertemplate=(
                "<b>%{text}</b>"
                "<extra></extra>"
            )
        )
    )


    # Current station

    fig.add_trace(
        go.Scatter(
            x=[
                x_values[
                    current_index
                ]
            ],
            y=[
                y_values[
                    current_index
                ]
            ],
            mode="markers",
            marker=dict(
                size=24
            ),
            name="Current"
        )
    )


    fig.update_layout(
        height=330,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        )
    )


    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )


    st.caption(
        "Route: "
        + " → ".join(route_stations)
    )


    # ========================================================
    # CONDITIONS
    # ========================================================

    st.markdown(
        '<div class="section-green">🌦️ Current Operating Conditions</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.metric(
            "🌧️ Rainfall",
            f"{rainfall:.0f} mm"
        )


    with c2:

        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C"
        )


    with c3:

        st.metric(
            "👁️ Visibility",
            f"{visibility:.0f} m"
        )


    with c4:

        st.metric(
            "🚦 Congestion",
            f"{congestion:.2f}"
        )


    with c5:

        st.metric(
            "🛤️ Section Time",
            f"{section_time:.1f} min"
        )


    # ========================================================
    # WHY PREDICTION
    # ========================================================

    st.markdown(
        '<div class="section-yellow">'
        '🔎 Why is RailCast predicting this?'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "RailCast combines current delay, historical running "
        "behaviour and operating conditions."
    )


    # Determine strongest visible signal

    factor_values = {

        "Current Delay":
            current_delay,

        "Rainfall":
            rainfall,

        "Low Visibility":
            max(
                0,
                1000 - visibility
            ),

        "Congestion":
            congestion * 100,

        "Historical Section Time":
            section_time
    }


    strongest_factor = max(
        factor_values,
        key=factor_values.get
    )


    if strongest_factor == "Current Delay":

        factor_detail = (
            f"The train currently has a delay of "
            f"{current_delay:.1f} minutes."
        )

        reason = (
            "Existing delay is an important signal for "
            "estimating downstream delay propagation."
        )


    elif strongest_factor == "Rainfall":

        factor_detail = (
            f"Rainfall is currently {rainfall:.0f} mm."
        )

        reason = (
            "Rainfall can contribute to slower running "
            "conditions and increased operational caution."
        )


    elif strongest_factor == "Low Visibility":

        factor_detail = (
            f"Visibility is currently {visibility:.0f} m."
        )

        reason = (
            "Reduced visibility can increase operational "
            "caution and affect running conditions."
        )


    elif strongest_factor == "Congestion":

        factor_detail = (
            f"Congestion score is {congestion:.2f}."
        )

        reason = (
            "Higher congestion can increase downstream "
            "running time and delay propagation."
        )


    else:

        factor_detail = (
            f"Historical section time is "
            f"{section_time:.1f} minutes."
        )

        reason = (
            "Historical running time helps estimate how "
            "quickly the train can cover the next section."
        )


    st.success(
        f"🎯 **RailCast Forecast: "
        f"{normal_prediction:.1f} minutes delay**"
    )


    st.write(
        f"**Main operating signal:** {strongest_factor}"
    )


    st.write(
        factor_detail
    )


    st.write(
        reason
    )


    st.markdown(
        "#### Supporting signals"
    )


    e1, e2, e3, e4 = st.columns(4)


    with e1:

        st.metric(
            "Current Delay",
            f"{current_delay:.1f} min"
        )


    with e2:

        st.metric(
            "Rainfall",
            f"{rainfall:.0f} mm"
        )


    with e3:

        st.metric(
            "Visibility",
            f"{visibility:.0f} m"
        )


    with e4:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    st.caption(
        f"Model validation reference: "
        f"MAE = {MODEL_MAE:.2f} minutes."
    )


    # ========================================================
    # DISRUPTION IMPACT
    # ========================================================

    if disruption != "None":

        st.markdown(
            '<div class="section-red">⚡ Disruption Impact</div>',
            unsafe_allow_html=True
        )


        impact = (
            disruption_prediction
            - normal_prediction
        )


        d1, d2, d3 = st.columns(3)


        with d1:

            st.metric(
                "Normal Forecast",
                f"{normal_prediction:.1f} min"
            )


        with d2:

            st.metric(
                "After Disruption",
                f"{disruption_prediction:.1f} min"
            )


        with d3:

            st.metric(
                "Change",
                f"{impact:+.1f} min"
            )


        if impact > 0:

            st.warning(
                f"**{disruption}** could increase the "
                f"predicted delay by approximately "
                f"**{impact:.1f} minutes**."
            )


    # ========================================================
    # FEEDBACK
    # ========================================================

    st.markdown(
        '<div class="section-pink">💬 Passenger Feedback</div>',
        unsafe_allow_html=True
    )


    feedback = st.radio(
        "How useful is this forecast?",
        [
            "👍 Useful",
            "😐 Not sure",
            "👎 Not useful"
        ],
        horizontal=True
    )


    if st.button(
        "Submit Feedback",
        type="primary"
    ):

        feedback_file = (
            BASE_DIR / "passenger_feedback.csv"
        )


        feedback_data = pd.DataFrame(
            [{
                "timestamp":
                    current_time.isoformat(),

                "train":
                    selected_train,

                "station":
                    station,

                "next_station":
                    next_station,

                "prediction":
                    normal_prediction,

                "expected_section_time":
                    expected_section_time,

                "predicted_section_time":
                    predicted_section_time,

                "feedback":
                    feedback
            }]
        )


        try:

            if feedback_file.exists():

                feedback_data.to_csv(
                    feedback_file,
                    mode="a",
                    header=False,
                    index=False
                )

            else:

                feedback_data.to_csv(
                    feedback_file,
                    index=False
                )


            st.success(
                "Thank you — feedback recorded."
            )


        except Exception:

            st.info(
                "Feedback captured for this session."
            )


# ============================================================
# STATION OPERATOR
# ============================================================

elif role == "🏢 Station Operator":

    st.markdown(
        '<div class="section-teal">'
        '🏢 Station Operator Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Passenger communication and station readiness.'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Train",
            selected_train
        )


    with c2:

        st.metric(
            "Current Delay",
            f"{current_delay:.1f} min"
        )


    with c3:

        st.metric(
            "Forecast",
            f"{normal_prediction:.1f} min"
        )


    with c4:

        st.metric(
            "Risk",
            risk
        )


    if normal_prediction >= 15:

        st.error(
            "🚨 High passenger-impact delay expected. "
            "Prepare proactive announcements and crowd management."
        )

    elif normal_prediction >= 5:

        st.warning(
            "⚠️ Moderate delay expected. "
            "Passenger information should be updated."
        )

    else:

        st.success(
            "✅ Low delay risk at the current journey point."
        )


    # --------------------------------------------------------
    # ANNOUNCEMENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-blue">'
        '📢 Suggested Passenger Announcement'
        '</div>',
        unsafe_allow_html=True
    )


    announcement = (
        f"Train {selected_train} is currently at "
        f"{station}. RailCast forecasts approximately "
        f"{normal_prediction:.1f} minutes of delay toward "
        f"{next_station}."
    )


    st.text_area(
        "Announcement",
        announcement,
        height=100
    )


    # --------------------------------------------------------
    # STATION CONDITIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-green">'
        '🌦️ Station Conditions'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Rainfall",
            f"{rainfall:.0f} mm"
        )


    with c2:

        st.metric(
            "Visibility",
            f"{visibility:.0f} m"
        )


    with c3:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    with c4:

        st.metric(
            "Section Time",
            f"{section_time:.1f} min"
        )


# ============================================================
# TRAFFIC CONTROLLER
# ============================================================

elif role == "🚦 Traffic Controller":

    st.markdown(
        '<div class="section-blue">'
        '🚦 Traffic Controller Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Predictive operational intelligence for delay propagation.'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Current Delay",
            f"{current_delay:.1f} min"
        )


    with c2:

        st.metric(
            "Forecast",
            f"{normal_prediction:.1f} min"
        )


    with c3:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    with c4:

        st.metric(
            "Risk",
            risk
        )


    # --------------------------------------------------------
    # DELAY EVOLUTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-purple">'
        '📈 Delay Evolution'
        '</div>',
        unsafe_allow_html=True
    )


    chart_data = pd.DataFrame(
        {
            "Stage": [
                "Current",
                "RailCast Forecast"
            ],

            "Delay": [
                current_delay,
                normal_prediction
            ]
        }
    )


    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            x=chart_data["Stage"],
            y=chart_data["Delay"],
            text=[
                f"{x:.1f} min"
                for x in chart_data["Delay"]
            ],
            textposition="auto"
        )
    )


    fig.update_layout(
        height=330,
        showlegend=False,
        yaxis_title="Delay (minutes)"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    # --------------------------------------------------------
    # OPERATIONAL SCENARIO
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-orange">'
        '⚡ Operational Scenario'
        '</div>',
        unsafe_allow_html=True
    )


    controller_event = st.selectbox(
        "Scenario",
        disruption_options,
        key="controller_event"
    )


    if controller_event != "None":

        controller_prediction, _ = (
            predict_delay(
                selected_row,
                controller_event
            )
        )


        controller_impact = (
            controller_prediction
            - normal_prediction
        )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Scenario Forecast",
                f"{controller_prediction:.1f} min"
            )


        with c2:

            st.metric(
                "Forecast Change",
                f"{controller_impact:+.1f} min"
            )


        if controller_impact > 0:

            st.warning(
                f"{controller_event} may add approximately "
                f"{controller_impact:.1f} minutes."
            )


    # --------------------------------------------------------
    # OPERATIONAL SIGNALS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-sky">'
        '🧠 Operational Signals'
        '</div>',
        unsafe_allow_html=True
    )


    o1, o2, o3, o4, o5 = st.columns(5)


    with o1:

        st.metric(
            "Route Position",
            str(
                selected_row.get(
                    "station_sequence",
                    "-"
                )
            )
        )


    with o2:

        st.metric(
            "Distance",
            f"{float(selected_row.get('distance_to_next', 0)):.1f}"
        )


    with o3:

        st.metric(
            "Section Time",
            f"{section_time:.1f} min"
        )


    with o4:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    with o5:

        st.metric(
            "Forecast",
            f"{normal_prediction:.1f} min"
        )


# ============================================================
# MAINTENANCE OPERATOR
# ============================================================

else:

    st.markdown(
        '<div class="section-maintenance">'
        '🛠️ Maintenance Operator Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Infrastructure and environmental conditions affecting train movement.'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Visibility",
            f"{visibility:.0f} m"
        )


    with c2:

        st.metric(
            "Rainfall",
            f"{rainfall:.0f} mm"
        )


    with c3:

        st.metric(
            "Section Time",
            f"{section_time:.1f} min"
        )


    with c4:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    # --------------------------------------------------------
    # INFRASTRUCTURE SCENARIO
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-orange">'
        '🛤️ Infrastructure Scenario'
        '</div>',
        unsafe_allow_html=True
    )


    maintenance_event = st.selectbox(
        "Potential event",
        [
            "None",
            "🚧 Speed Restriction",
            "🛤️ Maintenance Block",
            "🚦 Signal Halt",
            "🌫️ Dense Fog"
        ],
        key="maintenance_event"
    )


    if maintenance_event != "None":

        maintenance_prediction, _ = (
            predict_delay(
                selected_row,
                maintenance_event
            )
        )


        maintenance_impact = (
            maintenance_prediction
            - normal_prediction
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Normal",
                f"{normal_prediction:.1f} min"
            )


        with c2:

            st.metric(
                "Scenario",
                f"{maintenance_prediction:.1f} min"
            )


        with c3:

            st.metric(
                "Impact",
                f"{maintenance_impact:+.1f} min"
            )


        if maintenance_impact > 0:

            st.warning(
                f"Expected delay increase: "
                f"{maintenance_impact:.1f} minutes."
            )


    # --------------------------------------------------------
    # ENVIRONMENTAL CONDITIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-green">'
        '🌦️ Environmental Conditions'
        '</div>',
        unsafe_allow_html=True
    )


    e1, e2, e3, e4 = st.columns(4)


    with e1:

        st.metric(
            "Temperature",
            f"{temperature:.1f} °C"
        )


    with e2:

        st.metric(
            "Rainfall",
            f"{rainfall:.0f} mm"
        )


    with e3:

        st.metric(
            "Visibility",
            f"{visibility:.0f} m"
        )


    with e4:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RailCast • Dynamic ETA Intelligence • SIH 2026"
)

st.caption(
    "Decision-support system. Final operational decisions "
    "remain with authorised railway personnel."
)
