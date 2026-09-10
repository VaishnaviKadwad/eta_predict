import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path
from PIL import Image
from datetime import datetime


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
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = BASE_DIR / "demo_data.csv"
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
RESIDUAL_MODEL_PATH = BASE_DIR / "residual_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
RESIDUAL_FEATURES_PATH = BASE_DIR / "residual_features.pkl"


# ============================================================
# FIND BANNER AUTOMATICALLY
# ============================================================

def find_banner():
    """
    Search for train_banner.png in several possible locations.
    This avoids problems caused by Streamlit Cloud working
    directories.
    """

    possible_paths = [
        BASE_DIR / "assets" / "train_banner.png",
        BASE_DIR / "train_banner.png",
        Path.cwd() / "assets" / "train_banner.png",
        Path.cwd() / "train_banner.png",
    ]

    # Direct paths first
    for path in possible_paths:
        if path.exists() and path.is_file():
            return path

    # Recursive search as final fallback
    try:
        for path in BASE_DIR.rglob("train_banner.png"):
            if path.is_file():
                return path
    except Exception:
        pass

    return None


BANNER_FILE = find_banner()


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hide unnecessary Streamlit decoration */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Section titles */

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 14px;
        opacity: 0.68;
        margin-bottom: 18px;
    }

    /* Hero */

    .hero {
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
        background: linear-gradient(
            100deg,
            #d62828 0%,
            #f4511e 38%,
            #f77f00 68%,
            #fcbf49 100%
        );
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 17px;
        margin-top: 8px;
        font-weight: 500;
    }

    .hero-tagline {
        font-size: 14px;
        margin-top: 12px;
        opacity: 0.95;
    }

    /* Cards */

    .info-card {
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        background: rgba(128,128,128,0.045);
    }

    .card-label {
        font-size: 13px;
        opacity: 0.65;
        margin-bottom: 7px;
    }

    .card-value {
        font-size: 29px;
        font-weight: 750;
    }

    .card-small {
        font-size: 13px;
        opacity: 0.65;
        margin-top: 5px;
    }

    /* Prediction */

    .prediction-card {
        border-radius: 18px;
        padding: 24px;
        border: 1px solid rgba(30,100,200,0.18);
        background: linear-gradient(
            135deg,
            rgba(30,100,200,0.08),
            rgba(0,160,180,0.05)
        );
    }

    .prediction-number {
        font-size: 42px;
        font-weight: 800;
    }

    /* Status */

    .status-low {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-weight: 700;
        background: rgba(46, 160, 67, 0.13);
        color: #2e8b57;
    }

    .status-medium {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-weight: 700;
        background: rgba(245, 166, 35, 0.16);
        color: #c47700;
    }

    .status-high {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-weight: 700;
        background: rgba(214, 40, 40, 0.14);
        color: #d62828;
    }

    /* Explanation */

    .explanation-box {
        border-radius: 18px;
        padding: 22px 24px;
        background: rgba(128,128,128,0.055);
        border: 1px solid rgba(128,128,128,0.17);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .explanation-title {
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .explanation-main {
        font-size: 16px;
        line-height: 1.55;
    }

    .reason-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        margin: 5px 5px 0 0;
        background: rgba(30,100,200,0.10);
        font-size: 13px;
    }

    /* Disruption */

    .impact-positive {
        font-size: 26px;
        font-weight: 800;
    }

    /* Role card */

    .role-header {
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 3px;
    }

    /* Footer */

    .railcast-footer {
        text-align: center;
        opacity: 0.55;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(128,128,128,0.18);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO BANNER
# ============================================================

if BANNER_FILE is not None:

    try:
        banner = Image.open(BANNER_FILE)
        banner.load()

        if banner.mode not in ("RGB", "RGBA"):
            banner = banner.convert("RGB")

        # Use Streamlit's native image renderer.
        # This avoids the previous raw-HTML problem.
        st.image(
            banner,
            width="stretch"
        )

    except Exception:
        # Clean fallback
        st.markdown(
            """
            <div class="hero">
                <div class="hero-title">🚆 RailCast</div>
                <div class="hero-subtitle">
                    Dynamic ETA Intelligence for Indian Railways
                </div>
                <div class="hero-tagline">
                    Predicting how delays evolve — before they arrive.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

else:

    # If the PNG genuinely isn't available in deployment,
    # show a designed fallback rather than breaking the app.
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🚆 RailCast</div>
            <div class="hero-subtitle">
                Dynamic ETA Intelligence for Indian Railways
            </div>
            <div class="hero-tagline">
                Predicting how delays evolve — before they arrive.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error("demo_data.csv was not found.")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    return data


df = load_data()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    residual_model = None
    feature_columns = None
    residual_features = None

    if RESIDUAL_MODEL_PATH.exists():
        residual_model = joblib.load(RESIDUAL_MODEL_PATH)

    if FEATURE_COLUMNS_PATH.exists():
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    if RESIDUAL_FEATURES_PATH.exists():
        residual_features = joblib.load(RESIDUAL_FEATURES_PATH)

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

    st.error(f"Unable to load model files: {e}")
    st.stop()


# ============================================================
# MODEL METRIC
# ============================================================

MODEL_MAE = 7.88


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.info(
    f"🚆 **Current Journey Point**\n\n"
    f"### {station} → {next_station}\n\n"
    f"Train **{selected_train}** • {selected_row['date']}"
)

    st.markdown("### DASHBOARDS")

    role = st.radio(
        "Select dashboard",
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
        f"{MODEL_MAE:.2f} min"
    )

    st.caption("XGBoost prediction engine")
    st.caption("Historical + operating-condition features")


# ============================================================
# CURRENT DATE / TIME
# ============================================================

now = datetime.now()

top_left, top_right = st.columns([4, 1])

with top_right:

    st.markdown(
        f"📅 **{now.strftime('%d %b %Y')}**"
    )

    st.markdown(
        f"🕐 **{now.strftime('%I:%M %p')}**"
    )


# ============================================================
# JOURNEY SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">🚆 Select Journey</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Choose a train and the current journey point to generate a dynamic forecast.'
    '</div>',
    unsafe_allow_html=True
)


# Train list

trains = (
    df["train"]
    .astype(str)
    .dropna()
    .unique()
    .tolist()
)

trains = sorted(trains)


if len(trains) == 0:
    st.error("No train records found in demo_data.csv.")
    st.stop()


selected_train = st.selectbox(
    "Train",
    trains
)


# Filter train

train_df = df[
    df["train"].astype(str) == str(selected_train)
].copy()


if train_df.empty:
    st.warning("No data found for this train.")
    st.stop()


# Journey point options

train_df["journey_label"] = (
    train_df["station"].astype(str)
    + " → "
    + train_df["next_station"].astype(str)
    + " ("
    + train_df["date"].astype(str)
    + ")"
)


journey_options = train_df["journey_label"].tolist()


selected_journey = st.selectbox(
    "Journey point",
    journey_options
)


selected_row = train_df[
    train_df["journey_label"] == selected_journey
].iloc[0].copy()


station = str(selected_row["station"])
next_station = str(selected_row["next_station"])


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(row, disruption=None):

    x = row.copy()

    # ----------------------------------------
    # Apply disruption modifications
    # ----------------------------------------

    if disruption == "🌫️ Dense Fog":

        if "visibility_m" in x:
            x["visibility_m"] = 150

        if "temperature_c" in x:
            x["temperature_c"] = -3


    elif disruption == "🌧️ Heavy Rain":

        if "rainfall_mm" in x:
            x["rainfall_mm"] = 80


    elif disruption == "🚧 Speed Restriction":

        if "historical_section_time" in x:
            x["historical_section_time"] = (
                float(x["historical_section_time"]) * 1.5
            )


    elif disruption == "🚦 Signal Halt":

        if "delay_minutes" in x:
            x["delay_minutes"] = (
                float(x["delay_minutes"]) + 25
            )


    elif disruption == "🚥 Track Congestion Spike":

        if "congestion_score" in x:
            x["congestion_score"] = 0.95


    elif disruption == "🛤️ Maintenance Block":

        if "delay_minutes" in x:
            x["delay_minutes"] = (
                float(x["delay_minutes"]) + 45
            )


    # ----------------------------------------
    # Build model dataframe
    # ----------------------------------------

    if feature_columns is not None:

        cols = list(feature_columns)

    else:

        # Fallback if feature_columns.pkl isn't available
        cols = [
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


    model_input = {}

    for col in cols:

        if col in x.index:

            value = x[col]

            try:
                model_input[col] = float(value)

            except Exception:
                model_input[col] = 0.0

        else:

            model_input[col] = 0.0


    X = pd.DataFrame(
        [model_input],
        columns=cols
    )

    return X, x


# ============================================================
# PREDICTION
# ============================================================

def get_prediction(row, disruption=None):

    X, modified_row = prepare_features(
        row,
        disruption
    )

    prediction = model.predict(X)

    predicted_delay = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    return predicted_delay, modified_row


try:

    normal_prediction, normal_row = get_prediction(
        selected_row,
        None
    )

except Exception as e:

    st.error(f"Prediction error: {e}")
    st.stop()


# Don't allow negative delay to look like a negative delay
normal_prediction = max(0.0, normal_prediction)


# ============================================================
# DISRUPTION SIMULATOR
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">⚡ What-if Disruption Simulator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
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


if disruption != "None":

    try:

        disruption_prediction, disrupted_row = get_prediction(
            selected_row,
            disruption
        )

        disruption_prediction = max(
            0.0,
            disruption_prediction
        )

    except Exception as e:

        st.error(f"Disruption prediction error: {e}")

        disruption_prediction = normal_prediction

else:

    disruption_prediction = None


# ============================================================
# DELAY RISK
# ============================================================

current_delay = float(
    selected_row.get("delay_minutes", 0)
)

current_delay = max(
    0.0,
    current_delay
)


def calculate_risk(delay):

    if delay < 5:
        return "LOW", "status-low"

    elif delay < 15:
        return "MEDIUM", "status-medium"

    else:
        return "HIGH", "status-high"


risk, risk_class = calculate_risk(
    normal_prediction
)


# ============================================================
# PASSENGER DASHBOARD
# ============================================================

if role == "👤 Passenger":

    st.divider()

    st.markdown(
        '<div class="section-title">🎫 Passenger View</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A simple view of where your train is and what to expect next.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Journey headline
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="prediction-card">

            <div style="font-size:14px;opacity:0.65;">
                CURRENT JOURNEY POINT
            </div>

            <div style="
                font-size:27px;
                font-weight:800;
                margin-top:5px;
            ">
                🚆 {station} → {next_station}
            </div>

            <div style="
                margin-top:7px;
                font-size:14px;
                opacity:0.65;
            ">
                Train {selected_train} • {selected_row["date"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("")


    # --------------------------------------------------------
    # Main forecast cards
    # --------------------------------------------------------

    if disruption == "None":

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current Delay",
                f"{current_delay:.1f} min"
            )

        with c2:

            st.metric(
                "RailCast Forecast",
                f"{normal_prediction:.1f} min"
            )

        with c3:

            low = normal_prediction - MODEL_MAE
            high = normal_prediction + MODEL_MAE

            st.metric(
                "Estimated Range",
                f"{low:.0f} → {high:.0f} min"
            )

        with c4:

            st.markdown(
                f"""
                <div class="{risk_class}">
                    {risk} RISK
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption("Expected delay risk")


    else:

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current Delay",
                f"{current_delay:.1f} min"
            )

        with c2:

            st.metric(
                "Normal Forecast",
                f"{normal_prediction:.1f} min"
            )

        with c3:

            st.metric(
                "Disruption Forecast",
                f"{disruption_prediction:.1f} min",
                delta=f"{disruption_prediction - normal_prediction:+.1f} min"
            )

        with c4:

            risk_after, risk_class_after = calculate_risk(
                disruption_prediction
            )

            st.markdown(
                f"""
                <div class="{risk_class_after}">
                    {risk_after} RISK
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption("After selected event")


    # --------------------------------------------------------
    # Passenger message
    # --------------------------------------------------------

    st.markdown("")

    if disruption == "None":

        if normal_prediction < 5:

            passenger_message = (
                "The train is currently expected to maintain a relatively "
                "low delay at this journey point."
            )

        elif normal_prediction < 15:

            passenger_message = (
                "A moderate delay is expected. Passengers should allow "
                "some additional time for onward travel."
            )

        else:

            passenger_message = (
                "A significant delay is forecast. Passengers with "
                "connections should consider allowing extra time."
            )

    else:

        impact = disruption_prediction - normal_prediction

        passenger_message = (
            f"The selected event could change the forecast by "
            f"{impact:+.1f} minutes at this journey point."
        )


    st.info(
        f"💡 **Passenger insight:** {passenger_message}"
    )


    # --------------------------------------------------------
    # ROUTE MAP
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🗺️ Journey Map</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Current journey point and route progression.'
        '</div>',
        unsafe_allow_html=True
    )


    # Get all stations for this train/date
    route_df = train_df[
        train_df["date"].astype(str)
        == str(selected_row["date"])
    ].copy()


    # Remove duplicates while keeping route order
    route_df = route_df.drop_duplicates(
        subset=["station", "next_station"]
    )


    route_stations = []

    for _, r in route_df.iterrows():

        s = str(r["station"])

        if s not in route_stations:
            route_stations.append(s)


        ns = str(r["next_station"])

        if ns not in route_stations:
            route_stations.append(ns)


    # Fallback
    if len(route_stations) < 2:

        route_stations = [
            station,
            next_station
        ]


    # Position of current station
    current_index = (
        route_stations.index(station)
        if station in route_stations
        else 0
    )


    # Schematic map because current demo CSV has no lat/lon
    x_positions = list(
        range(len(route_stations))
    )

    y_positions = [
        np.sin(i / 1.8) * 0.35
        for i in x_positions
    ]


    fig = go.Figure()


    # Route line
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="lines+markers",
            line=dict(
                width=5
            ),
            marker=dict(
                size=12
            ),
            hovertemplate=(
                "<b>%{text}</b>"
                "<extra></extra>"
            ),
            text=route_stations
        )
    )


    # Highlight current station
    fig.add_trace(
        go.Scatter(
            x=[x_positions[current_index]],
            y=[y_positions[current_index]],
            mode="markers",
            marker=dict(
                size=24,
                symbol="circle"
            ),
            hovertemplate=(
                "<b>Current journey point</b><br>"
                + station
                + "<extra></extra>"
            )
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


    # Route labels below map
    route_text = "  →  ".join(
        route_stations
    )

    st.caption(
        f"Route: {route_text}"
    )


    # --------------------------------------------------------
    # OPERATING CONDITIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🌦️ Current Operating Conditions</div>',
        unsafe_allow_html=True
    )


    condition_cols = st.columns(5)


    rainfall = float(
        selected_row.get("rainfall_mm", 0)
    )

    temperature = float(
        selected_row.get("temperature_c", 0)
    )

    visibility = float(
        selected_row.get("visibility_m", 0)
    )

    congestion = float(
        selected_row.get("congestion_score", 0)
    )

    section_time = float(
        selected_row.get(
            "historical_section_time",
            0
        )
    )


    with condition_cols[0]:

        st.metric(
            "🌧️ Rainfall",
            f"{rainfall:.0f} mm"
        )


    with condition_cols[1]:

        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C"
        )


    with condition_cols[2]:

        st.metric(
            "👁️ Visibility",
            f"{visibility:.0f} m"
        )


    with condition_cols[3]:

        st.metric(
            "🚦 Congestion",
            f"{congestion:.2f}"
        )


    with condition_cols[4]:

        st.metric(
            "🛤️ Section Time",
            f"{section_time:.1f} min"
        )


   # --------------------------------------------------------
# WHY THIS PREDICTION?
# --------------------------------------------------------

st.markdown("### 🔎 Why is RailCast predicting this?")

st.caption(
    "RailCast combines the train's current delay, route conditions, "
    "weather, congestion and historical section behaviour."
)


# Determine the strongest visible operating factor

factors = {
    "rainfall": rainfall,
    "low_visibility": max(0, 1000 - visibility),
    "congestion": congestion * 100,
    "current_delay": current_delay,
    "section_time": section_time
}

top_factor = max(
    factors,
    key=factors.get
)


if top_factor == "rainfall":

    factor_name = "Rainfall"
    factor_value = f"{rainfall:.0f} mm"
    explanation = (
        "Rainfall can contribute to slower movement and increased "
        "operational caution."
    )


elif top_factor == "low_visibility":

    factor_name = "Visibility"
    factor_value = f"{visibility:.0f} m"
    explanation = (
        "Reduced visibility can increase operational caution "
        "and affect running conditions."
    )


elif top_factor == "congestion":

    factor_name = "Congestion"
    factor_value = f"{congestion:.2f}"
    explanation = (
        "Higher congestion can increase downstream running time "
        "and delay propagation."
    )


elif top_factor == "current_delay":

    factor_name = "Current Delay"
    factor_value = f"{current_delay:.1f} min"
    explanation = (
        "The current delay is an important signal for estimating "
        "how much delay may continue downstream."
    )


else:

    factor_name = "Historical Section Time"
    factor_value = f"{section_time:.1f} min"
    explanation = (
        "Historical running time helps RailCast estimate how long "
        "the train is likely to take over the next section."
    )


# Main prediction message

st.success(
    f"🎯 **RailCast Forecast: {normal_prediction:.1f} minutes delay**"
)


st.write(
    f"RailCast is currently influenced most strongly by "
    f"**{factor_name} ({factor_value})**."
)


st.write(explanation)


# Supporting factors

st.markdown("#### Operating factors considered")

factor_col1, factor_col2, factor_col3, factor_col4 = st.columns(4)


with factor_col1:

    st.metric(
        "Current Delay",
        f"{current_delay:.1f} min"
    )


with factor_col2:

    st.metric(
        "Rainfall",
        f"{rainfall:.0f} mm"
    )


with factor_col3:

    st.metric(
        "Visibility",
        f"{visibility:.0f} m"
    )


with factor_col4:

    st.metric(
        "Congestion",
        f"{congestion:.2f}"
    )


st.caption(
    f"Model validation reference: MAE = {MODEL_MAE:.2f} minutes"
)

    # --------------------------------------------------------
    # DISRUPTION IMPACT
    # --------------------------------------------------------

    if disruption != "None":

        st.markdown(
            '<div class="section-title">⚡ Disruption Impact</div>',
            unsafe_allow_html=True
        )

        impact = (
            disruption_prediction
            - normal_prediction
        )


        impact_cols = st.columns(3)


        with impact_cols[0]:

            st.metric(
                "Normal Forecast",
                f"{normal_prediction:.1f} min"
            )


        with impact_cols[1]:

            st.metric(
                "After Disruption",
                f"{disruption_prediction:.1f} min"
            )


        with impact_cols[2]:

            st.metric(
                "Forecast Change",
                f"{impact:+.1f} min"
            )


        if impact > 0:

            st.warning(
                f"⚠️ **{disruption}** increases the predicted "
                f"delay by approximately **{impact:.1f} minutes**."
            )

        elif impact < 0:

            st.success(
                f"RailCast predicts a reduction of "
                f"{abs(impact):.1f} minutes under this scenario."
            )

        else:

            st.info(
                "The selected scenario does not materially change "
                "the current prediction."
            )


    # --------------------------------------------------------
    # PASSENGER FEEDBACK
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">💬 Passenger Feedback</div>',
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

        feedback_row = pd.DataFrame(
            [{
                "timestamp": datetime.now().isoformat(),
                "train": selected_train,
                "station": station,
                "next_station": next_station,
                "prediction": normal_prediction,
                "feedback": feedback
            }]
        )


        try:

            if feedback_file.exists():

                feedback_row.to_csv(
                    feedback_file,
                    mode="a",
                    header=False,
                    index=False
                )

            else:

                feedback_row.to_csv(
                    feedback_file,
                    index=False
                )

            st.success(
                "Thank you — your feedback has been recorded."
            )

        except Exception:

            st.info(
                "Feedback captured for this session."
            )


# ============================================================
# STATION OPERATOR DASHBOARD
# ============================================================

elif role == "🏢 Station Operator":

    st.divider()

    st.markdown(
        '<div class="section-title">🏢 Station Operator Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Operational visibility for passenger communication and station readiness.'
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


    st.markdown("")


    if normal_prediction >= 15:

        st.error(
            "🚨 High passenger-impact delay expected. "
            "Consider proactive announcements and crowd-management preparation."
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


    st.markdown("### 📢 Passenger Communication")


    message = (
        f"Train {selected_train} is currently at {station}. "
        f"RailCast forecasts approximately "
        f"{normal_prediction:.1f} minutes of delay toward "
        f"{next_station}."
    )


    st.text_area(
        "Suggested announcement",
        message,
        height=100
    )


    st.markdown("### 🌦️ Station Conditions")


    cols = st.columns(4)


    with cols[0]:

        st.metric(
            "Rainfall",
            f"{rainfall:.0f} mm"
        )


    with cols[1]:

        st.metric(
            "Visibility",
            f"{visibility:.0f} m"
        )


    with cols[2]:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    with cols[3]:

        st.metric(
            "Section Time",
            f"{section_time:.1f} min"
        )


# ============================================================
# TRAFFIC CONTROLLER DASHBOARD
# ============================================================

elif role == "🚦 Traffic Controller":

    st.divider()

    st.markdown(
        '<div class="section-title">🚦 Traffic Controller Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Predictive operational intelligence for delay propagation and disruption response.'
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
            "Forecast Delay",
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


    st.markdown("")


    # Delay comparison chart

    chart_df = pd.DataFrame(
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
            x=chart_df["Stage"],
            y=chart_df["Delay"],
            text=[
                f"{x:.1f} min"
                for x in chart_df["Delay"]
            ],
            textposition="auto"
        )
    )


    fig.update_layout(
        title="Delay Evolution",
        height=330,
        showlegend=False,
        yaxis_title="Delay (minutes)"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.markdown("### ⚡ Scenario Analysis")


    controller_disruption = st.selectbox(
        "Test operational scenario",
        [
            "None",
            "🌫️ Dense Fog",
            "🌧️ Heavy Rain",
            "🚧 Speed Restriction",
            "🚦 Signal Halt",
            "🚥 Track Congestion Spike",
            "🛤️ Maintenance Block"
        ],
        key="controller_disruption"
    )


    if controller_disruption != "None":

        scenario_prediction, _ = get_prediction(
            selected_row,
            controller_disruption
        )

        scenario_prediction = max(
            0,
            scenario_prediction
        )

        delta = (
            scenario_prediction
            - normal_prediction
        )


        st.metric(
            "Scenario Forecast",
            f"{scenario_prediction:.1f} min",
            delta=f"{delta:+.1f} min"
        )


        if delta > 0:

            st.warning(
                f"{controller_disruption} could add approximately "
                f"{delta:.1f} minutes to the forecast."
            )


    st.markdown("### 🧠 Operational Signals")


    op_cols = st.columns(5)


    with op_cols[0]:

        st.metric(
            "Route Position",
            str(
                selected_row.get(
                    "station_sequence",
                    "—"
                )
            )
        )


    with op_cols[1]:

        st.metric(
            "Distance",
            f"{float(selected_row.get('distance_to_next', 0)):.1f}"
        )


    with op_cols[2]:

        st.metric(
            "Section Time",
            f"{section_time:.1f} min"
        )


    with op_cols[3]:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


    with op_cols[4]:

        st.metric(
            "Forecast",
            f"{normal_prediction:.1f} min"
        )


# ============================================================
# MAINTENANCE OPERATOR DASHBOARD
# ============================================================

else:

    st.divider()

    st.markdown(
        '<div class="section-title">🛠️ Maintenance Operator Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Infrastructure and environmental conditions that can influence train running time.'
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


    st.markdown("### 🛤️ Infrastructure Scenario")


    maintenance_scenario = st.selectbox(
        "Potential infrastructure event",
        [
            "None",
            "🚧 Speed Restriction",
            "🛤️ Maintenance Block",
            "🚦 Signal Halt",
            "🌫️ Dense Fog"
        ],
        key="maintenance_scenario"
    )


    if maintenance_scenario != "None":

        maintenance_prediction, _ = get_prediction(
            selected_row,
            maintenance_scenario
        )

        maintenance_prediction = max(
            0,
            maintenance_prediction
        )


        impact = (
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
                f"{impact:+.1f} min"
            )


        if impact > 0:

            st.warning(
                f"⚠️ The scenario increases the forecast by "
                f"approximately {impact:.1f} minutes."
            )


    st.markdown("### 🌦️ Environmental Conditions")


    env_cols = st.columns(4)


    with env_cols[0]:

        st.metric(
            "Temperature",
            f"{temperature:.1f} °C"
        )


    with env_cols[1]:

        st.metric(
            "Rainfall",
            f"{rainfall:.0f} mm"
        )


    with env_cols[2]:

        st.metric(
            "Visibility",
            f"{visibility:.0f} m"
        )


    with env_cols[3]:

        st.metric(
            "Congestion",
            f"{congestion:.2f}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="railcast-footer">
        RailCast • Dynamic ETA Intelligence • SIH 2026
        <br>
        Decision support system — final operational decisions remain with authorised railway personnel.
    </div>
    """,
    unsafe_allow_html=True
)
