import os
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb


# ============================================================
# LOAD MODEL + DATA
# ============================================================

xgb_model = joblib.load("xgb_model.pkl")
residual_model = joblib.load("residual_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
residual_features = joblib.load("residual_features.pkl")
demo_data = pd.read_csv("demo_data.csv")

# Confirmed validation MAE
XGB_MAE = 7.88


# ============================================================
# FEATURE NAMES FOR EXPLANATION
# ============================================================

readable_names = {
    "rainfall_mm": "rainfall",
    "visibility_m": "low visibility / fog",
    "temperature_c": "temperature",
    "congestion_score": "congestion on this stretch",
    "historical_section_time": "typically slow section",
    "distance_to_next": "distance to next station",
    "station_sequence": "position along the route",
    "hour": "time of day",
    "day_of_week": "day of week",
}


# ============================================================
# EXPLAINABILITY FUNCTION
# ============================================================

def explain_row(feature_row):

    row_df = (
        feature_row[residual_features]
        .to_frame()
        .T
        .apply(pd.to_numeric)
    )

    dmat = xgb.DMatrix(
        row_df,
        feature_names=residual_features
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

    top_feature = residual_features[top_idx]

    direction = (
        "increasing the delay"
        if contribs[top_idx] > 0
        else "reducing the delay"
    )

    return (
        readable_names.get(
            top_feature,
            top_feature
        ),
        direction
    )


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #0b1118;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }


    /* ======================================================
       HEADER
       ====================================================== */

    .brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 3px;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background-color: #c62828;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .brand-name {
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -1.2px;
        color: #f4f7fa;
        line-height: 1;
    }

    .brand-subtitle {
        color: #8d99a6;
        font-size: 0.98rem;
        margin-left: 60px;
        margin-bottom: 27px;
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-heading {
        font-size: 1.15rem;
        font-weight: 750;
        color: #e8edf1;
        margin-top: 12px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #8996a3;
        font-size: 0.86rem;
        margin-bottom: 13px;
    }


    /* ======================================================
       JOURNEY CARDS
       ====================================================== */

    .journey-panel {
        background-color: #111923;
        border: 1px solid #26313c;
        border-radius: 11px;
        padding: 17px 19px;
        margin-top: 4px;
        margin-bottom: 7px;
        min-height: 80px;
    }

    .journey-label {
        color: #778592;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .journey-value {
        color: #f1f5f8;
        font-size: 1.13rem;
        font-weight: 700;
    }

    .journey-route {
        color: #f1f5f8;
        font-size: 1.28rem;
        font-weight: 800;
    }

    .journey-arrow {
        color: #d33a3a;
        padding: 0 7px;
    }


    /* ======================================================
       ROUTE PANEL
       ====================================================== */

    .route-panel {
        background-color: #101820;
        border: 1px solid #26323d;
        border-radius: 11px;
        padding: 17px 20px;
        margin-top: 7px;
        margin-bottom: 18px;
    }

    .route-heading {
        color: #9aa6b1;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .route-current {
        color: #e9eef2;
        font-size: 0.9rem;
        margin-top: 3px;
    }


    /* ======================================================
       PREDICTION CARDS
       ====================================================== */

    .prediction-card {
        background-color: #111923;
        border: 1px solid #293540;
        border-radius: 12px;
        padding: 19px 20px;
        min-height: 145px;
    }

    .prediction-card.primary {
        border-left: 4px solid #c62828;
    }

    .prediction-label {
        color: #84919d;
        font-size: 0.69rem;
        font-weight: 750;
        letter-spacing: 0.9px;
        text-transform: uppercase;
    }

    .prediction-number {
        color: #f5f7f9;
        font-size: 2.3rem;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 9px;
    }

    .prediction-unit {
        color: #8995a0;
        font-size: 0.84rem;
    }

    .prediction-delta {
        color: #e0b447;
        font-size: 0.84rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .prediction-note {
        color: #76838f;
        font-size: 0.73rem;
        margin-top: 6px;
    }


    /* ======================================================
       DISRUPTION PANEL
       ====================================================== */

    .scenario-panel {
        background-color: #191814;
        border: 1px solid #554a24;
        border-radius: 11px;
        padding: 14px 17px;
        margin: 8px 0 14px 0;
    }

    .scenario-title {
        color: #e2bd54;
        font-size: 0.9rem;
        font-weight: 750;
    }

    .scenario-text {
        color: #aaa38e;
        font-size: 0.82rem;
        margin-top: 4px;
        line-height: 1.45;
    }


    /* ======================================================
       IMPACT PANEL
       ====================================================== */

    .impact-panel {
        background-color: #111923;
        border: 1px solid #293540;
        border-radius: 11px;
        padding: 16px 19px;
        margin-top: 8px;
        margin-bottom: 15px;
    }

    .impact-title {
        color: #e8edf1;
        font-size: 0.9rem;
        font-weight: 750;
        margin-bottom: 9px;
    }

    .impact-row {
        display: flex;
        justify-content: space-between;
        color: #8995a0;
        font-size: 0.78rem;
    }

    .impact-number {
        color: #e8edf1;
        font-weight: 750;
    }


    /* ======================================================
       OPERATING CONDITIONS
       ====================================================== */

    .condition-card {
        background-color: #111923;
        border: 1px solid #293540;
        border-radius: 10px;
        padding: 13px 14px;
        min-height: 77px;
    }

    .condition-label {
        color: #788590;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    .condition-value {
        color: #edf1f4;
        font-size: 0.94rem;
        font-weight: 700;
        margin-top: 5px;
        line-height: 1.25;
    }


    /* ======================================================
       EXPLANATION
       ====================================================== */

    .explanation-panel {
        background-color: #101820;
        border: 1px solid #28343e;
        border-radius: 11px;
        padding: 17px 19px;
        margin-top: 10px;
        margin-bottom: 16px;
    }

    .explanation-title {
        color: #e8edf1;
        font-weight: 750;
        font-size: 0.93rem;
        margin-bottom: 7px;
    }

    .explanation-text {
        color: #a9b3bc;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .explanation-highlight {
        color: #f1f4f6;
        font-weight: 700;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #080d13;
        border-right: 1px solid #1e2933;
    }

    .sidebar-brand {
        color: #f0f3f5;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .sidebar-caption {
        color: #77838e;
        font-size: 0.79rem;
    }

    .sidebar-status {
        border: 1px solid #27323b;
        background-color: #10171e;
        border-radius: 9px;
        padding: 12px;
        margin-top: 12px;
    }

    .status-online {
        color: #62ae72;
        font-size: 0.76rem;
        font-weight: 750;
    }

    .status-detail {
        color: #788590;
        font-size: 0.72rem;
        margin-top: 3px;
    }


    /* ======================================================
       STREAMLIT CONTROLS
       ====================================================== */

    div[data-baseweb="select"] > div {
        background-color: #111923;
        border-color: #303c47;
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
    }

    .stMetric {
        background-color: #111923;
        border: 1px solid #293540;
        border-radius: 10px;
        padding: 10px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">🚆 RailCast</div>
    <div class="sidebar-caption">
        Dynamic ETA & Delay Intelligence
    </div>

    <div class="sidebar-status">
        <div class="status-online">● MODEL ONLINE</div>
        <div class="status-detail">
            XGBoost prediction engine
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.markdown("### Dashboard")

view_mode = st.sidebar.radio(
    "Dashboard view",
    [
        "Passenger",
        "Control Room / Officer"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown("### Model reference")

st.sidebar.caption(
    f"Validation MAE: {XGB_MAE:.2f} minutes"
)

st.sidebar.caption(
    "Historical data + operating-condition features"
)

st.sidebar.divider()

st.sidebar.caption(
    "Decision-support prototype. Operational actions "
    "remain with authorised railway staff."
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-row">

        <div class="brand-mark">
            🚆
        </div>

        <div class="brand-name">
            RailCast
        </div>

    </div>

    <div class="brand-subtitle">
        Dynamic ETA & Delay Intelligence
        · Predict the arrival. Understand the delay.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# JOURNEY SELECTION
# ============================================================

st.markdown(
    '<div class="section-heading">Journey</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Choose a train and the section of its journey to analyse.'
    '</div>',
    unsafe_allow_html=True
)

select_col1, select_col2 = st.columns(
    [1, 1.7]
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

with select_col1:

    train_options = (
        demo_data["train"]
        .dropna()
        .unique()
    )

    selected_train = st.selectbox(
        "Train",
        train_options
    )


# ------------------------------------------------------------
# FILTER TRAIN DATA
# ------------------------------------------------------------

train_rows = (
    demo_data[
        demo_data["train"] == selected_train
    ]
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# JOURNEY POINT
# ------------------------------------------------------------

with select_col2:

    row_index = st.selectbox(
        "Journey point",
        train_rows.index,
        format_func=lambda i:
            (
                f'{train_rows.loc[i, "station"]} '
                f'→ '
                f'{train_rows.loc[i, "next_station"]} '
                f'({train_rows.loc[i, "date"]})'
            )
    )


# ============================================================
# IMPORTANT:
# CREATE current_row BEFORE USING IT ANYWHERE
# ============================================================

current_row = (
    train_rows
    .loc[row_index]
    .copy()
)


# ============================================================
# CURRENT JOURNEY
# ============================================================

st.markdown(
    '<div class="section-heading">Current journey</div>',
    unsafe_allow_html=True
)

j1, j2, j3 = st.columns(
    [0.9, 1.5, 0.9]
)


# ------------------------------------------------------------
# TRAIN CARD
# ------------------------------------------------------------

with j1:

    st.markdown(
        f"""
        <div class="journey-panel">

            <div class="journey-label">
                Train
            </div>

            <div class="journey-value">
                🚆 {selected_train}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# ROUTE CARD
# ------------------------------------------------------------

with j2:

    st.markdown(
        f"""
        <div class="journey-panel">

            <div class="journey-label">
                Current section
            </div>

            <div class="journey-route">
                {current_row["station"]}
                <span class="journey-arrow">→</span>
                {current_row["next_station"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# DATE CARD
# ------------------------------------------------------------

with j3:

    st.markdown(
        f"""
        <div class="journey-panel">

            <div class="journey-label">
                Journey date
            </div>

            <div class="journey-value">
                📅 {current_row["date"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ROUTE PROGRESS
# ============================================================

st.markdown(
    """
    <div class="route-panel">

        <div class="route-heading">
            Route progress
        </div>

    """,
    unsafe_allow_html=True
)


try:

    current_position = float(
        current_row["station_sequence"]
    )

    sequence_values = pd.to_numeric(
        train_rows["station_sequence"],
        errors="coerce"
    )

    min_sequence = sequence_values.min()
    max_sequence = sequence_values.max()

    if (
        pd.notna(min_sequence)
        and
        pd.notna(max_sequence)
        and
        max_sequence > min_sequence
    ):

        progress = (
            (current_position - min_sequence)
            /
            (max_sequence - min_sequence)
        )

    else:

        progress = 0.5

    progress = max(
        0.0,
        min(1.0, progress)
    )

except Exception:

    progress = 0.5


route_stations = (
    train_rows["station"]
    .dropna()
    .astype(str)
    .tolist()
)


if len(route_stations) >= 2:

    first_station = route_stations[0]
    last_station = route_stations[-1]

else:

    first_station = str(
        current_row["station"]
    )

    last_station = str(
        current_row["next_station"]
    )


route_left, route_middle, route_right = st.columns(
    [1, 5, 1]
)


with route_left:

    st.markdown(
        f"""
        <div style="
            color:#e8edf1;
            font-size:0.86rem;
            font-weight:700;
            padding-top:8px;
        ">
            ● {first_station}
        </div>
        """,
        unsafe_allow_html=True
    )


with route_middle:

    st.progress(
        progress
    )

    progress_percent = int(
        progress * 100
    )

    st.markdown(
        f"""
        <div class="route-current">
            <b>{current_row["station"]}</b>
            → {current_row["next_station"]}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            {progress_percent}% along selected route
        </div>
        """,
        unsafe_allow_html=True
    )


with route_right:

    st.markdown(
        f"""
        <div style="
            color:#e8edf1;
            font-size:0.86rem;
            font-weight:700;
            text-align:right;
            padding-top:8px;
        ">
            {last_station} ●
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# WHAT-IF DISRUPTION SIMULATOR
# ============================================================

st.markdown(
    '<div class="section-heading">What-if analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Inject an operating disruption and see how the forecast changes.'
    '</div>',
    unsafe_allow_html=True
)


disruption = st.selectbox(
    "Select operating scenario",
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


# ============================================================
# CREATE SIMULATED ROW
# ============================================================

sim_row = current_row.copy()

disruption_note = ""

sim_weather_label = (
    current_row[
        "weather_condition_passenger"
    ]
)


# ------------------------------------------------------------
# FOG
# ------------------------------------------------------------

if disruption == "Fog":

    sim_row["visibility_m"] = 150

    sim_row["temperature_c"] = (
        sim_row["temperature_c"] - 3
    )

    disruption_note = (
        "Visibility reduced to 150 m, "
        "representing dense fog conditions."
    )

    sim_weather_label = "Foggy"


# ------------------------------------------------------------
# HEAVY RAIN
# ------------------------------------------------------------

elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80 mm, "
        "representing storm-level rainfall."
    )

    sim_weather_label = "Heavy Rain / Storm"


# ------------------------------------------------------------
# SPEED RESTRICTION
# ------------------------------------------------------------

elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"]
        * 1.5
    )

    disruption_note = (
        "Expected section running time increased "
        "by 50% because of a temporary speed restriction."
    )

    sim_weather_label = (
        "Speed Restriction in effect"
    )


# ------------------------------------------------------------
# SIGNAL HALT
# ------------------------------------------------------------

elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"]
        + 25
    )

    disruption_note = (
        "An additional 25-minute signal halt "
        "was introduced."
    )

    sim_weather_label = (
        "Signal Halt in effect"
    )


# ------------------------------------------------------------
# CONGESTION
# ------------------------------------------------------------

elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    disruption_note = (
        "Downstream congestion increased "
        "to a near-maximum level."
    )

    sim_weather_label = (
        "Track Congestion Spike"
    )


# ------------------------------------------------------------
# MAINTENANCE
# ------------------------------------------------------------

elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"]
        + 45
    )

    disruption_note = (
        "An unscheduled maintenance block "
        "adding 45 minutes was introduced."
    )

    sim_weather_label = (
        "Maintenance Block in effect"
    )


# ============================================================
# BASELINE PREDICTION
# ============================================================

X_original = (
    pd.DataFrame(
        [
            current_row[
                feature_columns
            ]
        ]
    )
    .apply(pd.to_numeric)
)


original_predicted_delay = (
    xgb_model.predict(
        X_original
    )[0]
)


# ============================================================
# DISRUPTION PREDICTION
# ============================================================

X_input = (
    pd.DataFrame(
        [
            sim_row[
                feature_columns
            ]
        ]
    )
    .apply(pd.to_numeric)
)


predicted_delay = (
    xgb_model.predict(
        X_input
    )[0]
)


# ============================================================
# DELAY DIFFERENCE
# ============================================================

delay_change = (
    predicted_delay
    -
    original_predicted_delay
)


# ============================================================
# SCENARIO INFORMATION
# ============================================================

if disruption != "None":

    st.markdown(
        f"""
        <div class="scenario-panel">

            <div class="scenario-title">
                ⚠ {disruption}
            </div>

            <div class="scenario-text">
                {disruption_note}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ETA FORECAST
# ============================================================

st.markdown(
    '<div class="section-heading">ETA forecast</div>',
    unsafe_allow_html=True
)


p1, p2, p3 = st.columns(
    [1, 1, 0.9]
)


# ------------------------------------------------------------
# CURRENT CONDITIONS
# ------------------------------------------------------------

with p1:

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Current conditions
            </div>

            <div class="prediction-number">
                {original_predicted_delay:.1f}
            </div>

            <div class="prediction-unit">
                minutes predicted delay
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# SCENARIO PREDICTION
# ------------------------------------------------------------

with p2:

    if disruption != "None":

        st.markdown(
            f"""
            <div class="prediction-card primary">

                <div class="prediction-label">
                    With selected disruption
                </div>

                <div class="prediction-number">
                    {predicted_delay:.1f}
                </div>

                <div class="prediction-unit">
                    minutes predicted delay
                </div>

                <div class="prediction-delta">
                    {delay_change:+.1f} min impact
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="prediction-card primary">

                <div class="prediction-label">
                    Current prediction
                </div>

                <div class="prediction-number">
                    {predicted_delay:.1f}
                </div>

                <div class="prediction-unit">
                    minutes predicted delay
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# ESTIMATED RANGE
# ------------------------------------------------------------

with p3:

    lower_range = (
        predicted_delay - XGB_MAE
    )

    upper_range = (
        predicted_delay + XGB_MAE
    )

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Estimated range
            </div>

            <div
                class="prediction-number"
                style="font-size:1.8rem;"
            >
                {lower_range:.0f}–{upper_range:.0f}
            </div>

            <div class="prediction-unit">
                minutes
            </div>

            <div class="prediction-note">
                Based on validation MAE of
                ±{XGB_MAE:.2f} min
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISRUPTION IMPACT VISUAL
# ============================================================

if disruption != "None":

    denominator = max(
        abs(original_predicted_delay) + 20,
        20
    )

    impact_ratio = (
        abs(delay_change)
        /
        denominator
    )

    impact_ratio = min(
        impact_ratio,
        1.0
    )

    st.markdown(
        f"""
        <div class="impact-panel">

            <div class="impact-title">
                Disruption impact
            </div>

            <div class="impact-row">

                <span>
                    Normal forecast
                </span>

                <span class="impact-number">
                    {original_predicted_delay:.1f} min
                </span>

            </div>

            <div style="
                margin:10px 0;
                background:#27323c;
                height:9px;
                border-radius:8px;
                overflow:hidden;
            ">

                <div style="
                    width:{impact_ratio * 100:.1f}%;
                    height:100%;
                    background:#c62828;
                    border-radius:8px;
                ">
                </div>

            </div>

            <div class="impact-row">

                <span>
                    Scenario forecast
                </span>

                <span class="impact-number">
                    {predicted_delay:.1f} min
                </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OPERATING CONDITIONS
# ============================================================

st.markdown(
    '<div class="section-heading">Operating conditions</div>',
    unsafe_allow_html=True
)


condition1, condition2, condition3, condition4 = st.columns(4)


# ------------------------------------------------------------
# WEATHER
# ------------------------------------------------------------

with condition1:

    st.markdown(
        f"""
        <div class="condition-card">

            <div class="condition-label">
                Weather
            </div>

            <div class="condition-value">
                {sim_weather_label}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# VISIBILITY
# ------------------------------------------------------------

with condition2:

    visibility = sim_row.get(
        "visibility_m",
        None
    )

    if pd.notna(visibility):

        visibility_text = (
            f"{float(visibility):.0f} m"
        )

    else:

        visibility_text = "—"

    st.markdown(
        f"""
        <div class="condition-card">

            <div class="condition-label">
                Visibility
            </div>

            <div class="condition-value">
                {visibility_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# CONGESTION
# ------------------------------------------------------------

with condition3:

    congestion = sim_row.get(
        "congestion_score",
        None
    )

    if pd.notna(congestion):

        congestion_text = (
            f"{float(congestion):.2f}"
        )

    else:

        congestion_text = "—"

    st.markdown(
        f"""
        <div class="condition-card">

            <div class="condition-label">
                Congestion
            </div>

            <div class="condition-value">
                {congestion_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# SECTION TIME
# ------------------------------------------------------------

with condition4:

    section_time = sim_row.get(
        "historical_section_time",
        None
    )

    if pd.notna(section_time):

        section_text = (
            f"{float(section_time):.1f} min"
        )

    else:

        section_text = "—"

    st.markdown(
        f"""
        <div class="condition-card">

            <div class="condition-label">
                Section time
            </div>

            <div class="condition-value">
                {section_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

top_feature, direction = explain_row(
    sim_row
)


if view_mode == "Passenger":

    st.markdown(
        """
        <div class="explanation-panel">

            <div class="explanation-title">
                ℹ Why this prediction?
            </div>

        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="explanation-text">

            The forecast is mainly influenced by
            <span class="explanation-highlight">
                {top_feature}
            </span>,
            which is currently
            <span class="explanation-highlight">
                {direction}
            </span>.

            <br><br>

            Current operating condition:
            <span class="explanation-highlight">
                {sim_weather_label}
            </span>.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


else:

    st.markdown(
        """
        <div class="explanation-panel">

            <div class="explanation-title">
                🔧 Control-room analysis
            </div>

        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="explanation-text">

            Primary contributing feature:
            <span class="explanation-highlight">
                {top_feature}
            </span>
            ({direction}).

            <br><br>

            Weather:
            <span class="explanation-highlight">
                {sim_weather_label}
            </span>

            <br><br>

            Congestion score:
            <span class="explanation-highlight">
                {sim_row["congestion_score"]:.2f}
            </span>

            &nbsp;&nbsp; | &nbsp;&nbsp;

            Historical section time:
            <span class="explanation-highlight">
                {sim_row["historical_section_time"]:.1f} min
            </span>

            &nbsp;&nbsp; | &nbsp;&nbsp;

            Route position:
            <span class="explanation-highlight">
                {sim_row["station_sequence"]}
            </span>

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PASSENGER FEEDBACK
# ============================================================

st.markdown(
    '<div class="section-heading">Close the prediction loop</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Record the actual delay experienced. This creates a feedback '
    'dataset for future evaluation and model improvement.'
    '</div>',
    unsafe_allow_html=True
)


feedback_col1, feedback_col2 = st.columns(
    [1, 2]
)


with feedback_col1:

    actual_delay_input = st.number_input(
        "Actual delay experienced (minutes)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )


with feedback_col2:

    st.write("")

    if st.button(
        "Submit actual delay",
        type="primary"
    ):

        feedback_row = pd.DataFrame(
            [
                {
                    "train": selected_train,
                    "station": current_row["station"],
                    "predicted_delay": predicted_delay,
                    "actual_delay": actual_delay_input,
                    "submitted_at": pd.Timestamp.now()
                }
            ]
        )

        feedback_row.to_csv(
            "passenger_feedback.csv",
            mode="a",
            header=not os.path.exists(
                "passenger_feedback.csv"
            ),
            index=False
        )

        st.success(
            "Actual delay recorded successfully."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64717d;
        font-size:0.75rem;
        padding:7px 0 15px 0;
    ">

        RailCast · Dynamic ETA Forecasting Prototype
        &nbsp; • &nbsp;
        Predict → Explain → Simulate → Learn

    </div>
    """,
    unsafe_allow_html=True
)
