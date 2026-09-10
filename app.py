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

# Confirmed model MAE from your evaluation
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


def explain_row(feature_row):
    """
    Use the residual model's XGBoost contributions
    to identify the strongest feature affecting the prediction.
    """

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
        readable_names.get(top_feature, top_feature),
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
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background: #0b1118;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Remove excessive Streamlit spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }


    /* -------------------------------------------------------
       HEADER
    ------------------------------------------------------- */

    .brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background: #c62828;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
    }

    .brand-name {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f4f7fa;
        line-height: 1;
    }

    .brand-subtitle {
        color: #8d99a6;
        font-size: 0.98rem;
        margin: 5px 0 25px 60px;
    }


    /* -------------------------------------------------------
       SECTION HEADERS
       ------------------------------------------------------- */

    .section-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e7edf2;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .section-description {
        color: #8996a3;
        font-size: 0.88rem;
        margin-bottom: 14px;
    }


    /* -------------------------------------------------------
       JOURNEY CARD
       ------------------------------------------------------- */

    .journey-panel {
        background: #111923;
        border: 1px solid #26313c;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 5px 0 18px 0;
    }

    .journey-label {
        color: #778592;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .journey-value {
        color: #f1f5f8;
        font-size: 1.18rem;
        font-weight: 700;
    }

    .journey-route {
        color: #f1f5f8;
        font-size: 1.35rem;
        font-weight: 750;
    }

    .journey-arrow {
        color: #c62828;
        padding: 0 8px;
    }


    /* -------------------------------------------------------
       ROUTE GRAPHIC
       ------------------------------------------------------- */

    .route-box {
        background: #0f171f;
        border: 1px solid #25313c;
        border-radius: 12px;
        padding: 20px 22px 17px 22px;
        margin: 8px 0 20px 0;
    }

    .route-title {
        color: #aab5bf;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 17px;
    }

    .route-line {
        position: relative;
        height: 42px;
        margin: 0 22px;
    }

    .route-track {
        position: absolute;
        left: 0;
        right: 0;
        top: 13px;
        height: 3px;
        background: #3a4652;
    }

    .route-progress {
        position: absolute;
        left: 0;
        top: 13px;
        height: 3px;
        background: #c62828;
    }

    .station {
        position: absolute;
        top: 5px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 3px solid #0f171f;
        background: #66727e;
        box-shadow: 0 0 0 1px #66727e;
    }

    .station.active {
        background: #c62828;
        box-shadow: 0 0 0 2px #c62828;
    }

    .station-label {
        position: absolute;
        top: 29px;
        font-size: 0.73rem;
        color: #84909b;
        white-space: nowrap;
    }

    .station-label.active-label {
        color: #f0f3f5;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       PREDICTION CARDS
       ------------------------------------------------------- */

    .prediction-card {
        background: #111923;
        border: 1px solid #293540;
        border-radius: 12px;
        padding: 20px;
        min-height: 150px;
    }

    .prediction-card.highlight {
        border-left: 4px solid #c62828;
    }

    .prediction-label {
        color: #84919d;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .prediction-number {
        color: #f5f7f9;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 9px 0 3px 0;
    }

    .prediction-unit {
        color: #88949f;
        font-size: 0.88rem;
    }

    .prediction-delta {
        color: #d8a12d;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 8px;
    }


    /* -------------------------------------------------------
       IMPACT BAR
       ------------------------------------------------------- */

    .impact-panel {
        background: #111923;
        border: 1px solid #293540;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 8px;
    }

    .impact-title {
        color: #e7edf2;
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 10px;
    }

    .impact-track {
        height: 9px;
        background: #27323c;
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
    }

    .impact-fill {
        height: 100%;
        background: #c62828;
        border-radius: 10px;
    }

    .impact-caption {
        color: #7f8b96;
        font-size: 0.78rem;
    }


    /* -------------------------------------------------------
       INFO / EXPLANATION
       ------------------------------------------------------- */

    .explanation-panel {
        background: #101820;
        border: 1px solid #28343e;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 10px;
    }

    .explanation-title {
        color: #e8edf1;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .explanation-text {
        color: #a9b3bc;
        font-size: 0.91rem;
        line-height: 1.5;
    }

    .explanation-highlight {
        color: #f1f4f6;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       OPERATING CONDITIONS
       ------------------------------------------------------- */

    .condition-card {
        background: #111923;
        border: 1px solid #293540;
        border-radius: 10px;
        padding: 13px 15px;
        min-height: 80px;
    }

    .condition-label {
        color: #7f8b96;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .condition-value {
        color: #edf1f4;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 4px;
    }


    /* -------------------------------------------------------
       SCENARIO
       ------------------------------------------------------- */

    .scenario-panel {
        background: #181713;
        border: 1px solid #554922;
        border-radius: 12px;
        padding: 15px 18px;
        margin: 10px 0 15px 0;
    }

    .scenario-title {
        color: #e2bf5a;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .scenario-text {
        color: #aaa28c;
        font-size: 0.84rem;
        margin-top: 5px;
    }


    /* -------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #080d13;
        border-right: 1px solid #1e2933;
    }

    .sidebar-brand {
        color: #f0f3f5;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .sidebar-caption {
        color: #77838e;
        font-size: 0.8rem;
    }

    .sidebar-status {
        border: 1px solid #27323b;
        background: #10171e;
        border-radius: 9px;
        padding: 12px;
        margin-top: 12px;
    }

    .status-dot {
        color: #58a66b;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .status-detail {
        color: #788590;
        font-size: 0.74rem;
        margin-top: 3px;
    }


    /* -------------------------------------------------------
       STREAMLIT CONTROLS
       ------------------------------------------------------- */

    div[data-baseweb="select"] > div {
        background: #111923;
        border-color: #303c47;
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
    }

    .stMetric {
        background: #111923;
        border: 1px solid #293540;
        border-radius: 10px;
        padding: 10px;
    }

    /* Hide Streamlit footer */
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
        <div class="status-dot">● MODEL ONLINE</div>
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
    ["Passenger", "Control Room / Officer"],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown("### Model reference")

st.sidebar.caption(
    f"Test MAE: ±{XGB_MAE:.2f} minutes"
)

st.sidebar.caption(
    "Historical data + operating-condition features"
)

st.sidebar.divider()

st.sidebar.caption(
    "RailCast is a decision-support prototype. "
    "Operational actions remain with authorised railway staff."
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">🚆</div>
        <div class="brand-name">RailCast</div>
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
    'Choose a train and the section of its journey you want to analyse.'
    '</div>',
    unsafe_allow_html=True
)

select_col1, select_col2 = st.columns([1, 1.6])

with select_col1:

    train_options = demo_data["train"].unique()

    selected_train = st.selectbox(
        "Train",
        train_options
    )


train_rows = (
    demo_data[
        demo_data["train"] == selected_train
    ]
    .reset_index(drop=True)
)


with select_col2:

    row_index = st.selectbox(
        "Journey point",
        train_rows.index,
        format_func=lambda i:
            f'{train_rows.loc[i, "station"]} → '
            f'{train_rows.loc[i, "next_station"]} '
            f'({train_rows.loc[i, "date"]})'
    )


# ============================================================
# IMPORTANT:
# current_row MUST be created before using current_row[...]
# ============================================================

current_row = train_rows.loc[row_index].copy()


# ============================================================
# JOURNEY SUMMARY
# ============================================================

st.markdown(
    '<div class="section-heading">Current journey</div>',
    unsafe_allow_html=True
)

j1, j2, j3 = st.columns([0.9, 1.5, 0.9])

with j1:
    st.markdown(
        f"""
        <div class="journey-panel">
            <div class="journey-label">Train</div>
            <div class="journey-value">🚆 {selected_train}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with j2:
    st.markdown(
        f"""
        <div class="journey-panel">
            <div class="journey-label">Current section</div>
            <div class="journey-route">
                {current_row["station"]}
                <span class="journey-arrow">→</span>
                {current_row["next_station"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with j3:
    st.markdown(
        f"""
        <div class="journey-panel">
            <div class="journey-label">Journey date</div>
            <div class="journey-value">📅 {current_row["date"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ROUTE VISUALISATION
# ============================================================

try:

    current_position = float(current_row["station_sequence"])

    sequence_values = pd.to_numeric(
        train_rows["station_sequence"],
        errors="coerce"
    )

    min_sequence = sequence_values.min()
    max_sequence = sequence_values.max()

    if max_sequence > min_sequence:
        progress = (
            (current_position - min_sequence)
            / (max_sequence - min_sequence)
        )
    else:
        progress = 0.5

    progress = max(0.08, min(0.92, progress))

except Exception:
    progress = 0.5


route_stations = train_rows["station"].dropna().astype(str).tolist()

if len(route_stations) >= 2:

    first_station = route_stations[0]
    last_station = route_stations[-1]

    progress_percent = progress * 100

    route_html = f"""
    <div class="route-box">

        <div class="route-title">
            Route progress
        </div>

        <div class="route-line">

            <div class="route-track"></div>

            <div
                class="route-progress"
                style="width:{progress_percent:.1f}%">
            </div>

            <div
                class="station active"
                style="left:0%;">
            </div>

            <div
                class="station active"
                style="left:{progress_percent:.1f}%;
                       transform:translateX(-50%);">
            </div>

            <div
                class="station"
                style="right:0%;">
            </div>

            <div
                class="station-label active-label"
                style="left:0%;">
                {first_station}
            </div>

            <div
                class="station-label active-label"
                style="left:{progress_percent:.1f}%;
                       transform:translateX(-50%);">
                {current_row["station"]}
            </div>

            <div
                class="station-label"
                style="right:0%;">
                {last_station}
            </div>

        </div>

    </div>
    """

    st.markdown(
        route_html,
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
    'Inject a disruption and see how the predicted delay changes.'
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

sim_weather_label = current_row[
    "weather_condition_passenger"
]


if disruption == "Fog":

    sim_row["visibility_m"] = 150

    sim_row["temperature_c"] = (
        sim_row["temperature_c"] - 3
    )

    disruption_note = (
        "Visibility reduced to 150 m, representing dense fog."
    )

    sim_weather_label = "Foggy"


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80 mm, representing "
        "storm-level rainfall."
    )

    sim_weather_label = "Heavy Rain / Storm"


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"] * 1.5
    )

    disruption_note = (
        "Expected section running time increased by 50% "
        "because of a temporary speed restriction."
    )

    sim_weather_label = "Speed Restriction in effect"


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 25
    )

    disruption_note = (
        "An additional 25-minute signal halt was introduced."
    )

    sim_weather_label = "Signal Halt in effect"


elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    disruption_note = (
        "Downstream congestion increased to a near-maximum level."
    )

    sim_weather_label = "Track Congestion Spike"


elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 45
    )

    disruption_note = (
        "An unscheduled maintenance block adding 45 minutes "
        "was introduced."
    )

    sim_weather_label = "Maintenance Block in effect"


# ============================================================
# PREDICTIONS
# ============================================================

X_original = (
    pd.DataFrame(
        [current_row[feature_columns]]
    )
    .apply(pd.to_numeric)
)

original_predicted_delay = (
    xgb_model.predict(X_original)[0]
)


X_input = (
    pd.DataFrame(
        [sim_row[feature_columns]]
    )
    .apply(pd.to_numeric)
)

predicted_delay = (
    xgb_model.predict(X_input)[0]
)


delay_change = (
    predicted_delay -
    original_predicted_delay
)


# ============================================================
# SCENARIO MESSAGE
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
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-heading">ETA forecast</div>',
    unsafe_allow_html=True
)


p1, p2, p3 = st.columns([1, 1, 0.85])


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


with p2:

    if disruption != "None":

        st.markdown(
            f"""
            <div class="prediction-card highlight">

                <div class="prediction-label">
                    With {disruption}
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
            <div class="prediction-card highlight">

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


with p3:

    lower = predicted_delay - XGB_MAE
    upper = predicted_delay + XGB_MAE

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Estimated range
            </div>

            <div class="prediction-number"
                 style="font-size:1.75rem;">
                {lower:.0f}–{upper:.0f}
            </div>

            <div class="prediction-unit">
                minutes
            </div>

            <div class="prediction-delta"
                 style="color:#8a98a5;">
                based on ±{XGB_MAE:.2f} min MAE
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DELAY IMPACT GRAPHIC
# ============================================================

if disruption != "None":

    # Keep the bar visually meaningful without inventing a maximum.
    impact_ratio = abs(delay_change) / max(
        abs(original_predicted_delay) + 20,
        20
    )

    impact_ratio = min(impact_ratio, 1.0)

    st.markdown(
        f"""
        <div class="impact-panel">

            <div class="impact-title">
                Disruption impact
            </div>

            <div class="impact-track">
                <div
                    class="impact-fill"
                    style="width:{impact_ratio * 100:.1f}%;">
                </div>
            </div>

            <div class="impact-caption">
                Forecast changes from
                <b>{original_predicted_delay:.1f} min</b>
                to
                <b>{predicted_delay:.1f} min</b>
                under the selected scenario.
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

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        f"""
        <div class="condition-card">
            <div class="condition-label">Weather</div>
            <div class="condition-value">
                {sim_weather_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    visibility = sim_row.get(
        "visibility_m",
        None
    )

    if pd.notna(visibility):
        visibility_text = f"{float(visibility):.0f} m"
    else:
        visibility_text = "—"

    st.markdown(
        f"""
        <div class="condition-card">
            <div class="condition-label">Visibility</div>
            <div class="condition-value">
                {visibility_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    congestion = sim_row.get(
        "congestion_score",
        None
    )

    if pd.notna(congestion):
        congestion_text = f"{float(congestion):.2f}"
    else:
        congestion_text = "—"

    st.markdown(
        f"""
        <div class="condition-card">
            <div class="condition-label">Congestion</div>
            <div class="condition-value">
                {congestion_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    section_time = sim_row.get(
        "historical_section_time",
        None
    )

    if pd.notna(section_time):
        section_text = f"{float(section_time):.1f} min"
    else:
        section_text = "—"

    st.markdown(
        f"""
        <div class="condition-card">
            <div class="condition-label">Section time</div>
            <div class="condition-value">
                {section_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WHY DID THE PREDICTION CHANGE?
# ============================================================

top_feature, direction = explain_row(sim_row)


if view_mode == "Passenger":

    st.markdown(
        """
        <div class="explanation-panel">

            <div class="explanation-title">
                Why this prediction?
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

            Weather condition:
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
    'Record the actual delay experienced by the passenger. '
    'This creates a feedback dataset for future model evaluation and retraining.'
    '</div>',
    unsafe_allow_html=True
)

feedback_col1, feedback_col2 = st.columns([1, 2])

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
            [{
                "train": selected_train,
                "station": current_row["station"],
                "predicted_delay": predicted_delay,
                "actual_delay": actual_delay_input,
                "submitted_at": pd.Timestamp.now()
            }]
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
        font-size:0.76rem;
        padding:8px 0 15px 0;
    ">
        RailCast · Dynamic ETA Forecasting Prototype
        &nbsp;•&nbsp;
        Predict → Explain → Simulate → Learn
    </div>
    """,
    unsafe_allow_html=True
)
