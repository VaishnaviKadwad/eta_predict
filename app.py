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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RailCast | Dynamic ETA Intelligence",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FEATURE NAMES
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
# EXPLAINABILITY
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
# CUSTOM CSS
#
# ONLY styling here.
# NO HTML CARDS ARE USED ANYWHERE ELSE.
# ============================================================

st.markdown(
    """
    <style>

    /* Overall background */
    .stApp {
        background-color: #0b1118;
    }

    /* Main content */
    .block-container {
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #080d13;
        border-right: 1px solid #202a34;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        font-size: 0.92rem;
    }

    /* Main headings */
    h1, h2, h3 {
        letter-spacing: -0.3px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #111923;
        border: 1px solid #293540;
        border-radius: 10px;
        padding: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #111923;
        border-color: #303c47;
    }

    /* Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #101820;
        border-color: #293540;
        border-radius: 10px;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div {
        border-radius: 10px;
    }

    /* Hide footer */
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

st.sidebar.title("🚆 RailCast")

st.sidebar.caption(
    "Dynamic ETA & Delay Intelligence"
)

st.sidebar.divider()

st.sidebar.subheader("Dashboard")

view_mode = st.sidebar.radio(
    "Choose dashboard",
    [
        "Passenger",
        "Control Room / Officer"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Model")

st.sidebar.metric(
    "Validation MAE",
    f"{XGB_MAE:.2f} min"
)

st.sidebar.caption(
    "XGBoost prediction engine"
)

st.sidebar.caption(
    "Historical data + operating-condition features"
)

st.sidebar.divider()

st.sidebar.info(
    "Decision-support prototype. "
    "Operational actions remain with authorised railway staff."
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🚆 RailCast")

st.caption(
    "Dynamic ETA & Delay Intelligence  •  "
    "Predict the arrival. Understand the delay."
)


# ============================================================
# JOURNEY SELECTION
# ============================================================

st.header("Journey")

st.caption(
    "Select a train and a point along its journey."
)

select1, select2 = st.columns(
    [1, 2]
)


with select1:

    train_options = (
        demo_data["train"]
        .dropna()
        .unique()
    )

    selected_train = st.selectbox(
        "Train",
        train_options
    )


# Filter train

train_rows = (
    demo_data[
        demo_data["train"] == selected_train
    ]
    .reset_index(drop=True)
)


with select2:

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
# CURRENT ROW
# ============================================================

current_row = (
    train_rows
    .loc[row_index]
    .copy()
)


# ============================================================
# CURRENT JOURNEY SUMMARY
# ============================================================

st.subheader("Current journey")

j1, j2, j3 = st.columns(3)


with j1:

    st.metric(
        "Train",
        f"🚆 {selected_train}"
    )


with j2:

    st.metric(
        "Current section",
        f'{current_row["station"]} → '
        f'{current_row["next_station"]}'
    )


with j3:

    st.metric(
        "Journey date",
        str(current_row["date"])
    )


# ============================================================
# ROUTE PROGRESS
# ============================================================

st.subheader("Route progress")

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
        and pd.notna(max_sequence)
        and max_sequence > min_sequence
    ):

        progress = (
            current_position - min_sequence
        ) / (
            max_sequence - min_sequence
        )

    else:

        progress = 0.5

except Exception:

    progress = 0.5


progress = max(
    0.0,
    min(1.0, progress)
)


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


r1, r2, r3 = st.columns(
    [1, 6, 1]
)


with r1:

    st.write(
        f"**● {first_station}**"
    )


with r2:

    st.progress(
        progress
    )

    st.caption(
        f"Current: **{current_row['station']} → "
        f"{current_row['next_station']}**  "
        f"•  {progress * 100:.0f}% along selected route"
    )


with r3:

    st.write(
        f"**{last_station} ●**"
    )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

st.divider()

st.header("What-if analysis")

st.caption(
    "Introduce a disruption and see how the predicted delay changes."
)


disruption = st.selectbox(
    "Operating scenario",
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
# SIMULATED ROW
# ============================================================

sim_row = current_row.copy()

disruption_note = ""

sim_weather_label = (
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
        "Visibility reduced to 150 m, "
        "representing dense fog conditions."
    )

    sim_weather_label = "Foggy"


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80 mm, "
        "representing storm-level rainfall."
    )

    sim_weather_label = "Heavy Rain / Storm"


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"] * 1.5
    )

    disruption_note = (
        "Expected section running time increased "
        "by 50% because of a temporary speed restriction."
    )

    sim_weather_label = (
        "Speed Restriction in effect"
    )


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 25
    )

    disruption_note = (
        "An additional 25-minute signal halt "
        "was introduced."
    )

    sim_weather_label = (
        "Signal Halt in effect"
    )


elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    disruption_note = (
        "Downstream congestion increased "
        "to a near-maximum level."
    )

    sim_weather_label = (
        "Track Congestion Spike"
    )


elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"] + 45
    )

    disruption_note = (
        "An unscheduled maintenance block "
        "adding 45 minutes was introduced."
    )

    sim_weather_label = (
        "Maintenance Block in effect"
    )


# ============================================================
# PREDICTION
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

original_predicted_delay = float(
    xgb_model.predict(
        X_original
    )[0]
)


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

predicted_delay = float(
    xgb_model.predict(
        X_input
    )[0]
)


delay_change = (
    predicted_delay
    -
    original_predicted_delay
)


# ============================================================
# SCENARIO MESSAGE
# ============================================================

if disruption != "None":

    st.warning(
        f"**{disruption}**  \n"
        f"{disruption_note}"
    )


# ============================================================
# ETA FORECAST
# ============================================================

st.header("ETA forecast")


p1, p2, p3 = st.columns(3)


with p1:

    st.metric(
        "Normal conditions",
        f"{original_predicted_delay:.1f} min"
    )

    st.caption(
        "Predicted delay"
    )


with p2:

    if disruption != "None":

        st.metric(
            "With disruption",
            f"{predicted_delay:.1f} min",
            delta=f"{delay_change:+.1f} min"
        )

    else:

        st.metric(
            "Current prediction",
            f"{predicted_delay:.1f} min"
        )

    st.caption(
        "Predicted delay"
    )


with p3:

    lower_range = (
        predicted_delay - XGB_MAE
    )

    upper_range = (
        predicted_delay + XGB_MAE
    )

    st.metric(
        "Estimated range",
        f"{lower_range:.0f}–{upper_range:.0f} min"
    )

    st.caption(
        f"Based on validation MAE ±{XGB_MAE:.2f} min"
    )


# ============================================================
# DELAY IMPACT GRAPH
# ============================================================

if disruption != "None":

    st.subheader("Delay impact")

    chart_data = pd.DataFrame(
        {
            "Forecast": [
                original_predicted_delay,
                predicted_delay
            ]
        },
        index=[
            "Normal",
            "Scenario"
        ]
    )

    st.bar_chart(
        chart_data,
        height=230
    )

    if delay_change > 0:

        st.error(
            f"The selected disruption increases the "
            f"forecast by **{delay_change:.1f} minutes**."
        )

    elif delay_change < 0:

        st.success(
            f"The selected scenario reduces the "
            f"forecast by **{abs(delay_change):.1f} minutes**."
        )

    else:

        st.info(
            "The selected scenario does not change "
            "the model's forecast for this journey point."
        )


# ============================================================
# OPERATING CONDITIONS
# ============================================================

st.subheader("Operating conditions")

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Weather",
        str(sim_weather_label)
    )


with c2:

    visibility = sim_row.get(
        "visibility_m",
        None
    )

    if pd.notna(visibility):

        st.metric(
            "Visibility",
            f"{float(visibility):.0f} m"
        )

    else:

        st.metric(
            "Visibility",
            "—"
        )


with c3:

    congestion = sim_row.get(
        "congestion_score",
        None
    )

    if pd.notna(congestion):

        st.metric(
            "Congestion",
            f"{float(congestion):.2f}"
        )

    else:

        st.metric(
            "Congestion",
            "—"
        )


with c4:

    section_time = sim_row.get(
        "historical_section_time",
        None
    )

    if pd.notna(section_time):

        st.metric(
            "Section time",
            f"{float(section_time):.1f} min"
        )

    else:

        st.metric(
            "Section time",
            "—"
        )


# ============================================================
# RISK INDICATOR
# ============================================================

st.subheader("Delay risk")


if predicted_delay < 5:

    risk_level = "LOW"

    st.success(
        f"🟢 **{risk_level} RISK** — "
        "The current forecast indicates a relatively small delay."
    )

elif predicted_delay < 15:

    risk_level = "MODERATE"

    st.warning(
        f"🟡 **{risk_level} RISK** — "
        "The forecast indicates a noticeable delay."
    )

else:

    risk_level = "HIGH"

    st.error(
        f"🔴 **{risk_level} RISK** — "
        "The forecast indicates a significant delay."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

top_feature, direction = explain_row(
    sim_row
)


st.subheader("Why this prediction?")


if view_mode == "Passenger":

    with st.container(border=True):

        st.write(
            f"The forecast is mainly influenced by "
            f"**{top_feature}**, which is currently "
            f"**{direction}**."
        )

        st.caption(
            f"Current operating condition: {sim_weather_label}"
        )


else:

    with st.container(border=True):

        st.write(
            f"### 🔧 Control-room analysis"
        )

        st.write(
            f"Primary contributing feature: "
            f"**{top_feature}** "
            f"({direction})."
        )

        officer1, officer2, officer3 = st.columns(3)

        with officer1:

            st.metric(
                "Congestion score",
                f'{float(sim_row["congestion_score"]):.2f}'
            )

        with officer2:

            st.metric(
                "Historical section time",
                f'{float(sim_row["historical_section_time"]):.1f} min'
            )

        with officer3:

            st.metric(
                "Route position",
                str(sim_row["station_sequence"])
            )

        st.caption(
            f"Operating condition: {sim_weather_label}"
        )


# ============================================================
# PASSENGER FEEDBACK
# ============================================================

st.divider()

st.header("Prediction feedback")

st.caption(
    "Record the actual delay experienced. "
    "This creates a feedback dataset for future evaluation "
    "and model improvement."
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

st.caption(
    "RailCast · Dynamic ETA Forecasting Prototype "
    "• Predict → Explain → Simulate → Learn"
)
