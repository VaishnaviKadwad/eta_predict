import os

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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
# LOAD MODEL + DATA
# ============================================================

xgb_model = joblib.load("xgb_model.pkl")
residual_model = joblib.load("residual_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
residual_features = joblib.load("residual_features.pkl")

demo_data = pd.read_csv("demo_data.csv")

XGB_MAE = 7.88


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
# HELPERS
# ============================================================

def find_column(dataframe, candidates):
    """
    Finds the first matching column from a list of possible
    coordinate column names.
    """

    lower_map = {
        str(col).lower(): col
        for col in dataframe.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def get_coordinate_columns(dataframe):

    latitude_candidates = [
        "latitude",
        "lat",
        "station_lat",
        "station_latitude",
        "station_latitude_deg",
        "lat_deg"
    ]

    longitude_candidates = [
        "longitude",
        "lon",
        "lng",
        "station_lon",
        "station_longitude",
        "station_longitude_deg",
        "lon_deg"
    ]

    lat_col = find_column(
        dataframe,
        latitude_candidates
    )

    lon_col = find_column(
        dataframe,
        longitude_candidates
    )

    return lat_col, lon_col


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


def get_risk(delay):

    if delay < 5:
        return "LOW"

    if delay < 15:
        return "MODERATE"

    return "HIGH"


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
    "View",
    [
        "Passenger",
        "Control Room / Officer"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Model reference")

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

st.sidebar.caption(
    "Decision-support prototype. "
    "Operational actions remain with authorised railway staff."
)


# ============================================================
# HEADER
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
    "Choose a train and the section of its journey."
)

col_train, col_journey = st.columns(
    [1, 2]
)


with col_train:

    train_options = (
        demo_data["train"]
        .dropna()
        .unique()
    )

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


with col_journey:

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


current_row = (
    train_rows
    .loc[row_index]
    .copy()
)


# ============================================================
# CURRENT JOURNEY
# ============================================================

st.subheader("Current journey")

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Train",
        str(selected_train)
    )


with c2:

    st.metric(
        "Current station",
        str(current_row["station"])
    )


with c3:

    st.metric(
        "Next station",
        str(current_row["next_station"])
    )


with c4:

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
    min(
        1.0,
        progress
    )
)


route_stations = (
    train_rows["station"]
    .dropna()
    .astype(str)
    .tolist()
)


if route_stations:

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
        f"{progress * 100:.0f}% of selected route"
    )


with r3:

    st.write(
        f"**{last_station} ●**"
    )


# ============================================================
# MAP
# ============================================================

st.divider()

st.header("🗺️ Route map")

lat_col, lon_col = get_coordinate_columns(
    train_rows
)


if (
    lat_col is not None
    and lon_col is not None
):

    map_data = train_rows.copy()

    map_data["latitude"] = pd.to_numeric(
        map_data[lat_col],
        errors="coerce"
    )

    map_data["longitude"] = pd.to_numeric(
        map_data[lon_col],
        errors="coerce"
    )

    map_data = map_data.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    if len(map_data) > 0:

        if PLOTLY_AVAILABLE:

            fig = go.Figure()


            # ------------------------------------------------
            # ROUTE LINE
            # ------------------------------------------------

            fig.add_trace(
                go.Scattermap(
                    lat=map_data["latitude"],
                    lon=map_data["longitude"],
                    mode="lines",
                    line=dict(
                        width=4
                    ),
                    name="Train route",
                    hoverinfo="skip"
                )
            )


            # ------------------------------------------------
            # STATIONS
            # ------------------------------------------------

            fig.add_trace(
                go.Scattermap(
                    lat=map_data["latitude"],
                    lon=map_data["longitude"],
                    mode="markers+text",
                    marker=dict(
                        size=8
                    ),
                    text=map_data["station"],
                    textposition="top center",
                    name="Stations",
                    customdata=map_data[
                        ["station"]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>"
                        "<extra></extra>"
                    )
                )
            )


            # ------------------------------------------------
            # CURRENT TRAIN POSITION
            # ------------------------------------------------

            current_map = map_data[
                map_data["station"]
                ==
                current_row["station"]
            ]


            if len(current_map) == 0:

                current_map = map_data[
                    map_data.index
                    ==
                    map_data.index[
                        min(
                            row_index,
                            len(map_data) - 1
                        )
                    ]
                ]


            if len(current_map) > 0:

                fig.add_trace(
                    go.Scattermap(
                        lat=current_map["latitude"],
                        lon=current_map["longitude"],
                        mode="markers",
                        marker=dict(
                            size=18
                        ),
                        name="Current train",
                        hovertemplate=(
                            "<b>🚆 Current train</b>"
                            "<br>%{lat:.4f}, %{lon:.4f}"
                            "<extra></extra>"
                        )
                    )
                )


            # ------------------------------------------------
            # MAP LAYOUT
            # ------------------------------------------------

            center_lat = map_data[
                "latitude"
            ].mean()

            center_lon = map_data[
                "longitude"
            ].mean()


            fig.update_layout(
                map=dict(
                    style="open-street-map",
                    center=dict(
                        lat=center_lat,
                        lon=center_lon
                    ),
                    zoom=5
                ),
                height=500,
                margin=dict(
                    l=0,
                    r=0,
                    t=0,
                    b=0
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.01,
                    xanchor="left",
                    x=0.01
                )
            )


            st.plotly_chart(
                fig,
                width="stretch",
                theme="streamlit"
            )


        else:

            st.warning(
                "Plotly is not installed. "
                "Add plotly to requirements.txt to enable "
                "the interactive route map."
            )

    else:

        st.info(
            "Station coordinates were found, but no valid "
            "coordinates are available for this train."
        )


else:

    st.info(
        "Interactive map will appear when the dataset "
        "contains station latitude and longitude columns."
    )

    st.caption(
        "Expected examples: latitude/longitude, "
        "station_lat/station_lon or "
        "station_latitude/station_longitude."
    )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

st.divider()

st.header("⚠️ What-if disruption simulator")

st.caption(
    "Test how an operating disruption could affect the ETA."
)


disruption = st.selectbox(
    "Simulated event",
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
# SIMULATION
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
        "Visibility reduced to 150 m."
    )

    sim_weather_label = "Foggy"


elif disruption == "Heavy Rain / Storm":

    sim_row["rainfall_mm"] = 80

    disruption_note = (
        "Rainfall increased to 80 mm."
    )

    sim_weather_label = "Heavy Rain / Storm"


elif disruption == "Speed Restriction":

    sim_row["historical_section_time"] = (
        sim_row["historical_section_time"]
        * 1.5
    )

    disruption_note = (
        "Expected section running time increased by 50%."
    )

    sim_weather_label = (
        "Speed restriction in effect"
    )


elif disruption == "Signal Halt / Unscheduled Stoppage":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"]
        + 25
    )

    disruption_note = (
        "Additional 25-minute signal halt introduced."
    )

    sim_weather_label = (
        "Signal halt in effect"
    )


elif disruption == "Track Congestion Spike":

    sim_row["congestion_score"] = 0.95

    disruption_note = (
        "Downstream congestion increased to near maximum."
    )

    sim_weather_label = (
        "Track congestion spike"
    )


elif disruption == "Unscheduled Maintenance Block":

    sim_row["delay_minutes"] = (
        sim_row["delay_minutes"]
        + 45
    )

    disruption_note = (
        "Unscheduled maintenance block adding 45 minutes."
    )

    sim_weather_label = (
        "Maintenance block in effect"
    )


if disruption != "None":

    st.warning(
        f"**{disruption}**  \n"
        f"{disruption_note}"
    )


# ============================================================
# PREDICTIONS
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
# ETA FORECAST
# ============================================================

st.header("ETA forecast")

p1, p2, p3 = st.columns(3)


with p1:

    st.metric(
        "Normal conditions",
        f"{original_predicted_delay:.1f} min"
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


with p3:

    lower = predicted_delay - XGB_MAE
    upper = predicted_delay + XGB_MAE

    st.metric(
        "Estimated range",
        f"{lower:.0f}–{upper:.0f} min"
    )


# ============================================================
# DELAY IMPACT GRAPH
# ============================================================

if disruption != "None":

    st.subheader("Delay impact")

    if PLOTLY_AVAILABLE:

        graph = go.Figure()

        graph.add_bar(
            x=[
                "Normal",
                "Scenario"
            ],
            y=[
                original_predicted_delay,
                predicted_delay
            ],
            text=[
                f"{original_predicted_delay:.1f} min",
                f"{predicted_delay:.1f} min"
            ],
            textposition="outside"
        )

        graph.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            yaxis_title="Predicted delay (minutes)",
            xaxis_title=""
        )

        st.plotly_chart(
            graph,
            width="stretch",
            theme="streamlit"
        )

    else:

        chart_df = pd.DataFrame(
            {
                "Predicted delay": [
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
            chart_df
        )


# ============================================================
# OPERATING CONDITIONS
# ============================================================

st.subheader("Operating conditions")

o1, o2, o3, o4 = st.columns(4)


with o1:

    st.metric(
        "Weather",
        str(sim_weather_label)
    )


with o2:

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


with o3:

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


with o4:

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
# DELAY RISK
# ============================================================

st.subheader("Delay risk")

risk = get_risk(
    predicted_delay
)


if risk == "LOW":

    st.success(
        "🟢 **LOW RISK** — "
        "The forecast indicates a relatively small delay."
    )


elif risk == "MODERATE":

    st.warning(
        "🟡 **MODERATE RISK** — "
        "The forecast indicates a noticeable delay."
    )


else:

    st.error(
        "🔴 **HIGH RISK** — "
        "The forecast indicates a significant delay."
    )


# ============================================================
# WHY THIS PREDICTION?
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
            f"Current condition: {sim_weather_label}"
        )


else:

    with st.container(border=True):

        st.write(
            "### 🔧 Control-room analysis"
        )

        st.write(
            f"Primary contributing feature: "
            f"**{top_feature}** "
            f"({direction})."
        )

        a1, a2, a3 = st.columns(3)


        with a1:

            st.metric(
                "Congestion",
                f'{float(sim_row["congestion_score"]):.2f}'
            )


        with a2:

            st.metric(
                "Section time",
                f'{float(sim_row["historical_section_time"]):.1f} min'
            )


        with a3:

            st.metric(
                "Route position",
                str(sim_row["station_sequence"])
            )


# ============================================================
# STATION TIMELINE
# ============================================================

st.divider()

st.header("Station timeline")

timeline_cols = [
    "station",
    "next_station",
    "station_sequence"
]

timeline_available = all(
    col in train_rows.columns
    for col in timeline_cols
)


if timeline_available:

    timeline = train_rows[
        timeline_cols
    ].copy()

    timeline = timeline.sort_values(
        "station_sequence"
    )

    timeline = timeline.reset_index(
        drop=True
    )

    timeline["Status"] = timeline[
        "station"
    ].apply(
        lambda x:
            "Current"
            if str(x)
            ==
            str(current_row["station"])
            else "Upcoming"
    )

    st.dataframe(
        timeline[
            [
                "station",
                "next_station",
                "station_sequence",
                "Status"
            ]
        ],
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "Station sequence information is not available "
        "for a timeline view."
    )


# ============================================================
# PASSENGER FEEDBACK
# ============================================================

st.divider()

st.header("Prediction feedback")

st.caption(
    "Record the actual delay experienced. "
    "This creates a feedback dataset for future evaluation."
)


feedback_col1, feedback_col2 = st.columns(
    [1, 2]
)


with feedback_col1:

    actual_delay_input = st.number_input(
        "Actual delay (minutes)",
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
    "RailCast • Predict → Explain → Simulate → Learn"
)
