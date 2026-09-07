import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# ---------- Load everything saved from Colab ----------
xgb_model = joblib.load("xgb_model.pkl")
residual_model = joblib.load("residual_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
residual_features = joblib.load("residual_features.pkl")
demo_data = pd.read_csv("demo_data.csv")

XGB_MAE = 7.88  # confirmed headline result (verified against full 286,070-record test set) — used for the confidence range

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
    # Force numeric dtype before building the DMatrix — a single-row Series pulled
    # from a mixed-type DataFrame defaults to dtype=object, which XGBoost's raw
    # DMatrix constructor rejects even though the values themselves are numbers.
    row_df = feature_row[residual_features].to_frame().T.apply(pd.to_numeric)
    dmat = xgb.DMatrix(row_df, feature_names=residual_features)
    contribs = residual_model.get_booster().predict(dmat, pred_contribs=True)[0][:-1]
    top_idx = abs(contribs).argmax()
    top_feature = residual_features[top_idx]
    direction = "increasing the delay" if contribs[top_idx] > 0 else "reducing the delay"
    return readable_names.get(top_feature, top_feature), direction

# ---------- App layout ----------
st.set_page_config(page_title="Train ETA Predictor", layout="centered")
st.title("🚆 Dynamic ETA Forecasting")

view_mode = st.sidebar.radio("Dashboard view", ["Passenger", "Control Room / Officer"])

train_options = demo_data["train"].unique()
selected_train = st.selectbox("Select a train", train_options)

train_rows = demo_data[demo_data["train"] == selected_train].reset_index(drop=True)
row_index = st.selectbox(
    "Select journey point",
    train_rows.index,
    format_func=lambda i: f'{train_rows.loc[i, "station"]} → {train_rows.loc[i, "next_station"]} ({train_rows.loc[i, "date"]})'
)
current_row = train_rows.loc[row_index].copy()

st.divider()

# ---------- Disruption injection (before prediction, so it affects the result) ----------
st.subheader("⚠️ Simulate a disruption")
disruption = st.selectbox(
    "Inject a live event",
    ["None", "Fog", "Heavy Rain / Storm", "Speed Restriction", "Signal Halt / Unscheduled Stoppage", "Track Congestion Spike", "Unscheduled Maintenance Block"]
)

sim_row = current_row.copy()
disruption_note = ""
sim_weather_label = current_row["weather_condition_passenger"]  # default: unchanged, real recorded weather

if disruption == "Fog":
    sim_row["visibility_m"] = 150
    sim_row["temperature_c"] = sim_row["temperature_c"] - 3
    disruption_note = "Visibility dropped to 150m (dense fog conditions)"
    sim_weather_label = "Foggy"
elif disruption == "Heavy Rain / Storm":
    sim_row["rainfall_mm"] = 80
    disruption_note = "Rainfall spiked to 80mm (storm-level rainfall)"
    sim_weather_label = "Heavy Rain / Storm"
elif disruption == "Speed Restriction":
    sim_row["historical_section_time"] = sim_row["historical_section_time"] * 1.5
    disruption_note = "Section running time increased 50% (temporary speed restriction)"
    sim_weather_label = "Speed Restriction in effect"
elif disruption == "Signal Halt / Unscheduled Stoppage":
    sim_row["delay_minutes"] = sim_row["delay_minutes"] + 25
    disruption_note = "Train held 25 extra minutes at a signal"
    sim_weather_label = "Signal Halt in effect"
elif disruption == "Track Congestion Spike":
    sim_row["congestion_score"] = 0.95
    disruption_note = "Downstream section congestion spiked to near-maximum"
    sim_weather_label = "Track Congestion Spike"
elif disruption == "Unscheduled Maintenance Block":
    sim_row["delay_minutes"] = sim_row["delay_minutes"] + 45
    disruption_note = "Unscheduled maintenance block adding 45 minutes"
    sim_weather_label = "Maintenance Block in effect"

# ---------- Predict: baseline (no disruption) vs current selection ----------
X_input = pd.DataFrame([sim_row[feature_columns]]).apply(pd.to_numeric)
predicted_delay = xgb_model.predict(X_input)[0]

X_original = pd.DataFrame([current_row[feature_columns]]).apply(pd.to_numeric)
original_predicted_delay = xgb_model.predict(X_original)[0]

st.divider()
st.subheader("📍 Prediction")

col1, col2 = st.columns(2)
col1.metric("Predicted delay (normal)", f"{original_predicted_delay:.1f} min")
col2.metric(
    "Predicted delay (with disruption)" if disruption != "None" else "Predicted delay",
    f"{predicted_delay:.1f} min",
    delta=f"{predicted_delay - original_predicted_delay:+.1f} min" if disruption != "None" else None
)

if disruption != "None":
    st.info(f"Simulated event: {disruption_note}")

st.write(f"**Confidence range:** {predicted_delay - XGB_MAE:.0f} to {predicted_delay + XGB_MAE:.0f} minutes")

# ---------- Explanation, and dual-audience view ----------
top_feature, direction = explain_row(sim_row)

if view_mode == "Passenger":
    st.subheader("ℹ️ Why this prediction?")
    st.write(f"Mainly driven by: **{top_feature}** ({direction})")
    st.write(f"Current weather: **{sim_weather_label}**")
else:
    st.subheader("🔧 Officer detail view")
    st.write(f"Mainly driven by: **{top_feature}** ({direction})")
    st.write(f"Weather (detailed): **{sim_weather_label}**")
    st.write(f"Congestion score: **{sim_row['congestion_score']:.2f}**")
    st.write(f"Historical section time: **{sim_row['historical_section_time']:.1f} min**")
    st.write(f"Route position (station sequence): **{sim_row['station_sequence']}**")

# ---------- Feedback (sets up the retraining demo we'll add after this) ----------
st.divider()
st.subheader("📝 Passenger feedback")
actual_delay_input = st.number_input("Actual delay experienced (minutes)", value=0)
if st.button("Submit feedback"):
    feedback_row = pd.DataFrame([{
        "train": selected_train,
        "station": current_row["station"],
        "predicted_delay": predicted_delay,
        "actual_delay": actual_delay_input,
        "submitted_at": pd.Timestamp.now()
    }])
    feedback_row.to_csv("passenger_feedback.csv", mode="a",
                         header=not __import__("os").path.exists("passenger_feedback.csv"), index=False)
    st.success("Thank you — feedback recorded.")