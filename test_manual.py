import json

import pandas as pd
import joblib

from feature_engineering import FeatureEngineer, RAW_FEATURES

print("Loading Champion Model and Encoders...")
try:
    model = joblib.load("champion_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    try:
        with open("feature_fill_values.json", "r", encoding="utf-8") as f:
            fill_values = json.load(f)
    except FileNotFoundError:
        fill_values = {}
except FileNotFoundError:
    print("Error: Could not find model or artifacts. Run preprocess.py and train_models.py first!")
    exit()

print("\n" + "="*50)
print("🧠 TEEN MENTAL HEALTH ANALYZER - MANUAL TESTING")
print("="*50)

# 1. Collect inputs manually
try:
    age = int(input("Enter Age (e.g., 16): "))
    gender = input("Enter Gender (female/male): ").strip().lower()
    daily_social_media_hours = float(input("Daily Social Media Hours (e.g., 5.5): "))
    platform_usage = input("Platform Usage (Instagram/TikTok/Both): ").strip().capitalize()
    sleep_hours = float(input("Sleep Hours (e.g., 6.5): "))
    screen_time_before_sleep = float(input("Screen Time Before Sleep (hours): "))
    academic_performance = float(input("Academic Performance (e.g., 3.5): "))
    physical_activity = float(input("Physical Activity (hours): "))
    social_interaction_level = input("Social Interaction Level (low/medium/high): ").strip().lower()
    stress_level = int(input("Stress Level (1-10): "))
    anxiety_level = int(input("Anxiety Level (1-10): "))
    addiction_level = int(input("Addiction Level (1-10): "))
except Exception as e:
    print("Invalid input. Please enter the correct data types.")
    exit()

# 2. Add Outlier Capping (Keep inputs within realistic training bounds)
age = max(10, min(age, 25))
daily_social_media_hours = max(0.0, min(daily_social_media_hours, 10.0))
sleep_hours = max(0.0, min(sleep_hours, 12.0))
screen_time_before_sleep = max(0.0, min(screen_time_before_sleep, 4.0))
academic_performance = max(0.0, min(academic_performance, 4.0))
physical_activity = max(0.0, min(physical_activity, 4.0))
stress_level = max(1, min(stress_level, 10))
anxiety_level = max(1, min(anxiety_level, 10))
addiction_level = max(1, min(addiction_level, 10))

def encode_value(encoder, value):
    if hasattr(encoder, "transform"):
        return int(encoder.transform([value])[0])
    if isinstance(encoder, dict):
        if value not in encoder:
            raise ValueError(f"Unknown category: {value}")
        return int(encoder[value])
    raise TypeError("Unsupported encoder type")


# 3. Clean and Encode categorical data safely
try:
    # Auto-correct common typos
    if gender.startswith('m'):
        gender = 'male'
    elif gender.startswith('f'):
        gender = 'female'

    if 'inst' in platform_usage.lower():
        platform_usage = 'Instagram'
    elif 'tik' in platform_usage.lower():
        platform_usage = 'TikTok'
    else:
        platform_usage = 'Both'

    if social_interaction_level.startswith('m'):
        social_interaction_level = 'medium'
    elif social_interaction_level.startswith('l'):
        social_interaction_level = 'low'
    elif social_interaction_level.startswith('h'):
        social_interaction_level = 'high'

    # Encode
    gender_encoded = encode_value(label_encoders['gender'], gender)
    platform_encoded = encode_value(label_encoders['platform_usage'], platform_usage)
    social_encoded = encode_value(label_encoders['social_interaction_level'], social_interaction_level)
except ValueError as e:
    print(f"\nEncoding Error: {e}")
    print("Make sure your spelling matches the expected categories!")
    exit()

# 4. Create DataFrame with engineered features
input_data = pd.DataFrame([[
    age, gender_encoded, daily_social_media_hours, platform_encoded,
    sleep_hours, screen_time_before_sleep, academic_performance,
    physical_activity, social_encoded, stress_level, anxiety_level, addiction_level
]], columns=RAW_FEATURES)

feature_engineer = FeatureEngineer(fill_values=fill_values)
input_data = feature_engineer.transform(input_data)
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# 5. Predict (calibrated probability)
print("\nAnalyzing lifestyle data...")
probability = float(model.predict_proba(input_data)[0][1])
probability = max(min(probability, 0.97), 0.03)

sleep_deficit = float(input_data.loc[0, "sleep_deficit"])
digital_overload = float(input_data.loc[0, "digital_overload"])
addiction = float(input_data.loc[0, "addiction_level"])

if probability < 0.33:
    risk_level = "Low"
elif probability < 0.66:
    risk_level = "Moderate"
else:
    risk_level = "High"

# Real-world logic check: avoid low-risk on combined high-risk patterns
logic_flag = sleep_deficit >= 2 and digital_overload >= 6 and addiction >= 7
if logic_flag and risk_level == "Low":
    risk_level = "Moderate"
    probability = max(probability, 0.55)
if logic_flag and probability > 0.9:
    probability = 0.85

risk_score = probability * 100

def get_key_drivers(sample_df, model_obj):
    feature_name_map = {
        "sleep_deficit": "Sleep deficit",
        "digital_overload": "Digital overload",
        "night_risk": "Late-night screen + sleep loss",
        "mental_pressure": "Stress + anxiety",
        "isolation_score": "Low social interaction",
        "addiction_impact": "Addiction impact",
        "lifestyle_imbalance": "Lifestyle imbalance",
        "screen_time_before_sleep": "Screen before sleep",
        "sleep_hours": "Sleep hours",
        "physical_activity": "Physical activity",
    }

    try:
        import shap

        base_model = model_obj
        if hasattr(model_obj, "calibrated_classifiers_"):
            base_model = model_obj.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(sample_df)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        contrib = pd.Series(shap_vals[0], index=sample_df.columns).abs()
        top = contrib.sort_values(ascending=False).head(3).index.tolist()
        return [feature_name_map.get(name, name.replace("_", " ").title()) for name in top]
    except Exception:
        base_model = model_obj
        if hasattr(model_obj, "calibrated_classifiers_"):
            base_model = model_obj.calibrated_classifiers_[0].estimator

        if hasattr(base_model, "feature_importances_"):
            imp = pd.Series(base_model.feature_importances_, index=sample_df.columns)
            contrib = (imp * sample_df.iloc[0].abs()).abs()
            top = contrib.sort_values(ascending=False).head(3).index.tolist()
            return [feature_name_map.get(name, name.replace("_", " ").title()) for name in top]

    return ["Sleep deficit", "Digital overload", "Stress + anxiety"]


drivers = get_key_drivers(input_data, model)

suggestions = []
if sleep_hours < 7:
    suggestions.append("Increase sleep to 7-8 hours consistently")
if screen_time_before_sleep > 2:
    suggestions.append("Reduce screen exposure 60-90 minutes before bed")
if digital_overload > 5:
    suggestions.append("Set daily screen limits and schedule offline breaks")
if physical_activity < 1:
    suggestions.append("Add regular physical activity to the routine")
if social_encoded >= 1:
    suggestions.append("Increase offline social interaction")
if stress_level >= 6 or anxiety_level >= 6:
    suggestions.append("Use stress-reduction routines or seek support")

suggestions = suggestions[:4]

print("\n" + "="*50)
print("RISK SUMMARY")
print("="*50)
print(f"Risk Level : {risk_level}")
print(f"Risk Score : {risk_score:.1f}% (calibrated)")

print("\nKey Drivers:")
for item in drivers:
    print(f"- {item}")

if suggestions:
    print("\nSuggestions:")
    for item in suggestions:
        print(f"- {item}")

if logic_flag:
    print("\nLogic Check: Combined late-night screen time, sleep deficit, and addiction patterns detected")

print("="*50 + "\n")
