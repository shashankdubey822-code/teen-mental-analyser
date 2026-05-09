import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd


FEATURE_LABELS = {
    "sleep_deficit": "Sleep Deficit",
    "digital_overload": "Digital Overload",
    "night_risk": "Night Risk",
    "mental_pressure": "Mental Pressure",
    "isolation_score": "Social Isolation",
    "addiction_impact": "Addiction Impact",
    "lifestyle_imbalance": "Lifestyle Imbalance",
    "screen_time_before_sleep": "Screen Before Sleep",
    "sleep_hours": "Sleep Hours",
    "physical_activity": "Physical Activity",
}


def clamp(value, low, high):
    return max(low, min(value, high))


def load_base_estimator(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def get_key_drivers(base_model, X_row):
    try:
        import shap

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        contrib = pd.Series(shap_vals[0], index=X_row.columns).abs()
    except Exception:
        if hasattr(base_model, "feature_importances_"):
            importances = pd.Series(base_model.feature_importances_, index=X_row.columns)
            contrib = (importances * X_row.iloc[0].abs()).abs()
        else:
            contrib = pd.Series(dtype=float)

    if contrib.empty:
        return {"Sleep Deficit": 18, "Digital Overload": 22, "Mental Pressure": 15}

    top = contrib.sort_values(ascending=False).head(5)
    total = float(top.sum()) or 1.0
    drivers = {}
    for name, value in top.items():
        label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
        drivers[label] = int(round((value / total) * 100))
    return drivers


def build_suggestions(row):
    suggestions = []
    if row.get("sleep_hours", 8) < 7:
        suggestions.append("Improve sleep to 7-8 hours daily")
    if row.get("screen_time_before_sleep", 0) > 2:
        suggestions.append("Reduce screen time before bed")
    if row.get("digital_overload", 0) > 6:
        suggestions.append("Set daily screen limits and schedule offline breaks")
    if row.get("physical_activity", 0) < 1:
        suggestions.append("Increase physical activity to 30+ min/day")
    if row.get("social_interaction_level", 0) >= 1:
        suggestions.append("Schedule regular social interactions")
    if row.get("stress_level", 0) >= 6 or row.get("anxiety_level", 0) >= 6:
        suggestions.append("Use stress-reduction routines or seek support")
    return suggestions[:4]


def main():
    parser = argparse.ArgumentParser(description="Export dashboard data to JSON")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index from X_test.csv")
    parser.add_argument("--output", default="frontend/dashboard_data.json", help="Output JSON path")
    args = parser.parse_args()

    model = joblib.load("champion_model.pkl")
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    X_test = pd.read_csv("X_test.csv").reindex(columns=feature_columns)
    sample_idx = clamp(args.sample_index, 0, len(X_test) - 1)
    X_row = X_test.iloc[[sample_idx]]

    prob = float(model.predict_proba(X_row)[0][1])
    risk_score = int(round(prob * 100))
    risk_score = clamp(risk_score, 3, 97)

    sleep_deficit = float(X_row.get("sleep_deficit", pd.Series([0])).iloc[0])
    digital_overload = float(X_row.get("digital_overload", pd.Series([0])).iloc[0])
    addiction = float(X_row.get("addiction_level", pd.Series([0])).iloc[0])

    risk_level = "Moderate"
    if risk_score < 33:
        risk_level = "Low"
    elif risk_score >= 66:
        risk_level = "High"

    if sleep_deficit >= 2 and digital_overload >= 6 and addiction >= 7 and risk_level == "Low":
        risk_level = "Moderate"
        risk_score = max(risk_score, 55)

    base_model = load_base_estimator(model)
    key_drivers = get_key_drivers(base_model, X_row)

    mental_pressure = float(X_row.get("mental_pressure", pd.Series([0])).iloc[0])
    mental_pressure_score = clamp(int(round(mental_pressure / 2)), 1, 10)

    social_level = float(X_row.get("social_interaction_level", pd.Series([0])).iloc[0])
    social_engagement = clamp(int(round(10 - (social_level * 3))), 1, 10)

    dashboard = {
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "keyDrivers": key_drivers,
        "lifestyleMetrics": {
            "sleep": float(X_row.get("sleep_hours", pd.Series([8])).iloc[0]),
            "digitalUsage": float(X_row.get("digital_overload", pd.Series([0])).iloc[0]),
            "mentalPressure": mental_pressure_score,
            "physicalActivity": float(X_row.get("physical_activity", pd.Series([0])).iloc[0]),
            "socialEngagement": social_engagement,
        },
        "suggestions": build_suggestions(X_row.iloc[0].to_dict()),
        "trend": [clamp(risk_score + delta, 0, 100) for delta in (-18, -12, -8, -4, -2, 0, 2)],
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard data saved to {args.output}")


if __name__ == "__main__":
    main()
