import json

import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from feature_engineering import FEATURE_GROUPS

SAMPLE_INDEX = 0


def load_base_estimator(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def evaluate_metrics(model, X_eval, y_eval):
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_eval, y_prob),
        "Brier": brier_score_loss(y_eval, y_prob),
    }, y_pred, y_prob


def get_feature_importance(base_model, feature_names):
    if hasattr(base_model, "feature_importances_"):
        return pd.Series(base_model.feature_importances_, index=feature_names)
    return None


def get_shap_importance(base_model, X_sample):
    try:
        import shap

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        mean_abs = pd.Series(abs(shap_vals).mean(axis=0), index=X_sample.columns)
        return mean_abs.sort_values(ascending=False)
    except Exception:
        return None


def get_local_drivers(base_model, X_row, fallback_importance):
    try:
        import shap

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        contrib = pd.Series(shap_vals[0], index=X_row.columns).abs()
        return contrib.sort_values(ascending=False).head(5).index.tolist()
    except Exception:
        if fallback_importance is not None:
            weights = (fallback_importance * X_row.iloc[0].abs()).abs()
            return weights.sort_values(ascending=False).head(5).index.tolist()
    return []


print("Loading artifacts...")
model = joblib.load("champion_model.pkl")

with open("feature_columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

X_test = pd.read_csv("X_test.csv").reindex(columns=feature_columns)
y_test = pd.read_csv("y_test.csv").values.ravel()

champion_name = "Champion"
try:
    with open("champion_model_name.txt", "r", encoding="utf-8") as f:
        champion_name = f.read().strip() or "Champion"
except FileNotFoundError:
    pass

print("\n" + "="*60)
print(f"EVALUATION REPORT — {champion_name}")
print("="*60)

metrics, y_pred, y_prob = evaluate_metrics(model, X_test, y_test)
print(pd.Series(metrics).round(4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if pd.io.common.file_exists("model_comparison_results.csv"):
    print("\nModel Comparison (from training run):")
    print(pd.read_csv("model_comparison_results.csv").round(4))

base_model = load_base_estimator(model)
fi = get_feature_importance(base_model, feature_columns)
shap_global = get_shap_importance(base_model, X_test)

print("\n" + "="*60)
print("GLOBAL EXPLANATION")
print("="*60)

if shap_global is not None:
    print("Top Features (SHAP):")
    print(shap_global.head(10).round(4))
elif fi is not None:
    print("Top Features (Model Importance):")
    print(fi.sort_values(ascending=False).head(10).round(4))
else:
    print("No feature importance available for this model.")

if fi is not None:
    print("\nFeature Group Summary (Importance Sum):")
    group_scores = {}
    for group, features in FEATURE_GROUPS.items():
        group_scores[group] = float(fi[fi.index.isin(features)].sum())
    print(pd.Series(group_scores).sort_values(ascending=False).round(4))

print("\n" + "="*60)
print("LOCAL EXPLANATION (SAMPLE)")
print("="*60)

sample_idx = min(SAMPLE_INDEX, len(X_test) - 1)
X_row = X_test.iloc[[sample_idx]]
local_drivers = get_local_drivers(base_model, X_row, fi)

prob = float(model.predict_proba(X_row)[0][1])
if prob < 0.33:
    risk_level = "Low"
elif prob < 0.66:
    risk_level = "Moderate"
else:
    risk_level = "High"

print(f"Sample Index: {sample_idx}")
print(f"Risk Level: {risk_level}")
print(f"Risk Score: {prob*100:.1f}% (calibrated)")
if local_drivers:
    print("Key Drivers:")
    for name in local_drivers:
        print(f"- {name}")
else:
    print("Key Drivers: unavailable")

sleep_deficit = float(X_row["sleep_deficit"].iloc[0]) if "sleep_deficit" in X_row.columns else 0
digital_overload = float(X_row["digital_overload"].iloc[0]) if "digital_overload" in X_row.columns else 0
addiction = float(X_row["addiction_level"].iloc[0]) if "addiction_level" in X_row.columns else 0

if sleep_deficit >= 2 and digital_overload >= 6 and addiction >= 7:
    print("Logic Check: high-risk combo detected (sleep deficit + digital overload + addiction)")
