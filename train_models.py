import json

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Load the preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

with open('feature_columns.json', 'r', encoding='utf-8') as f:
    feature_columns = json.load(f)
X_train = X_train.reindex(columns=feature_columns)
X_test = X_test.reindex(columns=feature_columns)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Monotonic constraints for behavior-aligned learning
mono_map = {
    'daily_social_media_hours': 1,
    'sleep_hours': -1,
    'screen_time_before_sleep': 1,
    'academic_performance': -1,
    'physical_activity': -1,
    'social_interaction_level': 1,
    'stress_level': 1,
    'anxiety_level': 1,
    'addiction_level': 1,
    'sleep_deficit': 1,
    'digital_overload': 1,
    'night_risk': 1,
    'mental_pressure': 1,
    'isolation_score': 1,
    'addiction_impact': 1,
    'lifestyle_imbalance': 1,
}
mono_constraints = "(" + ",".join(str(mono_map.get(col, 0)) for col in feature_columns) + ")"

calibration_method = "isotonic"

def evaluate_model(model, X_eval, y_eval):
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1-Score": f1_score(y_eval, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_eval, y_prob),
        "Brier": brier_score_loss(y_eval, y_prob),
    }


def tune_model(name, base_model, param_distributions):
    print(f"\nTuning {name} with RandomizedSearchCV...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(f"tuning_results_{name}.csv", index=False)
    print(f"  Best CV F1: {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")
    return search.best_estimator_, search.best_score_


# 3. Baseline models
baseline_models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0, iterations=400, learning_rate=0.05),
}

baseline_cv = {}
for name, model in baseline_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    baseline_cv[name] = scores.mean()

# 4. Hyperparameter tuning
xgb_base = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    monotone_constraints=mono_constraints,
)
xgb_params = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
}

lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
lgbm_params = {
    "n_estimators": [200, 400, 600],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 6, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [20, 40],
}

best_xgb, xgb_cv = tune_model("xgboost", xgb_base, xgb_params)
best_lgbm, lgbm_cv = tune_model("lightgbm", lgbm_base, lgbm_params)

# 5. Calibrate and evaluate all models
models = {
    "Random Forest": baseline_models["Random Forest"],
    "CatBoost": baseline_models["CatBoost"],
    "XGBoost (Tuned)": best_xgb,
    "LightGBM (Tuned)": best_lgbm,
}

results = []
calibrated_models = {}

print("\nStarting Model Training and Evaluation...\n")
for name, model in models.items():
    print(f"Training {name}...")
    calibrated = CalibratedClassifierCV(model, method=calibration_method, cv=3)
    calibrated.fit(X_train, y_train)
    calibrated_models[name] = calibrated

    metrics = evaluate_model(calibrated, X_test, y_test)
    metrics["Model"] = name
    metrics["CV_F1"] = (
        xgb_cv if name == "XGBoost (Tuned)" else
        lgbm_cv if name == "LightGBM (Tuned)" else
        baseline_cv.get(name)
    )
    results.append(metrics)
    print(f"✓ {name} completed.\n")

# 6. Display results
results_df = pd.DataFrame(results).set_index("Model")
print("="*60)
print("FINAL MODEL EVALUATION RESULTS (TEST SET)")
print("="*60)
print(results_df.round(4))
print("="*60)

results_df.to_csv("model_comparison_results.csv")
print("\nResults saved to 'model_comparison_results.csv'")

champion_name = results_df.sort_values(["F1-Score", "ROC-AUC"], ascending=False).index[0]
champion_model = calibrated_models[champion_name]

print("\n" + "="*60)
print(f"CHAMPION MODEL (CALIBRATED): {champion_name}")
print("="*60)
champion_prob = champion_model.predict_proba(X_test)[:, 1]
champion_pred = (champion_prob >= 0.5).astype(int)
print(classification_report(y_test, champion_pred, target_names=["Low", "High"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, champion_pred))
print("="*60)

# 7. Save the calibrated champion model
import joblib
joblib.dump(champion_model, "champion_model.pkl")
with open("champion_model_name.txt", "w", encoding="utf-8") as f:
    f.write(champion_name)
print("Champion Model saved to 'champion_model.pkl'")
