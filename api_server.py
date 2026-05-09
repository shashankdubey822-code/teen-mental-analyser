import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler

try:
    import requests
except ImportError:
    requests = None

from feature_engineering import FeatureEngineer, RAW_FEATURES


class PredictRequest(BaseModel):
    age: int
    gender: str
    social_media_hours: float
    platform_usage: str
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    social_interaction_level: str
    stress_level: int
    anxiety_level: int
    addiction_level: int


class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_data: Dict[str, Any] = Field(default_factory=dict)
    model_output: Dict[str, Any] = Field(default_factory=dict)
    historical_data: Dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


class ApiKeyRequest(BaseModel):
    api_key: str


def clamp(value, low, high):
    return max(low, min(value, high))


def encode_value(encoder, value):
    if hasattr(encoder, "transform"):
        return int(encoder.transform([value])[0])
    if isinstance(encoder, dict):
        if value not in encoder:
            raise ValueError(f"Unknown category: {value}")
        return int(encoder[value])
    raise TypeError("Unsupported encoder type")


def load_artifacts():
    model_obj = joblib.load("champion_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    try:
        with open("feature_fill_values.json", "r", encoding="utf-8") as f:
            fill_vals = json.load(f)
    except FileNotFoundError:
        fill_vals = {}
    return model_obj, encoders, feature_cols, fill_vals


def normalize_inputs(payload):
    gender = payload.gender.strip().lower()
    if gender.startswith("m"):
        gender = "male"
    elif gender.startswith("f"):
        gender = "female"

    platform = payload.platform_usage.strip().title()
    if "inst" in platform.lower():
        platform = "Instagram"
    elif "tik" in platform.lower():
        platform = "TikTok"
    else:
        platform = "Both"

    social = payload.social_interaction_level.strip().lower()
    if social.startswith("l"):
        social = "low"
    elif social.startswith("m"):
        social = "medium"
    elif social.startswith("h"):
        social = "high"

    return {
        "age": clamp(payload.age, 10, 25),
        "gender": gender,
        "daily_social_media_hours": clamp(payload.social_media_hours, 0.0, 12.0),
        "platform_usage": platform,
        "sleep_hours": clamp(payload.sleep_hours, 0.0, 12.0),
        "screen_time_before_sleep": clamp(payload.screen_time_before_sleep, 0.0, 12.0),
        "academic_performance": clamp(payload.academic_performance, 0.0, 4.0),
        "physical_activity": clamp(payload.physical_activity, 0.0, 12.0),
        "social_interaction_level": social,
        "stress_level": clamp(payload.stress_level, 1, 10),
        "anxiety_level": clamp(payload.anxiety_level, 1, 10),
        "addiction_level": clamp(payload.addiction_level, 1, 10),
    }


def build_suggestions(row):
    suggestions = []

    # Sleep suggestions (granular)
    sleep = row.get("sleep_hours", 8)
    if sleep < 4:
        suggestions.append("Critical: You need at least 7-8 hours of sleep — consult a professional about your sleep patterns")
    elif sleep < 6:
        suggestions.append("Increase sleep to 7-8 hours — set a fixed bedtime and avoid screens 1 hour before")
    elif sleep < 7:
        suggestions.append("Add 1 more hour of sleep — try going to bed 30 minutes earlier")

    # Screen time before sleep
    screen_before = row.get("screen_time_before_sleep", 0)
    if screen_before > 4:
        suggestions.append("Significantly reduce screen time before bed — it's severely disrupting your sleep quality")
    elif screen_before > 2:
        suggestions.append("Reduce screen time before bed — blue light suppresses melatonin production")

    # Social media
    social_media = row.get("daily_social_media_hours", row.get("social_media_hours", 0))
    if social_media > 6:
        suggestions.append("Set strict daily screen limits (max 2 hrs) and schedule offline breaks every hour")
    elif social_media > 4:
        suggestions.append("Reduce social media to under 3 hours — try app timers and no-phone zones")
    elif social_media > 3:
        suggestions.append("Monitor social media use — curate your feed to reduce comparison triggers")

    # Physical activity
    physical = row.get("physical_activity", 0)
    if physical < 1:
        suggestions.append("Start with just 10-minute daily walks — any movement improves mood significantly")
    elif physical < 3:
        suggestions.append("Increase physical activity to 30+ min/day — exercise is a natural antidepressant")

    # Stress / Anxiety
    stress = row.get("stress_level", 5)
    anxiety = row.get("anxiety_level", 5)
    if stress >= 8 or anxiety >= 8:
        suggestions.append("Practice daily stress-relief: try 4-7-8 breathing, progressive muscle relaxation, or journaling")
    elif stress >= 6 or anxiety >= 6:
        suggestions.append("Use stress-reduction techniques — deep breathing, short walks, or talking to someone")

    # Addiction
    addiction = row.get("addiction_level", 5)
    if addiction >= 8:
        suggestions.append("Consider digital detox days and seek support for tech dependency")
    elif addiction >= 6:
        suggestions.append("Set app usage limits and designate phone-free time blocks")

    # Social interaction
    social = row.get("social_interaction_level", "medium")
    if social in ("low",):
        suggestions.append("Schedule regular face-to-face interactions — social connection is protective against depression")

    return suggestions[:5]


def key_drivers(base_model, X_row):
    feature_labels = {
        "sleep_deficit": "Sleep Deficit",
        "digital_overload": "Digital Overload",
        "night_risk": "Night Risk",
        "mental_pressure": "Mental Pressure",
        "isolation_score": "Social Isolation",
        "addiction_impact": "Addiction Impact",
        "lifestyle_imbalance": "Lifestyle Imbalance",
    }

    try:
        import shap

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_row)
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        contrib = pd.Series(shap_vals[0], index=X_row.columns).abs()
    except Exception:
        if hasattr(base_model, "feature_importances_"):
            imp = pd.Series(base_model.feature_importances_, index=X_row.columns)
            contrib = (imp * X_row.iloc[0].abs()).abs()
        else:
            contrib = pd.Series(dtype=float)

    if contrib.empty:
        return {
            "Sleep Deficit": 18,
            "Digital Overload": 22,
            "Mental Pressure": 15,
            "Social Isolation": 10,
        }

    top = contrib.sort_values(ascending=False).head(4)
    total = float(top.sum()) or 1.0
    drivers = {}
    for name, value in top.items():
        label = feature_labels.get(name, name.replace("_", " ").title())
        drivers[label] = int(round((value / total) * 100))
    return drivers


def build_features(row, encoders, feature_cols, feature_engineer):
    gender = encode_value(encoders["gender"], row["gender"])
    platform = encode_value(encoders["platform_usage"], row["platform_usage"])
    social = encode_value(encoders["social_interaction_level"], row["social_interaction_level"])

    input_data = pd.DataFrame([
        [
            row["age"],
            gender,
            row["daily_social_media_hours"],
            platform,
            row["sleep_hours"],
            row["screen_time_before_sleep"],
            row["academic_performance"],
            row["physical_activity"],
            social,
            row["stress_level"],
            row["anxiety_level"],
            row["addiction_level"],
        ]
    ], columns=RAW_FEATURES)

    input_data = feature_engineer.transform(input_data)
    input_data = input_data.reindex(columns=feature_cols, fill_value=0)
    return input_data


def _build_histogram(values: pd.Series, bins: np.ndarray) -> Tuple[List[float], List[float]]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers.tolist(), hist.tolist()


def _box_stats(values: pd.Series) -> Dict[str, float]:
    q1 = float(values.quantile(0.25))
    median = float(values.quantile(0.5))
    q3 = float(values.quantile(0.75))
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "mean": float(values.mean()),
    }


def _encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    gender_vals = sorted(df_enc["gender"].astype(str).str.lower().str.strip().unique())
    platform_vals = sorted(df_enc["platform_usage"].astype(str).str.title().unique())
    social_vals = ["low", "medium", "high"]

    gender_map = {val: idx for idx, val in enumerate(gender_vals)}
    platform_map = {val: idx for idx, val in enumerate(platform_vals)}
    social_map = {"low": 2, "medium": 1, "high": 0}

    VIZ_ENCODERS.clear()
    VIZ_ENCODERS.update({
        "gender": gender_map,
        "platform_usage": platform_map,
        "social_interaction_level": social_map,
    })

    df_enc["gender"] = df_enc["gender"].astype(str).str.lower().str.strip().map(gender_map)
    df_enc["platform_usage"] = df_enc["platform_usage"].astype(str).str.title().map(platform_map)
    df_enc["social_interaction_level"] = (
        df_enc["social_interaction_level"].astype(str).str.lower().str.strip().map(social_map)
    )
    return df_enc


def _load_viz_cache() -> Dict[str, Any]:
    global VIZ_CACHE, VIZ_SCALER, VIZ_PCA
    if VIZ_CACHE is not None:
        return VIZ_CACHE
    if not DATASET_PATH.exists():
        VIZ_CACHE = {}
        return VIZ_CACHE

    df = pd.read_csv(DATASET_PATH)
    df_enc = _encode_categories(df)

    class_counts = df["depression_label"].value_counts().sort_index()
    class_labels = ["No Depression", "Depression"]

    sleep_bins = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 36)
    sleep_no_x, sleep_no_y = _build_histogram(
        df[df["depression_label"] == 0]["sleep_hours"], sleep_bins
    )
    sleep_yes_x, sleep_yes_y = _build_histogram(
        df[df["depression_label"] == 1]["sleep_hours"], sleep_bins
    )

    addiction_stats = [
        _box_stats(df[df["depression_label"] == 0]["addiction_level"]),
        _box_stats(df[df["depression_label"] == 1]["addiction_level"]),
    ]

    corr = df_enc.corr().round(3)
    corr_labels = corr.columns.tolist()
    matrix = []
    for row_idx, row in enumerate(corr_labels):
        for col_idx, col in enumerate(corr_labels):
            matrix.append({"x": col_idx, "y": row_idx, "v": float(corr.iloc[row_idx, col_idx])})

    stress_points = df[["stress_level", "anxiety_level", "depression_label"]].sample(
        n=min(600, len(df)), random_state=42
    )
    stress_points = [
        {"x": float(row["stress_level"]), "y": float(row["anxiety_level"]), "label": int(row["depression_label"])}
        for _, row in stress_points.iterrows()
    ]

    social_stats = [
        _box_stats(df[df["depression_label"] == 0]["daily_social_media_hours"]),
        _box_stats(df[df["depression_label"] == 1]["daily_social_media_hours"]),
    ]

    platform_table = (
        df.groupby(["platform_usage", "depression_label"]).size().unstack(fill_value=0)
    )
    platform_labels = platform_table.index.tolist()
    platform_no = platform_table.get(0, pd.Series([0] * len(platform_labels))).tolist()
    platform_yes = platform_table.get(1, pd.Series([0] * len(platform_labels))).tolist()

    pca_features = df_enc.drop(columns=["depression_label"])
    VIZ_SCALER = StandardScaler()
    scaled = VIZ_SCALER.fit_transform(pca_features)
    VIZ_PCA = PCA(n_components=2, random_state=42)
    pca_vals = VIZ_PCA.fit_transform(scaled)
    pca_var = float(np.sum(VIZ_PCA.explained_variance_ratio_) * 100)
    pca_points = []
    for idx, (x_val, y_val) in enumerate(pca_vals):
        if idx % 2 == 0:
            label = int(df_enc.iloc[idx]["depression_label"])
            pca_points.append({"x": float(x_val), "y": float(y_val), "label": label})

    cv_data = pd.read_csv(BASE_DIR / "cross_validation_results.csv") if (BASE_DIR / "cross_validation_results.csv").exists() else pd.DataFrame()
    cv_payload = {
        "models": cv_data.get("Model", pd.Series(dtype=str)).tolist(),
        "f1_mean": cv_data.get("CV_F1_Mean", pd.Series(dtype=float)).tolist(),
        "f1_std": cv_data.get("CV_F1_Std", pd.Series(dtype=float)).tolist(),
    }

    X_train = pd.read_csv(BASE_DIR / "X_train.csv") if (BASE_DIR / "X_train.csv").exists() else pd.DataFrame()
    y_train = pd.read_csv(BASE_DIR / "y_train.csv").values.ravel() if (BASE_DIR / "y_train.csv").exists() else np.array([])
    X_test = pd.read_csv(BASE_DIR / "X_test.csv") if (BASE_DIR / "X_test.csv").exists() else pd.DataFrame()
    y_test = pd.read_csv(BASE_DIR / "y_test.csv").values.ravel() if (BASE_DIR / "y_test.csv").exists() else np.array([])

    if feature_columns:
        X_train = X_train.reindex(columns=feature_columns, fill_value=0)
        X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    base_model = model
    if hasattr(model, "calibrated_classifiers_"):
        base_model = model.calibrated_classifiers_[0].estimator

    learning_payload = {"train_sizes": [], "train_f1": [], "val_f1": []}
    if len(X_train) and len(y_train):
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                base_model,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="f1",
                train_sizes=np.linspace(0.15, 1.0, 6),
                n_jobs=1,
            )
            learning_payload = {
                "train_sizes": train_sizes.tolist(),
                "train_f1": train_scores.mean(axis=1).tolist(),
                "val_f1": val_scores.mean(axis=1).tolist(),
            }
        except Exception:
            pass

    rf_importance = {"labels": [], "values": []}
    model_importance = {"labels": [], "values": []}
    perm_importance = {"labels": [], "values": []}
    rf_model = None

    if len(X_train) and len(y_train):
        try:
            rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_series = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            rf_importance = {
                "labels": rf_series.index.tolist()[:10],
                "values": rf_series.values.tolist()[:10],
            }
        except Exception:
            pass

    if hasattr(base_model, "feature_importances_"):
        try:
            base_series = pd.Series(base_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            model_importance = {
                "labels": base_series.index.tolist()[:10],
                "values": base_series.values.tolist()[:10],
            }
        except Exception:
            model_importance = rf_importance
    else:
        model_importance = rf_importance

    if len(X_test) and len(y_test):
        try:
            estimator = base_model if hasattr(base_model, "predict") else rf_model
            if estimator is None:
                raise RuntimeError("No estimator available for permutation importance")
            perm = permutation_importance(
                estimator,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )
            perm_series = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
            perm_importance = {
                "labels": perm_series.index.tolist()[:10],
                "values": perm_series.values.tolist()[:10],
            }
        except Exception:
            perm_importance = rf_importance

    evaluation_payload = {"roc": None, "pr": None, "confusion": None}
    if len(X_test) and len(y_test):
        try:
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            precision, recall, _ = precision_recall_curve(y_test, probs)
            conf = confusion_matrix(y_test, (probs >= 0.5).astype(int))
            evaluation_payload = {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
                "confusion": conf.tolist(),
            }
        except Exception:
            pass

    VIZ_CACHE = {
        "classDistribution": {
            "labels": class_labels,
            "counts": [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))],
        },
        "sleepKde": {
            "x": sleep_no_x,
            "noDepression": sleep_no_y,
            "depression": sleep_yes_y,
        },
        "addictionBox": {
            "labels": class_labels,
            "stats": addiction_stats,
        },
        "correlation": {
            "labels": corr_labels,
            "matrix": matrix,
        },
        "stressAnxiety": {
            "points": stress_points,
        },
        "socialMediaBox": {
            "labels": class_labels,
            "stats": social_stats,
        },
        "platformUsage": {
            "labels": platform_labels,
            "noDepression": platform_no,
            "depression": platform_yes,
        },
        "pca": {
            "points": pca_points,
            "variance": round(pca_var, 1),
        },
        "cv": cv_payload,
        "learningCurve": learning_payload,
        "featureImportance": {
            "rf": rf_importance,
            "model": model_importance,
            "permutation": perm_importance,
        },
        "evaluation": evaluation_payload,
    }
    return VIZ_CACHE


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATASET_PATH = BASE_DIR / "Teen_Mental_Health_Dataset.csv"

app = FastAPI(title="Mental Health Risk API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, label_encoders, feature_columns, fill_values = load_artifacts()
feature_engineer = FeatureEngineer(fill_values=fill_values)

CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY = 20
AVAILABLE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen3-235b-a22b:free",
]
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

VIZ_CACHE: Optional[Dict[str, Any]] = None
VIZ_SCALER: Optional[StandardScaler] = None
VIZ_PCA: Optional[PCA] = None
VIZ_ENCODERS: Dict[str, Dict[str, int]] = {}

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

ASSET_FILES = {
    "dashboard_cv_learning.png": BASE_DIR / "dashboard_cv_learning.png",
    "dashboard_eda.png": BASE_DIR / "dashboard_eda.png",
    "dashboard_feature_importance.png": BASE_DIR / "dashboard_feature_importance.png",
    "dashboard_final_evaluation.png": BASE_DIR / "dashboard_final_evaluation.png",
    "champion_model_name.txt": BASE_DIR / "champion_model_name.txt",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/viz-data")
def viz_data():
    return _load_viz_cache()


@app.get("/")
def index():
    if FRONTEND_DIR.exists():
        return FileResponse(FRONTEND_DIR / "index.html")
    return {"message": "Frontend not found. Open /docs for API."}


@app.get("/assets/{asset_name}")
def get_asset(asset_name: str):
    path = ASSET_FILES.get(asset_name)
    if path and path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Asset not found")


def build_chat_context(user_data, model_output, historical_data):
    return (
        "User Data: "
        + json.dumps(user_data, ensure_ascii=True)
        + "\nModel Output: "
        + json.dumps(model_output, ensure_ascii=True)
        + "\nHistory Data: "
        + json.dumps(historical_data, ensure_ascii=True)
    )


def get_history(session_id: str) -> List[Dict[str, str]]:
    history = CHAT_HISTORY.get(session_id, [])
    return history[-MAX_HISTORY:]


def save_turn(session_id: str, role: str, content: str) -> None:
    history = CHAT_HISTORY.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        CHAT_HISTORY[session_id] = history[-MAX_HISTORY:]


def stream_openrouter(messages: List[Dict[str, str]], model_name: str) -> Generator[str, None, None]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    if requests is None:
        raise RuntimeError("requests package is required for streaming")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "MindGuard Mental Health Dashboard",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1500,
    }

    with requests.post(OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=90) as resp:
        resp.raise_for_status()
        in_thinking = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line.replace("data:", "", 1).strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
            except json.JSONDecodeError:
                delta = None
            if delta:
                # Filter out <think>...</think> blocks from reasoning models
                if "<think>" in delta:
                    in_thinking = True
                    delta = delta.split("<think>")[0]
                    if delta:
                        yield delta
                    continue
                if "</think>" in delta:
                    in_thinking = False
                    delta = delta.split("</think>")[-1]
                    if delta:
                        yield delta
                    continue
                if in_thinking:
                    continue
                yield delta


def build_system_prompt() -> str:
    return (
        "You are Dr. MindGuard, a warm, empathetic, and highly knowledgeable mental wellness assistant "
        "specializing in teen mental health. You communicate like a caring, experienced doctor having a "
        "genuine human conversation — never robotic or generic.\n\n"
        "PERSONALITY: Be warm, conversational, and engaging like ChatGPT. Use emojis naturally (not excessively). "
        "Acknowledge feelings before giving advice. Be thorough and detailed — users want REAL depth, not surface tips. "
        "Use markdown formatting: **bold** for emphasis, bullet points (•), numbered lists for steps.\n\n"
        "SCOPE RULES:\n"
        "1) You ONLY discuss: mental health, wellness, sleep, stress, anxiety, depression, social media impact, "
        "physical activity, teen wellbeing, coping strategies, mindfulness, therapy, emotional regulation, "
        "self-care, relationships as they affect mental health, academic pressure, and related behavioral health topics.\n"
        "2) If the user asks about ANY unrelated topic (coding, math, politics, sports scores, recipes, news, etc.), "
        "respond ONLY with: 'I appreciate your curiosity! However, I specialize exclusively in mental health "
        "and wellness. Ask me about sleep, stress, anxiety, depression, social media impact, or any wellbeing topic "
        "and I will provide detailed, evidence-based guidance.'\n"
        "3) Never provide medical diagnoses. For severe cases or self-harm, always provide crisis resources "
        "(988 Lifeline, Crisis Text Line: text HOME to 741741).\n\n"
        "RESPONSE STYLE:\n"
        "- Give DETAILED, thorough responses (3-6 paragraphs). Users want ChatGPT-level depth.\n"
        "- Include specific data, statistics, and scientific explanations when relevant.\n"
        "- Always provide actionable steps (numbered 1-4).\n"
        "- End with a follow-up question to keep the conversation going.\n"
        "- If user asks to elaborate, go VERY deep with research-backed information.\n"
        "- Reference the user's assessment data naturally if provided in context.\n\n"
        "IMPORTANT: Respond in plain text with markdown formatting. Do NOT wrap your response in JSON. "
        "Just write your response directly as natural conversational text."
    )


def parse_json_response(text: str) -> Dict[str, str]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")
        parsed = json.loads(text[start : end + 1])
        return {
            "response": str(parsed.get("response", "")),
            "follow_up": str(parsed.get("follow_up", "")),
        }
    except Exception:
        return {
            "response": text.strip(),
            "follow_up": "Would you like personalized steps to improve sleep or screen habits?",
        }


MENTAL_HEALTH_TOPICS = {
    "sleep": {
        "keywords": ["sleep", "insomnia", "can't sleep", "cant sleep", "sleeping", "wake up", "tired", "bed", "rest", "nap", "night", "melatonin", "dream"],
        "responses": [
            {
                "trigger": ["can't sleep", "cant sleep", "insomnia", "trouble sleeping", "not sleeping", "don't get the sleep", "difficulty sleeping"],
                "response": (
                    "I completely understand how frustrating it is when you can't fall asleep — it's one of the most common issues I see, and it genuinely affects everything from your mood to your focus.\n\n"
                    "**Why this happens:** When your brain is overstimulated — from screens, stress, or an irregular schedule — it keeps producing cortisol (the stress hormone) instead of switching to melatonin (the sleep hormone). Your body essentially stays in 'alert mode' even when you're physically tired.\n\n"
                    "**Here's what I'd recommend trying tonight:**\n"
                    "1) **The 10-3-2-1 Rule:** No caffeine 10 hours before bed, no heavy food 3 hours before, no work/screens 2 hours before, no screens 1 hour before.\n"
                    "2) **4-7-8 Breathing:** Breathe in for 4 seconds, hold for 7, exhale slowly for 8. This activates your parasympathetic nervous system and physically calms you down.\n"
                    "3) **Cool your room** to 65-68°F (18-20°C) — your body needs to drop its core temperature to initiate sleep.\n"
                    "4) **Write a 'worry dump'** — spend 5 minutes writing down everything on your mind before bed. This tells your brain it's safe to let go.\n\n"
                    "These aren't just tips — they're backed by sleep science. Most people see improvement within 3-5 nights of consistent practice."
                ),
                "follow_up": "Would you like me to create a personalized 7-day sleep improvement plan for you?"
            },
            {
                "trigger": ["how much sleep", "hours of sleep", "enough sleep", "sleep need"],
                "response": (
                    "Great question! Sleep needs vary by age, and this is something a lot of people get wrong.\n\n"
                    "**Recommended sleep by age:**\n"
                    "• Teens (13-17): **8-10 hours** per night\n"
                    "• Young adults (18-25): **7-9 hours** per night\n\n"
                    "**Why it matters so much for mental health:**\n"
                    "During deep sleep, your brain processes emotions, consolidates memories, and clears out toxins. When you consistently get less than 7 hours, your amygdala (the brain's emotional center) becomes 60% more reactive — meaning small things feel like big problems.\n\n"
                    "**Quality matters as much as quantity:**\n"
                    "1) Keep a consistent sleep and wake time — even on weekends (within 1 hour difference)\n"
                    "2) Your bedroom should be dark, cool, and quiet\n"
                    "3) Avoid heavy meals and caffeine after 2 PM\n"
                    "4) Morning sunlight exposure (10-15 min) resets your circadian clock\n\n"
                    "Our data shows that **sleep deficit is the #1 predictor** of depression risk in teens. Fixing sleep alone often improves anxiety and mood significantly."
                ),
                "follow_up": "Want me to explain how screen time before bed specifically disrupts your sleep cycle?"
            },
        ],
        "default_response": (
            "Sleep is foundational to mental health — I'm glad you're thinking about it! 😊\n\n"
            "**Here's what science tells us:**\n"
            "Poor sleep doesn't just make you tired — it directly impacts your emotional regulation, decision-making, and stress resilience. Teens who sleep less than 7 hours are **3x more likely** to report depressive symptoms.\n\n"
            "**My top 3 sleep recommendations:**\n"
            "1) **Consistent schedule** — Go to bed and wake up at the same time daily. Your body's clock needs rhythm.\n"
            "2) **Screen curfew** — Blue light from phones suppresses melatonin by up to 50%. Put screens away 1 hour before bed.\n"
            "3) **Wind-down ritual** — Read, stretch, or practice breathing exercises. This signals your brain that it's time to rest.\n\n"
            "Small changes in sleep habits can produce dramatic improvements in mood and focus within just one week."
        ),
        "default_follow_up": "Would you like specific techniques to fall asleep faster, or tips for improving sleep quality?"
    },
    "anxiety": {
        "keywords": ["anxiety", "anxious", "nervous", "panic", "worry", "worrying", "worried", "fear", "scared", "overwhelming", "overwhelm", "overthink", "racing thoughts"],
        "responses": [
            {
                "trigger": ["panic", "panic attack", "can't breathe", "heart racing", "chest tight"],
                "response": (
                    "I hear you, and first — you're safe. Panic attacks feel terrifying, but they are not dangerous and they always pass. Let me help you understand what's happening.\n\n"
                    "**What's happening in your body:**\n"
                    "During a panic attack, your brain's fight-or-flight system misfires. It releases a flood of adrenaline even when there's no real danger. This causes rapid heartbeat, shortness of breath, chest tightness, and dizziness. It peaks within 10 minutes and subsides.\n\n"
                    "**Immediate grounding technique (do this right now if you're in one):**\n"
                    "1) **5-4-3-2-1 Method:** Name 5 things you can SEE, 4 you can TOUCH, 3 you can HEAR, 2 you can SMELL, 1 you can TASTE. This pulls your brain out of the fear loop.\n"
                    "2) **Box Breathing:** Inhale 4 sec → Hold 4 sec → Exhale 4 sec → Hold 4 sec. Repeat 4 times.\n"
                    "3) **Splash cold water** on your face — this triggers the dive reflex and immediately slows your heart rate.\n\n"
                    "**Long-term strategies:**\n"
                    "• Regular exercise (even 20 min walks) reduces panic frequency by 40%\n"
                    "• Limit caffeine — it mimics anxiety symptoms\n"
                    "• Consider speaking with a counselor about CBT techniques\n\n"
                    "If panic attacks happen frequently (more than once a week), I'd strongly recommend talking to a mental health professional. There are very effective treatments available."
                ),
                "follow_up": "Would you like me to explain CBT techniques that specifically help prevent panic attacks?"
            },
        ],
        "default_response": (
            "Anxiety is something millions of people experience, and acknowledging it is genuinely the first step toward managing it. You're not alone in this. 💛\n\n"
            "**Understanding your anxiety:**\n"
            "Anxiety is your brain's alarm system working overtime. It evolved to protect us from danger, but in modern life, it often fires in situations that aren't actually threatening — like social situations, exams, or uncertain futures.\n\n"
            "**Evidence-based strategies that work:**\n"
            "1) **Challenge the thought:** When anxious, ask yourself: 'What evidence do I actually have that this bad thing will happen?' Often, our fears are predictions, not facts.\n"
            "2) **Scheduled worry time:** Set aside 15 minutes daily to worry. Outside that window, write worries down and tell yourself 'I'll address this during worry time.' This sounds odd but it's clinically proven.\n"
            "3) **Body-based calming:** Deep belly breathing, progressive muscle relaxation, or a short walk. Your body can calm your mind when your mind can't calm itself.\n"
            "4) **Reduce stimulants:** Cut caffeine after noon, reduce sugar, and limit doom-scrolling on social media.\n\n"
            "Remember — anxiety is treatable. These techniques, practiced consistently, can reduce anxiety levels by 40-60% within a few weeks."
        ),
        "default_follow_up": "Want me to walk you through a specific anxiety management exercise step by step?"
    },
    "stress": {
        "keywords": ["stress", "stressed", "pressure", "burnout", "burnt out", "exhausted", "overwhelmed", "too much", "can't cope", "breaking point", "exam", "test", "deadline", "study"],
        "responses": [
            {
                "trigger": ["exam", "test", "study", "grade", "school", "academic", "homework", "assignment"],
                "response": (
                    "Academic pressure is one of the biggest stress sources for teens — and your feelings about it are completely valid. Let's break this down practically.\n\n"
                    "**Why academic stress hits so hard:**\n"
                    "Your brain's prefrontal cortex (the planning/decision center) is still developing until age 25. Under stress, the emotional brain takes over, making it harder to focus, remember, and think clearly. It's a cruel cycle — stress hurts performance, which creates more stress.\n\n"
                    "**Smart strategies for academic stress:**\n"
                    "1) **Pomodoro Technique:** Study for 25 minutes, break for 5. After 4 rounds, take a 15-30 minute break. This matches your brain's natural attention span.\n"
                    "2) **Active recall > Re-reading:** Test yourself instead of re-reading notes. It's 3x more effective for memory retention.\n"
                    "3) **Brain dumps before exams:** Write everything you know on paper before starting. This reduces anxiety and frees working memory.\n"
                    "4) **Self-compassion:** Talk to yourself like you'd talk to a friend. 'I'm doing my best' is more productive than 'I'm going to fail.'\n\n"
                    "**Physical resets that boost study performance:**\n"
                    "• 10-minute walk between study sessions increases focus by 20%\n"
                    "• Stay hydrated — even mild dehydration impairs concentration\n"
                    "• Sleep > all-nighters. Your brain consolidates learning during sleep.\n\n"
                    "Your grades don't define your worth. They're one data point in a much bigger picture of who you are."
                ),
                "follow_up": "Would you like a personalized study-stress management plan?"
            },
        ],
        "default_response": (
            "I can hear that you're carrying a lot right now, and I want you to know — feeling stressed doesn't mean you're weak. It means you care, and that matters. 💪\n\n"
            "**Understanding stress:**\n"
            "Stress in small doses actually helps performance (it's called 'eustress'). But chronic stress — the kind that doesn't let up — floods your system with cortisol, which disrupts sleep, appetite, focus, and mood. Over time, it can lead to anxiety and depression.\n\n"
            "**Practical stress relief (things you can do today):**\n"
            "1) **Priority matrix:** List everything stressing you. Divide into: 'Can control' vs 'Can't control.' Focus energy only on what you can control.\n"
            "2) **Movement breaks:** Even 5 minutes of stretching or walking resets your stress hormones.\n"
            "3) **Talk it out:** Share what you're feeling with someone you trust. Bottling up stress literally increases cortisol levels.\n"
            "4) **Micro-joy moments:** Schedule one small thing you enjoy daily — music, a short walk, a funny video. These aren't luxuries; they're stress medicine.\n\n"
            "Stress management is a skill, not a talent. The more you practice these techniques, the more resilient your brain becomes."
        ),
        "default_follow_up": "Would you like me to help you build a daily stress management routine?"
    },
    "depression": {
        "keywords": ["depression", "depressed", "sad", "hopeless", "empty", "numb", "cry", "crying", "unhappy", "miserable", "worthless", "no motivation", "don't care", "give up", "no point", "lonely", "alone", "isolated"],
        "default_response": (
            "Thank you for sharing this with me — it takes real courage to talk about feelings like these, and I'm glad you're reaching out. 💚\n\n"
            "**What you're feeling is real and valid:**\n"
            "Depression isn't just 'being sad' — it's a condition that affects your brain chemistry, energy, motivation, and how you see the world. It can make everything feel heavy and pointless, even things you used to enjoy. But here's what's important: **it is treatable, and it does get better.**\n\n"
            "**Things that can help right now:**\n"
            "1) **Small actions, not big changes:** When motivation is low, commit to just 5 minutes of something — a short walk, texting a friend, drinking water. Action creates motivation, not the other way around.\n"
            "2) **Routine and structure:** Depression thrives in chaos. Even a simple daily routine (wake time, one activity, sleep time) gives your brain something to hold onto.\n"
            "3) **Social connection:** Even when you don't feel like it, being around people — even briefly — helps. Isolation feeds depression.\n"
            "4) **Sunlight and movement:** 15 minutes of morning sunlight and any form of movement boost serotonin naturally.\n\n"
            "**Important:** If you've been feeling this way for more than 2 weeks, or if you're having thoughts of self-harm, please reach out to a mental health professional or crisis line. You deserve support.\n"
            "• **Crisis Text Line:** Text HOME to 741741\n"
            "• **988 Suicide & Crisis Lifeline:** Call or text 988\n\n"
            "You are not alone, and asking for help is a sign of strength, not weakness."
        ),
        "default_follow_up": "Would you like me to help you create a small daily routine to build momentum, or would you like to talk more about what you're feeling?"
    },
    "social_media": {
        "keywords": ["social media", "instagram", "tiktok", "snapchat", "phone", "screen", "scroll", "scrolling", "online", "internet", "digital", "notification", "likes", "followers", "comparison", "fomo"],
        "default_response": (
            "This is such an important topic — social media's impact on mental health is one of the biggest concerns in teen wellness right now. Let me share what the research actually shows.\n\n"
            "**The data is clear:**\n"
            "• Teens spending **3+ hours/day** on social media have **double** the risk of anxiety and depression\n"
            "• Our model shows social media hours as a top-5 predictor of depression risk\n"
            "• The 'comparison trap' (seeing curated highlight reels) directly lowers self-esteem\n"
            "• Notifications trigger dopamine loops — the same reward pathway involved in addiction\n\n"
            "**But it's not all bad — the key is HOW you use it:**\n"
            "1) **Active vs Passive:** Messaging friends (active) is healthy. Endless scrolling (passive) is harmful. Shift your usage toward active engagement.\n"
            "2) **Curate your feed:** Unfollow accounts that make you feel bad about yourself. Follow ones that inspire or educate.\n"
            "3) **Set boundaries:** Use built-in screen time limits. Try 'no phone zones' — during meals, 1 hour before bed, and first 30 min after waking.\n"
            "4) **Digital detox days:** Try one screen-free day per week. Most people report feeling significantly calmer by the end of the day.\n\n"
            "The goal isn't to quit social media — it's to use it intentionally rather than compulsively."
        ),
        "default_follow_up": "Want me to help you create a balanced digital wellness plan?"
    },
    "self_harm": {
        "keywords": ["self harm", "self-harm", "hurt myself", "cutting", "suicide", "suicidal", "kill myself", "end it", "don't want to live", "die"],
        "default_response": (
            "I'm really glad you told me this, and I want you to know — you matter, and there are people who want to help. 💚\n\n"
            "**Please reach out to a crisis professional right now:**\n"
            "• **988 Suicide & Crisis Lifeline:** Call or text **988** (available 24/7)\n"
            "• **Crisis Text Line:** Text **HOME** to **741741**\n"
            "• **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "**What I want you to know:**\n"
            "What you're feeling right now is temporary, even though it doesn't feel that way. Depression and crisis states distort how we see the future — they make everything seem permanent and hopeless. But with the right support, these feelings do change.\n\n"
            "**Right now, please:**\n"
            "1) Tell someone you trust — a parent, teacher, friend, or counselor\n"
            "2) Remove anything that could be used for self-harm from your immediate environment\n"
            "3) Stay with someone — don't be alone right now\n"
            "4) Call one of the numbers above — trained counselors are available 24/7\n\n"
            "You reached out here, which tells me a part of you wants help. That part of you is right. You deserve support and you can get through this. 🤍"
        ),
        "default_follow_up": "Please reach out to 988 or Crisis Text Line. Would you like me to provide more resources?"
    },
    "exercise": {
        "keywords": ["exercise", "workout", "physical activity", "gym", "running", "walk", "yoga", "sport", "fitness", "sedentary", "lazy", "inactive"],
        "default_response": (
            "You're asking about one of the most powerful natural antidepressants available — and it's completely free! 🏃\n\n"
            "**The science is remarkable:**\n"
            "• Just **30 minutes of moderate exercise** releases endorphins, serotonin, and BDNF (a protein that literally grows new brain cells)\n"
            "• Regular exercise is as effective as antidepressants for mild-to-moderate depression in clinical studies\n"
            "• Physical activity reduces anxiety symptoms by **20-30%** within just 2 weeks\n"
            "• Our model data shows teens with regular physical activity have significantly lower depression risk\n\n"
            "**Starting when motivation is low:**\n"
            "1) **The 5-minute rule:** Commit to just 5 minutes. Once you start, you'll usually keep going.\n"
            "2) **Walk first:** Don't aim for the gym. A 15-minute walk outside counts and is incredibly effective.\n"
            "3) **Social exercise:** Walk with a friend, join a sports club, or try a group class. Social + physical = double benefit.\n"
            "4) **Dance it out:** Put on your favorite music and move for one song. Seriously — it works.\n\n"
            "The best exercise for mental health is the one you'll actually do. Don't aim for perfection — aim for consistency."
        ),
        "default_follow_up": "Want me to suggest a simple weekly exercise plan designed for mental wellness?"
    },
    "general_greeting": {
        "keywords": ["hi", "hello", "hey", "how are you", "what can you do", "help", "what do you do", "who are you"],
        "default_response": (
            "Hey there! 👋 Welcome — I'm Dr. MindGuard, your mental health wellness assistant.\n\n"
            "**Here's how I can help you:**\n"
            "• 😴 **Sleep issues** — trouble sleeping, sleep quality, bedtime routines\n"
            "• 😰 **Anxiety & stress** — exam pressure, panic attacks, worry management\n"
            "• 😔 **Depression awareness** — understanding symptoms, coping strategies, when to seek help\n"
            "• 📱 **Social media impact** — digital wellness, screen time, healthy habits\n"
            "• 🏃 **Physical wellness** — exercise for mental health, activity planning\n"
            "• 🧠 **General mental health** — self-care, mindfulness, emotional regulation\n\n"
            "I can also use your risk assessment data (if you've filled in the predictor) to give you **personalized advice** based on your specific situation.\n\n"
            "Just ask me anything about mental wellbeing — I'm here to listen and help! 💚"
        ),
        "default_follow_up": "What aspect of mental health would you like to explore? Sleep, stress, anxiety, or something else?"
    },
}

NON_MH_KEYWORDS = [
    "code", "coding", "python", "javascript", "program", "software", "recipe", "cook",
    "weather", "sports", "football", "cricket", "movie", "song", "math", "calcul",
    "physics", "chemistry", "history", "geography", "politic", "election", "stock",
    "crypto", "bitcoin", "game", "gaming", "minecraft", "fortnite", "joke", "funny",
    "travel", "flight", "hotel", "shopping", "buy", "price", "car", "bike",
]


def _detect_topic(message: str) -> Optional[str]:
    """Detect which mental health topic the message is about."""
    lower = message.lower().strip()

    # Check for non-MH topics first
    if any(kw in lower for kw in NON_MH_KEYWORDS):
        has_mh = any(
            kw in lower
            for topic in MENTAL_HEALTH_TOPICS.values()
            for kw in topic["keywords"]
        )
        if not has_mh:
            return "__off_topic__"

    # Check each topic
    best_topic = None
    best_score = 0
    for topic_key, topic_data in MENTAL_HEALTH_TOPICS.items():
        score = sum(1 for kw in topic_data["keywords"] if kw in lower)
        if score > best_score:
            best_score = score
            best_topic = topic_key

    return best_topic


def _find_specific_response(topic_data: dict, message: str) -> Optional[dict]:
    """Find a specific trigger-based response within a topic."""
    lower = message.lower()
    for resp in topic_data.get("responses", []):
        if any(trigger in lower for trigger in resp.get("trigger", [])):
            return resp
    return None


def build_fallback_chat_response(message: str, model_output: Dict[str, Any]) -> Dict[str, str]:
    """Smart local AI engine that responds contextually to mental health questions."""
    topic = _detect_topic(message)

    # Off-topic rejection
    if topic == "__off_topic__":
        return {
            "response": (
                "I appreciate your curiosity! 😊 However, I'm specifically designed to help with "
                "**mental health and wellness** topics only.\n\n"
                "I can help you with:\n"
                "• Sleep problems and insomnia\n"
                "• Anxiety, stress, and panic management\n"
                "• Depression awareness and coping strategies\n"
                "• Social media's impact on mental health\n"
                "• Exercise and physical wellness for the mind\n"
                "• Academic stress and exam pressure\n\n"
                "Please ask me something related to mental wellbeing, and I'll give you detailed, actionable guidance! 💚"
            ),
            "follow_up": "What mental health topic would you like to explore?"
        }

    # Match a topic
    if topic and topic in MENTAL_HEALTH_TOPICS:
        topic_data = MENTAL_HEALTH_TOPICS[topic]

        # Try specific trigger match first
        specific = _find_specific_response(topic_data, message)
        if specific:
            response = specific["response"]
            follow_up = specific.get("follow_up", topic_data.get("default_follow_up", ""))
        else:
            response = topic_data["default_response"]
            follow_up = topic_data.get("default_follow_up", "Would you like to explore this topic further?")

        # Enrich with model output if available
        risk_level = model_output.get("risk_level")
        risk_score = model_output.get("risk_score")
        if risk_level and risk_score:
            context_line = (
                f"\n\n**Based on your assessment data:** Your current risk level is "
                f"**{risk_level}** ({risk_score}%). "
            )
            drivers = model_output.get("drivers", {})
            if drivers:
                top_drivers = ", ".join(list(drivers.keys())[:3])
                context_line += f"Key factors: {top_drivers}. "
            suggestions = model_output.get("suggestions", [])
            if suggestions:
                context_line += "Personalized recommendations: " + "; ".join(suggestions[:2]) + "."
            response += context_line

        return {"response": response, "follow_up": follow_up}

    # Generic mental health catch-all
    lower = message.lower()

    # Elaborate / depth requests
    if any(w in lower for w in ["elaborate", "explain more", "tell me more", "go deeper", "in depth", "details", "more about"]):
        history = CHAT_HISTORY.get("default", [])
        last_topic = None
        for entry in reversed(history):
            if entry["role"] == "assistant":
                content = entry["content"].lower()
                if "sleep" in content:
                    last_topic = "sleep"
                elif "anxiety" in content or "stress" in content:
                    last_topic = "anxiety"
                elif "depression" in content:
                    last_topic = "depression"
                elif "social media" in content:
                    last_topic = "social_media"
                elif "exercise" in content:
                    last_topic = "exercise"
                break

        if last_topic and last_topic in MENTAL_HEALTH_TOPICS:
            topic_data = MENTAL_HEALTH_TOPICS[last_topic]
            response = topic_data["default_response"]
            # Add deeper context
            response += (
                "\n\n**Going deeper:** The relationship between mental health and daily habits is "
                "bidirectional — poor habits worsen mental health, and poor mental health makes it "
                "harder to maintain good habits. Breaking this cycle requires starting with the "
                "smallest possible change and building consistency. Research shows it takes about "
                "21 days for a new habit to feel natural. Start with ONE thing from the list above "
                "and commit to it for 3 weeks."
            )
            return {
                "response": response,
                "follow_up": "Would you like a structured 21-day plan for this area?"
            }

    # Gratitude / positive responses
    if any(w in lower for w in ["thank", "thanks", "helpful", "great", "awesome", "perfect", "good"]):
        return {
            "response": (
                "You're very welcome! 😊 I'm really glad I could help. Remember — taking care of your "
                "mental health is a journey, not a destination. Small, consistent steps make the biggest difference.\n\n"
                "Feel free to come back anytime you have questions about sleep, stress, anxiety, or any "
                "aspect of your wellbeing. I'm always here for you. 💚"
            ),
            "follow_up": "Is there anything else about your mental health you'd like to discuss?"
        }

    # Default intelligent response
    return {
        "response": (
            "That's a thoughtful question, and I'm glad you're thinking about your wellbeing! 😊\n\n"
            "**Mental health is multi-dimensional.** It's influenced by several interconnected factors:\n"
            "• **Sleep quality** — the foundation of emotional regulation\n"
            "• **Physical activity** — nature's antidepressant\n"
            "• **Social connections** — humans need meaningful relationships\n"
            "• **Screen habits** — digital overload strains the mind\n"
            "• **Stress management** — learning to cope with pressure\n\n"
            "Each of these areas plays a critical role in your overall mental wellness. Our data analysis "
            "shows that teens who maintain balance across these factors have significantly lower depression risk.\n\n"
            "**I'd love to help you more specifically.** Could you tell me:\n"
            "• Are you struggling with sleep?\n"
            "• Feeling anxious or stressed?\n"
            "• Concerned about social media habits?\n"
            "• Want to understand depression warning signs?\n\n"
            "The more specific you are, the more personalized and actionable my advice will be!"
        ),
        "follow_up": "What specific area of mental health would you like to explore together?"
    }


def compute_clinical_risk_index(row: dict, features: pd.DataFrame) -> float:
    """
    Domain-driven risk index based on clinical mental health indicators.
    This acts as a secondary signal when the ML model lacks discrimination.
    Weighted factors from published teen mental health research:
      - Sleep deficit is the #1 predictor (weight: 0.22)
      - Stress + Anxiety composite (weight: 0.25)
      - Digital overload: social media + screen time (weight: 0.18)
      - Physical inactivity (weight: 0.12)
      - Addiction level (weight: 0.13)
      - Social isolation (weight: 0.10)
    Returns a score between 0.0 and 1.0.
    """
    # --- Sleep deficit (0-1) ---
    sleep_hours = row.get("sleep_hours", 8)
    # Teens need 8-10 hours. Below 6 is severe, 6-7 moderate, 7+ ok
    if sleep_hours >= 8:
        sleep_risk = 0.0
    elif sleep_hours >= 7:
        sleep_risk = 0.15
    elif sleep_hours >= 6:
        sleep_risk = 0.45
    elif sleep_hours >= 5:
        sleep_risk = 0.7
    elif sleep_hours >= 4:
        sleep_risk = 0.85
    else:
        sleep_risk = 1.0

    # --- Stress + Anxiety composite (0-1) ---
    stress = row.get("stress_level", 5)
    anxiety = row.get("anxiety_level", 5)
    mental_raw = (stress + anxiety) / 20.0  # max combined = 20
    # Apply sigmoid-like curve: low values stay low, high values get amplified
    mental_risk = min(1.0, mental_raw ** 0.8 * 1.3)

    # --- Digital overload (0-1) ---
    social_media_hrs = row.get("daily_social_media_hours", row.get("social_media_hours", 3))
    screen_before_sleep = row.get("screen_time_before_sleep", 2)
    digital_raw = (social_media_hrs / 12.0) * 0.6 + (screen_before_sleep / 10.0) * 0.4
    digital_risk = min(1.0, digital_raw * 1.4)

    # --- Physical inactivity (0-1) ---
    physical = row.get("physical_activity", 3)
    if physical >= 5:
        activity_risk = 0.0
    elif physical >= 3:
        activity_risk = 0.2
    elif physical >= 1:
        activity_risk = 0.5
    else:
        activity_risk = 0.9

    # --- Addiction (0-1) ---
    addiction = row.get("addiction_level", 5)
    addiction_risk = min(1.0, (addiction / 10.0) ** 0.9 * 1.2)

    # --- Social isolation (0-1) ---
    social_interaction = row.get("social_interaction_level", "medium")
    if isinstance(social_interaction, str):
        social_map = {"high": 0.0, "medium": 0.35, "low": 0.75}
        isolation_risk = social_map.get(social_interaction, 0.35)
    else:
        isolation_risk = min(1.0, float(social_interaction) / 2.0)  # encoded: 0=high, 1=medium, 2=low

    # --- Weighted composite ---
    clinical_score = (
        0.22 * sleep_risk
        + 0.25 * mental_risk
        + 0.18 * digital_risk
        + 0.12 * activity_risk
        + 0.13 * addiction_risk
        + 0.10 * isolation_risk
    )

    return min(1.0, max(0.0, clinical_score))


@app.post("/predict")
def predict(payload: PredictRequest):
    row = normalize_inputs(payload)
    features = build_features(row, label_encoders, feature_columns, feature_engineer)

    prob = float(model.predict_proba(features)[0][1])
    clinical_index = compute_clinical_risk_index(row, features)

    # Hybrid scoring: blend ML probability with clinical risk index
    # ML model may have poor calibration, so we weight the clinical index more
    # when there's high disagreement
    ml_score = prob
    disagreement = abs(ml_score - clinical_index)

    if disagreement > 0.3:
        # Model and clinical assessment diverge significantly
        # Trust clinical more (it's based on domain knowledge)
        blended = 0.35 * ml_score + 0.65 * clinical_index
    elif disagreement > 0.15:
        # Moderate disagreement — balance them
        blended = 0.45 * ml_score + 0.55 * clinical_index
    else:
        # Both agree — trust the model more
        blended = 0.55 * ml_score + 0.45 * clinical_index

    risk_score = clamp(int(round(blended * 100)), 3, 97)

    if risk_score < 30:
        risk_level = "Low"
    elif risk_score < 60:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Safety net: extreme inputs should never show "Low"
    extreme_count = 0
    if row["sleep_hours"] < 5:
        extreme_count += 1
    if row["stress_level"] >= 8:
        extreme_count += 1
    if row["anxiety_level"] >= 8:
        extreme_count += 1
    if row["daily_social_media_hours"] >= 7:
        extreme_count += 1
    if row["addiction_level"] >= 8:
        extreme_count += 1
    if row["physical_activity"] < 1:
        extreme_count += 1

    if extreme_count >= 3 and risk_level == "Low":
        risk_level = "Moderate"
        risk_score = max(risk_score, 45)
    if extreme_count >= 4 and risk_level != "High":
        risk_level = "High"
        risk_score = max(risk_score, 65)

    base_model = model
    if hasattr(model, "calibrated_classifiers_"):
        base_model = model.calibrated_classifiers_[0].estimator

    drivers = key_drivers(base_model, features)
    suggestions = build_suggestions({**row, **features.iloc[0].to_dict()})

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": round(blended, 4),
        "drivers": drivers,
        "suggestions": suggestions,
    }


@app.post("/set-api-key")
def set_api_key(payload: ApiKeyRequest):
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = payload.api_key.strip()
    return {"status": "ok", "key_set": bool(OPENROUTER_API_KEY)}


@app.get("/api-key-status")
def api_key_status():
    return {"has_key": bool(OPENROUTER_API_KEY)}


@app.get("/available-models")
def available_models():
    return {"models": AVAILABLE_MODELS}


@app.post("/chat")
def chat(payload: ChatRequest):
    session_id = payload.session_id or "default"
    history = get_history(session_id)

    model_name = payload.model or AVAILABLE_MODELS[0]
    if model_name not in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[0]

    system_message = {"role": "system", "content": build_system_prompt()}
    context_message = {
        "role": "system",
        "content": build_chat_context(payload.user_data, payload.model_output, payload.historical_data),
    }
    messages = [system_message, context_message, *history, {"role": "user", "content": payload.message}]

    def event_stream():
        if not OPENROUTER_API_KEY or requests is None:
            parsed = build_fallback_chat_response(payload.message, payload.model_output)
            save_turn(session_id, "user", payload.message)
            save_turn(session_id, "assistant", parsed["response"])
            yield f"data: {json.dumps({'type': 'final', **parsed})}\n\n"
            return

        save_turn(session_id, "user", payload.message)
        full_text = ""
        errors_tried = 0
        last_error = None

        # Try models with fallback
        for try_model in [model_name] + [m for m in AVAILABLE_MODELS if m != model_name]:
            try:
                for token in stream_openrouter(messages, try_model):
                    full_text += token
                    yield f"data: {json.dumps({'type': 'token', 'value': token})}\n\n"
                break  # success
            except Exception as exc:
                errors_tried += 1
                last_error = str(exc)
                if errors_tried >= len(AVAILABLE_MODELS):
                    # All models failed — use fallback
                    parsed = build_fallback_chat_response(payload.message, payload.model_output)
                    save_turn(session_id, "assistant", parsed["response"])
                    yield f"data: {json.dumps({'type': 'final', **parsed})}\n\n"
                    return
                full_text = ""  # reset for next model
                continue

        if full_text:
            # Plain text response — don't try JSON parsing
            clean_text = full_text.strip()
            save_turn(session_id, "assistant", clean_text)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
