from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

RAW_FEATURES = [
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
    "stress_level",
    "anxiety_level",
    "addiction_level",
]

ENGINEERED_FEATURES = [
    "sleep_deficit",
    "digital_overload",
    "night_risk",
    "mental_pressure",
    "isolation_score",
    "addiction_impact",
    "lifestyle_imbalance",
]

FEATURE_GROUPS = {
    "digital_behavior": ["digital_overload", "night_risk"],
    "sleep_health": ["sleep_deficit"],
    "mental_state": ["mental_pressure"],
    "social_behavior": ["isolation_score"],
    "risk_amplifiers": ["addiction_impact"],
    "lifestyle_balance": ["lifestyle_imbalance"],
}

SOCIAL_INTERACTION_MAP = {"low": 2, "medium": 1, "high": 0}
SOCIAL_MEDIA_ALIASES = ("daily_social_media_hours", "social_media_hours")


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column for {label}. Tried: {', '.join(candidates)}")


def _safe_numeric(series: pd.Series, fill_value: Optional[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if fill_value is None or pd.isna(fill_value):
        fill_value = numeric.median()
    if pd.isna(fill_value):
        fill_value = 0.0
    numeric = numeric.fillna(fill_value)
    return numeric.clip(lower=0)


def _encode_social_interaction(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        encoded = pd.to_numeric(series, errors="coerce")
        return encoded.fillna(mapping["medium"]).clip(lower=0)

    cleaned = series.astype(str).str.lower().str.strip()
    cleaned = cleaned.where(cleaned.isin(mapping), "medium")
    return cleaned.map(mapping).fillna(mapping["medium"]).astype(float)


class FeatureEngineer:
    def __init__(
        self,
        ideal_sleep: float = 8.0,
        social_interaction_map: Optional[Dict[str, int]] = None,
        fill_values: Optional[Dict[str, float]] = None,
        social_media_aliases: Optional[Iterable[str]] = None,
    ) -> None:
        self.ideal_sleep = ideal_sleep
        self.social_interaction_map = social_interaction_map or SOCIAL_INTERACTION_MAP
        self.social_media_aliases = tuple(social_media_aliases or SOCIAL_MEDIA_ALIASES)
        self.fill_values: Dict[str, float] = fill_values or {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        social_media_col = _resolve_column(df, self.social_media_aliases, "social_media_hours")
        numeric_cols = [
            social_media_col,
            "sleep_hours",
            "screen_time_before_sleep",
            "stress_level",
            "anxiety_level",
            "addiction_level",
            "physical_activity",
        ]

        for col in numeric_cols:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                median = numeric.median()
                if pd.isna(median):
                    median = 0.0
                self.fill_values[col] = float(median)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        social_media_col = _resolve_column(df_out, self.social_media_aliases, "social_media_hours")
        social_media = _safe_numeric(df_out[social_media_col], self.fill_values.get(social_media_col))
        screen_before = _safe_numeric(
            df_out["screen_time_before_sleep"],
            self.fill_values.get("screen_time_before_sleep"),
        )
        sleep_hours = _safe_numeric(df_out["sleep_hours"], self.fill_values.get("sleep_hours"))
        stress = _safe_numeric(df_out["stress_level"], self.fill_values.get("stress_level"))
        anxiety = _safe_numeric(df_out["anxiety_level"], self.fill_values.get("anxiety_level"))
        addiction = _safe_numeric(df_out["addiction_level"], self.fill_values.get("addiction_level"))
        physical = _safe_numeric(df_out["physical_activity"], self.fill_values.get("physical_activity"))
        social_encoded = _encode_social_interaction(
            df_out["social_interaction_level"],
            self.social_interaction_map,
        )

        df_out["sleep_deficit"] = (self.ideal_sleep - sleep_hours).clip(lower=0)
        df_out["digital_overload"] = social_media + screen_before
        df_out["night_risk"] = screen_before * df_out["sleep_deficit"]
        df_out["mental_pressure"] = stress + anxiety
        df_out["isolation_score"] = social_encoded * df_out["digital_overload"]
        df_out["addiction_impact"] = addiction * screen_before
        df_out["lifestyle_imbalance"] = df_out["digital_overload"] - physical

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def add_engineered_features(
    df: pd.DataFrame,
    feature_engineer: Optional[FeatureEngineer] = None,
    fit: bool = False,
) -> pd.DataFrame:
    engineer = feature_engineer or FeatureEngineer()
    if fit or not engineer.fill_values:
        engineer.fit(df)
    return engineer.transform(df)
