"""Predictive models, anomaly scoring, hybrid rules, and RUL estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import DATA_DIR, REPORT_DIR, RANDOM_STATE


FEATURE_COLS = ["temperature", "vibration", "current", "pressure", "rpm", "alarm_count"]


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def risk_band_from_score(score: float) -> str:
    score = safe_float(score)
    if score >= 75:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def add_sequence_split(df: pd.DataFrame, split_mode: str = "asset") -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    for col in FEATURE_COLS + ["failure_label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + ["failure_label"]).copy()
    df["failure_label"] = df["failure_label"].astype(int)

    if split_mode == "global":
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["seq_id"] = np.arange(len(df))
        df["seq_count"] = len(df)
        df["seq_pct"] = df["seq_id"] / max(len(df) - 1, 1)
    else:
        df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
        df["seq_id"] = df.groupby("asset_id").cumcount()
        df["seq_count"] = df.groupby("asset_id")["seq_id"].transform("max") + 1
        df["seq_pct"] = df["seq_id"] / df["seq_count"].clip(lower=1)
    return df


def add_failure_target(df: pd.DataFrame, target_mode: str = "future_7d", horizon_steps: int = 168) -> pd.DataFrame:
    df = df.copy()
    if target_mode == "direct":
        df["failure_in_7d"] = df["failure_label"].astype(int)
        return df

    targets = []
    for _, group in df.groupby("asset_id", sort=False):
        y = group["failure_label"].astype(int)
        future_target = (
            y.iloc[::-1]
            .rolling(window=horizon_steps + 1, min_periods=1)
            .max()
            .iloc[::-1]
            .astype(int)
        )
        targets.append(future_target)
    df["failure_in_7d"] = pd.concat(targets).sort_index().astype(int)
    return df


def tune_threshold(y_true, proba, target_recall: float = 0.75) -> float:
    y_true = pd.Series(y_true).astype(int)
    if y_true.nunique() < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5

    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(precision[valid])]
        return float(thresholds[best_idx])

    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    return float(thresholds[int(np.argmax(f1))])


def slope_values(series: pd.Series, window: int = 24) -> pd.Series:
    out = []
    values = series.astype(float)
    for i in range(len(values)):
        x = values.iloc[max(0, i - window + 1) : i + 1].dropna().values
        out.append(0.0 if len(x) < 4 else float(np.polyfit(np.arange(len(x)), x, 1)[0]))
    return pd.Series(out, index=series.index)


def quick_slope(values) -> float:
    values = pd.Series(values).astype(float).dropna().values
    if len(values) < 4:
        return 0.0
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def train_model_block(df: pd.DataFrame, label: str, target_mode: str, split_mode: str) -> tuple:
    df = add_sequence_split(df, split_mode=split_mode)
    df = add_failure_target(df, target_mode=target_mode)

    x = df[FEATURE_COLS].copy()
    y = df["failure_in_7d"].astype(int)
    train_mask = df["seq_pct"] <= 0.70
    val_mask = (df["seq_pct"] > 0.70) & (df["seq_pct"] <= 0.85)
    test_mask = df["seq_pct"] > 0.85

    risk_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=7,
                    min_samples_leaf=6,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    anomaly_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", IsolationForest(n_estimators=250, contamination=0.08, random_state=RANDOM_STATE)),
        ]
    )

    risk_model.fit(x.loc[train_mask], y.loc[train_mask])
    anomaly_model.fit(x.loc[train_mask])

    threshold = 0.5
    if val_mask.sum() > 0:
        threshold = tune_threshold(y.loc[val_mask], risk_model.predict_proba(x.loc[val_mask])[:, 1])

    metrics = {
        "label": label,
        "target_mode": target_mode,
        "split_mode": split_mode,
        "best_threshold": float(threshold),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "positive_rate": float(y.mean()),
    }

    for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
        if mask.sum() > 0 and y.loc[mask].nunique() > 1:
            proba = risk_model.predict_proba(x.loc[mask])[:, 1]
            pred = (proba >= threshold).astype(int)
            metrics[f"{split_name}_roc_auc"] = float(roc_auc_score(y.loc[mask], proba))
            metrics[f"{split_name}_classification_report"] = classification_report(y.loc[mask], pred, output_dict=True)
        else:
            metrics[f"{split_name}_roc_auc"] = None

    return df, risk_model, anomaly_model, threshold, metrics


@dataclass
class ModelManager:
    steel_df: pd.DataFrame | None = None
    public_df: pd.DataFrame | None = None
    asset_health: pd.DataFrame | None = None
    steel_risk_model: object | None = None
    steel_anomaly_model: object | None = None
    public_risk_model: object | None = None
    public_anomaly_model: object | None = None
    steel_threshold: float = 0.5
    public_threshold: float = 0.5
    model_summary: dict | None = None

    def train_or_load(self, force: bool = False) -> "ModelManager":
        scored_path = DATA_DIR / "steel_sensor_logs_scored.csv"
        health_path = DATA_DIR / "asset_health_summary.csv"
        summary_path = REPORT_DIR / "model_summary.json"

        if not force and scored_path.exists() and health_path.exists():
            self.steel_df = pd.read_csv(scored_path)
            self.asset_health = pd.read_csv(health_path)
            public_path = DATA_DIR / "public_ai4i_common_schema.csv"
            self.public_df = pd.read_csv(public_path) if public_path.exists() else pd.DataFrame()
            if summary_path.exists():
                self.model_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.steel_threshold = float(self.model_summary.get("steel_threshold", 0.5))
                self.public_threshold = float(self.model_summary.get("public_threshold", 0.5))
            else:
                self.model_summary = {
                    "startup_mode": "precomputed_scored_data",
                    "important_note": "Cloud app reused committed scored steel data for fast startup.",
                }
            return self

        steel_raw = pd.read_csv(DATA_DIR / "steel_sensor_logs.csv")
        public_path = DATA_DIR / "public_ai4i_common_schema.csv"
        public_raw = pd.read_csv(public_path) if public_path.exists() else pd.DataFrame()

        if not public_raw.empty and public_raw["failure_label"].nunique() > 1:
            self.public_df, self.public_risk_model, self.public_anomaly_model, self.public_threshold, public_metrics = train_model_block(
                public_raw, "public_ai4i_benchmark", target_mode="direct", split_mode="global"
            )
        else:
            self.public_df = public_raw
            public_metrics = {"label": "public_ai4i_benchmark", "available": False}

        self.steel_df, self.steel_risk_model, self.steel_anomaly_model, self.steel_threshold, steel_metrics = train_model_block(
            steel_raw, "steel_demo_app", target_mode="future_7d", split_mode="asset"
        )
        self._score_steel()

        self.model_summary = {
            "public_model": public_metrics,
            "steel_demo_model": steel_metrics,
            "important_note": "AI4I is a leakage-free public benchmark. Steel app decisions use a separate steel model plus operational rules.",
            "steel_threshold": float(self.steel_threshold),
            "public_threshold": float(self.public_threshold),
            "hybrid_formula": "0.45 * ML failure risk + 0.55 * operational rule score",
        }
        (REPORT_DIR / "model_summary.json").write_text(json.dumps(self.model_summary, indent=2), encoding="utf-8")
        self.steel_df.to_csv(scored_path, index=False)
        self.asset_health.to_csv(health_path, index=False)
        return self

    def _score_steel(self) -> None:
        assert self.steel_df is not None
        x = self.steel_df[FEATURE_COLS].copy()
        self.steel_df["ml_failure_risk"] = self.steel_risk_model.predict_proba(x)[:, 1]
        self.steel_df["ml_failure_pred"] = (self.steel_df["ml_failure_risk"] >= self.steel_threshold).astype(int)
        self.steel_df["anomaly_score"] = -self.steel_anomaly_model.decision_function(x)
        self.steel_df["is_anomaly"] = (self.steel_anomaly_model.predict(x) == -1).astype(int)

        for col in ["temperature", "vibration", "pressure", "current"]:
            self.steel_df[f"{col}_slope_24h"] = 0.0
            for _, idx in self.steel_df.groupby("asset_id").groups.items():
                self.steel_df.loc[idx, f"{col}_slope_24h"] = slope_values(self.steel_df.loc[idx, col]).values

        self.steel_df["anomaly_events_24h"] = (
            self.steel_df.groupby("asset_id")["is_anomaly"].rolling(24, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        self.steel_df["operational_rule_score"] = self.steel_df.apply(self.operational_rule_score, axis=1)
        self.steel_df["hybrid_health_score"] = (
            0.45 * (self.steel_df["ml_failure_risk"] * 100) + 0.55 * self.steel_df["operational_rule_score"]
        ).clip(0, 100)
        self.steel_df["hybrid_failure_risk"] = (self.steel_df["hybrid_health_score"] / 100).round(4)
        self.steel_df["failure_risk"] = self.steel_df["hybrid_failure_risk"]
        self.steel_df["failure_pred"] = (self.steel_df["hybrid_health_score"] >= 55).astype(int)
        self.steel_df["risk_band"] = self.steel_df["hybrid_health_score"].apply(risk_band_from_score)
        self.steel_df["estimated_rul_days"] = self.steel_df.apply(self.estimate_rul, axis=1)
        self.refresh_asset_health()

    def refresh_asset_health(self) -> None:
        assert self.steel_df is not None
        tmp = self.steel_df.copy()
        tmp["_parsed_ts"] = pd.to_datetime(tmp["timestamp"], format="mixed", errors="coerce")
        tmp = tmp.sort_values(["asset_id", "_parsed_ts"]).drop(columns=["_parsed_ts"])
        self.asset_health = (
            tmp.groupby("asset_id")
            .tail(1)
            .sort_values(["hybrid_health_score", "estimated_rul_days"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def operational_rule_score(self, row) -> float:
        delay_map = {}
        delay_path = DATA_DIR / "delay_logs.csv"
        if delay_path.exists():
            delay_map = pd.read_csv(delay_path).set_index("asset_id")["delay_hours"].to_dict()

        typ = str(row.get("asset_type", "")).lower()
        criticality = str(row.get("criticality", "medium")).lower()
        temp = safe_float(row.get("temperature"))
        vib = safe_float(row.get("vibration"))
        current = safe_float(row.get("current"))
        pressure = safe_float(row.get("pressure"), 8)
        alarms = safe_float(row.get("alarm_count"))
        anomalies = safe_float(row.get("anomaly_events_24h"))
        delay = safe_float(delay_map.get(row.get("asset_id"), 0))
        score = 0.0

        if "gearbox" in typ:
            score += 30 if vib >= 7 else 0
            score += 20 if vib >= 9 else 0
            score += 10 if vib >= 10 else 0
            score += 8 if temp >= 65 else 0
        if "motor" in typ:
            score += 25 if temp >= 80 else 0
            score += 15 if temp >= 85 else 0
            score += 12 if current >= 80 else 0
            score += 8 if vib >= 5 else 0
        if "pump" in typ:
            score += 22 if pressure <= 6 else 0
            score += 12 if vib >= 5 else 0
            score += 8 if current >= 75 else 0
        if "hydraulic" in typ:
            score += 25 if pressure <= 6 else 0
            score += 10 if temp >= 65 else 0
            score += 8 if alarms >= 2 else 0

        score += 8 if alarms >= 2 else 0
        score += 8 if alarms >= 4 else 0
        score += 15 if anomalies >= 6 else 0
        score += 10 if anomalies >= 18 else 0
        score += {"critical": 15, "high": 10, "medium": 5, "low": 0}.get(criticality, 5)
        score += min(delay * 2.0, 12)
        return round(float(np.clip(score, 0, 100)), 2)

    def estimate_rul(self, row) -> float:
        risk = safe_float(row.get("hybrid_failure_risk", row.get("failure_risk", 0)))
        anomaly_events = safe_float(row.get("anomaly_events_24h", 0))
        criticality = str(row.get("criticality", "medium")).lower()
        rul = 50 * (1 - risk)
        rul -= anomaly_events * 0.45
        rul -= max(0, safe_float(row.get("temperature_slope_24h", 0))) * 4.0
        rul -= max(0, safe_float(row.get("vibration_slope_24h", 0))) * 8.0
        rul -= max(0, -safe_float(row.get("pressure_slope_24h", 0))) * 5.0
        rul -= {"critical": 5, "high": 3, "medium": 1, "low": 0}.get(criticality, 1)
        return round(float(np.clip(rul, 1.0, 60.0)), 1)

    def score_live_alert(self, row: dict) -> dict:
        assert self.steel_df is not None
        if self.steel_risk_model is None or self.steel_anomaly_model is None:
            self.train_or_load(force=True)
        prev = self.steel_df[self.steel_df["asset_id"] == row["asset_id"]].tail(23).copy()
        x = pd.DataFrame([row])[FEATURE_COLS].copy()
        row["ml_failure_risk"] = float(self.steel_risk_model.predict_proba(x)[0, 1])
        row["ml_failure_pred"] = int(row["ml_failure_risk"] >= self.steel_threshold)
        row["anomaly_score"] = float(-self.steel_anomaly_model.decision_function(x)[0])
        row["is_anomaly"] = int(self.steel_anomaly_model.predict(x)[0] == -1)
        row["anomaly_events_24h"] = int(prev["is_anomaly"].sum() + row["is_anomaly"])

        for col in ["temperature", "vibration", "pressure", "current"]:
            row[f"{col}_slope_24h"] = round(quick_slope(list(prev[col].values) + [row[col]]), 4)

        row["operational_rule_score"] = self.operational_rule_score(row)
        row["hybrid_health_score"] = round(
            float(np.clip(0.45 * (row["ml_failure_risk"] * 100) + 0.55 * row["operational_rule_score"], 0, 100)),
            2,
        )
        row["hybrid_failure_risk"] = round(row["hybrid_health_score"] / 100, 4)
        row["failure_risk"] = row["hybrid_failure_risk"]
        row["failure_pred"] = int(row["hybrid_health_score"] >= 55)
        row["risk_band"] = risk_band_from_score(row["hybrid_health_score"])
        row["estimated_rul_days"] = self.estimate_rul(row)

        self.steel_df = pd.concat([self.steel_df, pd.DataFrame([row])], ignore_index=True, sort=False)
        self.steel_df.to_csv(DATA_DIR / "steel_sensor_logs_scored.csv", index=False)
        self.steel_df.to_csv(DATA_DIR / "sensor_logs_scored.csv", index=False)
        self.refresh_asset_health()
        self.asset_health.to_csv(DATA_DIR / "asset_health_summary.csv", index=False)
        return row
