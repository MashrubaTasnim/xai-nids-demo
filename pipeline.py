"""
XAI-NIDS inference pipeline.

Reproduces, step by step, the preprocessing used in the UNSW-NB15 training notebook:

    raw record
      -> add_features (8 engineered features; byte_ratio clipped at the TRAIN 99th percentile)
      -> drop id / attack_cat / label
      -> LabelEncoder per categorical column (fit on train; unseen category -> -1)
      -> median imputation (fit on train)
      -> StandardScaler (fit on the balanced train set)
      -> XGBoost  ->  SHAP TreeExplainer

No Streamlit imports here, so this module can be tested on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ART_DIR = Path(__file__).parent / "artifacts"

ENGINEERED = [
    "total_bytes", "total_packets", "bytes_per_second", "packets_per_second",
    "byte_ratio", "packet_ratio", "src_header_ratio", "payload_diff",
]


@dataclass
class Artifacts:
    model: object
    explainer: object
    encoders: dict
    imputer: object
    scaler: object
    feature_names: list
    meta: dict

    @property
    def cat_cols(self) -> list:
        return list(self.meta["cat_cols"])

    @property
    def raw_features(self) -> list:
        """The 42 columns a user has to supply (model features minus the engineered ones)."""
        return [f for f in self.feature_names if f not in ENGINEERED]

    @property
    def num_raw_features(self) -> list:
        return [f for f in self.raw_features if f not in self.cat_cols]


def load_artifacts(art_dir: Path = ART_DIR) -> Artifacts:
    with open(art_dir / "metadata.json") as f:
        meta = json.load(f)
    return Artifacts(
        model=joblib.load(art_dir / "best_model.pkl"),
        explainer=joblib.load(art_dir / "shap_explainer.pkl"),
        encoders=joblib.load(art_dir / "label_encoders.pkl"),
        imputer=joblib.load(art_dir / "imputer.pkl"),
        scaler=joblib.load(art_dir / "scaler.pkl"),
        feature_names=list(joblib.load(art_dir / "feature_names.pkl")),
        meta=meta,
    )


def add_features(df: pd.DataFrame, eps: float, byte_ratio_cap: float) -> pd.DataFrame:
    """Same formulas as `add_features` in the notebook (Section 3.2)."""
    df = df.copy()
    df["total_bytes"] = df["sbytes"] + df["dbytes"]
    df["total_packets"] = df["spkts"] + df["dpkts"]
    df["bytes_per_second"] = df["total_bytes"] / (df["dur"] + eps)
    df["packets_per_second"] = df["total_packets"] / (df["dur"] + eps)
    df["byte_ratio"] = df["sbytes"] / (df["dbytes"] + 1)
    df["packet_ratio"] = df["spkts"] / (df["dpkts"] + 1)
    df["src_header_ratio"] = df["sload"] / (df["bytes_per_second"] + eps)
    df["payload_diff"] = (df["sbytes"] - df["dbytes"]).abs()
    df["byte_ratio"] = df["byte_ratio"].clip(upper=byte_ratio_cap)
    return df


def preprocess(raw_df: pd.DataFrame, art: Artifacts):
    """
    raw_df: rows in the original UNSW-NB15 schema (extra columns such as id / attack_cat /
            label are allowed and ignored).
    Returns (X_display, X_scaled):
        X_display - imputed, UNscaled feature matrix (for showing human-readable values)
        X_scaled  - what the model actually receives
    """
    df = raw_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in art.raw_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Expected the UNSW-NB15 raw schema (e.g. the official CSV files)."
        )

    for c in art.num_raw_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_features(df, art.meta["eps"], art.meta["byte_ratio_cap"])
    X = df[art.feature_names].copy()  # also fixes column order and drops id/attack_cat/label

    for col in art.cat_cols:
        mapping = {cls: idx for idx, cls in enumerate(art.encoders[col].classes_)}
        X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    X_imp = art.imputer.transform(X)
    X_scaled = art.scaler.transform(X_imp)
    return X_imp, X_scaled


def predict_proba_attack(art: Artifacts, X_scaled: np.ndarray) -> np.ndarray:
    return art.model.predict_proba(X_scaled)[:, 1]


def shap_values(art: Artifacts, X_scaled: np.ndarray) -> np.ndarray:
    """Returns an (n_rows, n_features) array of SHAP values (log-odds of 'Attack')."""
    sv = art.explainer.shap_values(X_scaled)
    if isinstance(sv, list):          # some versions return one array per class
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[..., -1]
    return sv


def base_value(art: Artifacts) -> float:
    return float(np.ravel(art.explainer.expected_value)[-1])


def decode_row_for_display(X_imp_row: np.ndarray, art: Artifacts) -> list:
    """Human-readable values for one row: decode categorical codes back to their names."""
    out = []
    for name, val in zip(art.feature_names, X_imp_row):
        if name in art.cat_cols:
            classes = art.encoders[name].classes_
            idx = int(round(val))
            out.append(classes[idx] if 0 <= idx < len(classes) else "(unseen)")
        else:
            out.append(float(val))
    return out


def top_contributions(sv_row: np.ndarray, display_values: list, art: Artifacts, k: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": art.feature_names,
        "value": [str(v) if isinstance(v, str) else f"{v:,.6g}" for v in display_values],
        "shap": sv_row,
    })
    df["abs"] = df["shap"].abs()
    df = df.sort_values("abs", ascending=False).head(k).drop(columns="abs").reset_index(drop=True)
    return df


def default_raw_row(art: Artifacts) -> dict:
    """Training-set medians for every raw feature (categoricals decoded to their names)."""
    stats = dict(zip(art.feature_names, art.imputer.statistics_))
    row = {}
    for f in art.raw_features:
        if f in art.cat_cols:
            classes = art.encoders[f].classes_
            idx = int(np.clip(round(stats[f]), 0, len(classes) - 1))
            row[f] = classes[idx]
        else:
            row[f] = float(stats[f])
    return row
