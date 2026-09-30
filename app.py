from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

APP_VERSION = "1.0.0"
RANDOM_SEED = 42
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_ALIASES = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "september": 9, "sept": 9, "septiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}
REQUIRED_SHEETS = {"Actuals", "PY", "PM Database"}
COST_KEYWORDS = ("labor", "utilit", "muv", "faulty", "maintenance", "loss", "cost", "expense", "nicc", "fixed")
HIGHER_KEYWORDS = ("volume", "saving", "productiv", "ebitda", "revenue", "throughput")
CAT_FEATURES = ["BU", "POD", "Plant", "Cost_Bucket"]
NUM_FEATURES = [
    "Month", "Month_sin", "Month_cos", "Prior_Year_2025", "Current_Forecast_8_4",
    "Actual_YTD_Mean", "Actual_YTD_Sum", "Actual_Last", "Actual_Trend", "History_Count",
] + [f"Actual_{m}" for m in MONTHS[:8]]

TEXT = {
    "ES": {
        "title": "Performance Model · 8+4 Intelligence",
        "subtitle": "Pronóstico financiero 2026 por Planta × Cost Bucket × Mes",
        "upload": "1 · Cargar y validar",
        "upload_help": "Excel original, CSV normalizado o plantilla estructurada. El archivo fuente nunca se guarda en el proyecto.",
        "quality": "Calidad",
        "config": "Configuración",
        "train": "Entrenar / actualizar modelo",
        "dashboard": "Dashboard",
        "explain": "Explicabilidad",
        "diagnostics": "Diagnósticos",
        "downloads": "Descargas",
        "no_data": "Carga el Excel original o un CSV normalizado para comenzar.",
        "valid": "Datos validados y normalizados correctamente.",
        "mode": "Modo",
        "executive": "Ejecutivo",
        "advanced": "Avanzado",
        "reference": "Referencia principal",
        "forecast_ref": "Current Forecast 8+4",
        "py_ref": "Prior Year 2025",
        "pct_threshold": "Umbral porcentual",
        "materiality": "Materialidad (USD)",
        "filters": "Filtros globales",
        "all": "Todos",
        "scenario": "Escenarios what-if",
        "actual_adj": "Ajustes enero–agosto",
        "trained": "Modelo entrenado y resultados actualizados.",
        "save_model": "Descargar modelo entrenado",
        "reuse_model": "Reutilizar modelo (.joblib)",
        "new_categories": "Se detectaron categorías nuevas. Se aplicará respaldo jerárquico, menor confianza y se recomienda reentrenar.",
        "incompatible": "El modelo cargado no es compatible con el esquema o la versión de la aplicación.",
        "newer": "Los datos cargados son más recientes que los usados por el modelo. Se recomienda reentrenar.",
        "favorable": "Favorable", "risk": "En riesgo", "unfavorable": "Desfavorable",
        "train_first": "Entrena un modelo o carga uno compatible.",
        "unit_note": "Valores de origen normalizados a miles de USD; la interfaz y las descargas monetarias muestran USD.",
    },
    "EN": {
        "title": "Performance Model · 8+4 Intelligence",
        "subtitle": "2026 financial forecast by Plant × Cost Bucket × Month",
        "upload": "1 · Upload and validate",
        "upload_help": "Original Excel, normalized CSV, or structured template. The source file is never stored in the project.",
        "quality": "Quality",
        "config": "Configuration",
        "train": "Train / refresh model",
        "dashboard": "Dashboard",
        "explain": "Explainability",
        "diagnostics": "Diagnostics",
        "downloads": "Downloads",
        "no_data": "Upload the original Excel or a normalized CSV to begin.",
        "valid": "Data validated and normalized successfully.",
        "mode": "Mode",
        "executive": "Executive",
        "advanced": "Advanced",
        "reference": "Primary reference",
        "forecast_ref": "Current Forecast 8+4",
        "py_ref": "Prior Year 2025",
        "pct_threshold": "Percentage threshold",
        "materiality": "Materiality (USD)",
        "filters": "Global filters",
        "all": "All",
        "scenario": "What-if scenarios",
        "actual_adj": "January–August adjustments",
        "trained": "Model trained and results refreshed.",
        "save_model": "Download trained model",
        "reuse_model": "Reuse model (.joblib)",
        "new_categories": "New categories detected. Hierarchical fallback and lower confidence will be used; retraining is recommended.",
        "incompatible": "The uploaded model is incompatible with the schema or app version.",
        "newer": "Loaded data is newer than the model training data. Retraining is recommended.",
        "favorable": "Favorable", "risk": "At risk", "unfavorable": "Unfavorable",
        "train_first": "Train a model or load a compatible one.",
        "unit_note": "Source values are normalized to USD thousands; the UI and monetary downloads display USD.",
    },
}


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return " ".join(str(value).strip().split()) or "UNKNOWN"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_text(c).replace("\n", " ") for c in out.columns]
    return out


def norm_key(value: Any) -> str:
    return clean_text(value).upper()


def to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def file_bytes(uploaded: Any) -> bytes:
    if isinstance(uploaded, bytes):
        return uploaded
    if hasattr(uploaded, "getvalue"):
        return uploaded.getvalue()
    if hasattr(uploaded, "read"):
        pos = uploaded.tell() if hasattr(uploaded, "tell") else 0
        data = uploaded.read()
        if hasattr(uploaded, "seek"):
            uploaded.seek(pos)
        return data
    with open(uploaded, "rb") as handle:
        return handle.read()


def schema_hash(panel: pd.DataFrame) -> str:
    cols = sorted(panel.columns.astype(str).tolist())
    keys = sorted(panel["Cost_Bucket"].dropna().astype(str).unique().tolist())
    return hashlib.sha256(json.dumps([cols, keys]).encode()).hexdigest()[:16]


def month_to_num(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)) and 1 <= float(value) <= 12:
        return int(value)
    text = clean_text(value).lower()
    if text in MONTH_ALIASES:
        return MONTH_ALIASES[text]
    try:
        parsed = pd.to_datetime(value)
        return int(parsed.month)
    except Exception:
        return np.nan


def metric_nature(bucket: str) -> str:
    key = clean_text(bucket).lower()
    if any(k in key for k in HIGHER_KEYWORDS):
        return "Higher is favorable / Mayor es favorable"
    if any(k in key for k in COST_KEYWORDS):
        return "Lower cost/loss is favorable / Menor costo o pérdida es favorable"
    return "Higher EBITDA impact is favorable / Mayor impacto EBITDA es favorable"


def _read_excel(data: bytes, sheet: str, header: int) -> pd.DataFrame:
    return clean_columns(pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=header, engine="openpyxl"))


def _wide_to_long(df: pd.DataFrame, id_cols: list[str], month_cols: list[str], value_name: str) -> pd.DataFrame:
    available = [c for c in month_cols if c in df.columns]
    out = df[id_cols + available].melt(id_vars=id_cols, value_vars=available, var_name="Month_Name", value_name=value_name)
    out["Month"] = out["Month_Name"].map({m: i + 1 for i, m in enumerate(month_cols)})
    out[value_name] = to_number(out[value_name])
    return out.drop(columns="Month_Name")


def parse_original_excel(data: bytes) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    missing_sheets = sorted(REQUIRED_SHEETS - set(xl.sheet_names))
    if missing_sheets:
        raise ValueError(f"Missing required sheets / Faltan hojas requeridas: {', '.join(missing_sheets)}")

    actual = _read_excel(data, "Actuals", 3).dropna(how="all")
    prior = _read_excel(data, "PY", 3).dropna(how="all")
    forecast = _read_excel(data, "PM Database", 2).dropna(how="all")

    required_actual = {"Plant", "Database Cost Bucket", *MONTHS}
    required_forecast = {"Plant", "Cost Bucket", *MONTH_FULL}
    if not required_actual.issubset(actual.columns) or not required_actual.issubset(prior.columns):
        raise ValueError("Actuals/PY columns are incompatible with the expected workbook structure.")
    if not required_forecast.issubset(forecast.columns):
        raise ValueError("PM Database columns are incompatible with the expected workbook structure.")

    exact_dupes = int(actual.duplicated().sum() + prior.duplicated().sum() + forecast.duplicated().sum())
    actual = actual.drop_duplicates()
    prior = prior.drop_duplicates()
    forecast = forecast.drop_duplicates()

    a = _wide_to_long(actual, ["Plant", "Database Cost Bucket"], MONTHS, "Actual_2026")
    p = _wide_to_long(prior, ["Plant", "Database Cost Bucket"], MONTHS, "Prior_Year_2025")
    a = a.rename(columns={"Database Cost Bucket": "Cost_Bucket"})
    p = p.rename(columns={"Database Cost Bucket": "Cost_Bucket"})

    fmap_cols = [c for c in ["BU", "Pod", "Plant", "Cost Bucket"] if c in forecast.columns]
    f = _wide_to_long(forecast, fmap_cols, MONTH_FULL, "Current_Forecast_8_4")
    f = f.rename(columns={"Pod": "POD", "Cost Bucket": "Cost_Bucket"})

    for frame in (a, p, f):
        frame["Plant"] = frame["Plant"].map(norm_key)
        frame["Cost_Bucket"] = frame["Cost_Bucket"].map(clean_text)
        frame["Cost_Key"] = frame["Cost_Bucket"].map(norm_key)

    unmapped_cost_rows = int((a["Cost_Key"] == "UNKNOWN").sum() + (p["Cost_Key"] == "UNKNOWN").sum())
    a = a[a["Cost_Key"] != "UNKNOWN"]
    p = p[p["Cost_Key"] != "UNKNOWN"]
    f = f[f["Cost_Key"] != "UNKNOWN"]

    a = a.groupby(["Plant", "Cost_Key", "Month"], as_index=False)["Actual_2026"].sum(min_count=1)
    p = p.groupby(["Plant", "Cost_Key", "Month"], as_index=False)["Prior_Year_2025"].sum(min_count=1)
    f["BU"] = f["BU"].map(clean_text) if "BU" in f.columns else "UNKNOWN"
    f["POD"] = f["POD"].map(clean_text) if "POD" in f.columns else "UNKNOWN"
    mapping = f.sort_values(["Plant", "Cost_Key"]).drop_duplicates(["Plant", "Cost_Key"])[["Plant", "Cost_Key", "BU", "POD", "Cost_Bucket"]]
    fg = f.groupby(["Plant", "Cost_Key", "Month"], as_index=False)["Current_Forecast_8_4"].sum(min_count=1)

    keys = pd.concat([a[["Plant", "Cost_Key", "Month"]], p[["Plant", "Cost_Key", "Month"]], fg[["Plant", "Cost_Key", "Month"]]], ignore_index=True).drop_duplicates()
    panel = keys.merge(a, on=["Plant", "Cost_Key", "Month"], how="left")
    panel = panel.merge(p, on=["Plant", "Cost_Key", "Month"], how="left")
    panel = panel.merge(fg, on=["Plant", "Cost_Key", "Month"], how="left")
    panel = panel.merge(mapping, on=["Plant", "Cost_Key"], how="left")
    panel["Cost_Bucket"] = panel["Cost_Bucket"].fillna(panel["Cost_Key"].str.title())

    plant_map = forecast.assign(Plant=forecast["Plant"].map(norm_key)).drop_duplicates("Plant")
    plant_map = plant_map.set_index("Plant")
    for col, source_col in [("BU", "BU"), ("POD", "Pod")]:
        fallback = panel["Plant"].map(plant_map[source_col].to_dict()) if source_col in plant_map else "UNKNOWN"
        panel[col] = panel[col].fillna(fallback).map(clean_text)

    panel["Actual_Impact"] = panel["Prior_Year_2025"] - panel["Actual_2026"]
    forecast_derived = int(panel["Current_Forecast_8_4"].isna().sum())
    panel["Current_Forecast_8_4"] = panel["Current_Forecast_8_4"].fillna(panel["Actual_Impact"])
    panel.loc[panel["Month"] > 8, ["Actual_2026", "Actual_Impact"]] = np.nan
    panel["Metric_Nature"] = panel["Cost_Bucket"].map(metric_nature)
    panel["Unit_Scale"] = 1000.0
    panel["Source"] = "Original Excel"
    panel = panel.sort_values(["BU", "POD", "Plant", "Cost_Bucket", "Month"]).reset_index(drop=True)

    missing_by_col = panel[["Actual_2026", "Prior_Year_2025", "Current_Forecast_8_4"]].isna().sum()
    quality = pd.DataFrame([
        ["Rows after normalization", len(panel), "info"],
        ["Exact duplicates removed", exact_dupes, "warning" if exact_dupes else "ok"],
        ["Plants", panel["Plant"].nunique(), "info"],
        ["Cost buckets", panel["Cost_Bucket"].nunique(), "info"],
        ["Rows excluded without cost-bucket mapping", unmapped_cost_rows, "warning" if unmapped_cost_rows else "ok"],
        ["Missing Actual 2026 (Jan-Aug)", int(panel.loc[panel.Month <= 8, "Actual_2026"].isna().sum()), "warning"],
        ["Missing Prior Year 2025", int(missing_by_col["Prior_Year_2025"]), "warning"],
        ["Missing Current Forecast 8+4", int(missing_by_col["Current_Forecast_8_4"]), "warning"],
        ["Forecast cells derived from workbook PY–Actual", forecast_derived, "info"],
        ["FY/Q1-Q4 aggregate columns excluded", 1, "ok"],
    ], columns=["Check", "Value", "Status"])
    meta = {
        "source_type": "original_excel", "unit_scale": 1000.0, "year": 2026,
        "sheets": xl.sheet_names, "file_hash": hashlib.sha256(data).hexdigest()[:16],
    }
    return panel, quality, meta


def parse_normalized_csv(data: bytes) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = clean_columns(pd.read_csv(io.BytesIO(data))).dropna(how="all")
    aliases = {
        "plant": "Plant", "planta": "Plant", "cost bucket": "Cost_Bucket", "cost_bucket": "Cost_Bucket",
        "month": "Month", "mes": "Month", "bu": "BU", "pod": "POD",
        "actual 2026": "Actual_2026", "actual_2026": "Actual_2026", "actuals 2026": "Actual_2026",
        "prior year 2025": "Prior_Year_2025", "prior_year_2025": "Prior_Year_2025", "py 2025": "Prior_Year_2025",
        "current forecast 8+4": "Current_Forecast_8_4", "current_forecast_8_4": "Current_Forecast_8_4", "forecast 8+4": "Current_Forecast_8_4",
        "unit scale": "Unit_Scale", "unit_scale": "Unit_Scale", "metric direction": "Metric_Nature", "metric_nature": "Metric_Nature",
    }
    raw = raw.rename(columns={c: aliases.get(c.lower().strip(), c) for c in raw.columns})
    required = {"Plant", "Cost_Bucket", "Month", "Actual_2026", "Prior_Year_2025", "Current_Forecast_8_4"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Missing CSV columns / Faltan columnas CSV: " + ", ".join(missing))
    duplicates = int(raw.duplicated().sum())
    panel = raw.drop_duplicates().copy()
    panel["Month"] = panel["Month"].map(month_to_num)
    invalid_months = int(panel["Month"].isna().sum())
    panel = panel.dropna(subset=["Month"])
    panel["Month"] = panel["Month"].astype(int)
    panel = panel[panel["Month"].between(1, 12)]
    for col in ["Actual_2026", "Prior_Year_2025", "Current_Forecast_8_4"]:
        panel[col] = to_number(panel[col])
    panel["Plant"] = panel["Plant"].map(norm_key)
    panel["Cost_Bucket"] = panel["Cost_Bucket"].map(clean_text)
    panel["Cost_Key"] = panel["Cost_Bucket"].map(norm_key)
    panel["BU"] = panel.get("BU", "UNKNOWN")
    panel["POD"] = panel.get("POD", panel["BU"])
    panel["BU"] = panel["BU"].map(clean_text)
    panel["POD"] = panel["POD"].map(clean_text)
    panel["Unit_Scale"] = to_number(panel.get("Unit_Scale", pd.Series(1000.0, index=panel.index))).fillna(1000.0)
    panel["Metric_Nature"] = panel.get("Metric_Nature", panel["Cost_Bucket"].map(metric_nature))
    panel["Metric_Nature"] = panel["Metric_Nature"].fillna(panel["Cost_Bucket"].map(metric_nature))
    panel["Actual_Impact"] = panel["Prior_Year_2025"] - panel["Actual_2026"]
    panel.loc[panel["Month"] > 8, ["Actual_2026", "Actual_Impact"]] = np.nan
    panel["Source"] = "Normalized CSV"
    panel = panel.sort_values(["Plant", "Cost_Bucket", "Month"]).reset_index(drop=True)
    quality = pd.DataFrame([
        ["Rows after normalization", len(panel), "info"],
        ["Exact duplicates removed", duplicates, "warning" if duplicates else "ok"],
        ["Invalid months removed", invalid_months, "warning" if invalid_months else "ok"],
        ["Missing Actual 2026 (Jan-Aug)", int(panel.loc[panel.Month <= 8, "Actual_2026"].isna().sum()), "warning"],
        ["Missing Prior Year 2025", int(panel["Prior_Year_2025"].isna().sum()), "warning"],
        ["Missing Current Forecast 8+4", int(panel["Current_Forecast_8_4"].isna().sum()), "warning"],
    ], columns=["Check", "Value", "Status"])
    meta = {"source_type": "normalized_csv", "unit_scale": float(panel["Unit_Scale"].median()), "year": 2026, "file_hash": hashlib.sha256(data).hexdigest()[:16]}
    return panel, quality, meta


def load_source(data: bytes, name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if name.lower().endswith(".csv"):
        return parse_normalized_csv(data)
    if name.lower().endswith((".xlsx", ".xlsm")):
        return parse_original_excel(data)
    raise ValueError("Unsupported file type / Tipo de archivo no compatible")


def apply_cleaning(panel: pd.DataFrame, imputation: str, outlier_mode: str, outlier_limit: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    numeric = ["Actual_2026", "Prior_Year_2025", "Current_Forecast_8_4", "Actual_Impact"]
    before_missing = int(out[numeric].isna().sum().sum())
    if imputation == "Zero / Cero":
        out[numeric] = out[numeric].fillna(0.0)
    elif imputation == "Group median / Mediana por grupo":
        for col in numeric:
            med = out.groupby(["BU", "Cost_Bucket", "Month"])[col].transform("median")
            out[col] = out[col].fillna(med).fillna(out[col].median()).fillna(0.0)
    else:
        out = out.dropna(subset=["Prior_Year_2025", "Current_Forecast_8_4"])
        out.loc[out.Month <= 8, "Actual_Impact"] = out.loc[out.Month <= 8, "Actual_Impact"].fillna(0.0)

    treated = 0
    if outlier_mode != "Keep / Conservar":
        for col in ["Actual_Impact", "Prior_Year_2025", "Current_Forecast_8_4"]:
            values = out[col].dropna()
            if values.empty:
                continue
            if outlier_mode == "Winsorize / Winsorizar":
                lo, hi = values.quantile([outlier_limit, 1 - outlier_limit])
            else:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            mask = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
            treated += int(mask.sum())
            out.loc[mask, col] = out.loc[mask, col].clip(lo, hi)
    report = pd.DataFrame([
        ["Missing numeric cells before imputation", before_missing],
        ["Rows retained", len(out)],
        ["Extreme values treated", treated],
        ["Imputation", imputation],
        ["Outlier treatment", outlier_mode],
    ], columns=["Cleaning step", "Result"])
    return out.reset_index(drop=True), report


def build_feature_rows(panel: pd.DataFrame, target_months: list[int], history_cutoff: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {(p, b): g.set_index("Month") for (p, b), g in panel.groupby(["Plant", "Cost_Bucket"], sort=False)}
    for _, r in panel[panel["Month"].isin(target_months)].iterrows():
        m = int(r["Month"])
        cutoff = min(m - 1, history_cutoff if history_cutoff is not None else m - 1)
        hist_group = grouped[(r["Plant"], r["Cost_Bucket"])]
        hist = hist_group.loc[hist_group.index <= cutoff, "Actual_Impact"].dropna().sort_index()
        record = {
            "BU": r["BU"], "POD": r["POD"], "Plant": r["Plant"], "Cost_Bucket": r["Cost_Bucket"],
            "Month": m, "Month_sin": math.sin(2 * math.pi * m / 12), "Month_cos": math.cos(2 * math.pi * m / 12),
            "Prior_Year_2025": r["Prior_Year_2025"], "Current_Forecast_8_4": r["Current_Forecast_8_4"],
            "Actual_YTD_Mean": hist.mean() if len(hist) else np.nan,
            "Actual_YTD_Sum": hist.sum() if len(hist) else 0.0,
            "Actual_Last": hist.iloc[-1] if len(hist) else np.nan,
            "Actual_Trend": (hist.iloc[-1] - hist.iloc[0]) / max(len(hist) - 1, 1) if len(hist) > 1 else 0.0,
            "History_Count": len(hist), "Target": r["Actual_Impact"], "Unit_Scale": r.get("Unit_Scale", 1000.0),
        }
        for idx, mon in enumerate(MONTHS[:8], start=1):
            record[f"Actual_{mon}"] = hist_group.at[idx, "Actual_Impact"] if idx <= cutoff and idx in hist_group.index else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def make_pipeline(cfg: dict[str, Any]) -> Pipeline:
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
    prep = ColumnTransformer([("cat", categorical, CAT_FEATURES), ("num", numeric, NUM_FEATURES)], remainder="drop")
    rf = RandomForestRegressor(
        n_estimators=int(cfg["n_estimators"]), max_depth=cfg["max_depth"], min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=cfg["max_features"], random_state=RANDOM_SEED, n_jobs=-1, bootstrap=True,
    )
    return Pipeline([("prep", prep), ("rf", rf)])


def fit_model(train: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    usable = train.dropna(subset=["Target"]).copy()
    if len(usable) < 25:
        raise ValueError("Insufficient training rows / Filas insuficientes para entrenar")
    pipe = make_pipeline(cfg)
    pipe.fit(usable[CAT_FEATURES + NUM_FEATURES], usable["Target"])
    fitted = pipe.predict(usable[CAT_FEATURES + NUM_FEATURES])
    residual = usable["Target"].to_numpy() - fitted
    residual_frame = usable[["Plant", "Cost_Bucket", "BU"]].copy()
    residual_frame["Residual"] = residual
    pb = residual_frame.groupby(["Plant", "Cost_Bucket"])["Residual"].agg(["mean", "count"]).to_dict("index")
    bb = residual_frame.groupby(["BU", "Cost_Bucket"])["Residual"].agg(["mean", "count"]).to_dict("index")
    categories = {c: sorted(usable[c].astype(str).unique().tolist()) for c in CAT_FEATURES}
    return {"pipeline": pipe, "plant_bucket": pb, "bu_bucket": bb, "categories": categories, "residuals": residual, "training_rows": len(usable)}


def predict_model(model: dict[str, Any], features: pd.DataFrame, bootstrap_samples: int = 0) -> pd.DataFrame:
    X = features[CAT_FEATURES + NUM_FEATURES]
    base = model["pipeline"].predict(X)
    adjusted, source, confidence, new_category = [], [], [], []
    for i, (_, row) in enumerate(features.iterrows()):
        pb = model["plant_bucket"].get((row["Plant"], row["Cost_Bucket"]))
        bb = model["bu_bucket"].get((row["BU"], row["Cost_Bucket"]))
        unknown = any(str(row[c]) not in model["categories"].get(c, []) for c in CAT_FEATURES)
        if pb and pb["count"] >= 2:
            pred, level, conf = base[i] + pb["mean"], "Plant / Cost Bucket", 0.90
        elif bb and bb["count"] >= 3:
            pred, level, conf = base[i] + bb["mean"], "BU / Cost Bucket", 0.75
        elif np.isfinite(base[i]):
            pred, level, conf = base[i], "Global model", 0.60
        else:
            pred, level, conf = row["Current_Forecast_8_4"], "Forecast 8+4 fallback", 0.35
        if unknown:
            conf *= 0.65
        adjusted.append(pred); source.append(level); confidence.append(conf); new_category.append(unknown)

    transformed = model["pipeline"].named_steps["prep"].transform(X)
    trees = model["pipeline"].named_steps["rf"].estimators_
    tree_matrix = np.vstack([tree.predict(transformed) for tree in trees])
    low = np.quantile(tree_matrix, 0.10, axis=0)
    high = np.quantile(tree_matrix, 0.90, axis=0)
    if bootstrap_samples:
        rng = np.random.default_rng(RANDOM_SEED)
        residuals = np.asarray(model.get("residuals", [0.0]))
        boot = rng.choice(residuals, size=(int(bootstrap_samples), len(base)), replace=True) + np.asarray(adjusted)
        low = np.minimum(low, np.quantile(boot, 0.10, axis=0))
        high = np.maximum(high, np.quantile(boot, 0.90, axis=0))
    return pd.DataFrame({
        "Predicted_Impact": adjusted, "Lower_80": low, "Upper_80": high, "Fallback_Level": source,
        "Confidence": confidence, "New_Category": new_category,
    }, index=features.index)


def metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    denom = np.maximum(np.abs(y_true), np.nanmedian(np.abs(y_true)) * 0.05 + 1e-6)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "MAPE": float(np.mean(np.abs(y_true - y_pred) / denom)),
    }


def temporal_backtest(panel: pd.DataFrame, cfg: dict[str, Any], strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    for test_month in [7, 8]:
        train_months = list(range(1, test_month))
        train = build_feature_rows(panel, train_months)
        if strategy == "Independent":
            train = train[train["Month"] % 4 == test_month % 4]
        test = build_feature_rows(panel, [test_month])
        try:
            model = fit_model(train, cfg)
            pred = predict_model(model, test)["Predicted_Impact"].to_numpy()
        except ValueError:
            pred = test["Current_Forecast_8_4"].fillna(0).to_numpy()
        frame = test[["Plant", "Cost_Bucket", "Month", "Target", "Current_Forecast_8_4"]].copy()
        frame["Prediction"] = pred
        frame["Strategy"] = strategy
        detail.append(frame)
    detail_df = pd.concat(detail, ignore_index=True)
    rows = []
    for (strategy_name, month), group in detail_df.groupby(["Strategy", "Month"]):
        metrics = metric_set(group["Target"].to_numpy(), group["Prediction"].to_numpy())
        rows.append({"Strategy": strategy_name, "Month": MONTHS[int(month) - 1], **metrics, "Rows": len(group)})
    overall = metric_set(detail_df["Target"].to_numpy(), detail_df["Prediction"].to_numpy())
    rows.append({"Strategy": strategy, "Month": "Overall", **overall, "Rows": len(detail_df)})
    return pd.DataFrame(rows), detail_df


def optimize_config(panel: pd.DataFrame, base_cfg: dict[str, Any], enabled: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = [base_cfg]
    if enabled:
        candidates = [
            {**base_cfg, "n_estimators": 120, "max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt"},
            {**base_cfg, "n_estimators": 180, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.7},
            {**base_cfg, "n_estimators": 220, "max_depth": None, "min_samples_leaf": 3, "max_features": 0.8},
            {**base_cfg, "n_estimators": 160, "max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt"},
        ]
    scored = []
    for i, cfg in enumerate(candidates, 1):
        metrics, _ = temporal_backtest(panel, cfg, "Unified")
        rmse = float(metrics.loc[metrics.Month == "Overall", "RMSE"].iloc[0])
        scored.append({"Candidate": i, "RMSE": rmse, **cfg})
    table = pd.DataFrame(scored).sort_values("RMSE")
    best = candidates[int(table.iloc[0]["Candidate"]) - 1]
    return best, table


def scenario_effect(row: pd.Series, scenarios: dict[str, float], volume_elasticity: float) -> float:
    bucket = clean_text(row["Cost_Bucket"]).lower()
    base = abs(float(row.get("Prior_Year_2025", 0) or 0))
    effect = base * scenarios.get("Volume", 0.0) * volume_elasticity
    for key in ["Labor", "Utilities", "MUV", "Faulty", "Maintenance"]:
        if key.lower() in bucket:
            effect -= base * scenarios.get(key, 0.0)
    return effect


def train_and_forecast(panel: pd.DataFrame, cfg: dict[str, Any], bootstrap_samples: int, scenarios: dict[str, float], month_adjustments: dict[int, float], volume_elasticity: float) -> dict[str, Any]:
    work = panel.copy()
    for month, change in month_adjustments.items():
        mask = work["Month"] == month
        work.loc[mask, "Actual_Impact"] *= 1 + change
    best_cfg, optimization = optimize_config(work, cfg, cfg.get("auto_optimize", False))
    metrics_u, detail_u = temporal_backtest(work, best_cfg, "Unified")
    metrics_i, detail_i = temporal_backtest(work, best_cfg, "Independent")
    rmse_u = float(metrics_u.loc[metrics_u.Month == "Overall", "RMSE"].iloc[0])
    rmse_i = float(metrics_i.loc[metrics_i.Month == "Overall", "RMSE"].iloc[0])
    selected = "Unified" if rmse_u <= rmse_i else "Independent"

    future = build_feature_rows(work, [9, 10, 11, 12], history_cutoff=8)
    future["Current_Forecast_8_4"] = future.apply(
        lambda r: r["Current_Forecast_8_4"] * (1 + scenarios.get("Volume", 0.0) * volume_elasticity), axis=1
    )
    predictions = []
    models: dict[str, Any] = {}
    if selected == "Unified":
        train = build_feature_rows(work, list(range(1, 9)))
        model = fit_model(train, best_cfg)
        pred = predict_model(model, future, bootstrap_samples)
        predictions.append(pd.concat([future.reset_index(drop=True), pred.reset_index(drop=True)], axis=1))
        models["Unified"] = model
    else:
        for month in [9, 10, 11, 12]:
            train = build_feature_rows(work, list(range(1, 9)))
            train = train[train["Month"] % 4 == month % 4]
            model = fit_model(train, best_cfg)
            fm = future[future["Month"] == month].copy()
            pred = predict_model(model, fm, bootstrap_samples)
            predictions.append(pd.concat([fm.reset_index(drop=True), pred.reset_index(drop=True)], axis=1))
            models[str(month)] = model
    result = pd.concat(predictions, ignore_index=True)
    result["Scenario_Effect"] = result.apply(lambda r: scenario_effect(r, scenarios, volume_elasticity), axis=1)
    for col in ["Predicted_Impact", "Lower_80", "Upper_80"]:
        result[col] += result["Scenario_Effect"]
    result["Metric_Nature"] = result["Cost_Bucket"].map(metric_nature)
    return {
        "models": models, "strategy": selected, "config": best_cfg, "predictions_raw": result,
        "backtest_metrics": pd.concat([metrics_u, metrics_i], ignore_index=True),
        "backtest_detail": pd.concat([detail_u, detail_i], ignore_index=True),
        "optimization": optimization, "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def classify_results(raw: pd.DataFrame, reference: str, pct_threshold: float, materiality_usd: float, language: str) -> pd.DataFrame:
    out = raw.copy()
    out["Reference_Impact"] = np.where(reference == "Current Forecast 8+4", out["Current_Forecast_8_4"], 0.0)
    out["Variance_to_Reference"] = out["Predicted_Impact"] - out["Reference_Impact"]
    scale = out.get("Unit_Scale", pd.Series(1000.0, index=out.index)).fillna(1000.0)
    out["Predicted_Impact_USD"] = out["Predicted_Impact"] * scale
    out["Lower_80_USD"] = out["Lower_80"] * scale
    out["Upper_80_USD"] = out["Upper_80"] * scale
    out["Forecast_8_4_USD"] = out["Current_Forecast_8_4"] * scale
    out["Prior_Year_2025_USD"] = out["Prior_Year_2025"] * scale
    out["Predicted_2026_Value_USD"] = (out["Prior_Year_2025"] - out["Predicted_Impact"]) * scale
    out["Variance_to_Reference_USD"] = out["Variance_to_Reference"] * scale
    denom = np.maximum(np.abs(out["Reference_Impact"] * scale), materiality_usd * 0.10)
    out["Variance_Pct"] = out["Variance_to_Reference_USD"] / denom
    significant = (out["Variance_to_Reference_USD"].abs() >= materiality_usd) & (out["Variance_Pct"].abs() >= pct_threshold)
    labels = TEXT[language]
    out["Status"] = np.where(
        significant & (out["Variance_to_Reference_USD"] > 0), labels["favorable"],
        np.where(significant & (out["Variance_to_Reference_USD"] < 0), labels["unfavorable"], labels["risk"]),
    )
    out["Traffic_Light"] = out["Status"].map({labels["favorable"]: "🟢", labels["risk"]: "🟡", labels["unfavorable"]: "🔴"})
    out["Reference"] = reference
    return out


def global_importance(model: dict[str, Any]) -> pd.DataFrame:
    pipe = model["pipeline"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    values = pipe.named_steps["rf"].feature_importances_
    imp = pd.DataFrame({"Feature": names, "Importance": values})
    imp["Feature"] = imp["Feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


def local_explanation(model: dict[str, Any], row: pd.DataFrame, reference_rows: pd.DataFrame) -> pd.DataFrame:
    X = row[CAT_FEATURES + NUM_FEATURES].copy()
    base_pred = float(model["pipeline"].predict(X)[0])
    items = []
    for feature in CAT_FEATURES + NUM_FEATURES:
        altered = X.copy()
        if feature in CAT_FEATURES:
            mode = reference_rows[feature].mode()
            altered[feature] = mode.iloc[0] if len(mode) else "UNKNOWN"
        else:
            altered[feature] = reference_rows[feature].median()
        changed = float(model["pipeline"].predict(altered)[0])
        items.append({"Feature": feature, "Contribution": base_pred - changed})
    return pd.DataFrame(items).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)


def build_template_csv() -> bytes:
    rows = []
    for month in range(1, 13):
        rows.append({
            "BU": "Example BU", "POD": "Example POD", "Plant": "EXAMPLE PLANT", "Cost_Bucket": "Labor",
            "Month": month, "Actual_2026": 0 if month <= 8 else "", "Prior_Year_2025": 0,
            "Current_Forecast_8_4": 0, "Unit_Scale": 1000,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def excel_export(predictions: pd.DataFrame, plant_summary: pd.DataFrame, quality: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        plant_summary.to_excel(writer, sheet_name="Plant Summary", index=False)
        quality.to_excel(writer, sheet_name="Data Quality", index=False)
        metrics.to_excel(writer, sheet_name="Backtesting", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="Configuration", index=False)
        book = writer.book
        header = book.add_format({"bold": True, "font_name": "Arial", "font_color": "#FFFFFF", "bg_color": "#17324D", "border": 0})
        body = book.add_format({"font_name": "Arial"})
        money = book.add_format({"font_name": "Arial", "num_format": "$#,##0;($#,##0);-"})
        pct = book.add_format({"font_name": "Arial", "num_format": "0.0%;(0.0%);-"})
        for name, sheet in writer.sheets.items():
            sheet.set_default_row(16)
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, max(1, sheet.dim_rowmax), max(0, sheet.dim_colmax))
            sheet.set_row(0, 22, header)
            sheet.set_column(0, max(0, sheet.dim_colmax), 18, body)
            frame = {"Predictions": predictions, "Plant Summary": plant_summary, "Data Quality": quality, "Backtesting": metrics}.get(name)
            if frame is not None:
                for idx, col in enumerate(frame.columns):
                    if "USD" in str(col): sheet.set_column(idx, idx, 18, money)
                    if "Pct" in str(col) or "MAPE" in str(col): sheet.set_column(idx, idx, 14, pct)
    return output.getvalue()


def serialize_artifact(bundle: dict[str, Any], panel: pd.DataFrame, meta: dict[str, Any]) -> bytes:
    payload = {
        "app_version": APP_VERSION, "schema_hash": schema_hash(panel), "trained_at": bundle["trained_at"],
        "data_year": meta.get("year", 2026), "source_hash": meta.get("file_hash"), "strategy": bundle["strategy"],
        "config": bundle["config"], "models": bundle["models"],
    }
    buffer = io.BytesIO(); joblib.dump(payload, buffer); return buffer.getvalue()


def load_artifact(data: bytes, panel: pd.DataFrame, meta: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    notes = []
    try:
        payload = joblib.load(io.BytesIO(data))
    except Exception as exc:
        return None, [f"Invalid model file: {exc}"]
    if payload.get("app_version") != APP_VERSION or payload.get("schema_hash") != schema_hash(panel):
        return None, ["incompatible"]
    if meta.get("year", 2026) > payload.get("data_year", 0):
        notes.append("newer")
    return payload, notes


def forecast_with_artifact(payload: dict[str, Any], panel: pd.DataFrame, bootstrap_samples: int, scenarios: dict[str, float], month_adjustments: dict[int, float], volume_elasticity: float) -> dict[str, Any]:
    work = panel.copy()
    for month, change in month_adjustments.items():
        work.loc[work.Month == month, "Actual_Impact"] *= 1 + change
    future = build_feature_rows(work, [9, 10, 11, 12], history_cutoff=8)
    future["Current_Forecast_8_4"] *= 1 + scenarios.get("Volume", 0.0) * volume_elasticity
    predictions = []
    if payload["strategy"] == "Unified":
        pred = predict_model(payload["models"]["Unified"], future, bootstrap_samples)
        predictions.append(pd.concat([future.reset_index(drop=True), pred.reset_index(drop=True)], axis=1))
    else:
        for month in [9, 10, 11, 12]:
            fm = future[future.Month == month].copy()
            pred = predict_model(payload["models"][str(month)], fm, bootstrap_samples)
            predictions.append(pd.concat([fm.reset_index(drop=True), pred.reset_index(drop=True)], axis=1))
    result = pd.concat(predictions, ignore_index=True)
    result["Scenario_Effect"] = result.apply(lambda r: scenario_effect(r, scenarios, volume_elasticity), axis=1)
    for col in ["Predicted_Impact", "Lower_80", "Upper_80"]: result[col] += result["Scenario_Effect"]
    result["Metric_Nature"] = result["Cost_Bucket"].map(metric_nature)
    return {"models": payload["models"], "strategy": payload["strategy"], "config": payload["config"], "predictions_raw": result, "trained_at": payload["trained_at"]}


def app_css() -> None:
    st.markdown("""
    <style>
    :root { --ink:#102A43; --navy:#17324D; --blue:#0B6E99; --mint:#37B89C; --fog:#F3F7FA; --amber:#F2B134; --red:#C94C4C; }
    .stApp { background: linear-gradient(180deg,#F7FAFC 0,#EEF4F7 100%); color:var(--ink); }
    h1,h2,h3 { font-family: "Aptos Display","Segoe UI",sans-serif; letter-spacing:-.02em; color:var(--navy); }
    p,div,label { font-family:"Aptos","Segoe UI",sans-serif; }
    .hero { border-left:8px solid var(--mint); padding:1.2rem 1.4rem; background:#fff; box-shadow:0 8px 24px rgba(16,42,67,.08); margin-bottom:1rem; }
    .hero .eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.14em; font-weight:700; font-size:.72rem; }
    .stepbar { display:flex; gap:.5rem; flex-wrap:wrap; margin:.4rem 0 1.2rem; }
    .step { background:#DCEAF1; color:#17324D; padding:.35rem .65rem; border-radius:4px; font-size:.78rem; font-weight:700; }
    [data-testid="stMetric"] { background:white; border:1px solid #D8E3EA; padding:.8rem; border-radius:6px; }
    [data-testid="stSidebar"] { background:#E8F0F4; }
    .small-note { color:#486581; font-size:.82rem; }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Performance Model 8+4", page_icon="📊", layout="wide")
    app_css()
    language = st.sidebar.segmented_control("Idioma / Language", ["ES", "EN"], default="ES")
    t = TEXT[language]
    st.markdown(f"<div class='hero'><div class='eyebrow'>MANUFACTURING FINANCE · 2026</div><h1>{t['title']}</h1><p>{t['subtitle']}</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='stepbar'><span class='step'>01 Upload</span><span class='step'>02 Validate</span><span class='step'>03 Quality</span><span class='step'>04 Configure</span><span class='step'>05 Train</span><span class='step'>06 Save & reuse</span></div>", unsafe_allow_html=True)

    uploaded = st.sidebar.file_uploader(t["upload"], type=["xlsx", "xlsm", "csv"], help=t["upload_help"])
    st.sidebar.download_button("CSV template / Plantilla CSV", build_template_csv(), "performance_model_template.csv", "text/csv")
    model_upload = st.sidebar.file_uploader(t["reuse_model"], type=["joblib"], key="model_upload")
    mode = st.sidebar.radio(t["mode"], [t["executive"], t["advanced"]], horizontal=True)

    if uploaded is None:
        st.info(t["no_data"])
        st.stop()

    data = uploaded.getvalue()
    source_key = hashlib.sha256(data).hexdigest()
    try:
        if st.session_state.get("source_key") != source_key:
            with st.spinner("Validating and normalizing / Validando y normalizando..."):
                panel, quality, meta = load_source(data, uploaded.name)
            st.session_state.update({"source_key": source_key, "panel": panel, "quality": quality, "meta": meta})
            st.session_state.pop("bundle", None); st.session_state.pop("artifact_payload", None)
        panel = st.session_state.panel.copy(); quality = st.session_state.quality.copy(); meta = st.session_state.meta.copy()
    except Exception as exc:
        st.error(str(exc)); st.stop()

    st.success(t["valid"])
    st.caption(t["unit_note"])

    with st.sidebar.expander(t["quality"], expanded=False):
        imputation = st.selectbox("Imputation / Imputación", ["Group median / Mediana por grupo", "Zero / Cero", "Drop incomplete / Eliminar incompletos"])
        outlier_mode = st.selectbox("Outliers / Extremos", ["Keep / Conservar", "Winsorize / Winsorizar", "Cap 3×IQR"])
        outlier_limit = st.slider("Winsor tail / Cola", 0.005, 0.05, 0.01, 0.005)
    clean_panel, cleaning_report = apply_cleaning(panel, imputation, outlier_mode, outlier_limit)

    st.sidebar.markdown(f"### {t['filters']}")
    bu_options = sorted(clean_panel.BU.unique().tolist())
    bu_sel = st.sidebar.multiselect("BU", bu_options, default=bu_options)
    pod_options = sorted(clean_panel[clean_panel.BU.isin(bu_sel)].POD.unique().tolist())
    pod_sel = st.sidebar.multiselect("POD", pod_options, default=pod_options)
    plant_options = sorted(clean_panel[clean_panel.POD.isin(pod_sel)].Plant.unique().tolist())
    plant_sel = st.sidebar.multiselect("Plant / Planta", plant_options, default=plant_options)
    filtered = clean_panel[clean_panel.BU.isin(bu_sel) & clean_panel.POD.isin(pod_sel) & clean_panel.Plant.isin(plant_sel)].copy()
    if filtered.empty:
        st.warning("Filters produce no rows / Los filtros no producen datos"); st.stop()

    st.sidebar.markdown(f"### {t['config']}")
    reference = st.sidebar.selectbox(t["reference"], [t["forecast_ref"], t["py_ref"]])
    pct_threshold = st.sidebar.slider(t["pct_threshold"], 0.0, 0.25, 0.05, 0.005)
    materiality_usd = st.sidebar.number_input(t["materiality"], min_value=0.0, value=50000.0, step=5000.0)

    with st.sidebar.expander(t["scenario"], expanded=False):
        scenarios = {}
        for driver in ["Volume", "Labor", "Utilities", "MUV", "Faulty", "Maintenance"]:
            scenarios[driver] = st.slider(f"{driver} %", -0.30, 0.30, 0.0, 0.01, key=f"sc_{driver}")
        volume_elasticity = st.slider("Volume elasticity / Elasticidad", 0.0, 1.5, 0.35, 0.05)
    with st.sidebar.expander(t["actual_adj"], expanded=False):
        month_adjustments = {i: st.slider(f"{MONTHS[i-1]} %", -0.30, 0.30, 0.0, 0.01, key=f"m_{i}") for i in range(1, 9)}

    cfg = {"n_estimators": 180, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.7, "auto_optimize": False}
    bootstrap_samples = 0
    if mode == t["advanced"]:
        with st.sidebar.expander("Model & optimization / Modelo y optimización", expanded=True):
            cfg["n_estimators"] = st.slider("Trees / Árboles", 50, 500, 180, 10)
            depth = st.slider("Max depth / Profundidad", 0, 30, 12, 1)
            cfg["max_depth"] = None if depth == 0 else depth
            cfg["min_samples_leaf"] = st.slider("Min samples leaf", 1, 15, 2)
            max_features_label = st.selectbox("Max features", ["70%", "80%", "sqrt", "100%"])
            cfg["max_features"] = {"70%": 0.7, "80%": 0.8, "sqrt": "sqrt", "100%": 1.0}[max_features_label]
            cfg["auto_optimize"] = st.checkbox("Limited automatic optimization / Optimización automática limitada", False)
            use_bootstrap = st.checkbox("Residual bootstrap intervals / Intervalos bootstrap", False)
            bootstrap_samples = st.slider("Bootstrap samples", 50, 500, 200, 50) if use_bootstrap else 0
    st.sidebar.caption(f"Fixed seed / Semilla fija: {RANDOM_SEED}")

    if model_upload is not None and st.session_state.get("model_key") != hashlib.sha256(model_upload.getvalue()).hexdigest():
        payload, notes = load_artifact(model_upload.getvalue(), filtered, meta)
        if payload is None:
            st.sidebar.error(t["incompatible"] if notes == ["incompatible"] else notes[0])
        else:
            st.session_state.artifact_payload = payload
            st.session_state.model_key = hashlib.sha256(model_upload.getvalue()).hexdigest()
            if "newer" in notes: st.sidebar.warning(t["newer"])

    train_clicked = st.sidebar.button(t["train"], type="primary", use_container_width=True)
    if train_clicked:
        with st.spinner("Training, backtesting and selecting strategy / Entrenando, validando y seleccionando estrategia..."):
            bundle = train_and_forecast(filtered, cfg, bootstrap_samples, scenarios, month_adjustments, volume_elasticity)
        st.session_state.bundle = bundle
        st.session_state.pop("artifact_payload", None)
        st.toast(t["trained"], icon="✅")

    if "artifact_payload" in st.session_state and not train_clicked:
        bundle = forecast_with_artifact(st.session_state.artifact_payload, filtered, bootstrap_samples, scenarios, month_adjustments, volume_elasticity)
    else:
        bundle = st.session_state.get("bundle")

    tabs = st.tabs([t["quality"], t["dashboard"], t["explain"], t["diagnostics"], t["downloads"]])
    with tabs[0]:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows / Filas", f"{len(filtered):,}")
        q2.metric("Plants / Plantas", filtered.Plant.nunique())
        q3.metric("Cost buckets", filtered.Cost_Bucket.nunique())
        q4.metric("Coverage / Cobertura", f"{filtered.Current_Forecast_8_4.notna().mean():.1%}")
        st.subheader("Validation report / Reporte de validación")
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.subheader("Controlled cleaning / Limpieza controlada")
        st.dataframe(cleaning_report, use_container_width=True, hide_index=True)
        with st.expander("Normalized preview / Vista normalizada"):
            st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

    if bundle is None:
        for tab in tabs[1:]:
            with tab: st.info(t["train_first"])
        return

    raw = bundle["predictions_raw"]
    results = classify_results(raw, reference, pct_threshold, materiality_usd, language)
    if results["New_Category"].any(): st.warning(t["new_categories"])
    plant_summary = results.groupby(["BU", "POD", "Plant", "Month"], as_index=False).agg(
        Predicted_Impact_USD=("Predicted_Impact_USD", "sum"), Forecast_8_4_USD=("Forecast_8_4_USD", "sum"),
        Variance_to_Reference_USD=("Variance_to_Reference_USD", "sum"), Lower_80_USD=("Lower_80_USD", "sum"), Upper_80_USD=("Upper_80_USD", "sum"),
        Confidence=("Confidence", "mean"),
    )

    with tabs[1]:
        total_pred = results.Predicted_Impact_USD.sum()
        total_ref = results.Forecast_8_4_USD.sum() if reference == t["forecast_ref"] else 0.0
        variance = results.Variance_to_Reference_USD.sum()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Predicted EBITDA impact", f"${total_pred:,.0f}")
        k2.metric(reference, f"${total_ref:,.0f}")
        k3.metric("Variance", f"${variance:,.0f}")
        k4.metric(t["favorable"], int((results.Status == t["favorable"]).sum()))
        k5.metric("Confidence / Confianza", f"{results.Confidence.mean():.0%}")
        st.caption(f"Selected strategy / Estrategia seleccionada: **{bundle['strategy']}** · 80% intervals · {t['unit_note']}")

        left, right = st.columns([1.5, 1])
        with left:
            selected_plant = st.selectbox("Series: Plant / Planta", sorted(results.Plant.unique()))
            hist = filtered[filtered.Plant == selected_plant].groupby("Month", as_index=False).agg(Actual_Impact=("Actual_Impact", "sum"), Forecast=("Current_Forecast_8_4", "sum"))
            fut = results[results.Plant == selected_plant].groupby("Month", as_index=False).agg(Prediction=("Predicted_Impact", "sum"), Lower=("Lower_80", "sum"), Upper=("Upper_80", "sum"), Forecast=("Current_Forecast_8_4", "sum"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.Month, y=hist.Actual_Impact * 1000, name="Actual impact", mode="lines+markers", line=dict(color="#17324D", width=3)))
            fig.add_trace(go.Scatter(x=fut.Month, y=fut.Upper * 1000, name="Upper 80%", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=fut.Month, y=fut.Lower * 1000, name="80% interval", fill="tonexty", line=dict(width=0), fillcolor="rgba(55,184,156,.20)"))
            fig.add_trace(go.Scatter(x=fut.Month, y=fut.Prediction * 1000, name="Model", mode="lines+markers", line=dict(color="#0B6E99", width=3)))
            fig.add_trace(go.Scatter(x=fut.Month, y=fut.Forecast * 1000, name="Forecast 8+4", mode="lines", line=dict(color="#F2B134", dash="dash")))
            fig.update_layout(title=f"{selected_plant} · monthly performance", xaxis=dict(tickmode="array", tickvals=list(range(1,13)), ticktext=MONTHS), yaxis_title="USD", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            rank = results.groupby("Plant", as_index=False).Variance_to_Reference_USD.sum().sort_values("Variance_to_Reference_USD")
            fig_rank = px.bar(rank, x="Variance_to_Reference_USD", y="Plant", orientation="h", color="Variance_to_Reference_USD", color_continuous_scale=["#C94C4C", "#E9EEF2", "#37B89C"], title="Plant ranking / Ranking de plantas")
            fig_rank.update_layout(coloraxis_showscale=False, xaxis_title="Variance (USD)")
            st.plotly_chart(fig_rank, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            heat = plant_summary.pivot_table(index="Plant", columns="Month", values="Variance_to_Reference_USD", aggfunc="sum").fillna(0)
            fig_heat = px.imshow(heat, aspect="auto", color_continuous_scale=["#C94C4C", "#F7FAFC", "#37B89C"], color_continuous_midpoint=0, title="Variance heatmap / Mapa de variaciones")
            fig_heat.update_xaxes(tickmode="array", tickvals=list(range(4)), ticktext=MONTHS[8:])
            st.plotly_chart(fig_heat, use_container_width=True)
        with c2:
            wf = results.groupby("Cost_Bucket", as_index=False).Variance_to_Reference_USD.sum().sort_values("Variance_to_Reference_USD", key=lambda s: s.abs(), ascending=False)
            fig_w = go.Figure(go.Waterfall(x=wf.Cost_Bucket, y=wf.Variance_to_Reference_USD, measure=["relative"] * len(wf), increasing={"marker":{"color":"#37B89C"}}, decreasing={"marker":{"color":"#C94C4C"}}))
            fig_w.update_layout(title="Cost bucket waterfall / Cascada", yaxis_title="USD")
            st.plotly_chart(fig_w, use_container_width=True)

        fig_cmp = px.scatter(results, x="Forecast_8_4_USD", y="Predicted_Impact_USD", color="Status", size=results.Variance_to_Reference_USD.abs().clip(lower=1), hover_data=["Plant", "Cost_Bucket", "Month", "Confidence"], color_discrete_map={t["favorable"]:"#37B89C", t["risk"]:"#F2B134", t["unfavorable"]:"#C94C4C"}, title="Model vs Current Forecast 8+4")
        st.plotly_chart(fig_cmp, use_container_width=True)
        display_cols = ["Traffic_Light", "BU", "POD", "Plant", "Cost_Bucket", "Month", "Predicted_Impact_USD", "Forecast_8_4_USD", "Variance_to_Reference_USD", "Variance_Pct", "Status", "Confidence", "Fallback_Level"]
        st.dataframe(results[display_cols].sort_values("Variance_to_Reference_USD", key=lambda s: s.abs(), ascending=False), use_container_width=True, hide_index=True, column_config={"Predicted_Impact_USD": st.column_config.NumberColumn(format="$%.0f"), "Forecast_8_4_USD": st.column_config.NumberColumn(format="$%.0f"), "Variance_to_Reference_USD": st.column_config.NumberColumn(format="$%.0f"), "Variance_Pct": st.column_config.NumberColumn(format="%.1f%%"), "Confidence": st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.0%%")})

    with tabs[2]:
        model_key = "Unified" if bundle["strategy"] == "Unified" else "9"
        model = bundle["models"][model_key]
        imp = global_importance(model).head(25)
        fig_imp = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Global feature importance / Importancia global", color_discrete_sequence=["#0B6E99"])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.subheader("Individual explanation / Explicación individual")
        choices = results.apply(lambda r: f"{r['Plant']} · {r['Cost_Bucket']} · {MONTHS[int(r['Month'])-1]}", axis=1)
        selected_idx = st.selectbox("Prediction / Predicción", list(results.index), format_func=lambda i: choices.loc[i])
        row = raw.loc[[selected_idx]]
        train_ref = build_feature_rows(filtered, list(range(1, 9)))
        local_model = bundle["models"]["Unified"] if bundle["strategy"] == "Unified" else bundle["models"][str(int(row.Month.iloc[0]))]
        local = local_explanation(local_model, row, train_ref).head(15)
        fig_local = px.bar(local.sort_values("Contribution"), x="Contribution", y="Feature", orientation="h", color="Contribution", color_continuous_scale=["#C94C4C", "#E9EEF2", "#37B89C"], title="Local contributions vs baseline / Contribuciones locales")
        fig_local.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_local, use_container_width=True)
        item = results.loc[selected_idx]
        direction = "above / por encima" if item.Variance_to_Reference_USD >= 0 else "below / por debajo"
        st.info(f"{item.Traffic_Light} {item.Plant} · {item.Cost_Bucket} · {MONTHS[int(item.Month)-1]}: predicted EBITDA impact ${item.Predicted_Impact_USD:,.0f}, {abs(item.Variance_to_Reference_USD):,.0f} USD {direction} {reference}. Confidence {item.Confidence:.0%}. Interpretation: {item.Metric_Nature}. Fallback: {item.Fallback_Level}.")

    with tabs[3]:
        if "backtest_metrics" not in bundle:
            st.info("Backtesting metrics are available after retraining / Las métricas están disponibles después de reentrenar")
        else:
            st.subheader("Temporal backtesting 8+4")
            st.dataframe(bundle["backtest_metrics"], use_container_width=True, hide_index=True)
            detail = bundle["backtest_detail"]
            detail["Residual"] = detail.Target - detail.Prediction
            d1, d2 = st.columns(2)
            with d1:
                fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(detail.Residual, kde=True, ax=ax, color="#0B6E99"); ax.set_title("Residual distribution"); ax.set_xlabel("Residual (USD thousands)"); st.pyplot(fig); plt.close(fig)
            with d2:
                fig, ax = plt.subplots(figsize=(7, 4)); sns.scatterplot(data=detail, x="Target", y="Prediction", hue="Strategy", ax=ax, palette=["#17324D", "#37B89C"]); lo=min(detail.Target.min(),detail.Prediction.min()); hi=max(detail.Target.max(),detail.Prediction.max()); ax.plot([lo,hi],[lo,hi],"--",color="#C94C4C"); ax.set_title("Actual vs prediction"); st.pyplot(fig); plt.close(fig)
            if mode == t["advanced"]:
                st.subheader("Limited optimization / Optimización limitada")
                st.dataframe(bundle["optimization"], use_container_width=True, hide_index=True)

    with tabs[4]:
        export_cols = [c for c in results.columns if c not in {"Target"}]
        excel = excel_export(results[export_cols], plant_summary, quality, bundle.get("backtest_metrics", pd.DataFrame()), {**bundle["config"], "strategy": bundle["strategy"], "reference": reference, "pct_threshold": pct_threshold, "materiality_usd": materiality_usd})
        st.download_button("Excel multi-sheet / Excel multipestaña", excel, "performance_model_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Predictions CSV", results[export_cols].to_csv(index=False).encode(), "predictions.csv", "text/csv", use_container_width=True)
        c2.download_button("Plant summary CSV", plant_summary.to_csv(index=False).encode(), "plant_summary.csv", "text/csv", use_container_width=True)
        c3.download_button("Quality CSV", quality.to_csv(index=False).encode(), "data_quality.csv", "text/csv", use_container_width=True)
        if "backtest_metrics" in bundle:
            st.download_button("Backtesting CSV", bundle["backtest_metrics"].to_csv(index=False).encode(), "backtesting.csv", "text/csv", use_container_width=True)
        if "backtest_metrics" in bundle:
            artifact = serialize_artifact(bundle, filtered, meta)
            st.download_button(t["save_model"], artifact, "performance_model_8_4.joblib", "application/octet-stream", use_container_width=True)


def run_self_test(path: str) -> None:
    data = file_bytes(path)
    panel, quality, meta = load_source(data, str(path))
    clean, _ = apply_cleaning(panel, "Group median / Mediana por grupo", "Keep / Conservar", 0.01)
    cfg = {"n_estimators": 40, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.7, "auto_optimize": False}
    bundle = train_and_forecast(clean, cfg, 0, {k: 0.0 for k in ["Volume", "Labor", "Utilities", "MUV", "Faulty", "Maintenance"]}, {i: 0.0 for i in range(1, 9)}, 0.35)
    results = classify_results(bundle["predictions_raw"], "Current Forecast 8+4", 0.05, 50000, "ES")
    checks = {
        "rows": len(panel), "plants": panel.Plant.nunique(), "cost_buckets": panel.Cost_Bucket.nunique(),
        "future_rows": len(results), "months": sorted(results.Month.unique().tolist()), "strategy": bundle["strategy"],
        "finite_predictions": bool(np.isfinite(results.Predicted_Impact_USD).all()),
        "quality_checks": len(quality), "schema_hash": schema_hash(panel), "source_hash": meta["file_hash"],
    }
    assert checks["months"] == [9, 10, 11, 12]
    assert checks["finite_predictions"] and checks["future_rows"] > 0
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        idx = sys.argv.index("--self-test")
        run_self_test(sys.argv[idx + 1])
    else:
        main()
