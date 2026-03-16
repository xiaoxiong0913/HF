from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request


APP_DIR = Path(__file__).resolve().parent
CARD_PATH = APP_DIR / "clinical_web_model_card.json"
HTML_PATH = APP_DIR / "clinical_risk_comparison.html"


def _dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "todense"):
        return np.asarray(matrix.todense())
    return np.asarray(matrix)


def _as_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).gt(0).astype(int)


def _score_from_bins(value: float, bins: tuple[tuple[float, float], ...]) -> float:
    if np.isnan(value):
        return 0.0
    for upper, points in bins:
        if value <= upper:
            return float(points)
    return 0.0


GWTG_AGE_BINS = (
    (19, 0), (29, 3), (39, 6), (49, 8), (59, 11), (69, 14), (79, 17), (89, 19),
    (99, 22), (109, 25), (119, 28), (129, 31), (139, 34), (np.inf, 36),
)
GWTG_SBP_BINS = (
    (49, 28), (59, 26), (69, 24), (79, 21), (89, 19), (99, 17), (109, 15), (119, 13),
    (129, 11), (139, 8), (149, 6), (159, 5), (169, 3), (179, 2), (189, 1), (199, 0), (np.inf, 0),
)
GWTG_HEART_RATE_BINS = (
    (69, 0), (74, 2), (79, 3), (84, 4), (89, 5), (94, 6), (99, 8), (104, 9),
    (109, 10), (114, 11), (119, 12), (124, 13), (129, 14), (134, 15), (139, 16), (144, 17), (np.inf, 18),
)
GWTG_BUN_BINS = (
    (9, 0), (19, 3), (29, 6), (39, 8), (49, 11), (59, 13), (69, 15), (79, 17),
    (89, 19), (99, 21), (109, 22), (119, 24), (np.inf, 25),
)
GWTG_SODIUM_BINS = (
    (-np.inf, 12), (121, 11), (123, 10), (125, 9), (127, 8), (129, 6), (131, 5),
    (133, 4), (135, 3), (137, 2), (139, 1), (np.inf, 0),
)
ADHERE_GROUP_LABELS = {
    1: "BUN < 43 and SBP >= 115",
    2: "BUN < 43 and SBP < 115",
    3: "BUN >= 43 and SBP >= 115",
    4: "BUN >= 43 and SBP < 115 and creatinine < 2.75",
    5: "BUN >= 43 and SBP < 115 and creatinine >= 2.75",
}


def compute_gwtg_hf_score(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)
    output["gwtg_age_points"] = df["age"].map(lambda value: _score_from_bins(float(value), GWTG_AGE_BINS))
    output["gwtg_sbp_points"] = df["sbp_mean_24h"].map(lambda value: _score_from_bins(float(value), GWTG_SBP_BINS))
    output["gwtg_heart_rate_points"] = df["heart_rate_mean_24h"].map(lambda value: _score_from_bins(float(value), GWTG_HEART_RATE_BINS))
    output["gwtg_bun_points"] = df["bun_max_24h"].map(lambda value: _score_from_bins(float(value), GWTG_BUN_BINS))
    output["gwtg_sodium_points"] = df["sodium_min_24h"].map(lambda value: _score_from_bins(float(value), GWTG_SODIUM_BINS))
    output["gwtg_copd_points"] = _as_binary(df.get("copd_flag", pd.Series(0, index=df.index))).astype(float) * 2.0
    race_series = df.get("race_ethnicity", pd.Series("Other/Unknown", index=df.index)).astype(str)
    output["gwtg_race_points"] = race_series.str.strip().eq("Black").astype(float) * 3.0
    point_columns = [column for column in output.columns if column.endswith("_points")]
    output["gwtg_score"] = output[point_columns].sum(axis=1)
    return output


def compute_adhere_score(df: pd.DataFrame) -> pd.DataFrame:
    bun = pd.to_numeric(df["bun_max_24h"], errors="coerce")
    sbp = pd.to_numeric(df["sbp_mean_24h"], errors="coerce")
    creatinine = pd.to_numeric(df["creatinine_max_24h"], errors="coerce")
    group = np.where(
        bun < 43,
        np.where(sbp >= 115, 1, 2),
        np.where(sbp >= 115, 3, np.where(creatinine < 2.75, 4, 5)),
    )
    output = pd.DataFrame(index=df.index)
    output["adhere_group"] = group.astype(int)
    output["adhere_label"] = output["adhere_group"].map(ADHERE_GROUP_LABELS)
    return output


def _predict_calibrated_probability(values: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    linear = calibrator["intercept"] + calibrator["coefficient"] * values
    return 1.0 / (1.0 + np.exp(-linear))


def _coerce_runtime_value(value: Any, feature_spec: dict[str, Any]) -> Any:
    if value in (None, "", "null"):
        return feature_spec["default"]
    if feature_spec["type"] == "select":
        option_values = [option["value"] for option in feature_spec.get("options", [])]
        if any(item in [0, 1, "0", "1"] for item in option_values):
            return float(value)
        return str(value)
    return float(value)


with CARD_PATH.open("r", encoding="utf-8") as fh:
    PAYLOAD = json.load(fh)
MODEL = joblib.load(APP_DIR / PAYLOAD["runtime"]["model_file"])
PREPROCESSOR = joblib.load(APP_DIR / PAYLOAD["runtime"]["preprocessor_file"])
TRANSFORMED_NAMES = PREPROCESSOR.get_feature_names_out().tolist()
RAW_FEATURE_ORDER = list(PAYLOAD["prediction_model"]["raw_feature_order"])
SELECTED_TRANSFORMED_FEATURES = list(PAYLOAD["prediction_model"]["selected_transformed_features"])
TRANSFORMED_INDEX_MAP = {name: index for index, name in enumerate(TRANSFORMED_NAMES)}
FEATURE_SPECS = {item["key"]: item for item in PAYLOAD["prediction_model"]["input_features"]}

app = Flask(__name__)


def _prediction_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    row = {feature: _coerce_runtime_value(inputs.get(feature), FEATURE_SPECS[feature]) for feature in RAW_FEATURE_ORDER}
    return pd.DataFrame([row], columns=RAW_FEATURE_ORDER)


def _predict(inputs: dict[str, Any]) -> dict[str, Any]:
    frame = _prediction_frame(inputs)
    dense = _dense(PREPROCESSOR.transform(frame))
    selected_idx = [TRANSFORMED_INDEX_MAP[name] for name in SELECTED_TRANSFORMED_FEATURES]
    risk = float(MODEL.predict_proba(dense[:, selected_idx])[:, 1][0])

    working = frame.copy()
    if "race_ethnicity" not in working.columns:
        working["race_ethnicity"] = str(inputs.get("race_ethnicity", PAYLOAD["defaults"]["race_ethnicity"]))
    if "copd_flag" not in working.columns:
        working["copd_flag"] = float(PAYLOAD["defaults"].get("copd_flag", 0.0))
    if "sodium_min_24h" not in working.columns:
        working["sodium_min_24h"] = float(PAYLOAD["defaults"].get("sodium_min_24h", 138.0))

    gwtg_score = compute_gwtg_hf_score(working)
    gwtg_prob = float(_predict_calibrated_probability(
        gwtg_score["gwtg_score"].to_numpy(dtype=float),
        PAYLOAD["gwtg_hf"]["calibrator"],
    )[0])
    adhere_score = compute_adhere_score(working)
    adhere_group = int(adhere_score["adhere_group"].iloc[0])

    top_drivers = ", ".join(item["label"] for item in PAYLOAD["prediction_model"]["top_drivers"][:3])
    model_name = PAYLOAD["prediction_model"]["name"]
    return {
        "prediction_model": {
            "title": f"{model_name} deployment model",
            "risk": risk,
            "details": [
                f"Model alignment: {PAYLOAD['prediction_model']['deployment_note']}",
                f"Input scope: {PAYLOAD['metadata']['prediction_model_feature_count']} harmonized first-day ICU variables",
                f"External validation AUROC: {PAYLOAD['prediction_model']['performance']['external_auroc']:.3f}",
                f"Leading drivers: {top_drivers}",
            ],
        },
        "gwtg_hf": {
            "risk": gwtg_prob,
            "details": [
                f"Point total: {float(gwtg_score['gwtg_score'].iloc[0]):.0f}",
                "Fixed variables: age, SBP, HR, BUN, sodium, COPD, race",
                "Benchmark style: database-adapted point-based bedside score",
            ],
        },
        "adhere": {
            "risk": float(PAYLOAD["adhere"]["group_risk"][str(adhere_group)]),
            "details": [
                f"Risk group: {adhere_group}",
                PAYLOAD["adhere"]["group_label"][str(adhere_group)],
                "Benchmark style: fixed bedside decision tree",
            ],
        },
    }


@app.get("/")
def index():
    return Response(HTML_PATH.read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/api/config")
def config():
    payload = dict(PAYLOAD)
    payload.pop("runtime", None)
    return jsonify(payload)


@app.post("/api/predict")
def predict():
    request_payload = request.get_json(silent=True) or {}
    return jsonify(_predict(request_payload.get("inputs", {})))


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": PAYLOAD["prediction_model"]["name"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8765")))
