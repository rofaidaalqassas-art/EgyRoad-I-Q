# -*- coding: utf-8 -*-
"""
ai_model.py
===========

AI wrapper for EgyRoad IQ.

Models:
    - final_injury_classifier_catboost.pkl
    - final_injuries_xgboost.pkl
    - final_fatalities_catboost.pkl
    - final_model_config.pkl

Main functions:
    1. predict()
    2. predict_impact()
    3. estimate_outcomes()
    4. explain()
    5. what_if()
    6. recommend_actions()
    7. full_report()

Important:
All models are complete sklearn-compatible pipelines.
Missing raw features are left as NaN and handled by the model pipeline.
"""

import json
import math
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd


ARTIFACTS_DIR = "artifacts"


RISK_BANDS = [
    (0.30, "منخفض"),
    (0.60, "متوسط"),
    (0.85, "مرتفع"),
    (1.01, "حرج"),
]


CONTROLLABLE_FACTORS = {
    "Lighting_Condition": {
        "label": "الإضاءة الليلية",
        "safe_value": "Lit",
        "action": "تركيب/تشغيل إضاءة ليلية على الطريق",
    },
    "Road_Surface_Condition": {
        "label": "حالة الرصف",
        "safe_value": "Good",
        "action": "صيانة وإعادة رصف الطريق",
    },
    "Technical_Inspection_Passed": {
        "label": "الفحص الفني للمركبة",
        "safe_value": "Yes",
        "action": "التأكد من الفحص الفني الدوري للمركبات",
    },
    "Overload_Flag": {
        "label": "تحميل زائد على المركبة",
        "safe_value": "No",
        "action": "منع تحميل المركبات فوق الحمولة المقررة",
    },
    "ABS_Equipped": {
        "label": "وجود نظام ABS",
        "safe_value": "Yes",
        "action": "التشجيع/الإلزام بمركبات مزودة بنظام ABS",
    },
    "Driver_Airbag_Equipped": {
        "label": "وجود وسادة هوائية",
        "safe_value": "Yes",
        "action": "التأكد من تجهيز المركبة بوسادة هوائية سليمة",
    },
    "Tire_Condition": {
        "label": "حالة الإطارات",
        "safe_value": "Good",
        "action": "فحص واستبدال الإطارات التالفة",
    },
    "Brake_Condition": {
        "label": "حالة الفرامل",
        "safe_value": "Good",
        "action": "صيانة نظام الفرامل بشكل دوري",
    },
    "Distracted_Driving_Mobile": {
        "label": "استخدام الموبايل أثناء القيادة",
        "safe_value": "No",
        "action": "حملات توعية وتغليظ عقوبة استخدام الموبايل أثناء القيادة",
    },
    "Seatbelt_Used": {
        "label": "استخدام حزام الأمان",
        "safe_value": "Yes",
        "action": "إلزام وتوعية باستخدام حزام الأمان",
    },
    "Helmet_Used": {
        "label": "استخدام الخوذة",
        "safe_value": "Yes",
        "action": "إلزام وتوعية باستخدام الخوذة لمستخدمي الدراجات النارية",
    },
    "Insurance_Coverage": {
        "label": "التأمين على المركبة",
        "safe_value": "Yes",
        "action": "التشجيع على التأمين الشامل للمركبات",
    },
    "Licensing_Status": {
        "label": "حالة ترخيص المركبة",
        "safe_value": "Valid",
        "action": "التأكد من سريان ترخيص المركبة",
    },
}


def _safe_float(value, default=0.0):
    """
    يحول أي قيمة إلى float آمن للـJSON.
    يمنع NaN و Infinity من الخروج للـFastAPI.
    """
    try:
        x = float(value)

        if not math.isfinite(x):
            return float(default)

        return x

    except (TypeError, ValueError):
        return float(default)


def _json_safe(value):
    """
    يحول النتائج recursively إلى قيم JSON آمنة.
    """
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        x = float(value)

        if not math.isfinite(x):
            return None

        return x

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def risk_label(prob: float) -> str:
    prob = _safe_float(prob, 0.0)

    for threshold, label in RISK_BANDS:
        if prob < threshold:
            return label

    return "حرج"


class AIModel:

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):

        self.dir = artifacts_dir

        classifier_path = os.path.join(
            self.dir,
            "final_injury_classifier_catboost.pkl"
        )

        injuries_path = os.path.join(
            self.dir,
            "final_injuries_xgboost.pkl"
        )

        fatalities_path = os.path.join(
            self.dir,
            "final_fatalities_catboost.pkl"
        )

        config_path = os.path.join(
            self.dir,
            "final_model_config.pkl"
        )

        for path in (
            classifier_path,
            injuries_path,
            fatalities_path,
            config_path,
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{path}' مش موجود. "
                    f"شغّلي train_model.py الأول عشان يبنى الموديلات."
                )

        self.classifier = joblib.load(classifier_path)
        self.injuries_model = joblib.load(injuries_path)
        self.fatalities_model = joblib.load(fatalities_path)
        self.config = joblib.load(config_path)

        self.classification_features = (
            self.config["classification"]["features"]
        )

        self.classification_threshold = _safe_float(
            self.config["classification"]["threshold"],
            0.55
        )

        self.impact_features = self.config["injuries"]["features"]

        self.feature_importance = self._load_json(
            "feature_importance.json"
        )

        self.outcome_lookup = self._load_json(
            "outcome_lookup.json"
        )

        self.global_outcome_avg = self._load_json(
            "global_outcome_avg.json"
        )

    def _load_json(self, filename):

        path = os.path.join(
            self.dir,
            filename
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    # ---------------------------------------------------------
    # DataFrame builder
    # ---------------------------------------------------------

    def _to_frame(
        self,
        scenario: dict,
        feature_list
    ) -> pd.DataFrame:

        row = {}

        for col in feature_list:
            value = scenario.get(col, np.nan)

            # None تتحول إلى NaN
            if value is None:
                value = np.nan

            row[col] = value

        return pd.DataFrame([row])

    # ---------------------------------------------------------
    # Derived impact features
    # ---------------------------------------------------------

    def _add_impact_derived_features(
        self,
        scenario: dict
    ) -> dict:

        scenario = dict(scenario)

        speed = scenario.get(
            "Impact_Speed_KMH"
        )

        limit = scenario.get(
            "Posted_Speed_Limit_KMH"
        )

        if (
            "Speed_Over_Limit_KMH" not in scenario
            and speed is not None
            and limit is not None
        ):
            try:
                speed_value = _safe_float(speed)
                limit_value = _safe_float(limit)

                scenario["Speed_Over_Limit_KMH"] = max(
                    0.0,
                    speed_value - limit_value
                )

            except Exception:
                pass

        if (
            "Speed_Ratio" not in scenario
            and speed is not None
            and limit is not None
        ):
            try:
                speed_value = _safe_float(speed)
                limit_value = _safe_float(limit)

                if limit_value > 0:
                    scenario["Speed_Ratio"] = (
                        speed_value / limit_value
                    )

            except Exception:
                pass

        return scenario

    # ---------------------------------------------------------
    # Classification probability
    # ---------------------------------------------------------

    def _predict_prob(
        self,
        scenario: dict
    ) -> float:

        frame = self._to_frame(
            scenario,
            self.classification_features
        )

        try:
            prediction = self.classifier.predict_proba(frame)

            probability = prediction[0, 1]

            return max(
                0.0,
                min(
                    1.0,
                    _safe_float(probability, 0.0)
                )
            )

        except Exception:
            # fallback آمن بدل انهيار الـAPI
            return 0.0

    # ---------------------------------------------------------
    # Model 1
    # ---------------------------------------------------------

    def predict(
        self,
        scenario: dict
    ) -> dict:

        prob = self._predict_prob(
            scenario
        )

        result = {
            "injury_probability": round(
                prob,
                4
            ),

            "has_injury_prediction": int(
                prob >= self.classification_threshold
            ),

            "risk_level": risk_label(
                prob
            ),

            "risk_score_0_100": round(
                prob * 100,
                1
            ),
        }

        return _json_safe(result)

    # ---------------------------------------------------------
    # Model 2
    # Accident Impact
    # ---------------------------------------------------------

    def predict_impact(
        self,
        scenario: dict
    ) -> dict:

        scenario = self._add_impact_derived_features(
            scenario
        )

        frame = self._to_frame(
            scenario,
            self.impact_features
        )

        # -------------------------------
        # Injuries
        # -------------------------------

        try:
            injuries_prediction = (
                self.injuries_model.predict(frame)
            )

            injuries_raw = _safe_float(
                injuries_prediction[0],
                0.0
            )

        except Exception:
            injuries_raw = 0.0

        # -------------------------------
        # Fatalities
        # -------------------------------

        try:
            fatalities_prediction = (
                self.fatalities_model.predict(frame)
            )

            fatalities_raw = _safe_float(
                fatalities_prediction[0],
                0.0
            )

        except Exception:
            fatalities_raw = 0.0

        # -------------------------------
        # Clean predictions
        # -------------------------------

        predicted_injuries = int(
            round(
                max(
                    0.0,
                    injuries_raw
                )
            )
        )

        predicted_fatalities = int(
            round(
                max(
                    0.0,
                    fatalities_raw
                )
            )
        )

        # -------------------------------
        # Impact Score
        # -------------------------------

        impact_score = (
            predicted_injuries * 8.0
            +
            predicted_fatalities * 25.0
        )

        impact_score = max(
            0.0,
            min(
                100.0,
                _safe_float(
                    impact_score,
                    0.0
                )
            )
        )

        # -------------------------------
        # Impact Level
        # -------------------------------

        if impact_score >= 60:
            impact_level = "مرتفع"

        elif impact_score >= 25:
            impact_level = "متوسط"

        else:
            impact_level = "منخفض"

        result = {
            "predicted_injuries": predicted_injuries,

            "predicted_fatalities": predicted_fatalities,

            "predicted_injuries_raw": round(
                injuries_raw,
                3
            ),

            "predicted_fatalities_raw": round(
                fatalities_raw,
                3
            ),

            "impact_score": round(
                impact_score,
                1
            ),

            "impact_level": impact_level,
        }

        # أهم نقطة:
        # ضمان عدم خروج NaN أو Infinity نهائيًا.
        return _json_safe(result)

    # ---------------------------------------------------------
    # Historical outcomes
    # ---------------------------------------------------------

    def estimate_outcomes(
        self,
        scenario: dict,
        risk_level: Optional[str] = None
    ) -> dict:

        governorate = scenario.get(
            "Governorate_EN"
        )

        if risk_level is None:
            risk_level = risk_label(
                self._predict_prob(
                    scenario
                )
            )

        gov_table = self.outcome_lookup.get(
            governorate,
            {}
        )

        stats = gov_table.get(
            risk_level
        )

        source = "governorate_risk_band"

        if stats is None:

            if gov_table:
                stats = next(
                    iter(
                        gov_table.values()
                    )
                )

                source = "governorate_overall"

            else:

                stats = self.global_outcome_avg

                source = "global_average"

        expected_fatalities = _safe_float(
            stats.get(
                "avg_fatalities",
                0
            )
        )

        expected_injuries = _safe_float(
            stats.get(
                "avg_injuries",
                0
            )
        )

        expected_loss = _safe_float(
            stats.get(
                "avg_economic_loss_egp",
                0
            )
        )

        result = {
            "expected_fatalities": round(
                expected_fatalities,
                2
            ),

            "expected_injuries": round(
                expected_injuries,
                2
            ),

            "expected_economic_loss_egp": round(
                expected_loss,
                0
            ),

            "estimate_source": source,
        }

        return _json_safe(result)

    # ---------------------------------------------------------
    # Explain
    # ---------------------------------------------------------

    def explain(
        self,
        scenario: dict,
        top_n: int = 5
    ) -> list:

        baseline_prob = self._predict_prob(
            scenario
        )

        results = []

        for field, meta in CONTROLLABLE_FACTORS.items():

            current_value = scenario.get(
                field,
                "Unknown"
            )

            if current_value == meta["safe_value"]:
                continue

            improved_scenario = dict(
                scenario
            )

            improved_scenario[field] = (
                meta["safe_value"]
            )

            improved_prob = self._predict_prob(
                improved_scenario
            )

            delta = (
                baseline_prob
                -
                improved_prob
            )

            if delta <= 0:
                continue

            results.append({
                "field": field,

                "label": meta["label"],

                "current_value": current_value,

                "suggested_value": meta["safe_value"],

                "impact_points": round(
                    delta * 100,
                    2
                ),

                "action": meta["action"],
            })

        results.sort(
            key=lambda r: r["impact_points"],
            reverse=True
        )

        return _json_safe(
            results[:top_n]
        )

    # ---------------------------------------------------------
    # Model 3
    # What-If
    # ---------------------------------------------------------

    def what_if(
        self,
        scenario: dict,
        changes: dict
    ) -> dict:

        baseline_prob = self._predict_prob(
            scenario
        )

        new_scenario = dict(
            scenario
        )

        new_scenario.update(
            changes
        )

        new_prob = self._predict_prob(
            new_scenario
        )

        improvement = (
            baseline_prob
            -
            new_prob
        ) * 100

        result = {
            "original": {
                "injury_probability": round(
                    baseline_prob,
                    4
                ),

                "risk_level": risk_label(
                    baseline_prob
                ),
            },

            "after_changes": {
                "injury_probability": round(
                    new_prob,
                    4
                ),

                "risk_level": risk_label(
                    new_prob
                ),
            },

            "improvement_points": round(
                _safe_float(
                    improvement,
                    0.0
                ),
                2
            ),

            "changes_applied": changes,
        }

        return _json_safe(
            result
        )

    # ---------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------

    def recommend_actions(
        self,
        scenario: dict,
        top_n: int = 5
    ) -> list:

        top_causes = self.explain(
            scenario,
            top_n=top_n
        )

        result = []

        for cause in top_causes:

            result.append({
                "factor": cause["label"],

                "action": cause["action"],

                "expected_impact_points": cause[
                    "impact_points"
                ],
            })

        return _json_safe(
            result
        )

    # ---------------------------------------------------------
    # Full Report
    # ---------------------------------------------------------

    def full_report(
        self,
        scenario: dict
    ) -> dict:

        prediction = self.predict(
            scenario
        )

        outcomes = self.estimate_outcomes(
            scenario,
            risk_level=prediction["risk_level"]
        )

        top_causes = self.explain(
            scenario
        )

        actions = [
            {
                "factor": cause["label"],

                "action": cause["action"],

                "expected_impact_points": cause[
                    "impact_points"
                ],
            }

            for cause in top_causes
        ]

        report = {
            "risk_assessment": prediction,

            "expected_outcomes": outcomes,

            "top_risk_factors": top_causes,

            "recommended_actions": actions,
        }

        if scenario.get(
            "Impact_Speed_KMH"
        ) is not None:

            report["impact_prediction"] = (
                self.predict_impact(
                    scenario
                )
            )

        return _json_safe(
            report
        )