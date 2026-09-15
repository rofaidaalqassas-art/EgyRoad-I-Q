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
    - Models are loaded once at startup.
    - Speed-derived features are ALWAYS recalculated.
    - Model 2 keeps RAW predictions for impact scoring.
    - Missing raw features remain NaN and are handled by
      the trained preprocessing pipelines.
"""

import json
import math
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd


ARTIFACTS_DIR = "artifacts"


# =========================================================
# RISK BANDS
# =========================================================

RISK_BANDS = [
    (0.30, "منخفض"),
    (0.60, "متوسط"),
    (0.85, "مرتفع"),
    (1.01, "حرج"),
]


# =========================================================
# CONTROLLABLE FACTORS
# =========================================================

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


# =========================================================
# HELPERS
# =========================================================

def _safe_float(value, default=0.0):
    """
    تحويل القيمة إلى float آمن.
    يمنع NaN و Infinity.
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
    تحويل النتائج recursively إلى JSON-safe values.
    """

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return [
            _json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        x = float(value)

        if not math.isfinite(x):
            return None

        return x

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

    return value


def risk_label(prob: float) -> str:

    prob = _safe_float(
        prob,
        0.0
    )

    for threshold, label in RISK_BANDS:

        if prob < threshold:
            return label

    return "حرج"


def _get_learned_categories(pipeline) -> dict:
    """
    استخراج القيم الفئوية التي تدرب عليها الـPipeline.
    """

    ohe = (
        pipeline
        .named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )

    cat_cols = (
        pipeline
        .named_steps["preprocessor"]
        .transformers_[1][2]
    )

    return {
        col: [
            str(v)
            for v in categories
        ]
        for col, categories in zip(
            cat_cols,
            ohe.categories_
        )
    }


# =========================================================
# AI MODEL
# =========================================================

class AIModel:

    def __init__(
        self,
        artifacts_dir: str = ARTIFACTS_DIR
    ):

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

        # =====================================================
        # CHECK ARTIFACTS
        # =====================================================

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

        # =====================================================
        # LOAD MODELS ONCE
        # =====================================================

        self.classifier = joblib.load(
            classifier_path
        )

        self.injuries_model = joblib.load(
            injuries_path
        )

        self.fatalities_model = joblib.load(
            fatalities_path
        )

        self.config = joblib.load(
            config_path
        )

        # =====================================================
        # FEATURES
        # =====================================================

        self.classification_features = (
            self.config[
                "classification"
            ][
                "features"
            ]
        )

        self.classification_threshold = _safe_float(
            self.config[
                "classification"
            ][
                "threshold"
            ],
            0.55
        )

        self.impact_features = (
            self.config[
                "injuries"
            ][
                "features"
            ]
        )

        # =====================================================
        # JSON ARTIFACTS
        # =====================================================

        self.feature_importance = (
            self._load_json(
                "feature_importance.json"
            )
        )

        self.outcome_lookup = (
            self._load_json(
                "outcome_lookup.json"
            )
        )

        self.global_outcome_avg = (
            self._load_json(
                "global_outcome_avg.json"
            )
        )

        # =====================================================
        # CATEGORICAL OPTIONS
        # =====================================================

        self.categorical_options = {}

        try:

            self.categorical_options.update(
                _get_learned_categories(
                    self.injuries_model
                )
            )

        except Exception as e:

            print(
                "[WARN] تعذر استخراج القيم الفئوية "
                f"من موديل الإصابات: {e}"
            )

        try:

            self.categorical_options.update(
                _get_learned_categories(
                    self.classifier
                )
            )

        except Exception as e:

            print(
                "[WARN] تعذر استخراج القيم الفئوية "
                f"من موديل التصنيف: {e}"
            )

        # =====================================================
        # STARTUP INFO
        # =====================================================

        print(
            "[INFO] AIModel loaded successfully."
        )

        print(
            "[INFO] Classification features: "
            f"{len(self.classification_features)}"
        )

        print(
            "[INFO] Impact features: "
            f"{len(self.impact_features)}"
        )

        print(
            "[INFO] Impact feature list:"
        )

        print(
            self.impact_features
        )

    # =========================================================
    # LOAD JSON
    # =========================================================

    def _load_json(
        self,
        filename
    ):

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

    # =========================================================
    # DATAFRAME BUILDER
    # =========================================================

    def _to_frame(
        self,
        scenario: dict,
        feature_list
    ) -> pd.DataFrame:

        row = {}

        for col in feature_list:

            value = scenario.get(
                col,
                np.nan
            )

            if value is None:
                value = np.nan

            row[col] = value

        return pd.DataFrame(
            [row]
        )

    # =========================================================
    # DERIVED SPEED FEATURES
    # =========================================================

    def _add_impact_derived_features(
        self,
        scenario: dict
    ) -> dict:
        """
        IMPORTANT:

        Always recalculate:

            Speed_Over_Limit_KMH
            Speed_Ratio

        whenever speed and speed limit exist.

        This prevents stale derived values after What-If.
        """

        scenario = dict(
            scenario
        )

        speed = scenario.get(
            "Impact_Speed_KMH"
        )

        limit = scenario.get(
            "Posted_Speed_Limit_KMH"
        )

        # -----------------------------------------------------
        # Normalize speed
        # -----------------------------------------------------

        speed_value = None

        if speed is not None:

            try:

                speed_value = float(
                    speed
                )

                if not math.isfinite(
                    speed_value
                ):
                    speed_value = None

            except (
                TypeError,
                ValueError
            ):

                speed_value = None

        # -----------------------------------------------------
        # Normalize limit
        # -----------------------------------------------------

        limit_value = None

        if limit is not None:

            try:

                limit_value = float(
                    limit
                )

                if not math.isfinite(
                    limit_value
                ):
                    limit_value = None

            except (
                TypeError,
                ValueError
            ):

                limit_value = None

        # -----------------------------------------------------
        # Save normalized values
        # -----------------------------------------------------

        if speed_value is not None:

            scenario[
                "Impact_Speed_KMH"
            ] = speed_value

        if limit_value is not None:

            scenario[
                "Posted_Speed_Limit_KMH"
            ] = limit_value

        # -----------------------------------------------------
        # ALWAYS RECOMPUTE DERIVED FEATURES
        # -----------------------------------------------------

        if (
            speed_value is not None
            and
            limit_value is not None
        ):

            scenario[
                "Speed_Over_Limit_KMH"
            ] = max(
                0.0,
                speed_value - limit_value
            )

            if limit_value > 0:

                scenario[
                    "Speed_Ratio"
                ] = (
                    speed_value
                    /
                    limit_value
                )

            else:

                scenario[
                    "Speed_Ratio"
                ] = np.nan

        else:

            # No valid pair => no derived values
            scenario.pop(
                "Speed_Over_Limit_KMH",
                None
            )

            scenario.pop(
                "Speed_Ratio",
                None
            )

        return scenario

    # =========================================================
    # CLASSIFICATION PROBABILITY
    # =========================================================

    def _predict_prob(
        self,
        scenario: dict
    ) -> float:

        scenario = dict(
            scenario
        )

        scenario = (
            self._add_impact_derived_features(
                scenario
            )
        )

        frame = self._to_frame(
            scenario,
            self.classification_features
        )

        try:

            prediction = (
                self.classifier.predict_proba(
                    frame
                )
            )

            probability = prediction[
                0,
                1
            ]

            probability = _safe_float(
                probability,
                0.0
            )

            probability = max(
                0.0,
                min(
                    1.0,
                    probability
                )
            )

            return probability

        except Exception as e:

            print(
                "[ERROR] Classification prediction failed:"
            )

            print(
                f"[ERROR] {type(e).__name__}: {e}"
            )

            print(
                "[ERROR] Scenario:"
            )

            print(
                scenario
            )

            print(
                "[ERROR] Classification features:"
            )

            print(
                self.classification_features
            )

            # مهم:
            # لا نخفي الخطأ ونرجع 0.
            # لأن 0 كان ممكن يخلي الـAPI يعطي نتيجة مضللة.
            raise

    # =========================================================
    # MODEL 1
    # =========================================================

    def predict(
        self,
        scenario: dict
    ) -> dict:

        prob = self._predict_prob(
            scenario
        )

        print(
            "[DEBUG] predict (classifier)"
        )

        print(
            f"[DEBUG] scenario={scenario}"
        )

        print(
            f"[DEBUG] probability={prob}"
        )

        result = {

            "injury_probability":
                round(
                    prob,
                    4
                ),

            "has_injury_prediction":
                int(
                    prob >=
                    self.classification_threshold
                ),

            "risk_level":
                risk_label(
                    prob
                ),

            "risk_score_0_100":
                round(
                    prob * 100,
                    1
                ),
        }

        return _json_safe(
            result
        )

    # =========================================================
    # MODEL 2
    # ACCIDENT IMPACT
    # =========================================================

    def predict_impact(
        self,
        scenario: dict
    ) -> dict:
        """
        Model 2:
            injuries regression
            fatalities regression

        IMPORTANT:
        Raw predictions are preserved for scoring.
        Display values are rounded separately.
        """

        # -----------------------------------------------------
        # Recalculate speed features
        # -----------------------------------------------------

        scenario = (
            self._add_impact_derived_features(
                scenario
            )
        )

        # -----------------------------------------------------
        # Debug: Present features
        # -----------------------------------------------------

        present_features = {
            feature: scenario.get(
                feature
            )
            for feature in self.impact_features
            if feature in scenario
        }

        missing_features = [
            feature
            for feature in self.impact_features
            if feature not in scenario
        ]

        print(
            "[DEBUG] ===== MODEL 2 INPUT ====="
        )

        print(
            "[DEBUG] Total impact features: "
            f"{len(self.impact_features)}"
        )

        print(
            "[DEBUG] Present features: "
            f"{len(present_features)}"
        )

        print(
            "[DEBUG] Missing features: "
            f"{len(missing_features)}"
        )

        print(
            "[DEBUG] Present values:"
        )

        print(
            present_features
        )

        if missing_features:

            print(
                "[DEBUG] Missing features:"
            )

            print(
                missing_features
            )

        # -----------------------------------------------------
        # Model frame
        # -----------------------------------------------------

        frame = self._to_frame(
            scenario,
            self.impact_features
        )

        # -----------------------------------------------------
        # Injuries
        # -----------------------------------------------------

        try:

            injuries_prediction = (
                self.injuries_model.predict(
                    frame
                )
            )

            injuries_raw = _safe_float(
                injuries_prediction[0],
                0.0
            )

        except Exception as e:

            print(
                "[ERROR] Injuries model prediction failed:"
            )

            print(
                f"[ERROR] {type(e).__name__}: {e}"
            )

            raise

        # -----------------------------------------------------
        # Fatalities
        # -----------------------------------------------------

        try:

            fatalities_prediction = (
                self.fatalities_model.predict(
                    frame
                )
            )

            fatalities_raw = _safe_float(
                fatalities_prediction[0],
                0.0
            )

        except Exception as e:

            print(
                "[ERROR] Fatalities model prediction failed:"
            )

            print(
                f"[ERROR] {type(e).__name__}: {e}"
            )

            raise

        # -----------------------------------------------------
        # Debug raw output
        # -----------------------------------------------------

        print(
            "[DEBUG] ===== MODEL 2 OUTPUT ====="
        )

        print(
            f"[DEBUG] injuries_raw="
            f"{injuries_raw}"
        )

        print(
            f"[DEBUG] fatalities_raw="
            f"{fatalities_raw}"
        )

        # -----------------------------------------------------
        # Display predictions
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # IMPACT SCORE
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # Use RAW model values.
        #
        # Example:
        #
        # 6.21 -> 6.87
        #
        # should NOT become:
        #
        # 6 -> 7
        #
        # and then lose the actual model difference.
        #

        impact_score = (
            injuries_raw * 8.0
            +
            fatalities_raw * 25.0
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

        # -----------------------------------------------------
        # IMPACT LEVEL
        # -----------------------------------------------------

        if impact_score >= 60:

            impact_level = "مرتفع"

        elif impact_score >= 25:

            impact_level = "متوسط"

        else:

            impact_level = "منخفض"

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        result = {

            "predicted_injuries":
                predicted_injuries,

            "predicted_fatalities":
                predicted_fatalities,

            "predicted_injuries_raw":
                round(
                    injuries_raw,
                    3
                ),

            "predicted_fatalities_raw":
                round(
                    fatalities_raw,
                    3
                ),

            "impact_score":
                round(
                    impact_score,
                    1
                ),

            "impact_level":
                impact_level,

            "features_used":
                len(
                    present_features
                ),

            "features_expected":
                len(
                    self.impact_features
                ),

            "missing_features":
                missing_features,

            # Speed debug
            "impact_speed":
                scenario.get(
                    "Impact_Speed_KMH"
                ),

            "posted_speed_limit":
                scenario.get(
                    "Posted_Speed_Limit_KMH"
                ),

            "speed_over_limit":
                scenario.get(
                    "Speed_Over_Limit_KMH"
                ),

            "speed_ratio":
                scenario.get(
                    "Speed_Ratio"
                ),
        }

        return _json_safe(
            result
        )

    # =========================================================
    # HISTORICAL OUTCOMES
    # =========================================================

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

        gov_table = (
            self.outcome_lookup.get(
                governorate,
                {}
            )
        )

        stats = gov_table.get(
            risk_level
        )

        source = (
            "governorate_risk_band"
        )

        if stats is None:

            if gov_table:

                stats = next(
                    iter(
                        gov_table.values()
                    )
                )

                source = (
                    "governorate_overall"
                )

            else:

                stats = (
                    self.global_outcome_avg
                )

                source = (
                    "global_average"
                )

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

            "expected_fatalities":
                round(
                    expected_fatalities,
                    2
                ),

            "expected_injuries":
                round(
                    expected_injuries,
                    2
                ),

            "expected_economic_loss_egp":
                round(
                    expected_loss,
                    0
                ),

            "estimate_source":
                source,
        }

        return _json_safe(
            result
        )

    # =========================================================
    # EXPLAIN
    # =========================================================

    def explain(
        self,
        scenario: dict,
        top_n: int = 5
    ) -> list:

        scenario = dict(
            scenario
        )

        scenario = (
            self._add_impact_derived_features(
                scenario
            )
        )

        baseline_prob = (
            self._predict_prob(
                scenario
            )
        )

        results = []

        for field, meta in (
            CONTROLLABLE_FACTORS.items()
        ):

            current_value = scenario.get(
                field,
                "Unknown"
            )

            if (
                current_value
                ==
                meta["safe_value"]
            ):

                continue

            improved_scenario = dict(
                scenario
            )

            improved_scenario[field] = (
                meta["safe_value"]
            )

            improved_prob = (
                self._predict_prob(
                    improved_scenario
                )
            )

            delta = (
                baseline_prob
                -
                improved_prob
            )

            if delta <= 0:

                continue

            results.append({

                "field":
                    field,

                "label":
                    meta["label"],

                "current_value":
                    current_value,

                "suggested_value":
                    meta["safe_value"],

                "impact_points":
                    round(
                        delta * 100,
                        2
                    ),

                "action":
                    meta["action"],
            })

        results.sort(
            key=lambda r:
                r["impact_points"],
            reverse=True
        )

        return _json_safe(
            results[:top_n]
        )

    # =========================================================
    # MODEL 3
    # WHAT-IF
    # =========================================================

    def what_if(
        self,
        scenario: dict,
        changes: dict
    ) -> dict:
        """
        Model 3 What-If.

        For categorical safety interventions:
            uses Model 1 classifier.

        For speed-related changes:
            the caller should preferably use predict_impact()
            because Model 2 was trained with Impact_Speed_KMH,
            Speed_Over_Limit_KMH and Speed_Ratio.

        This method still correctly recalculates all derived
        speed features whenever speed changes.
        """

        # =====================================================
        # BASELINE
        # =====================================================

        baseline_scenario = dict(
            scenario
        )

        baseline_scenario = (
            self._add_impact_derived_features(
                baseline_scenario
            )
        )

        baseline_prob = (
            self._predict_prob(
                baseline_scenario
            )
        )

        # =====================================================
        # APPLY CHANGES
        # =====================================================

        new_scenario = dict(
            baseline_scenario
        )

        new_scenario.update(
            changes
        )

        # =====================================================
        # REMOVE STALE DERIVED FEATURES
        # =====================================================

        if (
            "Impact_Speed_KMH"
            in changes
            or
            "Posted_Speed_Limit_KMH"
            in changes
        ):

            new_scenario.pop(
                "Speed_Over_Limit_KMH",
                None
            )

            new_scenario.pop(
                "Speed_Ratio",
                None
            )

        # =====================================================
        # REBUILD DERIVED FEATURES
        # =====================================================

        new_scenario = (
            self._add_impact_derived_features(
                new_scenario
            )
        )

        # =====================================================
        # NEW CLASSIFIER PREDICTION
        # =====================================================

        new_prob = (
            self._predict_prob(
                new_scenario
            )
        )

        # =====================================================
        # IMPROVEMENT
        # =====================================================

        improvement = (
            baseline_prob
            -
            new_prob
        ) * 100.0

        improvement = _safe_float(
            improvement,
            0.0
        )

        # =====================================================
        # DEBUG
        # =====================================================

        print(
            "[DEBUG] ===== WHAT-IF ====="
        )

        print(
            "[DEBUG] baseline speed="
            f"{baseline_scenario.get('Impact_Speed_KMH')}"
        )

        print(
            "[DEBUG] new speed="
            f"{new_scenario.get('Impact_Speed_KMH')}"
        )

        print(
            "[DEBUG] speed limit="
            f"{new_scenario.get('Posted_Speed_Limit_KMH')}"
        )

        print(
            "[DEBUG] baseline speed over limit="
            f"{baseline_scenario.get('Speed_Over_Limit_KMH')}"
        )

        print(
            "[DEBUG] new speed over limit="
            f"{new_scenario.get('Speed_Over_Limit_KMH')}"
        )

        print(
            "[DEBUG] baseline speed ratio="
            f"{baseline_scenario.get('Speed_Ratio')}"
        )

        print(
            "[DEBUG] new speed ratio="
            f"{new_scenario.get('Speed_Ratio')}"
        )

        print(
            "[DEBUG] baseline probability="
            f"{baseline_prob}"
        )

        print(
            "[DEBUG] new probability="
            f"{new_prob}"
        )

        print(
            "[DEBUG] improvement="
            f"{improvement}"
        )

        # =====================================================
        # RESULT
        # =====================================================

        result = {

            "original": {

                "injury_probability":
                    round(
                        baseline_prob,
                        4
                    ),

                "risk_level":
                    risk_label(
                        baseline_prob
                    ),

                "risk_score":
                    round(
                        baseline_prob * 100,
                        1
                    ),

                "impact_speed":
                    _safe_float(
                        baseline_scenario.get(
                            "Impact_Speed_KMH"
                        ),
                        0.0
                    ),

                "speed_over_limit":
                    _safe_float(
                        baseline_scenario.get(
                            "Speed_Over_Limit_KMH"
                        ),
                        0.0
                    ),

                "speed_ratio":
                    _safe_float(
                        baseline_scenario.get(
                            "Speed_Ratio"
                        ),
                        0.0
                    ),
            },

            "after_changes": {

                "injury_probability":
                    round(
                        new_prob,
                        4
                    ),

                "risk_level":
                    risk_label(
                        new_prob
                    ),

                "risk_score":
                    round(
                        new_prob * 100,
                        1
                    ),

                "impact_speed":
                    _safe_float(
                        new_scenario.get(
                            "Impact_Speed_KMH"
                        ),
                        0.0
                    ),

                "speed_over_limit":
                    _safe_float(
                        new_scenario.get(
                            "Speed_Over_Limit_KMH"
                        ),
                        0.0
                    ),

                "speed_ratio":
                    _safe_float(
                        new_scenario.get(
                            "Speed_Ratio"
                        ),
                        0.0
                    ),
            },

            "improvement_points":
                round(
                    improvement,
                    2
                ),

            "changes_applied":
                changes,

            "debug": {

                "baseline_speed":
                    baseline_scenario.get(
                        "Impact_Speed_KMH"
                    ),

                "new_speed":
                    new_scenario.get(
                        "Impact_Speed_KMH"
                    ),

                "speed_limit":
                    new_scenario.get(
                        "Posted_Speed_Limit_KMH"
                    ),

                "baseline_speed_over_limit":
                    baseline_scenario.get(
                        "Speed_Over_Limit_KMH"
                    ),

                "new_speed_over_limit":
                    new_scenario.get(
                        "Speed_Over_Limit_KMH"
                    ),

                "baseline_speed_ratio":
                    baseline_scenario.get(
                        "Speed_Ratio"
                    ),

                "new_speed_ratio":
                    new_scenario.get(
                        "Speed_Ratio"
                    ),

                "baseline_probability":
                    baseline_prob,

                "new_probability":
                    new_prob,
            },
        }

        return _json_safe(
            result
        )

    # =========================================================
    # RECOMMENDATIONS
    # =========================================================

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

                "factor":
                    cause["label"],

                "action":
                    cause["action"],

                "expected_impact_points":
                    cause[
                        "impact_points"
                    ],
            })

        return _json_safe(
            result
        )

    # =========================================================
    # FULL REPORT
    # =========================================================

    def full_report(
        self,
        scenario: dict
    ) -> dict:

        prediction = self.predict(
            scenario
        )

        outcomes = (
            self.estimate_outcomes(
                scenario,
                risk_level=
                    prediction[
                        "risk_level"
                    ]
            )
        )

        top_causes = self.explain(
            scenario
        )

        actions = [

            {
                "factor":
                    cause["label"],

                "action":
                    cause["action"],

                "expected_impact_points":
                    cause[
                        "impact_points"
                    ],
            }

            for cause in top_causes
        ]

        report = {

            "risk_assessment":
                prediction,

            "expected_outcomes":
                outcomes,

            "top_risk_factors":
                top_causes,

            "recommended_actions":
                actions,
        }

        if (
            scenario.get(
                "Impact_Speed_KMH"
            )
            is not None
        ):

            report[
                "impact_prediction"
            ] = self.predict_impact(
                scenario
            )

        return _json_safe(
            report
        )