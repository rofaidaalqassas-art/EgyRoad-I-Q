# -*- coding: utf-8 -*-
"""
ai_model.py
===========
غلاف (wrapper) حول الموديل المتدرب (injury_model.joblib) بيوفر بالظبط الأجزاء
المطلوبة فى وصف "Smart Accident Impact & Prevention AI":

  1. predict()            -> احتمالية الإصابة + مستوى الخطورة (label)
  2. estimate_outcomes()   -> توقع النتائج البشرية والاقتصادية (من جداول lookup تاريخية)
  3. explain()             -> أهم العوامل اللي رفعت الخطورة فى الحالة دي (local, مش عام)
  4. what_if()             -> تجربة سيناريو بديل (تغيير عوامل قابلة للتحكم) ومقارنته بالأصلي
  5. recommend_actions()   -> إجراءات مقترحة (مبنية على نفس نتيجة explain())

مفيش أى استدعاء للإكسيل هنا - كل حاجة بتتحمل من artifacts/ اللي بناها train_model.py.
"""

import json
import os
from typing import Optional

import joblib
import pandas as pd

ARTIFACTS_DIR = "artifacts"

RISK_BANDS = [(0.30, "منخفض"), (0.60, "متوسط"), (0.85, "مرتفع"), (1.01, "حرج")]

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


def risk_label(prob: float) -> str:
    for threshold, label in RISK_BANDS:
        if prob < threshold:
            return label
    return "حرج"


class AIModel:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.dir = artifacts_dir
        model_path = os.path.join(self.dir, "injury_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"'{model_path}' مش موجود. شغّلي train_model.py الأول عشان يبنى الموديل."
            )
        self.pipeline = joblib.load(model_path)
        self.feature_importance = self._load_json("feature_importance.json")
        self.outcome_lookup = self._load_json("outcome_lookup.json")
        self.global_outcome_avg = self._load_json("global_outcome_avg.json")
        features = self._load_json("model_features.json")
        self.numeric_features = features["numeric"]
        self.categorical_features = features["categorical"]
        self.all_features = self.numeric_features + self.categorical_features

    def _load_json(self, filename):
        path = os.path.join(self.dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _to_frame(self, scenario: dict) -> pd.DataFrame:
        row = {}
        for col in self.numeric_features:
            row[col] = scenario.get(col, 0)
        for col in self.categorical_features:
            row[col] = scenario.get(col, "Unknown")
        return pd.DataFrame([row])

    def _predict_prob(self, scenario: dict) -> float:
        frame = self._to_frame(scenario)
        return float(self.pipeline.predict_proba(frame)[0, 1])

    def predict(self, scenario: dict) -> dict:
        prob = self._predict_prob(scenario)
        label = risk_label(prob)
        return {
            "injury_probability": round(prob, 4),
            "risk_level": label,
            "risk_score_0_100": round(prob * 100, 1),
        }

    def estimate_outcomes(self, scenario: dict, risk_level: Optional[str] = None) -> dict:
        governorate = scenario.get("Governorate_EN")
        if risk_level is None:
            risk_level = risk_label(self._predict_prob(scenario))

        gov_table = self.outcome_lookup.get(governorate, {})
        stats = gov_table.get(risk_level)
        source = "governorate_risk_band"

        if stats is None:
            if gov_table:
                any_band = next(iter(gov_table.values()))
                stats = any_band
                source = "governorate_overall"
            else:
                stats = self.global_outcome_avg
                source = "global_average"

        return {
            "expected_fatalities": round(stats.get("avg_fatalities", 0), 2),
            "expected_injuries": round(stats.get("avg_injuries", 0), 2),
            "expected_economic_loss_egp": round(stats.get("avg_economic_loss_egp", 0), 0),
            "estimate_source": source,
        }

    def explain(self, scenario: dict, top_n: int = 5) -> list:
        baseline_prob = self._predict_prob(scenario)
        results = []

        for field, meta in CONTROLLABLE_FACTORS.items():
            current_value = scenario.get(field, "Unknown")
            if current_value == meta["safe_value"]:
                continue

            improved_scenario = dict(scenario)
            improved_scenario[field] = meta["safe_value"]
            improved_prob = self._predict_prob(improved_scenario)

            delta = baseline_prob - improved_prob
            if delta <= 0:
                continue

            results.append({
                "field": field,
                "label": meta["label"],
                "current_value": current_value,
                "suggested_value": meta["safe_value"],
                "impact_points": round(delta * 100, 2),
                "action": meta["action"],
            })

        results.sort(key=lambda r: r["impact_points"], reverse=True)
        return results[:top_n]

    def what_if(self, scenario: dict, changes: dict) -> dict:
        baseline_prob = self._predict_prob(scenario)
        new_scenario = dict(scenario)
        new_scenario.update(changes)
        new_prob = self._predict_prob(new_scenario)

        return {
            "original": {
                "injury_probability": round(baseline_prob, 4),
                "risk_level": risk_label(baseline_prob),
            },
            "after_changes": {
                "injury_probability": round(new_prob, 4),
                "risk_level": risk_label(new_prob),
            },
            "improvement_points": round((baseline_prob - new_prob) * 100, 2),
            "changes_applied": changes,
        }

    def recommend_actions(self, scenario: dict, top_n: int = 5) -> list:
        top_causes = self.explain(scenario, top_n=top_n)
        return [
            {"factor": c["label"], "action": c["action"], "expected_impact_points": c["impact_points"]}
            for c in top_causes
        ]

    def full_report(self, scenario: dict) -> dict:
        prediction = self.predict(scenario)
        outcomes = self.estimate_outcomes(scenario, risk_level=prediction["risk_level"])
        top_causes = self.explain(scenario)
        actions = [
            {"factor": c["label"], "action": c["action"], "expected_impact_points": c["impact_points"]}
            for c in top_causes
        ]
        return {
            "risk_assessment": prediction,
            "expected_outcomes": outcomes,
            "top_risk_factors": top_causes,
            "recommended_actions": actions,
        }