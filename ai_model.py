# -*- coding: utf-8 -*-
"""
ai_model.py
===========
غلاف (wrapper) حول الموديلات الثلاثة النهائية اللي درّبها train_model.py
(بنفس منطق نوتبوك "Final Selected ML Models"):

    - final_injury_classifier_catboost.pkl  -> احتمالية الإصابة (Has_Injury)
    - final_injuries_xgboost.pkl             -> عدد الإصابات المتوقع
    - final_fatalities_catboost.pkl          -> عدد الوفيات المتوقع
    - final_model_config.pkl                 -> params + threshold + feature lists

مهم: كل موديل عبارة عن Pipeline كامل (فيه الـ preprocessing جواه)، فأي
سيناريو بيتبعت له لازم يكون فيه الأعمدة الخام (raw) زي ما هي، من غير أي
تجهيز يدوي إضافي. الأعمدة الناقصة بيتم التعامل معاها تلقائيًا عن طريق
الـ imputer جوه كل Pipeline (median للأرقام / most_frequent للفئات).

بيوفر:
  1. predict()            -> احتمالية الإصابة + مستوى الخطورة (classification فقط)
  2. predict_impact()      -> عدد الإصابات/الوفيات المتوقع (regression - محتاج
                              Impact_Speed_KMH و Collision_Type وباقي IMPACT_FEATURES)
  3. estimate_outcomes()   -> توقع النتائج البشرية والاقتصادية (من جداول lookup تاريخية)
  4. explain()             -> أهم العوامل اللي رفعت الخطورة فى الحالة دي (local, مش عام)
  5. what_if()             -> تجربة سيناريو بديل (تغيير عوامل قابلة للتحكم) ومقارنته بالأصلي
  6. recommend_actions()   -> إجراءات مقترحة (مبنية على نفس نتيجة explain())
  7. full_report()          -> كل حاجة سوا (classification + outcomes + توصيات)

مفيش أى استدعاء للإكسيل هنا - كل حاجة بتتحمل من artifacts/ اللي بناها train_model.py.
"""

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = "artifacts"

# فئات خطورة عامة (لعرض النتيجة وربطها بجداول outcome_lookup) - مش لها علاقة
# بالـ classification threshold (0.55) اللي بيحدد Has_Injury نفسه.
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

        classifier_path = os.path.join(self.dir, "final_injury_classifier_catboost.pkl")
        injuries_path = os.path.join(self.dir, "final_injuries_xgboost.pkl")
        fatalities_path = os.path.join(self.dir, "final_fatalities_catboost.pkl")
        config_path = os.path.join(self.dir, "final_model_config.pkl")

        for path in (classifier_path, injuries_path, fatalities_path, config_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{path}' مش موجود. شغّلي train_model.py الأول عشان يبنى الموديلات."
                )

        self.classifier = joblib.load(classifier_path)
        self.injuries_model = joblib.load(injuries_path)
        self.fatalities_model = joblib.load(fatalities_path)
        self.config = joblib.load(config_path)

        self.classification_features = self.config["classification"]["features"]
        self.classification_threshold = self.config["classification"]["threshold"]
        self.impact_features = self.config["injuries"]["features"]

        # لسه محتاجين model_features.json (numeric/categorical) عشان
        # الـ explain()/what_if() تبني scenario كامل من غير ما تحتاج القيم
        # الرقمية والفئوية تتخلط ببعض.
        self.feature_importance = self._load_json("feature_importance.json")
        self.outcome_lookup = self._load_json("outcome_lookup.json")
        self.global_outcome_avg = self._load_json("global_outcome_avg.json")

    def _load_json(self, filename):
        path = os.path.join(self.dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # بناء الـ DataFrame المطلوب - الأعمدة الناقصة بتتسيب NaN عشان الـ
    # imputer جوه الـ Pipeline (median/most_frequent) هو اللي يتصرف فيها،
    # مش تخمين يدوي زي 0 أو "Unknown".
    # ------------------------------------------------------------------
    def _to_frame(self, scenario: dict, feature_list) -> pd.DataFrame:
        row = {col: scenario.get(col, np.nan) for col in feature_list}
        return pd.DataFrame([row])

    def _add_impact_derived_features(self, scenario: dict) -> dict:
        """يحسب Speed_Over_Limit_KMH و Speed_Ratio لو الأعمدة الأساسية متاحة."""
        scenario = dict(scenario)
        speed = scenario.get("Impact_Speed_KMH")
        limit = scenario.get("Posted_Speed_Limit_KMH")

        if "Speed_Over_Limit_KMH" not in scenario and speed is not None and limit is not None:
            scenario["Speed_Over_Limit_KMH"] = max(0.0, float(speed) - float(limit))

        if "Speed_Ratio" not in scenario and speed is not None and limit:
            scenario["Speed_Ratio"] = float(speed) / float(limit) if float(limit) > 0 else None

        return scenario

    def _predict_prob(self, scenario: dict) -> float:
        frame = self._to_frame(scenario, self.classification_features)
        return float(self.classifier.predict_proba(frame)[0, 1])

    # ------------------------------------------------------------------
    # Classification (Has_Injury)
    # ------------------------------------------------------------------
    def predict(self, scenario: dict) -> dict:
        prob = self._predict_prob(scenario)
        return {
            "injury_probability": round(prob, 4),
            "has_injury_prediction": int(prob >= self.classification_threshold),
            "risk_level": risk_label(prob),
            "risk_score_0_100": round(prob * 100, 1),
        }

    # ------------------------------------------------------------------
    # Regression (Injuries_Count / Fatalities_Count) - محتاج IMPACT_FEATURES
    # (فيها Impact_Speed_KMH و Collision_Type)
    # ------------------------------------------------------------------
    def predict_impact(self, scenario: dict) -> dict:
        scenario = self._add_impact_derived_features(scenario)
        frame = self._to_frame(scenario, self.impact_features)

        injuries_raw = float(self.injuries_model.predict(frame)[0])
        fatalities_raw = float(self.fatalities_model.predict(frame)[0])

        predicted_injuries = int(round(max(0.0, injuries_raw)))
        predicted_fatalities = int(round(max(0.0, fatalities_raw)))

        # درجة تأثير 0-100 مبنية على الإصابات/الوفيات المتوقعة (الوفيات لها
        # وزن أكبر). القيم دي تقريبية لعرض شريط/لون فى الواجهة فقط.
        impact_score = min(100.0, predicted_injuries * 8.0 + predicted_fatalities * 25.0)
        impact_level = (
            "مرتفع" if impact_score >= 60 else ("متوسط" if impact_score >= 25 else "منخفض")
        )

        return {
            "predicted_injuries": predicted_injuries,
            "predicted_fatalities": predicted_fatalities,
            "predicted_injuries_raw": round(injuries_raw, 3),
            "predicted_fatalities_raw": round(fatalities_raw, 3),
            "impact_score": round(impact_score, 1),
            "impact_level": impact_level,
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
                stats = next(iter(gov_table.values()))
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
        report = {
            "risk_assessment": prediction,
            "expected_outcomes": outcomes,
            "top_risk_factors": top_causes,
            "recommended_actions": actions,
        }

        # لو السيناريو فيه بيانات الاصطدام (Impact_Speed_KMH/Collision_Type)
        # ضيفي كمان توقع الإصابات/الوفيات الفعلي من موديلات الـ regression.
        if scenario.get("Impact_Speed_KMH") is not None:
            report["impact_prediction"] = self.predict_impact(scenario)

        return report
