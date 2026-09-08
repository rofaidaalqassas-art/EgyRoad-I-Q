# -*- coding: utf-8 -*-
"""
risk_model.py
=============
المعادلة اليدوية (weighted formula) القديمة - لسه مستخدمة بس فى:
  - /dashboard-stats و /roads (إحصائيات عامة للوحة متخذ القرار)
  - /predict القديم (تقييم سريع لمحافظة/رحلة بدون تفاصيل سائق/مركبة)

التنبؤ الحقيقي (AI) بقى فى ai_model.py + train_model.py.
"""

import json
import os

ARTIFACTS_DIR = "artifacts"

FACTOR_TO_COMPONENT = {
    "كاميرات المراقبة": "no_camera",
    "رادار سرعة": "speeding",
    "تحسين الإضاءة": "night_unlit",
    "حالة الرصف": "poor_surface",
    "زيادة الرقابة": "enforcement",
}

COMPONENT_WEIGHTS = {
    "severity": 0.30, "frequency": 0.20, "speeding": 0.15, "night_unlit": 0.10,
    "poor_surface": 0.10, "no_camera": 0.10, "black_spot": 0.05,
}


class RiskModel:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.dir = artifacts_dir
        self.road_risk = self._load("road_risk.json")
        self.governorate_risk = self._load("governorate_risk.json")
        self.factors = self._load("hour_weather_factors.json")
        self.dashboard_stats = self._load("dashboard_stats.json")
        self.roads_by_name = {r["Road_Name"]: r for r in self.road_risk}

    def _load(self, filename):
        path = os.path.join(self.dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' مش موجود. شغّلي data_prep.py الأول عشان يبنى ملفات الـ artifacts."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict_trip(self, governorates, hour_24, weather):
        govs = [g for g in governorates if g]
        gov_scores = [self.governorate_risk[g]["risk_score"] for g in govs if g in self.governorate_risk]
        base = sum(gov_scores) / len(gov_scores) if gov_scores else 50.0

        hour_key = str(int(hour_24) % 24)
        hour_mult = self.factors["hour_factor"].get(hour_key, 1.0)
        weather_mult = self.factors["weather_factor"].get(weather, 1.0)

        score = max(0.0, min(100.0, base * hour_mult * weather_mult))
        label = "منخفض" if score < 40 else ("متوسط" if score < 70 else "مرتفع")

        return {
            "overall_score": round(score, 1),
            "risk_score": round(score, 1),
            "overall_risk": label,
            "base_governorate_score": round(base, 1),
            "hour_multiplier": hour_mult,
            "weather_multiplier": weather_mult,
        }

    def what_if(self, road_name, factor_label, improvement_pct):
        road = self.roads_by_name.get(road_name)
        if road is None:
            raise KeyError(f"الطريق '{road_name}' مش موجود فى بيانات النموذج")

        pct = max(0.0, min(100.0, float(improvement_pct))) / 100.0
        comps = dict(road["components"])

        component_key = FACTOR_TO_COMPONENT.get(factor_label)
        if component_key == "enforcement":
            for k in ("speeding", "no_camera"):
                comps[k] = comps[k] * (1 - pct * 0.5)
        elif component_key in comps:
            comps[component_key] = comps[component_key] * (1 - pct)

        new_score = max(0.0, min(100.0, sum(comps[k] * COMPONENT_WEIGHTS[k] for k in COMPONENT_WEIGHTS)))

        return {
            "road_name": road_name,
            "factor": factor_label,
            "improvement_pct": improvement_pct,
            "original_risk_score": road["risk_score"],
            "estimated_new_risk_score": round(new_score, 1),
        }

    def get_dashboard_stats(self):
        return self.dashboard_stats

    def get_roads(self, governorate=None):
        if not governorate:
            return self.road_risk
        filtered = [r for r in self.road_risk if governorate in r["governorates"]]
        return filtered or self.road_risk[:5]