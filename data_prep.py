# -*- coding: utf-8 -*-
"""
data_prep.py
============
بيبني الإحصائيات المجمّعة (على مستوى الطريق/المحافظة) المستخدمة فى لوحة
متخذ القرار (/dashboard-stats, /roads). مش له علاقة بموديل الذكاء الاصطناعي
(ده فى train_model.py) - الاتنين شغالين مع بعض من غير تعارض.

شغّليه مرة واحدة (أو كل ما يتغيّر ملف البيانات):
    python data_prep.py
"""

import json
import os

import pandas as pd

EXCEL_PATH = os.environ.get("ACCIDENTS_XLSX", "accidents_data.xlsx")
OUT_DIR = "artifacts"

WEATHER_MAP = {"clear": "Clear", "rain": "Light Rain", "fog": "Fog", "dust": "Dusty"}

SEVERITY_WEIGHT = {
    "Fatal": 5.0, "Serious Injury": 3.0, "Minor Injury": 1.0, "Property Damage Only": 0.5,
}

WEIGHTS = {
    "severity": 0.30, "frequency": 0.20, "speeding": 0.15, "night_unlit": 0.10,
    "poor_surface": 0.10, "no_camera": 0.10, "black_spot": 0.05,
}


def minmax(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return series * 0 + 50.0
    return (series - lo) / (hi - lo) * 100.0


def load_data():
    xls = pd.ExcelFile(EXCEL_PATH)
    fact = pd.read_excel(xls, "Accidents")
    loc = pd.read_excel(xls, "Locations")
    return fact, loc


def build_component_table(df, loc):
    df = df.copy()
    df["sev_w"] = df["Severity_Level"].map(SEVERITY_WEIGHT).fillna(1.0)
    df["is_night_unlit"] = (df["Lighting_Condition"].astype(str).str.contains("Unlit", na=False)).astype(int)
    df["is_poor_surface"] = df["Road_Surface_Condition"].isin(["Poor", "Debris", "Rough"]).astype(int)
    df["is_no_camera"] = (df["Speed_Camera_Evidence"] == "NO").astype(int)

    black_spot_ids = set(loc.loc[loc["Is_Black_Spot"] == "YES", "Location_ID"]) if "Is_Black_Spot" in loc else set()
    df["is_black_spot"] = df["Location_ID"].isin(black_spot_ids).astype(int)

    aadt_by_loc = loc.set_index("Location_ID")["AADT_Volume"].to_dict()
    df["aadt"] = df["Location_ID"].map(aadt_by_loc)
    return df


def aggregate(df, group_col):
    g = df.groupby(group_col)
    out = pd.DataFrame({
        "accidents": g.size(),
        "fatalities": g["Fatalities_Count"].sum(),
        "injuries": g["Injuries_Count"].sum(),
        "avg_severity": g["sev_w"].mean(),
        "pct_night_unlit": g["is_night_unlit"].mean(),
        "pct_poor_surface": g["is_poor_surface"].mean(),
        "pct_no_camera": g["is_no_camera"].mean(),
        "pct_black_spot": g["is_black_spot"].mean(),
        "avg_aadt": g["aadt"].mean(),
    }).fillna(0)
    out["accidents_per_aadt"] = out["accidents"] / out["avg_aadt"].replace(0, pd.NA)
    out["accidents_per_aadt"] = out["accidents_per_aadt"].fillna(out["accidents_per_aadt"].median())
    return out


def score_table(agg):
    agg = agg.copy()
    agg["c_severity"] = minmax(agg["avg_severity"])
    agg["c_frequency"] = minmax(agg["accidents_per_aadt"])
    agg["c_night_unlit"] = minmax(agg["pct_night_unlit"])
    agg["c_poor_surface"] = minmax(agg["pct_poor_surface"])
    agg["c_no_camera"] = minmax(agg["pct_no_camera"])
    agg["c_black_spot"] = minmax(agg["pct_black_spot"])
    agg["c_speeding"] = 50.0  # مفيش Impact_Speed فى النسخة دي - قيمة محايدة

    agg["risk_score"] = (
        agg["c_severity"] * WEIGHTS["severity"]
        + agg["c_frequency"] * WEIGHTS["frequency"]
        + agg["c_speeding"] * WEIGHTS["speeding"]
        + agg["c_night_unlit"] * WEIGHTS["night_unlit"]
        + agg["c_poor_surface"] * WEIGHTS["poor_surface"]
        + agg["c_no_camera"] * WEIGHTS["no_camera"]
        + agg["c_black_spot"] * WEIGHTS["black_spot"]
    ).round(1)
    return agg


def build_hour_weather_factors(df):
    overall_mean = df["sev_w"].mean()
    by_hour = df.groupby("Hour_24")["sev_w"].mean()
    hour_factor = {int(k): v for k, v in (by_hour / overall_mean).round(3).to_dict().items()}

    by_weather = df.groupby("Weather_Condition")["sev_w"].mean()
    weather_factor_raw = (by_weather / overall_mean).round(3).to_dict()
    weather_factor = {ui: weather_factor_raw.get(data_key, 1.0) for ui, data_key in WEATHER_MAP.items()}
    return {"hour_factor": hour_factor, "weather_factor": weather_factor}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fact, loc = load_data()
    df = build_component_table(fact, loc)

    road_agg = aggregate(df, "Highway_Name")
    road_scored = score_table(road_agg)
    road_govs = df.groupby("Highway_Name")["Governorate_EN"].unique().apply(list)

    roads_out = []
    for name, row in road_scored.iterrows():
        roads_out.append({
            "Road_Name": name,
            "risk_score": float(row["risk_score"]),
            "accidents": int(row["accidents"]),
            "fatalities": int(row["fatalities"]),
            "injuries": int(row["injuries"]),
            "governorates": road_govs.get(name, []),
            "components": {
                "severity": round(float(row["c_severity"]), 1),
                "frequency": round(float(row["c_frequency"]), 1),
                "speeding": round(float(row["c_speeding"]), 1),
                "night_unlit": round(float(row["c_night_unlit"]), 1),
                "poor_surface": round(float(row["c_poor_surface"]), 1),
                "no_camera": round(float(row["c_no_camera"]), 1),
                "black_spot": round(float(row["c_black_spot"]), 1),
            },
        })
    roads_out.sort(key=lambda r: r["risk_score"], reverse=True)

    gov_agg = aggregate(df, "Governorate_EN")
    gov_scored = score_table(gov_agg)
    gov_out = {
        name: {
            "risk_score": float(row["risk_score"]),
            "accidents": int(row["accidents"]),
            "fatalities": int(row["fatalities"]),
            "injuries": int(row["injuries"]),
        }
        for name, row in gov_scored.iterrows()
    }

    factors = build_hour_weather_factors(df)

    dashboard_stats = {
        "total_accidents": int(len(df)),
        "total_fatalities": int(df["Fatalities_Count"].sum()),
        "total_injuries": int(df["Injuries_Count"].sum()),
        "top_roads": roads_out,
    }

    with open(f"{OUT_DIR}/road_risk.json", "w", encoding="utf-8") as f:
        json.dump(roads_out, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/governorate_risk.json", "w", encoding="utf-8") as f:
        json.dump(gov_out, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/hour_weather_factors.json", "w", encoding="utf-8") as f:
        json.dump(factors, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/dashboard_stats.json", "w", encoding="utf-8") as f:
        json.dump(dashboard_stats, f, ensure_ascii=False, indent=2)

    print(f"تم بناء الملفات فى {OUT_DIR}/ بنجاح:")
    print(f"  - {len(roads_out)} طريق")
    print(f"  - {len(gov_out)} محافظة")
    print(f"  - إجمالى الحوادث: {dashboard_stats['total_accidents']:,}")


if __name__ == "__main__":
    main()