# -*- coding: utf-8 -*-
"""
train_model.py
===============
ده السكريبت اللي بيبني "الذكاء الاصطناعي" الحقيقي (مش المعادلة اليدوية).

بياخد نفس منطق الـ notebook بتاعك بالظبط (نفس الـ feature engineering، نفس
الأعمدة المستبعدة، نفس الـ preprocessing) ويحوله لسكريبت جاهز للإنتاج:

1. يقرا الإكسيل (4 شيتات: Accidents / Vehicles / Drivers / Locations) ويعمل merge.
2. يجهّز نفس الـ features بالظبط اللي كانت فى الـ notebook.
3. يدرب RandomForestClassifier (نسخة متزنة: max_depth=15, min_samples_split=10,
   min_samples_leaf=5 - زي "Model 3" فى النوتبوك، أقل عرضة للـ overfitting من
   النسخة الافتراضية). لو حابة تجربي نسخة تانية، غيّري RF_PARAMS بس.
4. يحسب أهمية العوامل (feature importance).
5. يبني "جداول مرجعية" (lookup tables) لتقدير النتائج البشرية/الاقتصادية
   المتوقعة (متوسط القتلى/الإصابات/الخسارة الاقتصادية) لكل (محافظة + فئة خطورة)
   من البيانات التاريخية - عشان نقدر نجاوب "توقع النتائج البشرية والاقتصادية"
   بدون ما نحتاج موديل انحدار منفصل.
6. يحفظ كل حاجة فى مجلد artifacts/ عشان main.py يستخدمها فورًا.

التشغيل:
    python train_model.py

المخرجات (فى artifacts/):
    - injury_model.joblib        الـ Pipeline المدرّب كامل (preprocessing + classifier)
    - feature_importance.json    ترتيب العوامل حسب الأهمية
    - outcome_lookup.json        متوسطات القتلى/الإصابات/الخسارة الاقتصادية لكل (محافظة، فئة خطورة)
    - global_outcome_avg.json    متوسط عام (fallback لو مفيش بيانات كفاية لمحافظة معينة)
    - model_features.json        قائمة الأعمدة المطلوبة كمدخلات للتنبؤ (عشان main.py/الواجهة تعرف تبعت ايه)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCEL_PATH = os.environ.get("ACCIDENTS_XLSX", "accidents_data.xlsx")
OUT_DIR = "artifacts"

# نفس إعدادات الموديل "المتزن" (Model 3 فى النوتبوك) - أقل overfitting من الافتراضي
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",  # البيانات غالبًا فيها إصابات أقل من عدم إصابات
)

NUMERIC_FEATURES = [
    "Hour_24", "Year", "Month", "Day_of_Week",
    "KM_Marker", "Latitude", "Longitude", "AADT_Volume",
    "Posted_Speed_Limit_KMH", "Vehicle_Age_Years", "Age",
    "Driving_Experience_Years", "Monthly_Income_EGP",
    "Prior_Violations_Count", "Fatigue_Hours_Awake",
]

CATEGORICAL_FEATURES = [
    "Road_Type", "Highway_Name", "Governorate_EN",
    "Is_Urban_Road", "Is_Black_Spot",
    "Weather_Condition", "Lighting_Condition", "Road_Surface_Condition",
    "Vehicle_Category", "Make", "Model", "Licensing_Status",
    "Technical_Inspection_Passed", "Overload_Flag", "ABS_Equipped",
    "Driver_Airbag_Equipped", "Tire_Condition", "Brake_Condition",
    "Insurance_Coverage", "Fuel_Type",
    "Gender", "License_Category", "License_Status", "Occupation",
    "Distracted_Driving_Mobile", "Seatbelt_Used", "Helmet_Used",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Has_Injury"

REMOVE_COLUMNS = [
    "Accident_ID", "Report_Number", "Location_ID", "Primary_Vehicle_ID",
    "Primary_Driver_ID", "Vehicle_ID", "Driver_ID",
    "Governorate_AR", "Governorate_AR_Location", "Date_ID",
    "Injuries_Count", "Fatalities_Count", "Severity_Level",
    "Property_Damage_Cost_EGP", "Medical_Care_Cost_EGP",
    "Productivity_Loss_EGP", "Total_Economic_Loss_EGP",
    "Emergency_Response_Time_Min", "Fine_Amount_EGP",
    "Primary_Cause_ID", "Cause_Category", "Cause_Detail", "Collision_Type",
    "Traffic_Law_Applied", "Speed_Camera_Evidence",
    "City_District_EN", "Black_Spot_Name", "Manufacturing_Year",
    "Age_Group", "Impact_Speed_KMH", "Drug_Test_Result",
]

# فئات الخطورة المستخدمة فى كل حاجة (predict + lookup + الواجهة)
RISK_BANDS = [(0.30, "منخفض"), (0.60, "متوسط"), (0.85, "مرتفع"), (1.01, "حرج")]


def risk_label(prob: float) -> str:
    for threshold, label in RISK_BANDS:
        if prob < threshold:
            return label
    return "حرج"


def load_and_merge():
    accidents = pd.read_excel(EXCEL_PATH, sheet_name="Accidents")
    vehicles = pd.read_excel(EXCEL_PATH, sheet_name="Vehicles")
    drivers = pd.read_excel(EXCEL_PATH, sheet_name="Drivers")
    locations = pd.read_excel(EXCEL_PATH, sheet_name="Locations")

    accidents["Has_Injury"] = (accidents["Injuries_Count"] > 0).astype(int)
    accidents["Date_ID"] = pd.to_datetime(accidents["Date_ID"], errors="coerce")
    accidents["Year"] = accidents["Date_ID"].dt.year
    accidents["Month"] = accidents["Date_ID"].dt.month
    accidents["Day_of_Week"] = accidents["Date_ID"].dt.dayofweek

    df = accidents.merge(locations, on="Location_ID", how="left", suffixes=("", "_Location"))
    df = df.merge(vehicles, left_on="Primary_Vehicle_ID", right_on="Vehicle_ID", how="left", suffixes=("", "_Vehicle"))
    df = df.merge(drivers, left_on="Primary_Driver_ID", right_on="Driver_ID", how="left", suffixes=("", "_Driver"))

    outcome_cols = ["Fatalities_Count", "Injuries_Count", "Total_Economic_Loss_EGP", "Governorate_EN"]
    df_for_lookup = df[outcome_cols].copy()

    df_model = df.drop(columns=REMOVE_COLUMNS, errors="ignore")
    return df, df_model, df_for_lookup


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**RF_PARAMS)),
    ])


def build_outcome_lookup(df_for_lookup: pd.DataFrame, risk_probs: np.ndarray):
    tmp = df_for_lookup.copy()
    tmp["risk_band"] = [risk_label(p) for p in risk_probs]

    grouped = tmp.groupby(["Governorate_EN", "risk_band"]).agg(
        avg_fatalities=("Fatalities_Count", "mean"),
        avg_injuries=("Injuries_Count", "mean"),
        avg_economic_loss_egp=("Total_Economic_Loss_EGP", "mean"),
        n=("Fatalities_Count", "count"),
    ).round(2)

    lookup = {}
    for (gov, band), row in grouped.iterrows():
        lookup.setdefault(gov, {})[band] = {
            "avg_fatalities": float(row["avg_fatalities"]),
            "avg_injuries": float(row["avg_injuries"]),
            "avg_economic_loss_egp": float(row["avg_economic_loss_egp"]),
            "sample_size": int(row["n"]),
        }

    global_avg = {
        "avg_fatalities": float(tmp["Fatalities_Count"].mean()),
        "avg_injuries": float(tmp["Injuries_Count"].mean()),
        "avg_economic_loss_egp": float(tmp["Total_Economic_Loss_EGP"].mean()),
    }
    return lookup, global_avg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("بتقرا وتدمج البيانات...")
    df_raw, df_model, df_for_lookup = load_and_merge()

    X = df_model[ALL_FEATURES].copy()
    y = df_model[TARGET].copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("بيدرب RandomForest...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_pred = pipeline.predict(X_val)
    print("\n=== نتائج التحقق (Validation) ===")
    print(classification_report(y_val, val_pred, target_names=["No Injury", "Injury"]))
    print("ROC-AUC:", round(roc_auc_score(y_val, val_prob), 4))

    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_pred = pipeline.predict(X_test)
    print("\n=== نتائج الاختبار النهائي (Test) ===")
    print(classification_report(y_test, test_pred, target_names=["No Injury", "Injury"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    full_prob = pipeline.predict_proba(X)[:, 1]
    lookup, global_avg = build_outcome_lookup(df_for_lookup, full_prob)

    joblib.dump(pipeline, f"{OUT_DIR}/injury_model.joblib")
    with open(f"{OUT_DIR}/feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(fi.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/outcome_lookup.json", "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/global_outcome_avg.json", "w", encoding="utf-8") as f:
        json.dump(global_avg, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/model_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nتم الحفظ فى {OUT_DIR}/ :")
    print("  - injury_model.joblib")
    print("  - feature_importance.json")
    print("  - outcome_lookup.json")
    print("  - global_outcome_avg.json")
    print("  - model_features.json")


if __name__ == "__main__":
    main()