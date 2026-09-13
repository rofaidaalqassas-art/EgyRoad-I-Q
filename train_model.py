# -*- coding: utf-8 -*-
"""
train_model.py
===============
بيبني الموديلات الثلاثة النهائية بنفس منطق نوتبوك
"Final Selected ML Models — Egypt Road Accident Project" بالظبط
(نفس الـ features، نفس الأعمدة المستبعدة، نفس الـ preprocessing، نفس
الـ hyperparameters المُقفّلة، نفس الـ threshold).

الموديلات:
  1. CatBoost Classifier   -> Has_Injury (احتمالية الإصابة، threshold = 0.55)
  2. XGBoost Regressor     -> Injuries_Count
  3. CatBoost Regressor    -> Fatalities_Count

مهم جدًا: كل موديل بيتحفظ كـ Pipeline كامل (فيه الـ preprocessing جواه:
imputation + scaling + one-hot) - مش الـ estimator لوحده. أي كود هيستخدمهم
لازم يبعتلهم الـ raw features زي ما هي من غير أي preprocessing يدوي زيادة.

كمان فيه فرق مهم بين المجموعتين:
  - الـ classifier بياخد CLASSIFICATION_FEATURES بس (مفيهاش Impact_Speed_KMH
    ولا Collision_Type).
  - الـ regressors (injuries/fatalities) بياخدوا IMPACT_FEATURES اللي فيها
    Impact_Speed_KMH, Collision_Type, وفيتشرز مشتقة زي Speed_Over_Limit_KMH
    و Speed_Ratio.

التشغيل:
    python train_model.py

المخرجات (فى artifacts/):
    - final_injury_classifier_catboost.pkl   Pipeline التصنيف الكامل
    - final_injuries_xgboost.pkl             Pipeline انحدار الإصابات الكامل
    - final_fatalities_catboost.pkl          Pipeline انحدار الوفيات الكامل
    - final_model_config.pkl                 params + threshold + feature lists
    - feature_importance.json    ترتيب العوامل حسب الأهمية (من الـ classifier)
    - outcome_lookup.json        متوسطات القتلى/الإصابات/الخسارة لكل (محافظة، فئة خطورة)
    - global_outcome_avg.json    متوسط عام (fallback)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

EXCEL_PATH = os.environ.get("ACCIDENTS_XLSX", "accidents_data.xlsx")
OUT_DIR = "artifacts"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# فئات الخطورة (مستخدمة برضو فى ai_model.py وoutcome_lookup - مش لها علاقة
# بالـ CLASSIFICATION_THRESHOLD اللي بيحدد Has_Injury نفسه)
# ---------------------------------------------------------------------------
RISK_BANDS = [(0.30, "منخفض"), (0.60, "متوسط"), (0.85, "مرتفع"), (1.01, "حرج")]


def risk_label(prob: float) -> str:
    for threshold, label in RISK_BANDS:
        if prob < threshold:
            return label
    return "حرج"


# ---------------------------------------------------------------------------
# Classification features (identical to the notebook)
# ---------------------------------------------------------------------------
CLASSIFICATION_FORBIDDEN = [
    "Accident_ID", "Report_Number", "Location_ID",
    "Primary_Vehicle_ID", "Primary_Driver_ID",
    "Vehicle_ID", "Driver_ID",
    "Governorate_AR", "Governorate_AR_Location",
    "Date_ID",
    "Injuries_Count", "Fatalities_Count", "Severity_Level",
    "Property_Damage_Cost_EGP", "Medical_Care_Cost_EGP",
    "Productivity_Loss_EGP", "Total_Economic_Loss_EGP",
    "Emergency_Response_Time_Min", "Fine_Amount_EGP",
    "Primary_Cause_ID", "Cause_Category", "Cause_Detail",
    "Collision_Type", "Traffic_Law_Applied",
    "Speed_Camera_Evidence", "City_District_EN", "Black_Spot_Name",
    "Manufacturing_Year", "Age_Group",
    "Impact_Speed_KMH", "Drug_Test_Result",
]

TARGET_CLASS = "Has_Injury"

NUMERIC_FEATURES = [
    "Hour_24", "Year", "Month", "Day_of_Week",
    "KM_Marker", "Latitude", "Longitude",
    "AADT_Volume", "Posted_Speed_Limit_KMH",
    "Vehicle_Age_Years", "Age",
    "Driving_Experience_Years", "Monthly_Income_EGP",
    "Prior_Violations_Count", "Fatigue_Hours_Awake",
]

CATEGORICAL_FEATURES = [
    "Road_Type", "Highway_Name", "Governorate_EN",
    "Is_Urban_Road", "Is_Black_Spot",
    "Weather_Condition", "Lighting_Condition",
    "Road_Surface_Condition", "Vehicle_Category",
    "Make", "Model", "Licensing_Status",
    "Technical_Inspection_Passed", "Overload_Flag",
    "ABS_Equipped", "Driver_Airbag_Equipped",
    "Tire_Condition", "Brake_Condition",
    "Insurance_Coverage", "Fuel_Type", "Gender",
    "License_Category", "License_Status",
    "Occupation", "Distracted_Driving_Mobile",
    "Seatbelt_Used", "Helmet_Used",
]

CLASSIFICATION_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASSIFIER_PARAMS = {
    "depth": 7,
    "learning_rate": 0.08,
    "l2_leaf_reg": 5,
}
CLASSIFICATION_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Impact (regression) features (identical to the notebook)
# ---------------------------------------------------------------------------
IMPACT_NUMERIC_FEATURES = [
    "Hour_24", "Year", "Month", "Day_of_Week",
    "KM_Marker", "Latitude", "Longitude",
    "AADT_Volume", "Posted_Speed_Limit_KMH",
    "Impact_Speed_KMH", "Speed_Over_Limit_KMH",
    "Speed_Ratio", "Vehicles_Involved",
    "Vehicle_Age_Years", "Age",
    "Driving_Experience_Years", "Monthly_Income_EGP",
    "Prior_Violations_Count", "Fatigue_Hours_Awake",
]

IMPACT_CATEGORICAL_FEATURES = [
    "Road_Type", "Highway_Name", "Governorate_EN",
    "Is_Urban_Road", "Is_Black_Spot",
    "Weather_Condition", "Lighting_Condition",
    "Road_Surface_Condition", "Vehicle_Category",
    "Make", "Model", "Licensing_Status",
    "Technical_Inspection_Passed", "Overload_Flag",
    "ABS_Equipped", "Driver_Airbag_Equipped",
    "Tire_Condition", "Brake_Condition",
    "Insurance_Coverage", "Fuel_Type", "Gender",
    "License_Category", "License_Status",
    "Occupation", "Distracted_Driving_Mobile",
    "Seatbelt_Used", "Helmet_Used",
    "Collision_Type",
]

IMPACT_FEATURES = IMPACT_NUMERIC_FEATURES + IMPACT_CATEGORICAL_FEATURES
IMPACT_TARGETS = ["Injuries_Count", "Fatalities_Count"]

INJURIES_XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.10,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
}

FATALITIES_CATBOOST_PARAMS = {
    "iterations": 600,
    "depth": 7,
    "learning_rate": 0.08,
    "l2_leaf_reg": 3,
}


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

    master_df = accidents.merge(locations, on="Location_ID", how="left", suffixes=("", "_Location"))
    master_df = master_df.merge(
        vehicles, left_on="Primary_Vehicle_ID", right_on="Vehicle_ID",
        how="left", suffixes=("", "_Vehicle"),
    )
    master_df = master_df.merge(
        drivers, left_on="Primary_Driver_ID", right_on="Driver_ID",
        how="left", suffixes=("", "_Driver"),
    )
    return master_df


def build_classification_pipeline(estimator):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            CATEGORICAL_FEATURES,
        ),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def build_regression_pipeline(estimator):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            IMPACT_NUMERIC_FEATURES,
        ),
        (
            "categorical",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            IMPACT_CATEGORICAL_FEATURES,
        ),
    ])
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


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
    np.random.seed(RANDOM_STATE)

    print("بتقرا وتدمج البيانات...")
    master_df = load_and_merge()

    # ---------------- Classification ----------------
    classification_df = master_df.drop(columns=CLASSIFICATION_FORBIDDEN, errors="ignore").copy()
    X_class = classification_df[CLASSIFICATION_FEATURES].copy()
    y_class = classification_df[TARGET_CLASS].copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_class, y_class, test_size=0.20, random_state=RANDOM_STATE, stratify=y_class,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full,
    )
    print(f"Classification -> Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("بيدرب CatBoost Classifier...")
    final_classifier = build_classification_pipeline(
        CatBoostClassifier(**CLASSIFIER_PARAMS, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)
    )
    X_class_trainval = pd.concat([X_train, X_val], axis=0)
    y_class_trainval = pd.concat([y_train, y_val], axis=0)
    final_classifier.fit(X_class_trainval, y_class_trainval)

    test_prob = final_classifier.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= CLASSIFICATION_THRESHOLD).astype(int)
    print("\n=== Classification Test ===")
    print(classification_report(y_test, test_pred, target_names=["No Injury", "Injury"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))

    # ---------------- Regression (Injuries / Fatalities) ----------------
    impact_df = master_df.copy()
    impact_df["Speed_Over_Limit_KMH"] = (
        impact_df["Impact_Speed_KMH"] - impact_df["Posted_Speed_Limit_KMH"]
    ).clip(lower=0)
    impact_df["Speed_Ratio"] = np.where(
        impact_df["Posted_Speed_Limit_KMH"] > 0,
        impact_df["Impact_Speed_KMH"] / impact_df["Posted_Speed_Limit_KMH"],
        np.nan,
    )

    X_impact = impact_df[IMPACT_FEATURES].copy()
    Y_impact = impact_df[IMPACT_TARGETS].copy()

    # نفس تقسيم الـ classification بالظبط (نفس الـ index)
    X2_train = X_impact.loc[X_train.index].copy()
    X2_val = X_impact.loc[X_val.index].copy()
    X2_test = X_impact.loc[X_test.index].copy()
    Y2_train = Y_impact.loc[X_train.index].copy()
    Y2_val = Y_impact.loc[X_val.index].copy()
    Y2_test = Y_impact.loc[X_test.index].copy()

    X2_trainval = pd.concat([X2_train, X2_val], axis=0)
    Y2_trainval = pd.concat([Y2_train, Y2_val], axis=0)

    print("\nبيدرب XGBoost Regressor (Injuries_Count)...")
    final_injuries_model = build_regression_pipeline(
        XGBRegressor(**INJURIES_XGB_PARAMS, random_state=RANDOM_STATE, objective="reg:squarederror", n_jobs=-1)
    )
    final_injuries_model.fit(X2_trainval, Y2_trainval["Injuries_Count"])

    print("بيدرب CatBoost Regressor (Fatalities_Count)...")
    final_fatalities_model = build_regression_pipeline(
        CatBoostRegressor(
            **FATALITIES_CATBOOST_PARAMS, random_state=RANDOM_STATE,
            verbose=False, allow_writing_files=False, loss_function="RMSE",
        )
    )
    final_fatalities_model.fit(X2_trainval, Y2_trainval["Fatalities_Count"])

    # ---------------- feature importance (from the classifier) ----------------
    cat_feature_names = (
        final_classifier.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    feature_names = list(NUMERIC_FEATURES) + list(cat_feature_names)
    importances = final_classifier.named_steps["classifier"].get_feature_importance()
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ---------------- outcome lookup (built on the full dataset) ----------------
    outcome_cols = ["Fatalities_Count", "Injuries_Count", "Total_Economic_Loss_EGP", "Governorate_EN"]
    df_for_lookup = master_df[outcome_cols].copy()
    full_prob = final_classifier.predict_proba(X_class)[:, 1]
    lookup, global_avg = build_outcome_lookup(df_for_lookup, full_prob)

    # ---------------- save everything ----------------
    joblib.dump(final_classifier, f"{OUT_DIR}/final_injury_classifier_catboost.pkl")
    joblib.dump(final_injuries_model, f"{OUT_DIR}/final_injuries_xgboost.pkl")
    joblib.dump(final_fatalities_model, f"{OUT_DIR}/final_fatalities_catboost.pkl")

    model_config = {
        "random_state": RANDOM_STATE,
        "classification": {
            "model": "CatBoostClassifier",
            "params": CLASSIFIER_PARAMS,
            "threshold": CLASSIFICATION_THRESHOLD,
            "features": CLASSIFICATION_FEATURES,
        },
        "injuries": {
            "model": "XGBRegressor",
            "params": INJURIES_XGB_PARAMS,
            "features": IMPACT_FEATURES,
        },
        "fatalities": {
            "model": "CatBoostRegressor",
            "params": FATALITIES_CATBOOST_PARAMS,
            "features": IMPACT_FEATURES,
        },
    }
    joblib.dump(model_config, f"{OUT_DIR}/final_model_config.pkl")

    with open(f"{OUT_DIR}/feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(fi.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/outcome_lookup.json", "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/global_outcome_avg.json", "w", encoding="utf-8") as f:
        json.dump(global_avg, f, ensure_ascii=False, indent=2)

    print(f"\nتم الحفظ فى {OUT_DIR}/ :")
    print("  - final_injury_classifier_catboost.pkl")
    print("  - final_injuries_xgboost.pkl")
    print("  - final_fatalities_catboost.pkl")
    print("  - final_model_config.pkl")
    print("  - feature_importance.json")
    print("  - outcome_lookup.json")
    print("  - global_outcome_avg.json")


if __name__ == "__main__":
    main()
