


"""
risk_model.py
=============

موديل تقييم المخاطر القديم/الإحصائي لمشروع EgyRoad IQ.

مسؤول عن:
1. تحميل ملفات الـ artifacts.
2. تحميل بيانات الحوادث من Excel.
3. حساب تقييم الرحلة بالطريقة الإحصائية القديمة.
4. حساب What-If بالطريقة اليدوية.
5. بناء Scenario حقيقي من بيانات الحوادث لطريق معين.

ملاحظة:
التنبؤ الحقيقي بالـ AI موجود في:
    train_model.py
    ai_model.py

مهم:
هذا الملف لا يستورد نفسه، ولا يستورد main.py.
"""


import json
import os
import re

import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

ARTIFACTS_DIR = "artifacts"

ACCIDENTS_FILE = "accidents_data.xlsx"


# =========================================================
# WHAT-IF FACTORS
# =========================================================

FACTOR_TO_COMPONENT = {
    "كاميرات المراقبة": "no_camera",
    "رادار سرعة": "speeding",
    "تحسين الإضاءة": "night_unlit",
    "حالة الرصف": "poor_surface",
    "زيادة الرقابة": "enforcement",

    "Speed Reduction": "speeding",
}

# =========================================================
# RISK COMPONENT WEIGHTS
# =========================================================

COMPONENT_WEIGHTS = {
    "severity": 0.30,
    "frequency": 0.20,
    "speeding": 0.15,
    "night_unlit": 0.10,
    "poor_surface": 0.10,
    "no_camera": 0.10,
    "black_spot": 0.05,
}


# =========================================================
# RISK MODEL
# =========================================================

class RiskModel:

    def __init__(
        self,
        artifacts_dir=ARTIFACTS_DIR,
        accidents_file=ACCIDENTS_FILE,
    ):

        self.dir = artifacts_dir

        self.accidents_file = accidents_file

        # -------------------------------------------------
        # Load artifacts
        # -------------------------------------------------

        self.road_risk = self._load(
            "road_risk.json"
        )

        self.governorate_risk = self._load(
            "governorate_risk.json"
        )

        self.factors = self._load(
            "hour_weather_factors.json"
        )

        self.dashboard_stats = self._load(
            "dashboard_stats.json"
        )

        # -------------------------------------------------
        # Roads dictionary
        # -------------------------------------------------

        self.roads_by_name = {}

        for road in self.road_risk:

            road_name = road.get(
                "Road_Name",
                ""
            )

            if road_name:

                key = self._normalize_text(
                    road_name
                )

                self.roads_by_name[key] = road

        # -------------------------------------------------
        # Load real accident data
        # -------------------------------------------------

        self.accidents_data = (
            self._load_accidents_data()
        )


    # =====================================================
    # NORMALIZE TEXT
    # =====================================================

    @staticmethod
    def _normalize_text(value):

        if value is None:

            return ""

        text = str(value).strip().lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text


    # =====================================================
    # LOAD JSON FILE
    # =====================================================

    def _load(
        self,
        filename
    ):

        path = os.path.join(
            self.dir,
            filename
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"ملف artifacts غير موجود: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)


    # =====================================================
    # LOAD ACCIDENTS EXCEL
    # =====================================================

    def _load_accidents_data(self):

        if not os.path.exists(
            self.accidents_file
        ):

            print(
                f"[WARNING] ملف الحوادث غير موجود: "
                f"{self.accidents_file}"
            )

            return None

        try:

            # -------------------------------------------------
            # قراءة Excel
            # -------------------------------------------------

            excel_file = pd.ExcelFile(
                self.accidents_file
            )

            sheet_names = excel_file.sheet_names

            print(
                "[INFO] Excel sheets:",
                sheet_names
            )

            # -------------------------------------------------
            # Sheets مفضلة
            # -------------------------------------------------

            preferred_sheets = [
                "Accidents",
                "accidents",
                "Fact_Accidents",
                "FACT_ACCIDENTS",
                "Accident",
                "Data",
            ]

            selected_sheet = None

            normalized_sheets = {}

            for sheet in sheet_names:

                normalized_sheets[
                    self._normalize_text(sheet)
                ] = sheet

            # -------------------------------------------------
            # البحث عن Sheet معروف
            # -------------------------------------------------

            for preferred in preferred_sheets:

                key = self._normalize_text(
                    preferred
                )

                if key in normalized_sheets:

                    selected_sheet = (
                        normalized_sheets[key]
                    )

                    break

            # -------------------------------------------------
            # لو مش موجود، ابحث عن Sheet
            # يحتوي على عمود طريق
            # -------------------------------------------------

            if selected_sheet is None:

                road_candidates = {
                    "Highway_Name",
                    "Road_Name",
                    "Road",
                    "RoadName",
                    "Highway",
                }

                for sheet in sheet_names:

                    try:

                        temp_df = pd.read_excel(
                            excel_file,
                            sheet_name=sheet,
                            nrows=5,
                        )

                        temp_df.columns = [
                            str(column).strip()
                            for column in temp_df.columns
                        ]

                        columns = set(
                            temp_df.columns
                        )

                        if columns.intersection(
                            road_candidates
                        ):

                            selected_sheet = sheet

                            break

                    except Exception:

                        continue

            # -------------------------------------------------
            # لو لسه مش موجود، استخدم أول Sheet
            # -------------------------------------------------

            if selected_sheet is None:

                selected_sheet = sheet_names[0]

            print(
                f"[INFO] Using Excel sheet: "
                f"'{selected_sheet}'"
            )

            # -------------------------------------------------
            # قراءة البيانات - أعمدة build_average_scenario بس (زائد كل
            # أسماء عمود الطريق المحتملة اللي _find_road_column بيدوّر
            # عليها) بدل الشيت كامل - نفس الأسلوب المستخدم فى باقي الملف
            # (_load_journey_df/get_excel_data) لتقليل وقت التشغيل.
            # -------------------------------------------------

            _NEEDED_COLS = {
                "Vehicles_Involved", "Impact_Speed_KMH",
                "Posted_Speed_Limit_KMH", "Emergency_Response_Time_Min",
                "AADT_Volume", "Hour_24",
                "Weather_Condition", "Lighting_Condition",
                "Road_Surface_Condition", "Collision_Type", "Road_Type",
                "Governorate_EN", "Vehicle_Category",
                "Highway_Name", "Road_Name", "Road", "RoadName",
                "Highway", "Street_Name", "Road_Name_EN",
            }

            df = pd.read_excel(
                excel_file,
                sheet_name=selected_sheet,
                usecols=lambda c: c in _NEEDED_COLS,
            )

            # -------------------------------------------------
            # تنظيف أسماء الأعمدة
            # -------------------------------------------------

            df.columns = [
                str(column).strip()
                for column in df.columns
            ]

            print(
                f"[INFO] Loaded accidents data: "
                f"{len(df)} rows from "
                f"'{self.accidents_file}' "
                f"(sheet: {selected_sheet})"
            )

            print(
                "[INFO] Accident columns:"
            )

            print(
                list(df.columns)
            )

            return df

        except Exception as error:

            print(
                "[WARNING] Failed to load "
                f"accident data: {error}"
            )

            return None


    # =====================================================
    # FIND ROAD COLUMN
    # =====================================================

    def _find_road_column(
        self,
        df
    ):

        possible_columns = [
            "Highway_Name",
            "Road_Name",
            "Road",
            "RoadName",
            "Highway",
            "Street_Name",
            "Road_Name_EN",
        ]

        # -------------------------------------------------
        # Exact match
        # -------------------------------------------------

        for column in possible_columns:

            if column in df.columns:

                return column

        # -------------------------------------------------
        # Case-insensitive match
        # -------------------------------------------------

        normalized_columns = {}

        for column in df.columns:

            normalized_columns[
                self._normalize_text(column)
            ] = column

        for candidate in possible_columns:

            key = self._normalize_text(
                candidate
            )

            if key in normalized_columns:

                return normalized_columns[key]

        # -------------------------------------------------
        # Partial match
        # -------------------------------------------------

        for column in df.columns:

            normalized = self._normalize_text(
                column
            )

            if (
                "highway" in normalized
                or "road" in normalized
                or "street" in normalized
            ):

                return column

        return None


    # =====================================================
    # GET ROAD DATA
    # =====================================================

    def _get_road_dataframe(
        self,
        road_name
    ):

        if self.accidents_data is None:

            return None

        if self.accidents_data.empty:

            return None

        df = self.accidents_data.copy()

        road_column = self._find_road_column(
            df
        )

        if road_column is None:

            print(
                "[WARNING] لم يتم العثور على "
                "عمود اسم الطريق في بيانات الحوادث."
            )

            return None

        # -------------------------------------------------
        # Normalize road names
        # -------------------------------------------------

        df["_road_clean"] = (
            df[road_column]
            .fillna("")
            .astype(str)
            .map(self._normalize_text)
        )

        target = self._normalize_text(
            road_name
        )

        if not target:

            return None

        # -------------------------------------------------
        # Exact match
        # -------------------------------------------------

        exact = df[
            df["_road_clean"] == target
        ].copy()

        if not exact.empty:

            return exact

        # -------------------------------------------------
        # Contains match
        # -------------------------------------------------

        contains = df[
            df["_road_clean"].str.contains(
                target,
                na=False,
                regex=False,
            )
        ].copy()

        if not contains.empty:

            return contains

        # -------------------------------------------------
        # Reverse contains
        # -------------------------------------------------

        reverse = df[
            df["_road_clean"].apply(
                lambda value:
                    bool(value)
                    and (
                        target in value
                        or value in target
                    )
            )
        ].copy()

        if not reverse.empty:

            return reverse

        return None


    # =====================================================
    # BUILD AVERAGE SCENARIO
    # =====================================================

    def build_average_scenario(
        self,
        road_name
    ):

        """
        يبني Scenario حقيقي من بيانات الحوادث
        الخاصة بالطريق.

        Numeric:
            mean

        Categorical:
            mode
        """

        road_df = self._get_road_dataframe(
            road_name
        )

        if road_df is None:

            print(
                f"[WARNING] لا توجد بيانات "
                f"للطريق: {road_name}"
            )

            return None

        if road_df.empty:

            return None

        # -------------------------------------------------
        # Base scenario
        # -------------------------------------------------

        scenario = {
            "road_name": road_name,
            "source": "real_accident_data",
            "accident_count": int(
                len(road_df)
            ),
        }

        # -------------------------------------------------
        # Numeric columns
        # -------------------------------------------------

        numeric_columns = [
            "Vehicles_Involved",
            "Impact_Speed_KMH",
            "Posted_Speed_Limit_KMH",
            "Emergency_Response_Time_Min",
            "AADT_Volume",
            "Hour_24",
        ]

        for column in numeric_columns:

            if column not in road_df.columns:

                continue

            values = pd.to_numeric(
                road_df[column],
                errors="coerce",
            ).dropna()

            if values.empty:

                continue

            scenario[column] = round(
                float(values.mean()),
                2,
            )

        # -------------------------------------------------
        # Categorical columns
        # -------------------------------------------------

        categorical_columns = [
            "Weather_Condition",
            "Lighting_Condition",
            "Road_Surface_Condition",
            "Collision_Type",
            "Road_Type",
            "Governorate_EN",
            "Highway_Name",
            "Vehicle_Category",
        ]

        for column in categorical_columns:

            if column not in road_df.columns:

                continue

            values = (
                road_df[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            values = values[
                values != ""
            ]

            if values.empty:

                continue

            mode_values = values.mode()

            if not mode_values.empty:

                scenario[column] = str(
                    mode_values.iloc[0]
                )

        # -------------------------------------------------
        # Safe defaults
        # -------------------------------------------------

        defaults = {

            "Weather_Condition": "Clear",

            "Lighting_Condition": "Daylight",

            "Road_Surface_Condition": "Dry",

            "Collision_Type": "Rear-end",

            "Road_Type": "Urban",

            "Vehicles_Involved": 2,

            "Impact_Speed_KMH": 60,

            "Posted_Speed_Limit_KMH": 80,

            "Emergency_Response_Time_Min": 15,

            "Hour_24": 12,
        }

        for key, value in defaults.items():

            if key not in scenario:

                scenario[key] = value

        print(
            f"[INFO] Built average scenario "
            f"for '{road_name}':"
        )

        print(
            scenario
        )

        return scenario


    # =====================================================
    # PREDICT TRIP
    # =====================================================

    def predict_trip(
        self,
        governorates,
        hour_24,
        weather,
    ):

        if not isinstance(
            governorates,
            list
        ):

            governorates = [
                governorates
            ]

        gov_scores = []

        for governorate in governorates:

            if governorate in self.governorate_risk:

                score = self.governorate_risk[
                    governorate
                ].get(
                    "risk_score",
                    50.0,
                )

                try:

                    gov_scores.append(
                        float(score)
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        if gov_scores:

            base = (
                sum(gov_scores)
                / len(gov_scores)
            )

        else:

            base = 50.0

        # -------------------------------------------------
        # Hour factor
        # -------------------------------------------------

        hour_key = str(
            int(hour_24) % 24
        )

        hour_factor_data = (
            self.factors.get(
                "hour_factor",
                {}
            )
        )

        hour_mult = hour_factor_data.get(
            hour_key,
            1.0,
        )

        # -------------------------------------------------
        # Weather factor
        # -------------------------------------------------

        weather_factor_data = (
            self.factors.get(
                "weather_factor",
                {}
            )
        )

        weather_mult = (
            weather_factor_data.get(
                str(weather).strip().lower(),
                1.0,
            )
        )

        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        try:

            score = (
                float(base)
                * float(hour_mult)
                * float(weather_mult)
            )

        except (
            TypeError,
            ValueError,
        ):

            score = float(base)

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        # -------------------------------------------------
        # Risk label
        # -------------------------------------------------

        if score < 40:

            label = "منخفض"

        elif score < 70:

            label = "متوسط"

        else:

            label = "مرتفع"

        return {

            "overall_score":
                round(score, 1),

            "risk_score":
                round(score, 1),

            "overall_risk":
                label,

            "base_governorate_score":
                round(base, 1),

            "hour_multiplier":
                hour_mult,

            "weather_multiplier":
                weather_mult,
        }


    # =====================================================
    # WHAT-IF
    # =====================================================

    def what_if(
        self,
        road_name,
        factor_label,
        improvement_pct,
    ):

        road_key = self._normalize_text(
            road_name
        )

        road = self.roads_by_name.get(
            road_key
        )

        # -------------------------------------------------
        # Fallback search
        # -------------------------------------------------

        if road is None:

            for item in self.road_risk:

                item_name = item.get(
                    "Road_Name",
                    ""
                )

                if (
                    self._normalize_text(
                        item_name
                    )
                    == road_key
                ):

                    road = item

                    break

        if road is None:

            raise KeyError(
                f"الطريق '{road_name}' "
                "غير موجود في بيانات المخاطر."
            )

        # -------------------------------------------------
        # Improvement
        # -------------------------------------------------

        try:

            pct = float(
                improvement_pct
            )

        except (
            TypeError,
            ValueError,
        ):

            pct = 0.0

        pct = max(
            0.0,
            min(
                100.0,
                pct,
            ),
        ) / 100.0

        # -------------------------------------------------
        # Components
        # -------------------------------------------------

        components = dict(
            road.get(
                "components",
                {}
            )
        )

        component_key = (
            FACTOR_TO_COMPONENT.get(
                factor_label
            )
        )

        # -------------------------------------------------
        # Enforcement
        # -------------------------------------------------

        if component_key == "enforcement":

            for key in (
                "speeding",
                "no_camera",
            ):

                if key in components:

                    components[key] = (
                        float(
                            components[key]
                        )
                        * (
                            1
                            - (
                                pct
                                * 0.5
                            )
                        )
                    )

        # -------------------------------------------------
        # Normal factor
        # -------------------------------------------------

        elif (
            component_key
            and component_key in components
        ):

            components[
                component_key
            ] = (
                float(
                    components[
                        component_key
                    ]
                )
                * (
                    1 - pct
                )
            )

        # -------------------------------------------------
        # Calculate new risk
        # -------------------------------------------------

        try:
            original_score = float(
                road.get(
                    "risk_score",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            original_score = 0.0

        # Apply improvement according to the selected factor weight
        if (
            component_key
            and component_key in COMPONENT_WEIGHTS
        ):
            factor_weight = COMPONENT_WEIGHTS[
                component_key
            ]

            reduction = (
                pct
                * factor_weight
                * 100.0
            )

            new_score = (
                original_score
                - reduction
            )
        else:
            new_score = original_score

        new_score = max(
            0.0,
            min(
                100.0,
                new_score,
            ),
        )

        return {

            "road_name":
                road_name,

            "factor":
                factor_label,

            "improvement_pct":
                float(improvement_pct),

            "original_risk_score":
                round(
                    original_score,
                    1,
                ),

            "estimated_new_risk_score":
                round(
                    new_score,
                    1,
                ),
        }


    # =====================================================
    # DASHBOARD STATS
    # =====================================================

    def get_dashboard_stats(
        self
    ):

        return self.dashboard_stats


    # =====================================================
    # ROADS
    # =====================================================

    def get_roads(
        self,
        governorate=None,
    ):

        if not governorate:

            return self.road_risk

        filtered = []

        for road in self.road_risk:

            governorates = road.get(
                "governorates",
                []
            )

            if governorate in governorates:

                filtered.append(
                    road
                )

        if filtered:

            return filtered

        return self.road_risk[:5]
