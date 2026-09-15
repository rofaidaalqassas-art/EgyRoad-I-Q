# -*- coding: utf-8 -*-

"""
risk_model.py
=============

موديل تقييم المخاطر القديم/الإحصائي لمشروع EgyRoad IQ.

مسؤول عن:
1. تحميل ملفات الـ artifacts.
2. تحميل بيانات الحوادث من Excel.
3. دمج Fact_Accidents مع:
       Dim_Location
       Dim_Vehicle
       Dim_Driver
       Dim_Date
       Dim_Cause
4. حساب تقييم الرحلة بالطريقة الإحصائية القديمة.
5. حساب What-If بالطريقة اليدوية.
6. بناء Scenario حقيقي وكامل من بيانات الحوادث لطريق معين.

مهم:
- التنبؤ الحقيقي بالـ AI موجود في ai_model.py.
- هذا الملف لا يستورد main.py.
- هذا الملف لا يستورد نفسه.
- لا يتم اختراع قيم للـ ML features طالما يمكن استخراجها من Excel.
"""

import json
import os
import re

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

ARTIFACTS_DIR = "artifacts"

# الملف الأساسي الحالي
ACCIDENTS_FILE = "accidents_data1.xlsx"

# دعم الاسم القديم أيضًا
FALLBACK_ACCIDENTS_FILES = [
    "accidents_data.xlsx",
    "accidents_data1.xlsx",
]


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

        # -------------------------------------------------
        # Resolve Excel file
        # -------------------------------------------------

        self.accidents_file = self._resolve_accidents_file(
            accidents_file
        )

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
        # Load model config if available
        # -------------------------------------------------

        self.model_config = self._load_model_config()

        self.classification_features = []
        self.impact_features = []

        if self.model_config:

            try:

                self.classification_features = (
                    self.model_config
                    .get("classification", {})
                    .get("features", [])
                )

            except Exception:
                self.classification_features = []

            try:

                self.impact_features = (
                    self.model_config
                    .get("injuries", {})
                    .get("features", [])
                )

            except Exception:
                self.impact_features = []

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
        # Load and merge real accident data
        # -------------------------------------------------

        self.accidents_data = (
            self._load_accidents_data()
        )

        # -------------------------------------------------
        # Startup information
        # -------------------------------------------------

        print(
            "[INFO] RiskModel initialized."
        )

        print(
            "[INFO] Excel file:",
            self.accidents_file
        )

        if self.accidents_data is not None:

            print(
                "[INFO] Final merged accident rows:",
                len(self.accidents_data)
            )

            print(
                "[INFO] Final merged columns:",
                len(self.accidents_data.columns)
            )


    # =====================================================
    # NORMALIZE TEXT
    # =====================================================

    @staticmethod
    def _normalize_text(value):

        if value is None:
            return ""

        if isinstance(value, float) and pd.isna(value):
            return ""

        text = str(value).strip().lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text


    # =====================================================
    # RESOLVE EXCEL FILE
    # =====================================================

    def _resolve_accidents_file(
        self,
        requested_file
    ):

        # -------------------------------------------------
        # First requested file
        # -------------------------------------------------

        if requested_file and os.path.exists(
            requested_file
        ):

            return requested_file

        # -------------------------------------------------
        # Fallback files
        # -------------------------------------------------

        candidates = []

        if requested_file:
            candidates.append(
                requested_file
            )

        candidates.extend(
            FALLBACK_ACCIDENTS_FILES
        )

        for filename in candidates:

            if os.path.exists(filename):

                print(
                    "[INFO] Using Excel file:",
                    filename
                )

                return filename

        # -------------------------------------------------
        # Nothing found
        # -------------------------------------------------

        print(
            "[WARNING] لم يتم العثور على ملف Excel."
        )

        print(
            "[WARNING] الملفات التي تم البحث عنها:",
            candidates
        )

        return requested_file


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
    # LOAD MODEL CONFIG
    # =====================================================

    def _load_model_config(self):

        path = os.path.join(
            self.dir,
            "final_model_config.pkl"
        )

        if not os.path.exists(path):

            print(
                "[INFO] final_model_config.pkl غير موجود."
            )

            return None

        try:

            import joblib

            config = joblib.load(
                path
            )

            print(
                "[INFO] Loaded final_model_config.pkl"
            )

            return config

        except Exception as error:

            print(
                "[WARNING] Failed to load model config:",
                error
            )

            return None


    # =====================================================
    # READ SHEET SAFELY
    # =====================================================

    def _read_sheet(
        self,
        excel_file,
        sheet_name
    ):

        try:

            df = pd.read_excel(
                excel_file,
                sheet_name=sheet_name
            )

            df.columns = [
                str(column).strip()
                for column in df.columns
            ]

            return df

        except Exception as error:

            print(
                f"[WARNING] Failed reading sheet "
                f"'{sheet_name}': {error}"
            )

            return None


    # =====================================================
    # FIND SHEET
    # =====================================================

    def _find_sheet(
        self,
        sheet_names,
        candidates
    ):

        normalized = {}

        for sheet in sheet_names:

            normalized[
                self._normalize_text(sheet)
            ] = sheet

        for candidate in candidates:

            key = self._normalize_text(
                candidate
            )

            if key in normalized:

                return normalized[key]

        return None


    # =====================================================
    # LOAD DATE DIMENSION
    # =====================================================

    def _load_date_dimension(
        self,
        excel_file,
        sheet_name
    ):

        try:

            # -------------------------------------------------
            # اقرأ بدون header لأن Dim_Date في الملف الحالي
            # قد يكون أول صف فيها هو البيانات وليس header.
            # -------------------------------------------------

            raw = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                header=None
            )

            if raw.empty:

                return None

            # -------------------------------------------------
            # Expected structure:
            #
            # Date_Key
            # Date
            # Year
            # Month
            # Month_Name
            # Day
            # Day_of_Week
            # -------------------------------------------------

            expected_columns = [
                "Date_Key",
                "Date",
                "Year",
                "Month",
                "Month_Name",
                "Day",
                "Day_of_Week",
            ]

            # -------------------------------------------------
            # Detect whether first row is actually a header
            # -------------------------------------------------

            first_row = [
                self._normalize_text(value)
                for value in raw.iloc[0].tolist()
            ]

            header_keywords = {
                "date_key",
                "date",
                "year",
                "month",
                "month_name",
                "day",
                "day_of_week",
            }

            looks_like_header = bool(
                set(first_row)
                .intersection(
                    header_keywords
                )
            )

            if looks_like_header:

                raw = raw.iloc[1:].reset_index(
                    drop=True
                )

            # -------------------------------------------------
            # Keep expected number of columns
            # -------------------------------------------------

            if raw.shape[1] >= len(
                expected_columns
            ):

                raw = raw.iloc[
                    :,
                    :len(expected_columns)
                ]

                raw.columns = expected_columns

            else:

                print(
                    "[WARNING] Dim_Date columns أقل من المتوقع."
                )

                return None

            # -------------------------------------------------
            # Clean Date_Key
            # -------------------------------------------------

            raw["Date_Key"] = pd.to_numeric(
                raw["Date_Key"],
                errors="coerce"
            )

            raw = raw.dropna(
                subset=["Date_Key"]
            )

            raw["Date_Key"] = (
                raw["Date_Key"]
                .astype("Int64")
            )

            # -------------------------------------------------
            # Numeric date fields
            # -------------------------------------------------

            for column in [
                "Year",
                "Month",
                "Day",
                "Day_of_Week",
            ]:

                raw[column] = pd.to_numeric(
                    raw[column],
                    errors="coerce"
                )

            # -------------------------------------------------
            # Date field
            # -------------------------------------------------

            raw["Date"] = pd.to_datetime(
                raw["Date"],
                errors="coerce"
            )

            print(
                "[INFO] Dim_Date loaded:",
                len(raw),
                "rows"
            )

            return raw

        except Exception as error:

            print(
                "[WARNING] Failed loading Dim_Date:",
                error
            )

            return None


    # =====================================================
    # LOAD ACCIDENTS EXCEL
    # =====================================================

    def _load_accidents_data(self):

        if not self.accidents_file:

            return None

        if not os.path.exists(
            self.accidents_file
        ):

            print(
                "[WARNING] ملف الحوادث غير موجود:",
                self.accidents_file
            )

            return None

        try:

            excel_file = pd.ExcelFile(
                self.accidents_file
            )

            sheet_names = excel_file.sheet_names

            print(
                "[INFO] Excel sheets:",
                sheet_names
            )

            # =================================================
            # FACT ACCIDENTS
            # =================================================

            fact_sheet = self._find_sheet(
                sheet_names,
                [
                    "Fact_Accidents",
                    "FACT_ACCIDENTS",
                    "Accidents",
                    "accidents",
                    "Accident",
                    "Data",
                ]
            )

            if fact_sheet is None:

                print(
                    "[WARNING] لم يتم العثور على Fact_Accidents."
                )

                return None

            fact = self._read_sheet(
                excel_file,
                fact_sheet
            )

            if fact is None or fact.empty:

                print(
                    "[WARNING] Fact_Accidents فارغ."
                )

                return None

            print(
                "[INFO] Fact sheet:",
                fact_sheet
            )

            print(
                "[INFO] Fact rows:",
                len(fact)
            )

            # =================================================
            # LOCATION
            # =================================================

            location_sheet = self._find_sheet(
                sheet_names,
                [
                    "Dim_Location",
                    "Locations",
                    "Location",
                ]
            )

            if location_sheet:

                location = self._read_sheet(
                    excel_file,
                    location_sheet
                )

                if location is not None:

                    fact = self._merge_dimension(
                        fact,
                        location,
                        left_key="Location_ID",
                        right_key="Location_ID",
                        dimension_name="Dim_Location"
                    )

            # =================================================
            # VEHICLE
            # =================================================

            vehicle_sheet = self._find_sheet(
                sheet_names,
                [
                    "Dim_Vehicle",
                    "Vehicles",
                    "Vehicle",
                ]
            )

            if vehicle_sheet:

                vehicle = self._read_sheet(
                    excel_file,
                    vehicle_sheet
                )

                if vehicle is not None:

                    fact = self._merge_dimension(
                        fact,
                        vehicle,
                        left_key="Primary_Vehicle_ID",
                        right_key="Vehicle_ID",
                        dimension_name="Dim_Vehicle"
                    )

            # =================================================
            # DRIVER
            # =================================================

            driver_sheet = self._find_sheet(
                sheet_names,
                [
                    "Dim_Driver",
                    "Drivers",
                    "Driver",
                ]
            )

            if driver_sheet:

                driver = self._read_sheet(
                    excel_file,
                    driver_sheet
                )

                if driver is not None:

                    fact = self._merge_dimension(
                        fact,
                        driver,
                        left_key="Primary_Driver_ID",
                        right_key="Driver_ID",
                        dimension_name="Dim_Driver"
                    )

            # =================================================
            # DATE
            # =================================================

            date_sheet = self._find_sheet(
                sheet_names,
                [
                    "Dim_Date",
                    "Dates",
                    "Date",
                ]
            )

            if date_sheet:

                date = self._load_date_dimension(
                    excel_file,
                    date_sheet
                )

                if date is not None:

                    fact = self._merge_dimension(
                        fact,
                        date,
                        left_key="Date_ID",
                        right_key="Date_Key",
                        dimension_name="Dim_Date"
                    )

            # =================================================
            # CAUSE
            # =================================================

            cause_sheet = self._find_sheet(
                sheet_names,
                [
                    "Dim_Cause",
                    "Causes",
                    "Cause",
                ]
            )

            if cause_sheet:

                cause = self._read_sheet(
                    excel_file,
                    cause_sheet
                )

                if cause is not None:

                    fact = self._merge_dimension(
                        fact,
                        cause,
                        left_key="Primary_Cause_ID",
                        right_key="Cause_ID",
                        dimension_name="Dim_Cause"
                    )

            # =================================================
            # Remove duplicated columns created by merges
            # =================================================

            fact = self._clean_merged_columns(
                fact
            )

            # =================================================
            # Create engineered fields
            # =================================================

            fact = self._create_engineered_fields(
                fact
            )

            # =================================================
            # Clean column names
            # =================================================

            fact.columns = [
                str(column).strip()
                for column in fact.columns
            ]

            print(
                "[INFO] Final merged columns:"
            )

            print(
                list(fact.columns)
            )

            print(
                "[INFO] Final merged rows:",
                len(fact)
            )

            return fact

        except Exception as error:

            print(
                "[WARNING] Failed to load accident data:"
            )

            print(
                f"[WARNING] {type(error).__name__}: {error}"
            )

            return None


    # =====================================================
    # MERGE DIMENSION
    # =====================================================

    def _merge_dimension(
        self,
        fact,
        dimension,
        left_key,
        right_key,
        dimension_name
    ):

        if left_key not in fact.columns:

            print(
                f"[WARNING] {left_key} غير موجود في Fact."
            )

            return fact

        if right_key not in dimension.columns:

            print(
                f"[WARNING] {right_key} غير موجود في "
                f"{dimension_name}."
            )

            return fact

        left = fact.copy()
        right = dimension.copy()

        # -------------------------------------------------
        # Normalize keys
        # -------------------------------------------------

        left_key_values = pd.to_numeric(
            left[left_key],
            errors="coerce"
        )

        right_key_values = pd.to_numeric(
            right[right_key],
            errors="coerce"
        )

        left[left_key] = left_key_values
        right[right_key] = right_key_values

        # -------------------------------------------------
        # Remove duplicated key rows
        # -------------------------------------------------

        right = right.drop_duplicates(
            subset=[right_key],
            keep="first"
        )

        # -------------------------------------------------
        # Avoid duplicate columns
        # -------------------------------------------------

        duplicate_columns = [
            column
            for column in right.columns
            if column != right_key
            and column in left.columns
        ]

        if duplicate_columns:

            # Fact_Accidents is the authoritative source
            # for columns already existing there.
            right = right.drop(
                columns=duplicate_columns
            )

        # -------------------------------------------------
        # Merge
        # -------------------------------------------------

        merged = left.merge(
            right,
            how="left",
            left_on=left_key,
            right_on=right_key,
            sort=False
        )

        if right_key != left_key:

            if right_key in merged.columns:

                merged = merged.drop(
                    columns=[right_key]
                )

        print(
            f"[INFO] Merged {dimension_name}:",
            len(merged),
            "rows"
        )

        return merged


    # =====================================================
    # CLEAN MERGED COLUMNS
    # =====================================================

    def _clean_merged_columns(
        self,
        df
    ):

        df = df.copy()

        # -------------------------------------------------
        # Remove pandas merge suffixes if any
        # -------------------------------------------------

        rename_map = {}

        for column in df.columns:

            if column.endswith("_x"):

                base = column[:-2]

                if base not in df.columns:

                    rename_map[column] = base

            elif column.endswith("_y"):

                base = column[:-2]

                if base in df.columns:

                    continue

        if rename_map:

            df = df.rename(
                columns=rename_map
            )

        return df


    # =====================================================
    # ENGINEERED FIELDS
    # =====================================================

    def _create_engineered_fields(
        self,
        df
    ):

        df = df.copy()

        # -------------------------------------------------
        # Date fields
        # -------------------------------------------------

        if "Date" in df.columns:

            date_values = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            if "Year" not in df.columns:

                df["Year"] = (
                    date_values.dt.year
                )

            if "Month" not in df.columns:

                df["Month"] = (
                    date_values.dt.month
                )

            if "Day" not in df.columns:

                df["Day"] = (
                    date_values.dt.day
                )

            if "Day_of_Week" not in df.columns:

                # Monday=0 ... Sunday=6
                # Convert to 1...7 to match
                # common dataset representation.
                df["Day_of_Week"] = (
                    date_values.dt.dayofweek + 1
                )

        # -------------------------------------------------
        # Speed derived features
        # -------------------------------------------------

        if (
            "Impact_Speed_KMH" in df.columns
            and
            "Posted_Speed_Limit_KMH" in df.columns
        ):

            speed = pd.to_numeric(
                df["Impact_Speed_KMH"],
                errors="coerce"
            )

            limit = pd.to_numeric(
                df["Posted_Speed_Limit_KMH"],
                errors="coerce"
            )

            df["Speed_Over_Limit_KMH"] = (
                speed - limit
            ).clip(
                lower=0
            )

            df["Speed_Ratio"] = np.where(
                limit > 0,
                speed / limit,
                np.nan
            )

        # -------------------------------------------------
        # Hour
        # -------------------------------------------------

        if "Hour_24" in df.columns:

            df["Hour_24"] = pd.to_numeric(
                df["Hour_24"],
                errors="coerce"
            )

        return df


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
    # GET MODE
    # =====================================================

    @staticmethod
    def _get_mode(
        series
    ):

        if series is None:
            return None

        values = (
            series
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        if values.empty:
            return None

        modes = values.mode()

        if modes.empty:
            return None

        return str(
            modes.iloc[0]
        )


    # =====================================================
    # GET NUMERIC MEAN
    # =====================================================

    @staticmethod
    def _get_mean(
        series
    ):

        if series is None:
            return None

        values = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if values.empty:
            return None

        return round(
            float(values.mean()),
            2
        )


    # =====================================================
    # BUILD AVERAGE SCENARIO
    # =====================================================

    def build_average_scenario(
        self,
        road_name
    ):

        """
        يبني Scenario حقيقي وكامل من بيانات الحوادث
        الخاصة بالطريق.

        Numeric:
            mean

        Categorical:
            mode

        Location / Vehicle / Driver / Date:
            يتم أخذها من الـ merged dataset.

        لا يتم اختراع قيم ML features.
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

        # =================================================
        # BASE SCENARIO
        # =================================================

        scenario = {

            "road_name":
                road_name,

            "source":
                "real_accident_data",

            "accident_count":
                int(len(road_df)),
        }

        # =================================================
        # NUMERIC FEATURES
        # =================================================

        numeric_columns = [

            # Accident
            "Vehicles_Involved",
            "Impact_Speed_KMH",
            "Posted_Speed_Limit_KMH",
            "Emergency_Response_Time_Min",

            # Location
            "AADT_Volume",
            "KM_Marker",
            "Latitude",
            "Longitude",

            # Time
            "Hour_24",
            "Year",
            "Month",
            "Day",
            "Day_of_Week",

            # Vehicle
            "Vehicle_Age_Years",

            # Driver
            "Age",
            "Driving_Experience_Years",
            "Monthly_Income_EGP",
            "Prior_Violations_Count",
            "Fatigue_Hours_Awake",
        ]

        for column in numeric_columns:

            if column not in road_df.columns:

                continue

            value = self._get_mean(
                road_df[column]
            )

            if value is not None:

                scenario[column] = value

        # =================================================
        # CATEGORICAL FEATURES
        # =================================================

        categorical_columns = [

            # Accident / road
            "Road_Type",
            "Highway_Name",
            "Governorate_EN",
            "Collision_Type",
            "Weather_Condition",
            "Lighting_Condition",
            "Road_Surface_Condition",

            # Location
            "Is_Urban_Road",
            "Is_Black_Spot",

            # Vehicle
            "Vehicle_Category",
            "Make",
            "Model",
            "Licensing_Status",
            "Technical_Inspection_Passed",
            "Overload_Flag",
            "ABS_Equipped",
            "Driver_Airbag_Equipped",
            "Tire_Condition",
            "Brake_Condition",
            "Insurance_Coverage",
            "Fuel_Type",

            # Driver
            "Gender",
            "License_Category",
            "License_Status",
            "Occupation",
            "Distracted_Driving_Mobile",
            "Seatbelt_Used",
            "Helmet_Used",
        ]

        for column in categorical_columns:

            if column not in road_df.columns:

                continue

            value = self._get_mode(
                road_df[column]
            )

            if value is not None:

                scenario[column] = value

        # =================================================
        # DERIVED SPEED FEATURES
        # =================================================

        speed = scenario.get(
            "Impact_Speed_KMH"
        )

        speed_limit = scenario.get(
            "Posted_Speed_Limit_KMH"
        )

        if (
            speed is not None
            and
            speed_limit is not None
        ):

            try:

                speed = float(speed)
                speed_limit = float(
                    speed_limit
                )

                scenario[
                    "Speed_Over_Limit_KMH"
                ] = round(
                    max(
                        0.0,
                        speed - speed_limit
                    ),
                    2
                )

                if speed_limit > 0:

                    scenario[
                        "Speed_Ratio"
                    ] = round(
                        speed / speed_limit,
                        4
                    )

            except (
                TypeError,
                ValueError,
            ):

                pass

        # =================================================
        # COMPLETE SCENARIO FROM MODEL CONFIG
        # =================================================

        # لو model config موجود، اطبع الـ features الناقصة
        # بدل ما نخترع قيم.
        if self.impact_features:

            missing = [
                feature
                for feature in self.impact_features
                if feature not in scenario
            ]

            if missing:

                print(
                    "[WARNING] Missing AI impact features:"
                )

                print(
                    missing
                )

            else:

                print(
                    "[INFO] AI impact scenario is COMPLETE."
                )

        if self.classification_features:

            missing_classification = [
                feature
                for feature in self.classification_features
                if feature not in scenario
            ]

            if missing_classification:

                print(
                    "[WARNING] Missing AI classification features:"
                )

                print(
                    missing_classification
                )

            else:

                print(
                    "[INFO] AI classification scenario is COMPLETE."
                )

        # =================================================
        # OPTIONAL SAFE FALLBACKS
        # =================================================
        #
        # هذه ليست ML feature fabrication.
        # تستخدم فقط للحفاظ على endpoints القديمة
        # إذا كانت بعض الأعمدة العامة غير موجودة.
        #
        # لا نضع Defaults للـ driver/vehicle features.
        # =================================================

        if "Hour_24" not in scenario:

            scenario["Hour_24"] = 12

        # =================================================
        # PRINT FINAL SCENARIO
        # =================================================

        print(
            f"[INFO] Built complete average scenario "
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

        try:

            hour_key = str(
                int(hour_24) % 24
            )

        except (
            TypeError,
            ValueError,
        ):

            hour_key = "12"

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

        weather_key = (
            str(weather)
            .strip()
            .lower()
        )

        weather_mult = (
            weather_factor_data.get(
                weather_key,
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

                    try:

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

                    except (
                        TypeError,
                        ValueError,
                    ):

                        pass

        # -------------------------------------------------
        # Normal factor
        # -------------------------------------------------

        elif (
            component_key
            and component_key in components
        ):

            try:

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

            except (
                TypeError,
                ValueError,
            ):

                pass

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

        # -------------------------------------------------
        # Apply improvement according to factor weight
        # -------------------------------------------------

        if (
            component_key
            and component_key in COMPONENT_WEIGHTS
        ):

            factor_weight = (
                COMPONENT_WEIGHTS[
                    component_key
                ]
            )

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
                float(
                    improvement_pct
                ),

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