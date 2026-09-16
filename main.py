"""
main.py
=======
Backend FastAPI لواجهة EgyRoad IQ
"""

import os
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
import pandas as pd

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PROJECT IMPORTS
# ============================================================

from ai_model import CONTROLLABLE_FACTORS, AIModel
from risk_model import RiskModel
from storage import IncidentStore, UserStore

from ai_agent import (
    answer_question,
    classify_complaint,
    load_roadwise_cache,
)

from image_verification import (
    image_report_flag,
    verify_incident_image,
)


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

JWT_SECRET = os.environ.get(
    "ROADWISE_JWT_SECRET",
    "change-this-secret-in-production",
)

JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 12

LOCAL_DEMO_TOKEN = "local-demo-token"

LOCAL_DEMO_MODE = (
    os.environ.get(
        "ROADWISE_LOCAL_DEMO_MODE",
        "true",
    ).strip().lower()
    in ("1", "true", "yes", "on")
)

LOCAL_DEMO_USER = {
    "sub": "rofaida.alqassas@gmail.com",
    "role": "decision",
    "name": "rofaida amr",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="EgyRoad IQ API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_FRONTEND_CANDIDATES = [
    "EgyRoad_IQ.html",
    "EgyRoad_IQ (1).html",
]


# ============================================================
# GLOBAL MODELS / STORES
# ============================================================

print("[START] Loading EgyRoad IQ backend...")


risk_model = None
ai_model = None
AI_READY = False
roadwise_cache = None
ROADWISE_READY = False


users = UserStore(
    os.path.join(
        DATA_DIR,
        "users.json",
    )
)

incidents = IncidentStore(
    os.path.join(
        DATA_DIR,
        "incidents.json",
    )
)

security = HTTPBearer(auto_error=False)


def require_risk_model():
    if risk_model is None:
        raise HTTPException(
            status_code=503,
            detail="Risk model غير متاح حاليًا.",
        )

    return risk_model


def require_ai():
    if not AI_READY or ai_model is None:
        raise HTTPException(
            status_code=503,
            detail="موديل الذكاء الاصطناعي غير متاح حاليًا.",
        )

    return ai_model


def require_roadwise():
    if not ROADWISE_READY:
        raise HTTPException(
            status_code=503,
            detail="بيانات ROADWISE غير متاحة حاليًا.",
        )

    return roadwise_cache


# ============================================================
# STARTUP EVENT
# ============================================================
# التحميل التقيل كله (RiskModel/AIModel/roadwise_cache + قراءات
# accidents_data.xlsx التمهيدية) كان قبل كده بيحصل وقت استيراد الملف نفسه
# (module import) - يعني قبل ما uvicorn يقدر يفتح البورت أصلًا. وده كان
# سبب "Port scan timeout" ودقايق الانتظار الطويلة (حصل قبل كده لحد 19
# دقيقة) وقت ما الموقع "يصحى من النوم" بعد فترة خمول على Render Free.
#
# نقل التحميل هنا (FastAPI startup event) بيخلي uvicorn يفتح البورت فورًا
# بعد الـ import السريع، ثم يشغّل التحميل التقيل فى الخلفية بعد كده. أي
# طلب يوصل أثناء التحميل لسه شغال هياخد رسالة 503 واضحة بدل ما السيرفر
# كله يفضل معلّق من غير رد. لو اترجع الكود ده لوضعه القديم (تحميل مباشر
# فى مستوى الملف) هترجع مشكلة الـ Deploy البطيء اللي استغرقنا وقت طويل
# نشخصها ونصلحها.
# ============================================================

@app.on_event("startup")
def _load_heavy_dependencies():
    global risk_model, ai_model, AI_READY, roadwise_cache, ROADWISE_READY

    try:
        risk_model = RiskModel()
        print("[OK] RiskModel loaded.")
    except Exception as e:
        risk_model = None
        print(f"[ERROR] RiskModel failed: {e}")

    try:
        ai_model = AIModel()
        AI_READY = True
        print("[OK] AIModel loaded successfully.")
    except Exception as e:
        ai_model = None
        AI_READY = False
        print(f"[WARN] AI model unavailable: {e}")

    try:
        roadwise_cache = load_roadwise_cache()
        ROADWISE_READY = True
        print("[OK] ROADWISE cache loaded.")
    except Exception as e:
        roadwise_cache = None
        ROADWISE_READY = False
        print(f"[WARN] ROADWISE cache failed: {e}")

    try:
        _load_journey_df()
    except Exception as e:
        print(f"[WARN] Journey cache failed: {e}")

    try:
        get_excel_data()
    except Exception as e:
        print(f"[WARN] /api/data cache failed: {e}")


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    for name in _FRONTEND_CANDIDATES:

        path = os.path.join(BASE_DIR, name)

        if os.path.exists(path):
            return FileResponse(path)

    raise HTTPException(
        status_code=500,
        detail=(
            "ملف الواجهة غير موجود. "
            f"الأسماء المقبولة: {_FRONTEND_CANDIDATES}"
        ),
    )


# ============================================================
# PYDANTIC MODELS
# ============================================================

class RegisterBody(BaseModel):
    name: str
    email: str
    password: str
    national_id: Optional[str] = None
    role: str = "citizen"


class LoginBody(BaseModel):
    email: str
    password: str
    role: str = "citizen"


class PredictBody(BaseModel):
    governorates: List[str]
    hour_24: int
    weather: str


class WhatIfBody(BaseModel):
    road_name: str
    factor: str
    improvement_pct: Optional[float] = None


class ScenarioBody(BaseModel):
    scenario: Dict[str, Any]


class WhatIfAIBody(BaseModel):
    scenario: Dict[str, Any]
    changes: Dict[str, Any]


class ImpactBody(BaseModel):

    collision_type: Optional[str] = None
    vehicle_type: Optional[str] = None

    speed: Optional[float] = Field(
        default=None,
        ge=0,
    )

    vehicles_involved: Optional[int] = Field(
        default=None,
        ge=1,
    )

    posted_speed_limit: Optional[float] = Field(
        default=None,
        gt=0,
    )

    lighting: Optional[str] = None
    road_surface: Optional[str] = None
    weather: Optional[str] = None
    road_name: Optional[str] = None

    extra: Dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# AUTH
# ============================================================

def make_token(user: dict) -> str:

    payload = {
        "sub": user["email"],
        "role": user["role"],
        "name": user["name"],
        "exp": (
            datetime.now(timezone.utc)
            + timedelta(hours=JWT_EXPIRE_HOURS)
        ),
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGO,
    )


def get_current_user(
    creds: Optional[
        HTTPAuthorizationCredentials
    ] = Depends(security),
):

    if (
        LOCAL_DEMO_MODE
        and creds is not None
        and creds.credentials == LOCAL_DEMO_TOKEN
    ):
        return LOCAL_DEMO_USER

    if creds is None:

        if LOCAL_DEMO_MODE:
            return LOCAL_DEMO_USER

        raise HTTPException(
            status_code=401,
            detail="يجب تسجيل الدخول أولًا.",
        )

    try:

        payload = jwt.decode(
            creds.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGO],
        )

        return payload

    except jwt.PyJWTError:

        raise HTTPException(
            status_code=401,
            detail="جلسة الدخول غير صالحة.",
        )


def _public_user(user: dict) -> dict:

    return {
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
    }


def _require_decision_user(user: dict):

    if user["role"] not in (
        "decision",
        "admin",
    ):
        raise HTTPException(
            status_code=403,
            detail="مسموح فقط لمتخذي القرار.",
        )


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.post("/auth/register")
def register(
    body: RegisterBody,
    creds: Optional[
        HTTPAuthorizationCredentials
    ] = Depends(security),
):

    if users.get(body.email):

        raise HTTPException(
            status_code=409,
            detail="البريد الإلكتروني مسجّل بالفعل.",
        )

    role = body.role.strip().lower()

    if role == "citizen":

        if not (
            body.national_id
            and body.national_id.isdigit()
            and len(body.national_id) == 14
        ):
            raise HTTPException(
                status_code=400,
                detail="الرقم القومي يجب أن يتكون من 14 رقمًا.",
            )

        user = users.create(
            name=body.name,
            email=body.email,
            password=body.password,
            role="citizen",
            national_id=body.national_id,
        )

    elif role in ("decision", "admin"):

        current = get_current_user(creds)

        if current["role"] not in (
            "decision",
            "admin",
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "مسموح فقط لمتخذي القرار أو المدير."
                ),
            )

        user = users.create(
            name=body.name,
            email=body.email,
            password=body.password,
            role=role,
            national_id=None,
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="نوع الحساب غير معروف.",
        )

    return {
        "access_token": make_token(user),
        "user": _public_user(user),
    }


@app.post("/auth/login")
def login(body: LoginBody):

    user = users.get(body.email)

    if (
        not user
        or not users.verify_password(
            user,
            body.password,
        )
    ):

        raise HTTPException(
            status_code=401,
            detail=(
                "البريد الإلكتروني أو كلمة المرور غير صحيحة."
            ),
        )

    if user["role"] != body.role:

        raise HTTPException(
            status_code=403,
            detail=(
                "هذا الحساب غير مسجّل بهذه الصلاحية."
            ),
        )

    return {
        "access_token": make_token(user),
        "user": _public_user(user),
    }


# ============================================================
# BASIC PREDICT
# ============================================================

@app.post("/predict")
def predict(body: PredictBody):

    model = require_risk_model()

    if not 0 <= body.hour_24 <= 23:

        raise HTTPException(
            status_code=422,
            detail="الساعة يجب أن تكون بين 0 و23.",
        )

    return model.predict_trip(
        body.governorates,
        body.hour_24,
        body.weather,
    )


# ============================================================
# JOURNEY DATA
# ============================================================

_JOURNEY_DF = None
_JOURNEY_RAW_STATS = None
_JOURNEY_METRICS_CACHE = None


def _find_excel_path():

    candidates = [
        os.path.join(
            BASE_DIR,
            "accidents_data.xlsx",
        ),
        os.path.join(
            DATA_DIR,
            "accidents_data.xlsx",
        ),
        os.path.join(
            BASE_DIR,
            "111.xlsx",
        ),
    ]

    return next(
        (
            p
            for p in candidates
            if os.path.exists(p)
        ),
        None,
    )


def _load_journey_df():

    global _JOURNEY_DF

    if _JOURNEY_DF is not None:
        return _JOURNEY_DF

    excel_path = _find_excel_path()

    if not excel_path:

        raise HTTPException(
            status_code=500,
            detail="ملف بيانات الحوادث غير موجود.",
        )

    try:

        df = pd.read_excel(
            excel_path,
            sheet_name="Accidents",
            usecols=lambda c: c in (
                "Governorate_EN",
                "Highway_Name",
                "Fatalities_Count",
                "Injuries_Count",
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"تعذر قراءة بيانات الحوادث: {exc}",
        )

    required = [
        "Governorate_EN",
        "Highway_Name",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise HTTPException(
            status_code=500,
            detail=f"أعمدة مفقودة: {missing}",
        )

    for col in (
        "Governorate_EN",
        "Highway_Name",
    ):

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    for col in (
        "Fatalities_Count",
        "Injuries_Count",
    ):

        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    df = df[
        (df["Governorate_EN"] != "")
        & (df["Highway_Name"] != "")
    ].copy()

    _JOURNEY_DF = df

    print(
        f"[OK] Journey data cached: {len(df)} rows."
    )

    return df


def _road_raw_index(rows) -> float:

    accidents = len(rows)

    fatalities = rows[
        "Fatalities_Count"
    ].sum()

    injuries = rows[
        "Injuries_Count"
    ].sum()

    return (
        (
            fatalities * 5.0
        )
        + injuries
    ) / max(
        accidents,
        1,
    ) * 10.0


def _load_journey_raw_stats():

    global _JOURNEY_RAW_STATS

    if _JOURNEY_RAW_STATS is not None:
        return _JOURNEY_RAW_STATS

    df = _load_journey_df()

    raws = [
        _road_raw_index(rows)
        for _, rows in df.groupby(
            [
                "Governorate_EN",
                "Highway_Name",
            ],
            sort=False,
        )
    ]

    if not raws:

        _JOURNEY_RAW_STATS = (
            0.0,
            1.0,
        )

        return _JOURNEY_RAW_STATS

    series = pd.Series(
        raws,
        dtype="float64",
    )

    low = float(
        series.quantile(0.05)
    )

    high = float(
        series.quantile(0.95)
    )

    if high <= low:
        high = low + 1.0

    _JOURNEY_RAW_STATS = (
        low,
        high,
    )

    return _JOURNEY_RAW_STATS


def _calculate_road_metrics(
    rows,
    governorate,
    road_name,
):

    if rows.empty:
        return None

    accidents = int(len(rows))

    fatalities = int(
        rows["Fatalities_Count"].sum()
    )

    injuries = int(
        rows["Injuries_Count"].sum()
    )

    raw = _road_raw_index(rows)

    low, high = _load_journey_raw_stats()

    normalized = (
        (raw - low)
        / max(high - low, 1.0)
        * 100.0
    )

    risk_score = round(
        max(
            0.0,
            min(
                100.0,
                normalized,
            ),
        ),
        1,
    )

    if risk_score >= 75:
        level = "مرتفع جدًا"
    elif risk_score >= 55:
        level = "مرتفع"
    elif risk_score >= 35:
        level = "متوسط"
    else:
        level = "منخفض"

    return {
        "governorate": str(governorate).strip(),
        "road_name": str(road_name).strip(),
        "risk_score": risk_score,
        "risk_level": level,
        "accidents": accidents,
        "fatalities": fatalities,
        "injuries": injuries,
        "raw_index": round(raw, 2),
    }


def _build_journey_metrics_cache():

    global _JOURNEY_METRICS_CACHE

    if _JOURNEY_METRICS_CACHE is not None:
        return _JOURNEY_METRICS_CACHE

    df = _load_journey_df()

    cache = {}

    for (
        governorate,
        road_name,
    ), rows in df.groupby(
        [
            "Governorate_EN",
            "Highway_Name",
        ],
        sort=False,
    ):

        key = (
            str(governorate).strip().casefold(),
            str(road_name).strip().casefold(),
        )

        cache[key] = _calculate_road_metrics(
            rows,
            governorate,
            road_name,
        )

    _JOURNEY_METRICS_CACHE = cache

    print(
        f"[OK] Road metrics cached: {len(cache)} roads."
    )

    return cache


def _road_metrics(
    df,
    governorate,
    road_name,
):

    key = (
        str(governorate).strip().casefold(),
        str(road_name).strip().casefold(),
    )

    cache = _build_journey_metrics_cache()

    return cache.get(key)


# ============================================================
# JOURNEY ENDPOINTS
# ============================================================

class JourneyRequest(BaseModel):

    governorate: str
    road_name: str
    hour_24: int
    weather: str = "Clear"


_WEATHER_TRAINED_MAP = {
    "Clear": "Clear",
    "Rain": "Light Rain",
    "Fog": "Fog",
    "Dust": "Dusty",
}


def _ai_predict_road_risk(
    road_name: str,
    hour_24: int,
    weather: str,
):

    model = require_ai()
    rm = require_risk_model()

    scenario = rm.build_average_scenario(
        road_name
    )

    if not scenario:
        return None

    scenario["Hour_24"] = hour_24

    scenario[
        "Weather_Condition"
    ] = _WEATHER_TRAINED_MAP.get(
        weather,
        weather,
    )

    return model.predict(
        scenario
    )


class JourneyOptionsResponse(BaseModel):

    governorates: List[str]

    roads_by_governorate: Dict[
        str,
        List[str],
    ]


@app.get(
    "/journey-options",
    response_model=JourneyOptionsResponse,
)
def journey_options():

    df = _load_journey_df()

    grouped = {}

    for gov, g in df.groupby(
        "Governorate_EN",
        sort=True,
    ):

        roads = sorted(
            g[
                "Highway_Name"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        roads = [
            r
            for r in roads
            if r
            and str(r).lower() != "nan"
        ]

        if roads:
            grouped[gov] = roads

    return {
        "governorates": list(
            grouped.keys()
        ),
        "roads_by_governorate": grouped,
    }


@app.post(
    "/analyze-road-journey"
)
def analyze_road_journey(
    body: JourneyRequest,
):

    if not body.governorate.strip():

        raise HTTPException(
            status_code=422,
            detail="اختاري المحافظة.",
        )

    if not body.road_name.strip():

        raise HTTPException(
            status_code=422,
            detail="اختاري الطريق.",
        )

    if not 0 <= body.hour_24 <= 23:

        raise HTTPException(
            status_code=422,
            detail="الساعة يجب أن تكون بين 0 و23.",
        )

    df = _load_journey_df()

    metrics = _road_metrics(
        df,
        body.governorate,
        body.road_name,
    )

    if metrics is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "الطريق غير موجود داخل بيانات هذه المحافظة."
            ),
        )

    ai_result = None

    if AI_READY:

        try:

            ai_result = _ai_predict_road_risk(
                body.road_name,
                body.hour_24,
                body.weather,
            )

        except Exception as e:

            print(
                f"[WARN] Journey AI failed: {e}"
            )

    if ai_result:

        final_score = float(
            ai_result[
                "risk_score_0_100"
            ]
        )

        level = ai_result[
            "risk_level"
        ]

        source = (
            "final_injury_classifier_catboost.pkl "
            "(نموذج تعلّم آلي)"
        )

        model_used = "ai_classifier"

        injury_probability = ai_result[
            "injury_probability"
        ]

    else:

        try:

            rm = require_risk_model()

            context = rm.predict_trip(
                [
                    metrics[
                        "governorate"
                    ]
                ],
                body.hour_24,
                body.weather,
            )

            context_score = float(
                context.get(
                    "risk_score",
                    metrics["risk_score"],
                )
            )

        except Exception:

            context_score = metrics[
                "risk_score"
            ]

        final_score = round(
            min(
                100.0,
                max(
                    metrics["risk_score"],
                    context_score,
                ),
            ),
            1,
        )

        if final_score >= 75:
            level = "مرتفع جدًا"
        elif final_score >= 55:
            level = "مرتفع"
        elif final_score >= 35:
            level = "متوسط"
        else:
            level = "منخفض"

        source = (
            "accidents_data.xlsx / Accidents "
            "(مؤشر تاريخي)"
        )

        model_used = "historical_index"

        injury_probability = None

    result = {
        **metrics,

        "risk_score": final_score,

        "risk_level": level,

        "hour_24": body.hour_24,

        "weather": body.weather,

        "source": source,

        "model_used": model_used,

        "recommendation": (
            "الطريق مرتفع الخطورة؛ "
            "يفضل خفض السرعة وتجنب الظروف الجوية السيئة."
            if final_score >= 55
            else
            "مستوى الخطورة أقل، مع الالتزام بالسرعة الآمنة وتعليمات المرور."
        ),
    }

    if injury_probability is not None:
        result[
            "injury_probability"
        ] = injury_probability

    return result


@app.post("/best-route")
def best_route(
    body: JourneyRequest,
):

    df = _load_journey_df()

    gov = (
        body.governorate
        .strip()
        .casefold()
    )

    subset = df[
        df[
            "Governorate_EN"
        ].str.casefold()
        == gov
    ]

    if subset.empty:

        raise HTTPException(
            status_code=404,
            detail="لا توجد محافظة بهذا الاسم.",
        )

    options = []

    for road in sorted(
        subset[
            "Highway_Name"
        ].dropna().unique()
    ):

        m = _road_metrics(
            df,
            body.governorate,
            road,
        )

        if m:
            options.append(m)

    options.sort(
        key=lambda x: x["risk_score"]
    )

    if not options:

        raise HTTPException(
            status_code=404,
            detail="لا توجد طرق مدعومة.",
        )

    return {
        "governorate": body.governorate,

        "best_available_option": options[0],

        "alternatives": options[:5],

        "source": (
            "accidents_data.xlsx / Accidents"
        ),

        "note": (
            "الاختيار مبني على أقل Risk Score "
            "من الطرق الفعلية المسجلة."
        ),
    }


def _road_list_for_governorate(
    governorate
):

    df = _load_journey_df()

    gov = (
        str(governorate)
        .strip()
        .casefold()
    )

    subset = df[
        df[
            "Governorate_EN"
        ].str.casefold()
        == gov
    ]

    roads = []

    for road in sorted(
        subset[
            "Highway_Name"
        ].dropna().unique()
    ):

        m = _road_metrics(
            df,
            governorate,
            road,
        )

        if m:
            roads.append(m)

    return roads


@app.get("/governorate-roads")
def governorate_roads(
    governorate: str,
):

    roads = _road_list_for_governorate(
        governorate
    )

    roads.sort(
        key=lambda x: float(
            x.get(
                "risk_score",
                999,
            )
        )
    )

    return {
        "governorate": governorate,
        "count": len(roads),
        "roads": roads,
    }


# ============================================================
# WHAT-IF HELPERS
# ============================================================

_FACTOR_ALIASES = {
    "speed reduction": "Speed Reduction",
    "خفض السرعة": "Speed Reduction",
    "speed": "Speed Reduction",
    "السرعة": "Speed Reduction",
}


def _normalize_factor(factor: str):

    raw = str(factor).strip()

    alias = _FACTOR_ALIASES.get(
        raw.casefold()
    )

    if alias:
        return alias

    for key in CONTROLLABLE_FACTORS:

        if key.casefold() == raw.casefold():
            return key

    return raw


# ============================================================
# WHAT-IF
# ============================================================

@app.post("/what-if")
def what_if(
    body: WhatIfBody,
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    model = require_ai()
    rm = require_risk_model()

    factor = _normalize_factor(
        body.factor
    )

    # ========================================================
    # SPEED REDUCTION — MODEL 2
    # ========================================================

    if factor == "Speed Reduction":

        if body.improvement_pct is None:

            raise HTTPException(
                status_code=422,
                detail="لازم تحددي نسبة خفض السرعة.",
            )

        try:

            pct = float(
                body.improvement_pct
            )

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=422,
                detail="نسبة خفض السرعة يجب أن تكون رقمًا.",
            )

        if not 0 <= pct <= 100:

            raise HTTPException(
                status_code=422,
                detail=(
                    "نسبة خفض السرعة يجب أن تكون بين 0 و100."
                ),
            )

        try:

            scenario = rm.build_average_scenario(
                body.road_name
            )

        except Exception as e:

            print(
                f"[ERROR] Scenario failed: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="تعذر بناء سيناريو الطريق.",
            )

        if not scenario:

            raise HTTPException(
                status_code=404,
                detail=(
                    "لا توجد بيانات كافية عن هذا الطريق."
                ),
            )

        current_speed = scenario.get(
            "Impact_Speed_KMH"
        )

        speed_limit = scenario.get(
            "Posted_Speed_Limit_KMH"
        )

        try:

            current_speed = float(
                current_speed
            )

            speed_limit = float(
                speed_limit
            )

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    "بيانات السرعة أو الحد الأقصى غير صالحة."
                ),
            )

        if current_speed < 0:
            current_speed = 0.0

        if speed_limit <= 0:

            raise HTTPException(
                status_code=422,
                detail="الحد الأقصى للسرعة غير صالح.",
            )

        new_speed = max(
            0.0,
            current_speed
            * (
                1.0
                - pct / 100.0
            ),
        )

        before_scenario = dict(
            scenario
        )

        after_scenario = dict(
            scenario
        )

        before_scenario[
            "Impact_Speed_KMH"
        ] = current_speed

        after_scenario[
            "Impact_Speed_KMH"
        ] = new_speed

        before_scenario[
            "Posted_Speed_Limit_KMH"
        ] = speed_limit

        after_scenario[
            "Posted_Speed_Limit_KMH"
        ] = speed_limit

        try:

            before = model.predict_impact(
                before_scenario
            )

            after = model.predict_impact(
                after_scenario
            )

        except Exception as e:

            print(
                f"[ERROR] Speed What-If failed: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="تعذر تشغيل Model 2.",
            )

        original_score = float(
            before.get(
                "impact_score",
                0.0,
            )
        )

        new_score = float(
            after.get(
                "impact_score",
                0.0,
            )
        )

        improvement_points = round(
            original_score
            - new_score,
            2,
        )

        if original_score > 0:

            improvement_percent = round(
                (
                    improvement_points
                    / original_score
                )
                * 100.0,
                2,
            )

        else:

            improvement_percent = 0.0

        # ----------------------------------------------------
        # IMPORTANT:
        # We do not overwrite ML output.
        # If individual outcome moves unexpectedly while
        # total impact score improves, expose a warning.
        # ----------------------------------------------------

        behavior_note = None

        if (
            after.get(
                "expected_fatalities"
            ) is not None
            and before.get(
                "expected_fatalities"
            ) is not None
            and float(
                after.get(
                    "expected_fatalities",
                    0,
                )
            )
            >
            float(
                before.get(
                    "expected_fatalities",
                    0,
                )
            )
            and new_score < original_score
        ):

            behavior_note = (
                "النموذج يتوقع انخفاض التأثير الكلي، "
                "مع اختلاف اتجاه تقدير الوفيات بشكل منفصل؛ "
                "لذلك تم الإبقاء على مخرجات Model 2 كما هي "
                "دون تعديل يدوي."
            )

        return {

            "road_name":
                body.road_name,

            "factor":
                "Speed Reduction",

            "factor_label":
                "خفض السرعة",

            "action":
                f"خفض السرعة بنسبة {pct:.1f}%",

            "improvement_pct":
                pct,

            "original_speed":
                round(
                    current_speed,
                    1,
                ),

            "new_speed":
                round(
                    new_speed,
                    1,
                ),

            "speed_limit":
                round(
                    speed_limit,
                    1,
                ),

            "original_risk_score":
                round(
                    original_score,
                    2,
                ),

            "estimated_new_risk_score":
                round(
                    new_score,
                    2,
                ),

            "improvement_points":
                improvement_points,

            "improvement_percent":
                improvement_percent,

            "original_risk_level":
                before.get(
                    "impact_level"
                ),

            "new_risk_level":
                after.get(
                    "impact_level"
                ),

            "original_expected_injuries":
                before.get(
                    "expected_injuries"
                ),

            "new_expected_injuries":
                after.get(
                    "expected_injuries"
                ),

            "original_expected_fatalities":
                before.get(
                    "expected_fatalities"
                ),

            "new_expected_fatalities":
                after.get(
                    "expected_fatalities"
                ),

            "before":
                before,

            "after":
                after,

            "behavior_note":
                behavior_note,

            "debug":
                {
                    "model":
                        "Model 2",

                    "speed_feature":
                        "Impact_Speed_KMH",

                    "original_speed":
                        current_speed,

                    "new_speed":
                        new_speed,

                    "speed_limit":
                        speed_limit,

                    "original_speed_over_limit":
                        max(
                            0.0,
                            current_speed
                            - speed_limit,
                        ),

                    "new_speed_over_limit":
                        max(
                            0.0,
                            new_speed
                            - speed_limit,
                        ),

                    "original_speed_ratio":
                        current_speed
                        / speed_limit,

                    "new_speed_ratio":
                        new_speed
                        / speed_limit,

                    "speed_reduction_pct":
                        pct,

                    "features_used_before":
                        before.get(
                            "features_used"
                        ),

                    "features_used_after":
                        after.get(
                            "features_used"
                        ),

                    "features_expected":
                        before.get(
                            "features_expected"
                        ),

                    "missing_features":
                        before.get(
                            "missing_features"
                        ),
                },

            "source":
                (
                    "final_injuries_xgboost.pkl + "
                    "final_fatalities_catboost.pkl "
                    "(Model 2)"
                ),
        }

    # ========================================================
    # CATEGORICAL FACTORS — MODEL 1
    # ========================================================

    if factor not in CONTROLLABLE_FACTORS:

        raise HTTPException(
            status_code=422,
            detail=f"عامل غير معروف: {body.factor}",
        )

    try:

        scenario = rm.build_average_scenario(
            body.road_name
        )

    except Exception as e:

        print(
            f"[ERROR] Scenario failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="تعذر بناء السيناريو.",
        )

    if not scenario:

        raise HTTPException(
            status_code=404,
            detail=(
                "لا توجد بيانات كافية عن هذا الطريق."
            ),
        )

    meta = CONTROLLABLE_FACTORS[
        factor
    ]

    try:

        result = model.what_if(
            scenario,
            {
                factor:
                    meta["safe_value"]
            },
        )

    except Exception as e:

        print(
            f"[ERROR] Categorical What-If failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="تعذر تشغيل What-If.",
        )

    original_probability = float(
        result[
            "original"
        ][
            "injury_probability"
        ]
    )

    new_probability = float(
        result[
            "after_changes"
        ][
            "injury_probability"
        ]
    )

    return {

        "road_name":
            body.road_name,

        "factor":
            factor,

        "factor_label":
            meta["label"],

        "action":
            meta["action"],

        "original_risk_score":
            round(
                original_probability * 100,
                1,
            ),

        "estimated_new_risk_score":
            round(
                new_probability * 100,
                1,
            ),

        "original_risk_level":
            result[
                "original"
            ][
                "risk_level"
            ],

        "new_risk_level":
            result[
                "after_changes"
            ][
                "risk_level"
            ],

        "improvement_points":
            result[
                "improvement_points"
            ],

        "debug":
            result.get(
                "debug"
            ),

        "source":
            (
                "final_injury_classifier_catboost.pkl "
                "(Model 1)"
            ),
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard-stats")
def dashboard_stats(
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    return require_risk_model().get_dashboard_stats()


@app.get("/roads")
def roads(
    governorate: Optional[str] = None,
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    return require_risk_model().get_roads(
        governorate
    )


# ============================================================
# REPORT VERIFICATION
# ============================================================

_CLS_RANK = {
    "likely_false": 0,
    "needs_review": 1,
    "likely_valid": 2,
}


_LABELS_AR = {
    "likely_valid":
        "يبدو صحيحًا",

    "needs_review":
        "يحتاج مراجعة بشرية",

    "likely_false":
        "يبدو غير صحيح / مشبوه",
}


def _combine_verification(
    text_result: Dict[str, Any],
    image_flag: Optional[
        Dict[str, Any]
    ],
):

    overall = text_result.get(
        "classification",
        "needs_review",
    )

    if overall not in _CLS_RANK:
        overall = "needs_review"

    if image_flag:

        img_status = image_flag.get(
            "image_status"
        )

        if img_status in (
            "evidence_detected",
            "hazard_detected",
        ):

            rank = min(
                _CLS_RANK[overall] + 1,
                2,
            )

            overall = next(
                (
                    k
                    for k, v
                    in _CLS_RANK.items()
                    if v == rank
                ),
                "needs_review",
            )

        elif img_status == "irrelevant":

            if overall == "likely_valid":
                overall = "needs_review"

    return {

        "overall_classification":
            overall,

        "overall_label_ar":
            _LABELS_AR.get(
                overall,
                _LABELS_AR[
                    "needs_review"
                ],
            ),

        "text_verification":
            text_result,

        "image_verification":
            image_flag,
    }


def _classify_report_text(
    description: str,
    governorate: str = "",
    road_name: str = "",
) -> Dict[str, Any]:

    try:

        result = classify_complaint(
            description,
            governorate,
            road_name,
        )

    except Exception as e:

        print(
            f"[ERROR] Complaint classification failed: {e}"
        )

        result = {
            "classification":
                "needs_review",

            "confidence":
                0,

            "reason":
                "تعذر تحليل البلاغ آليًا.",
        }

    classification = result.get(
        "classification",
        "needs_review",
    )

    if classification not in _LABELS_AR:
        classification = "needs_review"

    return {

        "classification":
            classification,

        "classification_label_ar":
            _LABELS_AR[
                classification
            ],

        "confidence":
            result.get(
                "confidence",
                0,
            ),

        "reason":
            result.get(
                "reason",
                "",
            ),
    }


# ============================================================
# REPORT INCIDENT
# ============================================================

@app.post("/report-incident")
async def report_incident(
    governorate: str = Form(...),
    road_name: str = Form(...),
    description: str = Form(...),
    image: Optional[
        UploadFile
    ] = File(None),
    user=Depends(get_current_user),
):

    if not governorate.strip():

        raise HTTPException(
            status_code=422,
            detail="المحافظة مطلوبة.",
        )

    if not road_name.strip():

        raise HTTPException(
            status_code=422,
            detail="اسم الطريق مطلوب.",
        )

    if not description.strip():

        raise HTTPException(
            status_code=422,
            detail="وصف البلاغ مطلوب.",
        )

    image_path = None
    image_flag = None
    raw_bytes = None
    mime_type = "image/jpeg"

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if (
        image is not None
        and image.filename
    ):

        mime_type = (
            image.content_type
            or "image/jpeg"
        )

        if mime_type not in ALLOWED_IMAGE_TYPES:

            raise HTTPException(
                status_code=422,
                detail=(
                    "نوع الصورة غير مدعوم. "
                    "استخدمي JPG أو PNG أو WEBP."
                ),
            )

        raw_bytes = await image.read()

        if not raw_bytes:

            raise HTTPException(
                status_code=422,
                detail="الصورة فارغة.",
            )

        if len(raw_bytes) > MAX_IMAGE_SIZE:

            raise HTTPException(
                status_code=413,
                detail=(
                    "حجم الصورة كبير جدًا. "
                    "الحد الأقصى 10 MB."
                ),
            )

        ext = (
            os.path.splitext(
                image.filename
            )[1]
            or ".jpg"
        )

        image_path = os.path.join(
            UPLOADS_DIR,
            f"{uuid.uuid4().hex}{ext}",
        )

        with open(
            image_path,
            "wb",
        ) as f:

            f.write(raw_bytes)

    # --------------------------------------------------------
    # PARALLEL AI
    # --------------------------------------------------------

    tasks = [
        asyncio.to_thread(
            _classify_report_text,
            description,
            governorate,
            road_name,
        )
    ]

    if raw_bytes is not None:

        tasks.append(
            asyncio.to_thread(
                verify_incident_image,
                raw_bytes,
                mime_type,
            )
        )

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_result = results[0]

    if isinstance(
        text_result,
        Exception,
    ):

        text_result = {
            "classification":
                "needs_review",

            "classification_label_ar":
                _LABELS_AR[
                    "needs_review"
                ],

            "confidence":
                0,

            "reason":
                "تعذر تحليل البلاغ.",
        }

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if raw_bytes is not None:

        verification = results[1]

        if isinstance(
            verification,
            Exception,
        ):

            print(
                f"[ERROR] Image verification failed: {verification}"
            )

            image_flag = {
                "image_status":
                    "needs_review",

                "confidence":
                    0,

                "reason":
                    "تعذر فحص الصورة آليًا.",
            }

        else:

            image_flag = image_report_flag(
                verification
            )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    report_verification = (
        _combine_verification(
            text_result,
            image_flag,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    record = incidents.create(
        governorate=governorate,
        road_name=road_name,
        description=description,
        image_path=image_path,
        reported_by=user["sub"],
        image_verification=image_flag,
        report_verification=report_verification,
    )

    return {

        "incident_id":
            record["id"],

        "report_status":
            record["status"],

        "image_verification":
            image_flag,

        "report_verification":
            report_verification,
    }


# ============================================================
# COMPLAINTS
# ============================================================

@app.get("/admin/complaints")
def admin_complaints(
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    return incidents.list_all()


@app.get("/complaints")
def complaints_alias(
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    return incidents.list_all()


@app.get("/my-complaints")
def my_complaints(
    user=Depends(get_current_user),
):

    return incidents.list_by_user(
        user["sub"]
    )


class ClassifyIncidentBody(BaseModel):

    incident_id: str


@app.post("/classify-incident")
async def classify_incident(
    body: ClassifyIncidentBody,
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    record = incidents.get(
        body.incident_id
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="البلاغ غير موجود.",
        )

    # --------------------------------------------------------
    # Reuse previous classification.
    # This avoids calling Gemini again unnecessarily.
    # --------------------------------------------------------

    existing = record.get(
        "report_verification"
    )

    if existing:

        classification = existing.get(
            "overall_classification"
        )

        if classification in _LABELS_AR:

            text_result = (
                existing.get(
                    "text_verification"
                )
                or {
                    "classification":
                        classification,
                    "confidence":
                        0,
                    "reason":
                        "نتيجة محفوظة سابقًا.",
                }
            )

            return {

                "classification":
                    classification,

                "label":
                    existing.get(
                        "overall_label_ar",
                        _LABELS_AR[
                            classification
                        ],
                    ),

                "reason":
                    text_result.get(
                        "reason",
                        "",
                    ),

                "confidence":
                    text_result.get(
                        "confidence",
                        0,
                    ),

                "report_verification":
                    existing,
            }

    text_result = await asyncio.to_thread(
        _classify_report_text,
        record.get(
            "description",
            "",
        ),
        record.get(
            "governorate",
            "",
        ),
        record.get(
            "road_name",
            "",
        ),
    )

    report_verification = (
        _combine_verification(
            text_result,
            record.get(
                "image_verification"
            ),
        )
    )

    incidents.update(
        body.incident_id,
        report_verification=(
            report_verification
        ),
        classification=(
            report_verification[
                "overall_classification"
            ]
        ),
    )

    return {

        "classification":
            report_verification[
                "overall_classification"
            ],

        "label":
            report_verification[
                "overall_label_ar"
            ],

        "reason":
            text_result[
                "reason"
            ],

        "confidence":
            text_result[
                "confidence"
            ],

        "report_verification":
            report_verification,
    }


class ReviewIncidentBody(BaseModel):

    incident_id: str
    status: str
    classification: Optional[str] = None
    comment: Optional[str] = None


@app.post("/review-incident")
def review_incident(
    body: ReviewIncidentBody,
    user=Depends(get_current_user),
):

    _require_decision_user(user)

    record = incidents.get(
        body.incident_id
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="البلاغ غير موجود.",
        )

    fields: Dict[str, Any] = {
        "status":
            body.status,

        "reviewed_by":
            user["sub"],
    }

    if body.classification:

        if body.classification not in _LABELS_AR:

            raise HTTPException(
                status_code=422,
                detail="تصنيف البلاغ غير صالح.",
            )

        fields[
            "classification"
        ] = body.classification

    if body.comment is not None:

        fields[
            "decision_comment"
        ] = body.comment

    updated = incidents.update(
        body.incident_id,
        **fields,
    )

    return {
        "message":
            "تم حفظ قرار المراجعة",

        "incident":
            updated,
    }


@app.post("/verify-image")
async def verify_report_image(
    image: UploadFile = File(...),
    user=Depends(get_current_user),
):

    mime_type = (
        image.content_type
        or "image/jpeg"
    )

    if mime_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=422,
            detail=(
                "نوع الصورة غير مدعوم. "
                "استخدمي JPG أو PNG أو WEBP."
            ),
        )

    raw_bytes = await image.read()

    if not raw_bytes:

        raise HTTPException(
            status_code=422,
            detail="الصورة فارغة.",
        )

    if len(raw_bytes) > MAX_IMAGE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="حجم الصورة يتجاوز 10 MB.",
        )

    try:

        verification = await asyncio.to_thread(
            verify_incident_image,
            raw_bytes,
            mime_type,
        )

    except Exception as e:

        print(
            f"[ERROR] verify-image failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="تعذر فحص الصورة.",
        )

    return image_report_flag(
        verification
    )


# ============================================================
# REAL AI ENDPOINTS
# ============================================================

@app.post("/ai/predict")
def ai_predict(
    body: ScenarioBody
):

    return require_ai().predict(
        body.scenario
    )


@app.post("/ai/report")
def ai_report(
    body: ScenarioBody
):

    return require_ai().full_report(
        body.scenario
    )


@app.post("/ai/what-if")
def ai_what_if(
    body: WhatIfAIBody
):

    return require_ai().what_if(
        body.scenario,
        body.changes,
    )


# ============================================================
# MODEL 2 — ACCIDENT IMPACT
# ============================================================

_IMPACT_MAPPING = {

    "collision_type":
        "Collision_Type",

    "vehicle_type":
        "Vehicle_Category",

    "speed":
        "Impact_Speed_KMH",

    "vehicles_involved":
        "Vehicles_Involved",

    "posted_speed_limit":
        "Posted_Speed_Limit_KMH",

    "lighting":
        "Lighting_Condition",

    "road_surface":
        "Road_Surface_Condition",

    "weather":
        "Weather_Condition",
}


def _clean_extra(
    extra: Dict[str, Any]
) -> Dict[str, Any]:

    cleaned = {}

    for key, value in extra.items():

        if value is None:
            continue

        cleaned[str(key)] = value

    return cleaned


def _build_impact_scenario(
    body: ImpactBody,
) -> Dict[str, Any]:

    rm = require_risk_model()

    scenario: Dict[str, Any] = {}

    # --------------------------------------------------------
    # Road baseline
    # --------------------------------------------------------

    if body.road_name:

        try:

            road_scenario = (
                rm.build_average_scenario(
                    body.road_name
                )
            )

            if road_scenario:

                scenario.update(
                    road_scenario
                )

        except Exception as e:

            print(
                f"[WARN] Road scenario failed: {e}"
            )

    # --------------------------------------------------------
    # Explicit extra fields
    # --------------------------------------------------------

    if body.extra:

        scenario.update(
            _clean_extra(
                body.extra
            )
        )

    # --------------------------------------------------------
    # Frontend → ML mapping
    # --------------------------------------------------------

    body_dict = body.model_dump()

    for (
        frontend_field,
        ml_field,
    ) in _IMPACT_MAPPING.items():

        value = body_dict.get(
            frontend_field
        )

        if value is not None:

            scenario[
                ml_field
            ] = value

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    if "Impact_Speed_KMH" in scenario:

        try:

            speed = float(
                scenario[
                    "Impact_Speed_KMH"
                ]
            )

            if speed < 0:
                raise ValueError

            scenario[
                "Impact_Speed_KMH"
            ] = speed

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=422,
                detail="Impact speed غير صالحة.",
            )

    if "Posted_Speed_Limit_KMH" in scenario:

        try:

            limit = float(
                scenario[
                    "Posted_Speed_Limit_KMH"
                ]
            )

            if limit <= 0:
                raise ValueError

            scenario[
                "Posted_Speed_Limit_KMH"
            ] = limit

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=422,
                detail="Posted speed limit غير صالحة.",
            )

    if "Vehicles_Involved" in scenario:

        try:

            vehicles = int(
                float(
                    scenario[
                        "Vehicles_Involved"
                    ]
                )
            )

            if vehicles < 1:
                raise ValueError

            scenario[
                "Vehicles_Involved"
            ] = vehicles

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=422,
                detail="عدد المركبات غير صالح.",
            )

    return scenario


@app.post("/predict-impact")
def predict_impact(
    body: ImpactBody,
):

    model = require_ai()

    scenario = _build_impact_scenario(
        body
    )

    if not scenario:

        raise HTTPException(
            status_code=422,
            detail=(
                "لم يتم إرسال بيانات كافية "
                "لبناء سيناريو الحادث."
            ),
        )

    try:

        result = model.predict_impact(
            scenario
        )

    except Exception as e:

        print(
            f"[ERROR] predict-impact failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "تعذر تشغيل نموذج تأثير الحادث."
            ),
        )

    level = result.get(
        "impact_level"
    )

    if level == "مرتفع":

        recommendation = (
            "خفّضي السرعة والتزمي بالتباعد الآمن - "
            "عوامل التأثير الحالية مرتفعة."
        )

    elif level == "متوسط":

        recommendation = (
            "راجعي عوامل الخطر وقللي "
            "التدخلات عالية الخطورة."
        )

    else:

        recommendation = (
            "مستوى التأثير منخفض نسبيًا، "
            "مع الاستمرار في الالتزام بقواعد السلامة."
        )

    result[
        "recommendation"
    ] = recommendation

    # --------------------------------------------------------
    # Transparent diagnostics
    # --------------------------------------------------------

    result[
        "scenario_source"
    ] = (
        "road average + explicit frontend inputs"
        if body.road_name
        else
        "explicit frontend inputs"
    )

    result[
        "input_overrides"
    ] = {
        key: value
        for key, value
        in {
            "collision_type":
                body.collision_type,

            "vehicle_type":
                body.vehicle_type,

            "speed":
                body.speed,

            "vehicles_involved":
                body.vehicles_involved,

            "posted_speed_limit":
                body.posted_speed_limit,

            "lighting":
                body.lighting,

            "road_surface":
                body.road_surface,

            "weather":
                body.weather,
        }.items()
        if value is not None
    }

    return result


# ============================================================
# AI METADATA
# ============================================================

@app.get(
    "/ai/controllable-factors"
)
def ai_controllable_factors():

    return [

        {
            "field":
                field,

            "label":
                meta["label"],

            "safe_value":
                meta["safe_value"],

            "action":
                meta["action"],
        }

        for field, meta
        in CONTROLLABLE_FACTORS.items()
    ]


@app.get(
    "/ai/categorical-options"
)
def ai_categorical_options():

    return require_ai().categorical_options


@app.get(
    "/ai/feature-importance"
)
def ai_feature_importance(
    top: int = 15,
):

    top = max(
        1,
        min(
            int(top),
            100,
        ),
    )

    return require_ai().feature_importance[
        :top
    ]


# ============================================================
# EXCEL DATA
# ============================================================

_EXCEL_DATA_CACHE = None


def get_excel_data():

    global _EXCEL_DATA_CACHE

    if _EXCEL_DATA_CACHE is not None:
        return _EXCEL_DATA_CACHE

    # Reuse the already cached journey data.
    df = _load_journey_df()

    data = df[
        [
            "Fatalities_Count",
            "Injuries_Count",
        ]
    ].copy()

    records = data.to_dict(
        orient="records"
    )

    _EXCEL_DATA_CACHE = {

        "count":
            len(records),

        "columns":
            data.columns.tolist(),

        "data":
            records,
    }

    print(
        "[OK] /api/data cache loaded."
    )

    return _EXCEL_DATA_CACHE


@app.get("/api/data")
def read_data():

    return get_excel_data()


# ============================================================
# GEMINI AI ASSISTANT
# ============================================================

class AskAIRequest(BaseModel):

    question: str


@app.post("/ai-assistant")
async def ai_assistant(
    body: AskAIRequest
):

    question = body.question.strip()

    if not question:

        raise HTTPException(
            status_code=422,
            detail="اكتبي السؤال أولًا.",
        )

    cache = require_roadwise()

    try:

        # Gemini / agent execution is blocking,
        # so do not block FastAPI's event loop.
        answer = await asyncio.to_thread(
            answer_question,
            question,
            cache,
        )

        return {

            "success":
                True,

            "answer":
                answer,
        }

    except Exception as e:

        print(
            f"[ERROR] AI assistant failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "تعذر الحصول على إجابة المساعد."
            ),
        )


# ============================================================
# COMPLAINT CLASSIFICATION
# ============================================================

class ComplaintClassificationRequest(
    BaseModel
):

    description: str
    governorate: str = ""
    road_name: str = ""


@app.post("/classify-complaint")
async def classify_complaint_endpoint(
    data: ComplaintClassificationRequest,
):

    if not data.description.strip():

        raise HTTPException(
            status_code=422,
            detail="وصف البلاغ مطلوب.",
        )

    return await asyncio.to_thread(
        _classify_report_text,
        data.description,
        data.governorate,
        data.road_name,
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "ai_ready":
            AI_READY,

        "roadwise_ready":
            ROADWISE_READY,

        "risk_model_ready":
            risk_model is not None,

        "journey_data_ready":
            _JOURNEY_DF is not None,

        "version":
            "1.1.0",
    }


# ============================================================
# STARTUP
# ============================================================

print(
    "[READY] EgyRoad IQ backend initialization complete."
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    # IMPORTANT:
    # Pass the app object directly.
    # Do NOT use "main:app" here because Python would
    # import main.py again and initialize the models twice.

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )