"""
main.py
=======
باك اند FastAPI لواجهة EgyRoad IQ.

Endpoints المتاحة:
    POST /auth/register
    POST /auth/login
    POST /predict               <- تقييم سريع بالمعادلة اليدوية (محافظة/ساعة/طقس)
    POST /report-incident
    GET  /dashboard-stats
    GET  /roads
    POST /what-if                <- سيناريو الطرق (المعادلة اليدوية)

Endpoints الذكاء الاصطناعي الحقيقي:
    POST /ai/predict              -> احتمالية الإصابة + مستوى الخطورة
    POST /ai/report               -> التقرير الكامل
    POST /ai/what-if               -> تجربة سيناريو بديل
    GET  /ai/controllable-factors  -> العوامل القابلة للتحكم
    GET  /ai/feature-importance    -> أهمية العوامل بشكل عام

الأكواد المضافة حديثاً:
    GET  /api/data                 -> جلب بيانات ملف الإكسيل الأساسي في المشروع

تشغيل محلى:
    pip install -r requirements.txt
    python data_prep.py
    python train_model.py
    uvicorn main:app --reload --port 8000
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse
from pydantic import BaseModel

# لازم تتحمّل قبل أي import من ai_agent/image_verification، لأن الملفين دول
# بيعملوا genai.Client() وقت الـ import نفسه (module-level) - يعني بيدوروا
# على GEMINI_API_KEY / GOOGLE_API_KEY في os.environ فورًا. لو load_dotenv()
# اتنادت بعدهم أو متنادتش خالص، المفتاح المتخزّن في .env مش هيتلاقى، وهيرجع
# نفس خطأ "API key not valid" حتى لو المفتاح نفسه صحيح فعليًا.
#
# محتاجة: pip install python-dotenv (لو لسه مش متثبتة) + ملف .env جنب
# main.py فيه سطر GEMINI_API_KEY=...
from dotenv import load_dotenv
load_dotenv()

from ai_model import CONTROLLABLE_FACTORS, AIModel
from risk_model import RiskModel
from storage import IncidentStore, UserStore

from ai_agent import answer_question, classify_complaint, load_roadwise_cache
from image_verification import image_report_flag, verify_incident_image

JWT_SECRET = os.environ.get("ROADWISE_JWT_SECRET", "change-this-secret-in-production")
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 12
UPLOADS_DIR = "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="EgyRoad IQ API")
_FRONTEND_CANDIDATES = ["EgyRoad_IQ.html", "EgyRoad_IQ (1).html"]

@app.get("/")
def home():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in _FRONTEND_CANDIDATES:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(
        status_code=500,
        detail=f"ملف الواجهة غير موجود. لازم يكون أحد هذه الأسماء موجود جنب main.py: {_FRONTEND_CANDIDATES}",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

risk_model = RiskModel()
users = UserStore("data/users.json")
incidents = IncidentStore("data/incidents.json")
security = HTTPBearer(auto_error=False)

try:
    ai_model = AIModel()
    AI_READY = True
except FileNotFoundError:
    ai_model = None
    AI_READY = False


def require_ai():
    if not AI_READY:
        raise HTTPException(
            status_code=503,
            detail="موديل الذكاء الاصطناعي لسه مش موجود. شغّلي train_model.py الأول.",
        )
    return ai_model


# بيانات ROADWISE (accidents_data.xlsx) بتتحمّل مرة واحدة بس هنا عند
# تشغيل السيرفر - مش مع كل سؤال زي كان بيحصل قبل كده. لو الملف مش موجود
# وقت التشغيل، /ai-assistant هترجع خطأ واضح بدل ما السيرفر يقع بالكامل.
try:
    roadwise_cache = load_roadwise_cache()
    ROADWISE_READY = True
except Exception as e:
    roadwise_cache = None
    ROADWISE_READY = False
    print(f"[WARN] تعذر تحميل بيانات ROADWISE عند التشغيل: {e}")


def require_roadwise():
    if not ROADWISE_READY:
        raise HTTPException(
            status_code=503,
            detail="بيانات ROADWISE لسه مش محمّلة. تأكدي من وجود accidents_data.xlsx.",
        )
    return roadwise_cache


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
    improvement_pct: float


class ScenarioBody(BaseModel):
    scenario: Dict[str, Any]


class WhatIfAIBody(BaseModel):
    scenario: Dict[str, Any]
    changes: Dict[str, Any]


class ImpactBody(BaseModel):
    """
    بيانات بسيطة اللي واجهة "تقدير التأثير" (runImpactModel فى الـ HTML)
    بترسلها فعليًا. أي فيتشر تانى من IMPACT_FEATURES مش موجود هنا بيتسيب
    فاضي والـ Pipeline (imputer) هو اللي يتصرف فيه (median/most_frequent).
    """
    collision_type: Optional[str] = None
    vehicle_type: Optional[str] = None
    speed: Optional[float] = None
    vehicles_involved: Optional[int] = None
    # اختياري: لو الواجهة بعتت كمان أي فيتشرز تانية من IMPACT_FEATURES
    # (زي المحافظة/الإضاءة/حالة الطريق من فورم "تقييم عوامل الخطر")، بتتمرر
    # زي ما هي وتنضم لباقي السيناريو.
    extra: Dict[str, Any] = {}


def make_token(user: dict) -> str:
    payload = {
        "sub": user["email"],
        "role": user["role"],
        "name": user["name"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


LOCAL_DEMO_TOKEN = "local-demo-token"
LOCAL_DEMO_USER = {"sub": "rofaida.alqassas@gmail.com", "role": "decision", "name": "rofaida amr"}

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    # Local development mode: endpoints work without forcing login.
    LOCAL_DEV_MODE = True

    if LOCAL_DEV_MODE and creds is None:
        return LOCAL_DEMO_USER

    if creds is not None and creds.credentials == LOCAL_DEMO_TOKEN:
        return LOCAL_DEMO_USER

    if creds is None:
        raise HTTPException(status_code=401, detail="لازم تسجّلي الدخول الأول")

    try:
        payload = jwt.decode(
            creds.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGO]
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="جلسة الدخول غير صالحة، سجّلي الدخول تانى"
        )

    return payload


def _public_user(user: dict) -> dict:
    return {
        "name": user["name"],
        "email": user["email"],
        "role": user["role"]
    }


@app.post("/auth/register")
def register(
    body: RegisterBody,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    if users.get(body.email):
        raise HTTPException(
            status_code=409,
            detail="البريد الإلكتروني مسجّل بالفعل"
        )

    if body.role == "citizen":
        if not (
            body.national_id
            and body.national_id.isdigit()
            and len(body.national_id) == 14
        ):
            raise HTTPException(
                status_code=400,
                detail="الرقم القومي يجب أن يتكون من 14 رقمًا"
            )

        user = users.create(
            name=body.name,
            email=body.email,
            password=body.password,
            role="citizen",
            national_id=body.national_id
        )

    elif body.role in ("decision", "admin"):
        current = get_current_user(creds)

        if current["role"] not in ("decision", "admin"):
            raise HTTPException(
                status_code=403,
                detail="مسموح فقط لأعضاء لوحة متخذ القرار بإضافة أعضاء جدد"
            )

        user = users.create(
            name=body.name,
            email=body.email,
            password=body.password,
            role=body.role,
            national_id=None
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="نوع الحساب غير معروف"
        )

    token = make_token(user)

    return {
        "access_token": token,
        "user": _public_user(user)
    }


@app.post("/auth/login")
def login(body: LoginBody):
    user = users.get(body.email)

    if not user or not users.verify_password(user, body.password):
        raise HTTPException(
            status_code=401,
            detail="البريد الإلكتروني أو كلمة المرور غير صحيحة"
        )

    if user["role"] != body.role:
        raise HTTPException(
            status_code=403,
            detail="هذا الحساب غير مسجّل بهذه الصلاحية"
        )

    token = make_token(user)

    return {
        "access_token": token,
        "user": _public_user(user)
    }


@app.post("/predict")
def predict(body: PredictBody):
    return risk_model.predict_trip(
        body.governorates,
        body.hour_24,
        body.weather
    )


# ============================================================
# SMART JOURNEY — DATA-DRIVEN GOVERNORATE + ROAD
# ============================================================
_JOURNEY_DF = None
_JOURNEY_RAW_STATS = None  # (low, high) لتطبيع raw index - راجع _load_journey_raw_stats

def _load_journey_df():
    global _JOURNEY_DF
    if _JOURNEY_DF is not None:
        return _JOURNEY_DF
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "accidents_data.xlsx"),
        os.path.join(base_dir, "data", "accidents_data.xlsx"),
        os.path.join(base_dir, "111.xlsx"),
    ]
    excel_path = next((p for p in candidates if os.path.exists(p)), None)
    if not excel_path:
        raise HTTPException(status_code=500, detail="ملف بيانات الحوادث غير موجود داخل المشروع.")
    try:
        df = pd.read_excel(excel_path, sheet_name="Accidents")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"تعذر قراءة بيانات الحوادث: {exc}")
    required = ["Governorate_EN", "Highway_Name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"أعمدة البيانات المطلوبة غير موجودة: {missing}")
    for c in ["Governorate_EN", "Highway_Name"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    for c in ["Fatalities_Count", "Injuries_Count"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df[(df["Governorate_EN"] != "") & (df["Highway_Name"] != "")].copy()
    _JOURNEY_DF = df
    return df


def _road_raw_index(rows) -> float:
    """المؤشر الخام (raw) المشتق من نتائج الحوادث الفعلية لمجموعة صفوف
    طريق واحد. نفس صيغة الحساب القديمة (فتلات/إصابات مرجّحة لكل حادثة) -
    لسه مفيدة كترتيب نسبي بين الطرق، بس مش صالحة كـ0-100 مباشرة لأن
    متوسط raw عبر كل الطرق فعليًا بيتخطى الـ100 (راجع التعليق فى
    _load_journey_raw_stats تحت)."""
    accidents = len(rows)
    fatalities = rows["Fatalities_Count"].sum()
    injuries = rows["Injuries_Count"].sum()
    return ((fatalities * 5.0) + injuries) / max(accidents, 1) * 10.0


def _load_journey_raw_stats():
    """يحسب حدود التطبيع (5th/95th percentile) لمؤشر raw عبر كل الطرق
    الفعلية فى الداتا، مرة واحدة بس، ونكاشها.

    ليه مش زي القديم (min(100.0, raw))؟
    لأن متوسط raw لأي طريق عادي (مش أخطر طريق) بيتخطى الـ100 أصلاً فى
    الداتاست ده (متوسط وفيات/حادثة ≈ 0.79 ومتوسط إصابات/حادثة ≈ 8.4 ->
    raw ≈ 124 حتى للمتوسط العام). القص الثابت عند 100 كان بيخلي كل
    الطرق تقريبًا تطلع risk_score = 100 و"مرتفع جدًا"، من غير أي قدرة
    فعلية على التمييز بين طريق خطير وطريق أقل خطورة.

    الحل: بنحسب raw لكل طريق حقيقى فى الداتا مرة واحدة، وبعدين بنطبّع كل
    طريق نسبةً للـ5th/95th percentile الفعليين (مش أقل/أعلى قيمة
    بالظبط، عشان طريق واحد شاذ إحصائيًا ميضغطش باقي المقياس). النتيجة:
    أخطر الطرق فعليًا تاخد قريب من 100، الأقل خطورة تاخد قريب من صفر،
    والباقي يتوزع بينهم بمنطقية.
    """
    global _JOURNEY_RAW_STATS
    if _JOURNEY_RAW_STATS is not None:
        return _JOURNEY_RAW_STATS

    df = _load_journey_df()
    raws = [
        _road_raw_index(rows)
        for _, rows in df.groupby(["Governorate_EN", "Highway_Name"], sort=False)
    ]

    if not raws:
        _JOURNEY_RAW_STATS = (0.0, 1.0)
        return _JOURNEY_RAW_STATS

    series = pd.Series(raws, dtype="float64")
    low = float(series.quantile(0.05))
    high = float(series.quantile(0.95))
    if high - low < 1e-9:
        high = low + 1e-9

    _JOURNEY_RAW_STATS = (low, high)
    return _JOURNEY_RAW_STATS


def _road_metrics(df, governorate, road_name):
    g = str(governorate).strip()
    r = str(road_name).strip()
    rows = df[(df["Governorate_EN"].str.casefold() == g.casefold()) &
              (df["Highway_Name"].str.casefold() == r.casefold())]
    if rows.empty:
        return None
    accidents = int(len(rows))
    fatalities = int(rows["Fatalities_Count"].sum())
    injuries = int(rows["Injuries_Count"].sum())

    raw = _road_raw_index(rows)

    # تطبيع نسبي (percentile-based) بدل القص الثابت القديم عند 100 -
    # راجع _load_journey_raw_stats لشرح المشكلة والحل.
    low, high = _load_journey_raw_stats()
    normalized = (raw - low) / (high - low) * 100.0
    risk_score = round(max(0.0, min(100.0, normalized)), 1)

    if risk_score >= 75:
        level = "مرتفع جدًا"
    elif risk_score >= 55:
        level = "مرتفع"
    elif risk_score >= 35:
        level = "متوسط"
    else:
        level = "منخفض"
    return {
        "governorate": g,
        "road_name": r,
        "risk_score": risk_score,
        "risk_level": level,
        "accidents": accidents,
        "fatalities": fatalities,
        "injuries": injuries,
        "raw_index": round(raw, 2),
    }


class JourneyRequest(BaseModel):
    governorate: str
    road_name: str
    hour_24: int
    weather: str = "Clear"


class JourneyOptionsResponse(BaseModel):
    governorates: List[str]
    roads_by_governorate: Dict[str, List[str]]


@app.get("/journey-options", response_model=JourneyOptionsResponse)
def journey_options():
    df = _load_journey_df()
    grouped = {}
    for gov, g in df.groupby("Governorate_EN", sort=True):
        roads = sorted(g["Highway_Name"].dropna().unique().tolist())
        roads = [r for r in roads if r and r.lower() != "nan"]
        if roads:
            grouped[gov] = roads
    return {"governorates": list(grouped.keys()), "roads_by_governorate": grouped}


@app.post("/analyze-road-journey")
def analyze_road_journey(body: JourneyRequest):
    if not body.governorate.strip():
        raise HTTPException(status_code=422, detail="اختاري المحافظة.")
    if not body.road_name.strip():
        raise HTTPException(status_code=422, detail="اختاري الطريق.")
    if body.hour_24 < 0 or body.hour_24 > 23:
        raise HTTPException(status_code=422, detail="الساعة يجب أن تكون بين 0 و23.")
    df = _load_journey_df()
    metrics = _road_metrics(df, body.governorate, body.road_name)
    if metrics is None:
        raise HTTPException(status_code=404, detail="الطريق المختار غير موجود داخل بيانات الحوادث لهذه المحافظة.")

    # عامل الوقت/الطقس يأتي من RiskModel، بينما أرقام الطريق نفسها تأتي مباشرة من Excel.
    try:
        context = risk_model.predict_trip([metrics["governorate"]], body.hour_24, body.weather)
        context_score = float(context.get("risk_score", metrics["risk_score"]))
    except Exception:
        context_score = metrics["risk_score"]

    # نستخدم بيانات الطريق الفعلية كأساس، مع تعديل صغير فقط إذا كان سياق الوقت/الطقس أعلى.
    final_score = round(min(100.0, max(metrics["risk_score"], context_score)), 1)
    if final_score >= 75:
        level = "مرتفع جدًا"
    elif final_score >= 55:
        level = "مرتفع"
    elif final_score >= 35:
        level = "متوسط"
    else:
        level = "منخفض"

    return {
        **metrics,
        "risk_score": final_score,
        "risk_level": level,
        "hour_24": body.hour_24,
        "weather": body.weather,
        "source": "accidents_data.xlsx / Accidents",
        "recommendation": (
            "الطريق مرتفع الخطورة وفق نتائج الحوادث المسجلة؛ يفضّل خفض السرعة وتجنب الظروف الجوية السيئة."
            if final_score >= 55
            else "مستوى الخطورة أقل وفق بيانات الحوادث المسجلة، مع الالتزام بالسرعة الآمنة وتعليمات المرور."
        ),
    }


@app.post("/best-route")
def best_route(body: JourneyRequest):
    # اختيار أقل Risk Score من الطرق الفعلية داخل المحافظة المختارة.
    df = _load_journey_df()
    gov = body.governorate.strip().casefold()
    subset = df[df["Governorate_EN"].str.casefold() == gov]
    if subset.empty:
        raise HTTPException(status_code=404, detail="لا توجد محافظة بهذا الاسم في البيانات.")
    options = []
    for road in sorted(subset["Highway_Name"].unique()):
        m = _road_metrics(df, body.governorate, road)
        if m:
            options.append(m)
    options.sort(key=lambda x: x["risk_score"])
    if not options:
        raise HTTPException(status_code=404, detail="لا توجد طرق مدعومة في بيانات الحوادث لهذه المحافظة.")
    best = options[0]
    return {
        "governorate": body.governorate,
        "best_available_option": best,
        "alternatives": options[:5],
        "source": "accidents_data.xlsx / Accidents",
        "note": "الاختيار مبني على أقل Risk Score من الطرق الفعلية المسجلة داخل المحافظة.",
    }


def _road_list_for_governorate(governorate):
    """يبني قائمة بكل الطرق الفعلية المسجلة داخل محافظة معيّنة مع
    risk_score محسوب لكل واحد منها.

    ملحوظة: الدالة دي كانت متسخدمة فى /governorate-roads تحت من غير ما
    تكون معرّفة فى أي مكان فى الملف - أي نداء على الـendpoint ده كان
    هيطلع NameError دايمًا. تم تعريفها هنا بنفس منطق best_route."""
    df = _load_journey_df()
    gov = str(governorate).strip().casefold()
    subset = df[df["Governorate_EN"].str.casefold() == gov]
    roads = []
    for road in sorted(subset["Highway_Name"].unique()):
        m = _road_metrics(df, governorate, road)
        if m:
            roads.append(m)
    return roads


@app.get("/governorate-roads")
def governorate_roads(governorate: str):
    roads = _road_list_for_governorate(governorate)
    roads = sorted(roads, key=lambda x: float(x.get("risk_score", 999)))
    return {"governorate": governorate, "count": len(roads), "roads": roads}


@app.post("/what-if")
def what_if(
    body: WhatIfBody,
    user=Depends(get_current_user)
):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(
            status_code=403,
            detail="مسموح فقط لحسابات متخذي القرار"
        )

    try:
        return risk_model.what_if(
            body.road_name,
            body.factor,
            body.improvement_pct
        )
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )


@app.get("/dashboard-stats")
def dashboard_stats(user=Depends(get_current_user)):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(
            status_code=403,
            detail="مسموح فقط لحسابات متخذي القرار"
        )

    return risk_model.get_dashboard_stats()


@app.get("/roads")
def roads(
    governorate: Optional[str] = None,
    user=Depends(get_current_user)
):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(
            status_code=403,
            detail="مسموح فقط لحسابات متخذي القرار"
        )

    return risk_model.get_roads(governorate)


_CLS_RANK = {"likely_false": 0, "needs_review": 1, "likely_valid": 2}


def _combine_verification(text_result: Dict[str, Any], image_flag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """يدمج تصنيف النص مع فحص الصورة في تصنيف نهائي واحد للبلاغ (Report
    Verification)، بحرص شديد إن صورة "irrelevant" لوحدها ميقدرش يطلع منها
    حكم "likely_false" - أقصى حاجة يعملها إنه يخفض الثقة لـ needs_review،
    زي ما هو موضّح في image_verification.py. الصورة اللي بتثبت الحادث/الخطر
    ("evidence_detected"/"hazard_detected") ممكن ترفع مستوى الثقة بحد أقصى
    درجة واحدة فقط.
    """
    overall = text_result["classification"]

    if image_flag:
        img_status = image_flag.get("image_status")
        if img_status in ("evidence_detected", "hazard_detected"):
            # دعم إيجابي من الصورة -> نرفع درجة واحدة بحد أقصى
            rank = min(_CLS_RANK[overall] + 1, 2)
            overall = [k for k, v in _CLS_RANK.items() if v == rank][0]
        elif img_status == "irrelevant":
            # الصورة وحدها ميقدرش يطلع منها "likely_false" - أقصى تأثير
            # سلبي إنه يوقف overall عند "needs_review" لو كان "likely_valid"
            if overall == "likely_valid":
                overall = "needs_review"

    return {
        "overall_classification": overall,
        "overall_label_ar": _LABELS_AR[overall],
        "text_verification": text_result,
        "image_verification": image_flag,
    }


@app.post("/report-incident")
async def report_incident(
    governorate: str = Form(...),
    road_name: str = Form(...),
    description: str = Form(...),
    image: Optional[UploadFile] = File(None),
    user=Depends(get_current_user),
):
    image_path = None
    image_flag: Optional[Dict[str, Any]] = None

    if image is not None and image.filename:
        ext = os.path.splitext(image.filename)[1] or ".jpg"

        image_path = os.path.join(
            UPLOADS_DIR,
            f"{uuid.uuid4().hex}{ext}"
        )

        raw_bytes = await image.read()

        with open(image_path, "wb") as f:
            f.write(raw_bytes)

        # فحص فعلي للصورة عن طريق Gemini Vision - بدون أي بيانات وهمية.
        # لو الفحص فشل أو لم يوجد مفتاح API، النتيجة هتكون "needs_review"
        # بشكل صريح (راجع image_verification.py).
        verification = verify_incident_image(
            raw_bytes,
            mime_type=image.content_type or "image/jpeg",
        )
        image_flag = image_report_flag(verification)

    # تصنيف نص البلاغ بالتصنيف الثلاثي (likely_valid / needs_review /
    # likely_false)، ثم دمجه مع فحص الصورة في حكم واحد نهائي متحفّظ.
    text_result = _classify_report_text(description, governorate, road_name)
    report_verification = _combine_verification(text_result, image_flag)

    # ملاحظة: incidents.create لازم تدعم استقبال report_verification كـ
    # حقل إضافي في السجل (زيادة عن image_verification اللي كانت موجودة).
    # لو ظهر خطأ هنا، معناه IncidentStore في storage.py عندك محتاج تعديل
    # بسيط ليقبل **extra fields - ابعتيلي storage.py عشان أظبطه بالظبط
    # بدل ما أخمّن شكله.
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
        "incident_id": record["id"],
        "report_status": record["status"],
        "image_verification": image_flag,
        "report_verification": report_verification,
    }


@app.get("/admin/complaints")
def admin_complaints(user=Depends(get_current_user)):
    """قائمة كل البلاغات - للوحة متخذ القرار فقط. الواجهة (EgyRoad_IQ.html)
    بتنادي على الـendpoint ده بالاسم ده بالظبط."""
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")
    return incidents.list_all()


@app.get("/my-complaints")
def my_complaints(user=Depends(get_current_user)):
    """بلاغات المواطن نفسه فقط - بديل حقيقي لـlocalStorage، بيشتغل حتى لو
    المواطن غيّر متصفح أو جهاز."""
    return incidents.list_by_user(user["sub"])


class ClassifyIncidentBody(BaseModel):
    incident_id: str


@app.post("/classify-incident")
def classify_incident(body: ClassifyIncidentBody, user=Depends(get_current_user)):
    """إعادة تشغيل Report Verification AI على بلاغ محفوظ فعليًا (بدل ما
    ناخد نص خام زي /classify-complaint)، وتحديث السجل بأحدث تصنيف."""
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")

    record = incidents.get(body.incident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="البلاغ غير موجود")

    text_result = _classify_report_text(
        record.get("description", ""),
        record.get("governorate", ""),
        record.get("road_name", ""),
    )
    report_verification = _combine_verification(text_result, record.get("image_verification"))

    incidents.update(
        body.incident_id,
        report_verification=report_verification,
        classification=report_verification["overall_classification"],
    )

    return {
        "classification": report_verification["overall_classification"],
        "label": report_verification["overall_label_ar"],
        "reason": text_result["reason"],
        "confidence": text_result["confidence"],
        "report_verification": report_verification,
    }


class ReviewIncidentBody(BaseModel):
    incident_id: str
    status: str
    classification: Optional[str] = None


@app.post("/review-incident")
def review_incident(body: ReviewIncidentBody, user=Depends(get_current_user)):
    """قرار المراجعة البشري النهائي (متخذ القرار) - ده اللي بيغيّر status
    فعليًا، مش تصنيف الـAI لوحده."""
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")

    record = incidents.get(body.incident_id)
    if record is None:
        raise HTTPException(status_code=404, detail="البلاغ غير موجود")

    fields: Dict[str, Any] = {"status": body.status, "reviewed_by": user["sub"]}
    if body.classification:
        fields["classification"] = body.classification

    updated = incidents.update(body.incident_id, **fields)

    return {"message": "تم حفظ قرار المراجعة", "incident": updated}


@app.post("/verify-image")
async def verify_report_image(
    image: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """
    فحص أوّلي لصورة البلاغ قبل الإرسال (زر 'فحص البلاغ بالذكاء الاصطناعي').
    لا يتم حفظ أي بيانات هنا - فقط رأي الموديل في الصورة، عشان المواطن
    يقدر يشوف النتيجة قبل ما يقرر يبعت البلاغ أو لأ.
    """

    raw_bytes = await image.read()

    verification = verify_incident_image(
        raw_bytes,
        mime_type=image.content_type or "image/jpeg",
    )

    return image_report_flag(verification)


@app.post("/ai/predict")
def ai_predict(body: ScenarioBody):
    model = require_ai()
    return model.predict(body.scenario)


@app.post("/ai/report")
def ai_report(body: ScenarioBody):
    model = require_ai()
    return model.full_report(body.scenario)


@app.post("/ai/what-if")
def ai_what_if(body: WhatIfAIBody):
    model = require_ai()
    return model.what_if(
        body.scenario,
        body.changes
    )


@app.post("/predict-impact")
def predict_impact(body: ImpactBody):
    """
    تقدير عدد الإصابات/الوفيات المتوقع لسيناريو اصطدام معيّن، باستخدام
    final_injuries_xgboost.pkl و final_fatalities_catboost.pkl.
    هي الـ endpoint اللي واجهة runImpactModel() فى الـ HTML متوقعاها أصلاً
    (كانت بترجع خطأ وبتقع على fallback جافاسكريبت قبل كده).
    """
    model = require_ai()

    scenario: Dict[str, Any] = dict(body.extra or {})
    if body.collision_type is not None:
        scenario["Collision_Type"] = body.collision_type
    if body.vehicle_type is not None:
        scenario["Vehicle_Category"] = body.vehicle_type
    if body.speed is not None:
        scenario["Impact_Speed_KMH"] = body.speed
    if body.vehicles_involved is not None:
        scenario["Vehicles_Involved"] = body.vehicles_involved

    result = model.predict_impact(scenario)

    # تسمية متوافقة مع اللي الواجهة بتدور عليه فى الـ response
    # (d.impact_score / d.impact_level / d.recommendation)
    result["recommendation"] = (
        "خفّضي السرعة والتزمي بالتباعد الآمن - عوامل التأثير الحالية مرتفعة."
        if result["impact_level"] == "مرتفع"
        else "راجعي عوامل الخطر قبل اتخاذ القرار."
    )
    return result


@app.get("/ai/controllable-factors")
def ai_controllable_factors():
    return [
        {
            "field": field,
            "label": meta["label"],
            "safe_value": meta["safe_value"],
            "action": meta["action"]
        }
        for field, meta in CONTROLLABLE_FACTORS.items()
    ]


@app.get("/ai/feature-importance")
def ai_feature_importance(top: int = 15):
    model = require_ai()
    return model.feature_importance[:top]


@app.get("/complaints")
def complaints_alias(user=Depends(get_current_user)):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")
    return incidents.list_all()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_ready": AI_READY
    }


# ----------------------------------------------------
# قراءة ملف الإكسيل وربطه بالواجهات
# ----------------------------------------------------

_EXCEL_DATA_CACHE = None  # نتيجة get_excel_data() مكاشة - راجع التعليق تحت


def get_excel_data():
    """Return only the columns the frontend actually reads from /api/data.

    مكاش فى الذاكرة (زي _load_journey_df بالظبط) بدل ما يعيد قراءة/دمج
    ملف accidents_data.xlsx (~40 ألف صف) وتحويله لـ JSON فى كل طلب. القراءة
    والتحويل دول كانوا بيتكرروا مع كل GET /api/data، وده اللي كان بيسبب
    تخطي حد الذاكرة (Memory limit) على Render وإعادة تشغيل الخدمة.

    كانت الدالة بترجع الجدول كامل (Accidents مدموج مع كل أعمدة Locations،
    حوالي 27 عمود × 40 ألف صف). فحصت EgyRoad_IQ.html ولقيت إن d.data من
    /api/data مستخدم فى مكان واحد بس (loadExcelData/loadStats) وبس كـ
    احتياطي لو /dashboard-stats (السريع، المبني من artifacts/) فشل، وإن
    الوحيد اللي بيتقرا فعليًا من كل صف هو Fatalities_Count وInjuries_Count
    (زائد d.data.length نفسه). فبنرجع العمودين دول بس بدل الجدول كله - ده
    بيقلل حجم الاستجابة بشكل كبير جدًا (من ~27 عمود لعمودين) وبيلغي الحاجة
    لقراءة/دمج شيت Locations خالص، وده اللي كان بيسبب التأخير/الـ502 حتى
    بعد إصلاح الـ caching والـ NaN."""

    global _EXCEL_DATA_CACHE
    if _EXCEL_DATA_CACHE is not None:
        return _EXCEL_DATA_CACHE

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(
            base_dir,
            "accidents_data.xlsx"
        )

        accidents = pd.read_excel(
            excel_path,
            sheet_name="Accidents",
            usecols=lambda c: c in ("Fatalities_Count", "Injuries_Count")
        )

        for col in ("Fatalities_Count", "Injuries_Count"):
            if col not in accidents.columns:
                accidents[col] = 0
            accidents[col] = pd.to_numeric(accidents[col], errors="coerce").fillna(0)

        records = accidents.to_dict(orient="records")

        _EXCEL_DATA_CACHE = {
            "count": len(records),
            "columns": accidents.columns.tolist(),
            "data": records
        }
        return _EXCEL_DATA_CACHE

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"تعذر جلب بيانات الإكسيل: {str(e)}"
        )


@app.get("/api/data")
def read_data():
    """Return the unified Accidents + Locations dataset for the HTML frontend."""
    return get_excel_data()


# ----------------------------------------------------
# Gemini AI Assistant
# ----------------------------------------------------

class AskAIRequest(BaseModel):
    question: str


@app.post("/ai-assistant")
def ai_assistant(body: AskAIRequest):
    cache = require_roadwise()
    try:
        answer = answer_question(body.question, cache)

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
 # ----------------------------------------------------
# Complaint Classification
# ----------------------------------------------------

class ComplaintClassificationRequest(BaseModel):
    description: str
    governorate: str = ""
    road_name: str = ""


# تصنيفات "Report Verification AI" - مقصود إننا ما نطلعش حكم قاطع
# True/False من غير دليل كافي. القيمة الافتراضية عند أي شك أو خطأ هي
# "needs_review" دايمًا، مش "likely_false" - عشان ما نرفضش بلاغ حقيقي
# غلط لمجرد إن التصنيف الآلي فشل أو مش متأكد.
_LABELS_AR = {
    "likely_valid": "يبدو صحيحًا",
    "needs_review": "يحتاج مراجعة بشرية",
    "likely_false": "يبدو غير صحيح / مشبوه",
}


def _classify_report_text(description: str, governorate: str = "", road_name: str = "") -> Dict[str, Any]:
    """تصنيف نص البلاغ فقط (بدون الصورة) بالتصنيف الثلاثي. دالة داخلية
    مستقلة عشان تُستخدم هنا وفي /report-incident كمان من غير تكرار كود.

    ملحوظة مهمة: كانت الدالة دي قبل كده بتستخدم ask_gemini() - يعني كل
    تصنيف بلاغ كان فعليًا بيحمّل بيانات ROADWISE بالكامل (get_roadwise_summary)
    جوه برومبت "أجب عن سؤال متخذ القرار"، وبرومبت التصنيف ده كان بيتحط
    جوه البرومبت الكبير ده. ده كان سبب أساسي في بطء تصنيف الشكاوى
    ولاحتمال رجوع تصنيف غلط (بلاغ غير حقيقي يطلع "يبدو صحيحًا") بسبب
    تلخبط الموديل بين تعليمات مختلفة. دلوقتي بتستخدم classify_complaint()
    من ai_agent.py المستقلة تمامًا عن بيانات ROADWISE."""
    try:
        result = classify_complaint(description, governorate, road_name)
    except Exception as e:
        # أي فشل غير متوقع -> needs_review دايمًا، مش likely_false، عشان
        # الفشل التقني ما يترجمش لرفض بلاغ حقيقي.
        result = {
            "classification": "needs_review",
            "confidence": 0,
            "reason": f"تعذر تحليل البلاغ آليًا: {str(e)}",
        }

    return {
        "classification": result["classification"],
        "classification_label_ar": _LABELS_AR[result["classification"]],
        "confidence": result["confidence"],
        "reason": result["reason"],
    }


@app.post("/classify-complaint")
def classify_complaint_endpoint(data: ComplaintClassificationRequest):
    return _classify_report_text(data.description, data.governorate, data.road_name)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
