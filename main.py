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

from ai_model import CONTROLLABLE_FACTORS, AIModel
from risk_model import RiskModel
from storage import IncidentStore, UserStore

from ai_agent import ask_gemini
from image_verification import image_report_flag, verify_incident_image

JWT_SECRET = os.environ.get("ROADWISE_JWT_SECRET", "change-this-secret-in-production")
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 12
UPLOADS_DIR = "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="EgyRoad IQ API")
@app.get("/")
def home():
    return FileResponse("EgyRoad_IQ(1).html")

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
    if creds is None:
        raise HTTPException(status_code=401, detail="لازم تسجّلي الدخول الأول")

    if creds.credentials == LOCAL_DEMO_TOKEN:
        return LOCAL_DEMO_USER

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

    # ملاحظة: incidents.create لازم تدعم استقبال image_verification كـ
    # حقل إضافي في السجل. لو ظهر خطأ هنا، معناه IncidentStore في storage.py
    # عندك محتاج تعديل بسيط ليقبل **extra fields - ابعتيلي storage.py
    # عشان أظبطه بالظبط بدل ما أخمّن شكله.
    record = incidents.create(
        governorate=governorate,
        road_name=road_name,
        description=description,
        image_path=image_path,
        reported_by=user["sub"],
        image_verification=image_flag,
    )

    return {
        "incident_id": record["id"],
        "report_status": record["status"],
        "image_verification": image_flag,
    }


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_ready": AI_READY
    }


# ----------------------------------------------------
# قراءة ملف الإكسيل وربطه بالواجهات
# ----------------------------------------------------

def get_excel_data():
    """Load the accident records and enrich them with location attributes."""

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(
            base_dir,
            "accidents_data.xlsx"
        )

        xls = pd.ExcelFile(excel_path)

        accidents = pd.read_excel(
            xls,
            "Accidents"
        )

        locations = pd.read_excel(
            xls,
            "Locations"
        )

        location_cols = [
            "Location_ID",
            "AADT_Volume",
            "Is_Black_Spot",
            "Black_Spot_Name",
            "KM_Marker",
            "Latitude",
            "Longitude",
            "Is_Urban_Road"
        ]

        location_cols = [
            c for c in location_cols
            if c in locations.columns
        ]

        df = accidents.merge(
            locations[location_cols],
            on="Location_ID",
            how="left",
            suffixes=("", "_location")
        )

        # تحويل القيم إلى قيم آمنة للـ JSON
        df = df.where(
            pd.notna(df),
            None
        )

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        records = df.to_dict(
            orient="records"
        )

        return {
            "count": len(records),
            "columns": df.columns.tolist(),
            "data": records
        }

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
    try:
        answer = ask_gemini(body.question)

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


@app.post("/classify-complaint")
def classify_complaint(data: ComplaintClassificationRequest):
    try:
        prompt = f"""
أنت نظام ذكي لتحليل بلاغات حوادث ومخاطر الطرق في مصر.

بيانات البلاغ:

وصف البلاغ:
{data.description}

المحافظة:
{data.governorate}

الطريق:
{data.road_name}

صنف البلاغ إلى تصنيف واحد فقط:

موثوق
غير صحيح
يحتاج مراجعة

القواعد:

موثوق:
إذا كان البلاغ واضحًا ومنطقيًا ويتعلق بحادث أو خطر حقيقي
على الطريق مثل حفرة، حادث، سيارة متعطلة، طريق مغلق،
عمود ساقط، إشارة مرور تالفة، أو خطر مشابه.

غير صحيح:
إذا كان البلاغ لا علاقة له بالطرق أو المرور أو الحوادث،
أو كان واضحًا أنه محتوى عبثي.

يحتاج مراجعة:
إذا كانت المعلومات غير كافية للحكم.

أرجع النتيجة بهذا الشكل فقط:

classification: [موثوق أو غير صحيح أو يحتاج مراجعة]
confidence: [رقم من 0 إلى 100]
reason: [سبب مختصر باللغة العربية]

لا تضف أي معلومات أخرى.
"""

        answer = ask_gemini(prompt)

        text = answer.strip()

        # القيمة الافتراضية
        classification = "يحتاج مراجعة"

        # نحدد التصنيف
        if "غير صحيح" in text:
            classification = "غير صحيح"
        elif "موثوق" in text:
            classification = "موثوق"
        elif "يحتاج مراجعة" in text:
            classification = "يحتاج مراجعة"

        # استخراج confidence
        import re

        confidence = 0

        match = re.search(
            r"confidence\s*:\s*(\d+)",
            text,
            re.IGNORECASE
        )

        if match:
            confidence = int(match.group(1))

        # استخراج السبب
        reason = text

        reason_match = re.search(
            r"reason\s*:\s*(.*)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if reason_match:
            reason = reason_match.group(1).strip()

        return {
            "classification": classification,
            "confidence": confidence,
            "reason": reason
        }

    except Exception as e:

        return {
            "classification": "يحتاج مراجعة",
            "confidence": 0,
            "reason": f"تعذر تحليل البلاغ آليًا: {str(e)}"
        }
