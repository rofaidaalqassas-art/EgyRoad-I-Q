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
from pydantic import BaseModel

from ai_model import CONTROLLABLE_FACTORS, AIModel
from risk_model import RiskModel
from storage import IncidentStore, UserStore

JWT_SECRET = os.environ.get("ROADWISE_JWT_SECRET", "change-this-secret-in-production")
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 12
UPLOADS_DIR = "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="EgyRoad IQ API")
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
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="جلسة الدخول غير صالحة، سجّلي الدخول تانى")
    return payload


def _public_user(user: dict) -> dict:
    return {"name": user["name"], "email": user["email"], "role": user["role"]}


@app.post("/auth/register")
def register(body: RegisterBody, creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if users.get(body.email):
        raise HTTPException(status_code=409, detail="البريد الإلكتروني مسجّل بالفعل")

    if body.role == "citizen":
        if not (body.national_id and body.national_id.isdigit() and len(body.national_id) == 14):
            raise HTTPException(status_code=400, detail="الرقم القومي يجب أن يتكون من 14 رقمًا")
        user = users.create(name=body.name, email=body.email, password=body.password, role="citizen", national_id=body.national_id)
    elif body.role in ("decision", "admin"):
        current = get_current_user(creds)
        if current["role"] not in ("decision", "admin"):
            raise HTTPException(status_code=403, detail="مسموح فقط لأعضاء لوحة متخذ القرار بإضافة أعضاء جدد")
        user = users.create(name=body.name, email=body.email, password=body.password, role=body.role, national_id=None)
    else:
        raise HTTPException(status_code=400, detail="نوع الحساب غير معروف")

    token = make_token(user)
    return {"access_token": token, "user": _public_user(user)}


@app.post("/auth/login")
def login(body: LoginBody):
    user = users.get(body.email)
    if not user or not users.verify_password(user, body.password):
        raise HTTPException(status_code=401, detail="البريد الإلكتروني أو كلمة المرور غير صحيحة")
    if user["role"] != body.role:
        raise HTTPException(status_code=403, detail="هذا الحساب غير مسجّل بهذه الصلاحية")
    token = make_token(user)
    return {"access_token": token, "user": _public_user(user)}


@app.post("/predict")
def predict(body: PredictBody):
    return risk_model.predict_trip(body.governorates, body.hour_24, body.weather)


@app.post("/what-if")
def what_if(body: WhatIfBody, user=Depends(get_current_user)):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")
    try:
        return risk_model.what_if(body.road_name, body.factor, body.improvement_pct)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/dashboard-stats")
def dashboard_stats(user=Depends(get_current_user)):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")
    return risk_model.get_dashboard_stats()


@app.get("/roads")
def roads(governorate: Optional[str] = None, user=Depends(get_current_user)):
    if user["role"] not in ("decision", "admin"):
        raise HTTPException(status_code=403, detail="مسموح فقط لحسابات متخذي القرار")
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
    if image is not None and image.filename:
        ext = os.path.splitext(image.filename)[1] or ".jpg"
        image_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex}{ext}")
        with open(image_path, "wb") as f:
            f.write(await image.read())

    record = incidents.create(
        governorate=governorate, road_name=road_name, description=description,
        image_path=image_path, reported_by=user["sub"],
    )
    return {"incident_id": record["id"], "report_status": record["status"]}


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
    return model.what_if(body.scenario, body.changes)


@app.get("/ai/controllable-factors")
def ai_controllable_factors():
    return [
        {"field": field, "label": meta["label"], "safe_value": meta["safe_value"], "action": meta["action"]}
        for field, meta in CONTROLLABLE_FACTORS.items()
    ]


@app.get("/ai/feature-importance")
def ai_feature_importance(top: int = 15):
    model = require_ai()
    return model.feature_importance[:top]


@app.get("/health")
def health():
    return {"status": "ok", "ai_ready": AI_READY}


# ----------------------------------------------------
# الأكواد المضافة حديثاً لقراءة ملف الإكسيل وربطه بالواجهات
# ----------------------------------------------------

def get_excel_data():
    try:
        # قراءة ملف الإكسيل المرفق بالمشروع
        df = pd.read_excel("accidents_data.xlsx")
        
        # حماية البيانات: تحويل قيم التاريخ والوقت لنصوص حتى لا تسبب خطأ في الـ JSON للمتصفح
        for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
            df[col] = df[col].astype(str)
            
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": f"تعذر جلب ملف البيانات المطلوبة: {str(e)}"}


@app.get("/api/data")
def read_data():
    """
    Endpoint مخصص لجلب كامل بيانات ملف الإكسيل لربطها بلوحة التحكم
    """
    return get_excel_data()
