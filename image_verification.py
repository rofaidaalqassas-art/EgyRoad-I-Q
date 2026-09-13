"""
image_verification.py
======================
مرحلة "فحص الصورة" (Image Verification) لبلاغات المواطنين في EgyRoad IQ.

الفكرة:
    قبل ما نعتمد أي بلاغ، لازم نتأكد أول حاجة إن الصورة المرفقة فعلاً
    بتوثّق حادث/خطر مروري، مش صورة عشوائية. ده منفصل تمامًا عن تصنيف
    نص البلاغ نفسه، والقرار ده بيتحدد هنا فقط بناءً على محتوى الصورة.

القرار الممكن للصورة (image_status):
    - "evidence_detected"  -> صورة واضحة لحادث تصادم / سيارة متضررة
    - "hazard_detected"    -> صورة لخطر على الطريق (عائق، حفرة، إشارة تالفة...)
    - "irrelevant"         -> الصورة مالهاش علاقة بالمرور إطلاقًا
    - "needs_review"       -> الصورة مش واضحة / الدليل غير كافٍ / فشل الفحص

قاعدة أساسية (مهمة جدًا):
    ممنوع تمامًا إن الكود يوصف الصورة بـ "false" (بلاغ كاذب) لمجرد إنها
    مش بتثبت الحادث. أقصى حكم سلبي ممكن نطلعه من فحص الصورة وحده هو
    "irrelevant" أو "needs_review". أي حكم بـ "بلاغ كاذب" لازم ياخد في
    الاعتبار عناصر تانية (نص البلاغ، الموقع، بلاغات مضادة...) مش الصورة
    لوحدها، ولذلك مفيش دالة هنا بترجع "false" على الإطلاق.

ملاحظة توافق:
    ده بيستخدم نفس مكتبة google-genai ونفس أسلوب genai.Client() المستخدم
    في ai_agent.py بالظبط (client = genai.Client())، فمفتاح الـ API بياخده
    تلقائيًا من نفس متغير البيئة اللي شغالة بيه بالفعل - مفيش حاجة تتغيّر
    في إعدادات البيئة عندك.

لا بيانات وهمية:
    كل نتيجة هنا ناتجة عن استدعاء فعلي لموديل Gemini متعدد الوسائط
    (نفس الموديل المستخدم في ai_agent.py افتراضيًا، أو موديل يدعم رؤية
    الصور لو حددتِ واحد مختلف). لو مفيش مفتاح API، أو الاستدعاء فشل، أو
    رد الموديل مش JSON صالح -> الدالة ترجع "needs_review" بشكل صريح مع
    توضيح السبب، وأبدًا مش بتخترع نتيجة.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ----------------------------------------------------
# إعداد موديل Gemini - بنفس أسلوب ai_agent.py بالظبط
# ----------------------------------------------------
# ai_agent.py بيستخدم: from google import genai / client = genai.Client()
# / client.models.generate_content(model="gemini-3.6-flash", contents=...)
# فبنفس الطريقة هنا، عشان يشتغلوا بنفس المفتاح المُعد بالفعل في البيئة.
#
# ملحوظة: "gemini-3.6-flash" في ai_agent.py مستخدم لنص فقط. لو الموديل ده
# مش بيدعم فهم الصور عندك، غيّري القيمة في متغير البيئة GEMINI_VISION_MODEL
# لاسم موديل يدعم الرؤية (Vision) صراحة.
VISION_MODEL_NAME = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.6-flash")

_ALLOWED_STATUSES = {
    "evidence_detected",
    "hazard_detected",
    "irrelevant",
    "needs_review",
}

_VERIFICATION_PROMPT = """
انت جزء من نظام EgyRoad IQ لفحص بلاغات المواطنين عن حوادث/مخاطر الطرق في مصر.
مهمتك الوحيدة: افحص الصورة المرفقة فقط (من غير أي معلومات تانية) وحدد تصنيفها.

التصنيفات المسموح بيها فقط (لازم ترجع واحد منهم بالظبط في الحقل status):
- "evidence_detected": الصورة بتوضح بشكل واضح حادث تصادم / سيارة متضررة أو منقلبة
  بسبب حادث مروري.
- "hazard_detected": الصورة بتوضح خطر على الطريق بدون حادث فعلي (عائق، حفرة كبيرة،
  إشارة/عمود تالف، غرق شارع، ازدحام خطير...).
- "irrelevant": الصورة مالهاش أي علاقة بالمرور أو الطرق (طعام، أشخاص، حيوانات،
  screenshot غير متعلق، صورة عشوائية...).
- "needs_review": الصورة غير واضحة (مظلمة/ضبابية/بعيدة جدًا)، أو ممكن تكون متعلقة
  بالمرور لكن مش قادر تتأكد بثقة كافية، أو الملف مش صورة صالحة أصلاً.

قواعد صارمة:
1. لو مش متأكد بثقة كافية -> استخدم "needs_review" ولا تخمّن.
2. لا تستخدم أبدًا كلمة "false" أو تحكم إن البلاغ كاذب - مش من مهمتك، فقط صنّف الصورة.
3. رجّع إجابتك بصيغة JSON فقط بدون أي نص إضافي قبله أو بعده، بالشكل التالي بالظبط:

{
  "status": "<one of: evidence_detected, hazard_detected, irrelevant, needs_review>",
  "confidence": <رقم عشري بين 0 و 1>,
  "reason": "<جملة قصيرة بالعربي توضح سبب القرار>"
}
"""


@dataclass
class ImageVerificationResult:
    status: str
    confidence: float
    reason: str
    raw_model_output: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "error": self.error,
        }


def _needs_review(reason: str, raw: Optional[str] = None, error: Optional[str] = None) -> ImageVerificationResult:
    """نقطة رجوع موحّدة: أي حالة غير متأكدة أو فشل -> needs_review بدل بيانات ملفّقة."""
    return ImageVerificationResult(
        status="needs_review",
        confidence=0.0,
        reason=reason,
        raw_model_output=raw,
        error=error,
    )


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """يحاول يقرأ JSON من رد الموديل حتى لو حاطه جوه ```json ... ``` أو مع نص إضافي."""
    text = text.strip()

    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def verify_incident_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> ImageVerificationResult:
    """
    يفحص صورة بلاغ واحدة عن طريق نفس عميل Gemini المستخدم في ai_agent.py،
    ويرجع ImageVerificationResult بدون أي بيانات وهمية.

    Parameters
    ----------
    image_bytes: محتوى الصورة الخام (bytes)
    mime_type:   نوع الملف، مثال "image/jpeg" أو "image/png"
    """

    if not image_bytes:
        return _needs_review("لم يتم استلام أي بيانات صورة صالحة للفحص")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _needs_review(
            "مكتبة google-genai غير مثبتة (pip install google-genai)"
        )

    try:
        # نفس أسلوب ai_agent.py بالظبط: genai.Client() بياخد المفتاح
        # تلقائيًا من متغيرات البيئة المُعدة بالفعل عندك.
        client = genai.Client()

        response = client.models.generate_content(
            model=VISION_MODEL_NAME,
            contents=[
                _VERIFICATION_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
            ),
        )

        raw_text = (response.text or "").strip()

    except Exception as e:  # أي خطأ شبكة/API لا يتحول أبدًا لنتيجة ملفّقة
        return _needs_review(
            "تعذر الاتصال بموديل فحص الصور، يحتاج البلاغ مراجعة يدويًا",
            error=str(e),
        )

    parsed = _extract_json_block(raw_text)

    if not parsed:
        return _needs_review(
            "رد موديل فحص الصور لم يكن بصيغة JSON صالحة",
            raw=raw_text,
        )

    status = str(parsed.get("status", "")).strip()
    reason = str(parsed.get("reason", "")).strip() or "بدون توضيح من الموديل"

    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if status not in _ALLOWED_STATUSES:
        return _needs_review(
            f"الموديل رجّع تصنيف غير معروف ({status!r})، يحتاج مراجعة يدويًا",
            raw=raw_text,
        )

    if confidence < 0.4 and status != "needs_review":
        return ImageVerificationResult(
            status="needs_review",
            confidence=confidence,
            reason=f"ثقة الموديل منخفضة ({confidence:.2f}) على تصنيف '{status}', يحتاج مراجعة بشرية",
            raw_model_output=raw_text,
        )

    return ImageVerificationResult(
        status=status,
        confidence=confidence,
        reason=reason,
        raw_model_output=raw_text,
    )


# ----------------------------------------------------
# دمج نتيجة الصورة مع باقي بيانات البلاغ لإصدار قرار أوّلي
# ----------------------------------------------------
# ملاحظة: هذه الدالة تحدد فقط "قرار المرحلة الأولى" الخاص بمرفق الصورة
# ضمن البلاغ. القرار النهائي للبلاغ (Valid / Needs Review / False) لازم
# يدمج كمان نتيجة تصنيف نص البلاغ نفسه (لو موجود عندكم موديل نص منفصل)
# + بيانات الموقع/الوقت. الدمج الكامل مع تصنيف النص غير موجود في main.py
# الحالي، فلو حابة نضيفه محتاجين كود موديل تصنيف النص عندكم.

def image_report_flag(result: ImageVerificationResult) -> Dict[str, Any]:
    """
    يحوّل نتيجة فحص الصورة إلى Flag واضح يُرفق مع سجل البلاغ في التخزين،
    بدون أي حكم نهائي بـ "false" - القرار النهائي مسؤولية مرحلة تانية.
    """

    label_map = {
        "evidence_detected": "✅ Evidence detected",
        "hazard_detected": "⚠️ Road hazard detected",
        "irrelevant": "❌ Irrelevant",
        "needs_review": "🔎 Needs Review",
    }

    return {
        "image_status": result.status,
        "image_label": label_map.get(result.status, "🔎 Needs Review"),
        "image_confidence": round(result.confidence, 2),
        "image_reason": result.reason,
    }
