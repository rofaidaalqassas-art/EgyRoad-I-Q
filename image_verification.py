# -*- coding: utf-8 -*-

"""
image_verification.py
=====================

مرحلة فحص الصورة (Image Verification)
لبلاغات المواطنين في مشروع EgyRoad IQ.

الوظيفة:
- فحص الصورة باستخدام Gemini Vision.
- تحديد هل الصورة:
    evidence_detected
    hazard_detected
    irrelevant
    needs_review

مهم:
- هذا الملف لا يحكم أبدًا بأن البلاغ "false".
- الصورة وحدها لا تكفي للحكم النهائي على البلاغ.
- في حالة فشل Gemini أو عدم وضوح النتيجة:
    needs_review

نسخة محسنة للسرعة:
- استدعاء Gemini مرة واحدة فقط.
- بدون Retry.
- بدون Double Check.
- JSON فقط.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


# =========================================================
# GEMINI SETTINGS
# =========================================================

VISION_MODEL_NAME = os.environ.get(
    "GEMINI_VISION_MODEL",
    "gemini-3.6-flash",
)


# =========================================================
# ALLOWED STATUSES
# =========================================================

_ALLOWED_STATUSES = {
    "evidence_detected",
    "hazard_detected",
    "irrelevant",
    "needs_review",
}


# =========================================================
# CONFIDENCE SETTINGS
# =========================================================

# أقل ثقة تجعلنا نحول النتيجة إلى needs_review
MIN_CONFIDENCE_NEEDS_REVIEW = 0.40


# =========================================================
# GEMINI CLIENT
# =========================================================

# إنشاء Client مرة واحدة بدل إنشائه مع كل صورة.
try:
    client = genai.Client()
except Exception as e:
    client = None
    print(
        f"[WARN] Image verification client initialization failed: {e}"
    )


# =========================================================
# GEMINI PROMPT
# =========================================================

_VERIFICATION_PROMPT = """
أنت جزء من نظام EgyRoad IQ لفحص بلاغات المواطنين
عن حوادث ومخاطر الطرق في مصر.

مهمتك الوحيدة:

افحص الصورة المرفقة فقط، بدون الاعتماد على أي معلومات
خارج الصورة، وحدد تصنيف الصورة.

التصنيفات المسموح بها فقط:

1. "evidence_detected"

الصورة توضح بشكل واضح:
- حادث تصادم
- سيارة متضررة بسبب حادث
- سيارة منقلبة بسبب حادث مروري
- آثار واضحة لحادث مروري

2. "hazard_detected"

الصورة توضح خطرًا على الطريق بدون ضرورة وجود حادث فعلي، مثل:
- حفرة
- عائق على الطريق
- إشارة مرور تالفة
- عمود أو لوحة تالفة
- غرق أو تجمع مياه خطير
- طريق متضرر
- ازدحام أو وضع مروري خطير

3. "irrelevant"

الصورة ليس لها علاقة واضحة بالطرق أو المرور، مثل:
- طعام
- حيوانات
- صورة شخصية
- صورة عشوائية
- Screenshot غير متعلق بالمرور
- أي محتوى لا علاقة له بالطريق أو حادث أو خطر مروري

4. "needs_review"

استخدم هذا التصنيف عندما:
- الصورة مظلمة جدًا
- الصورة ضبابية
- الصورة بعيدة جدًا
- الصورة لا تسمح بالتأكد
- محتوى الصورة قد يكون متعلقًا بالمرور ولكن الدليل غير كافٍ
- الملف ليس صورة صالحة
- لا تستطيع تحديد التصنيف بثقة كافية

قواعد صارمة:
- إذا كنت غير متأكد، استخدم needs_review.
- لا تستخدم كلمة false إطلاقًا.
- لا تحكم بأن البلاغ كاذب.
- أنت تفحص الصورة فقط.
- يجب أن تكون الإجابة JSON فقط.
- لا تكتب أي شرح خارج JSON.

الشكل المطلوب:

{
    "status": "evidence_detected",
    "confidence": 0.95,
    "reason": "الصورة توضح سيارة متضررة نتيجة حادث تصادم"
}

يجب أن يكون status واحدًا فقط من:
evidence_detected
hazard_detected
irrelevant
needs_review

confidence يجب أن يكون رقمًا عشريًا بين 0 و 1.

reason يجب أن تكون جملة قصيرة باللغة العربية.
"""


# =========================================================
# RESULT CLASS
# =========================================================

@dataclass
class ImageVerificationResult:
    status: str
    confidence: float
    reason: str
    raw_model_output: Optional[str] = field(
        default=None,
        repr=False,
    )
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": round(
                float(self.confidence),
                2,
            ),
            "reason": self.reason,
            "error": self.error,
        }


# =========================================================
# NEEDS REVIEW HELPER
# =========================================================

def _needs_review(
    reason: str,
    raw: Optional[str] = None,
    error: Optional[str] = None,
) -> ImageVerificationResult:

    return ImageVerificationResult(
        status="needs_review",
        confidence=0.0,
        reason=reason,
        raw_model_output=raw,
        error=error,
    )


# =========================================================
# JSON EXTRACTION
# =========================================================

def _extract_json_block(
    text: str,
) -> Optional[Dict[str, Any]]:
    """
    يحاول استخراج JSON من رد Gemini.

    يدعم:
    1. JSON مباشر.
    2. JSON داخل ```json ... ```
    3. JSON وسط نص إضافي.
    """

    if not text:
        return None

    text = text.strip()

    # -----------------------------------------------------
    # 1. إزالة Markdown code fence
    # -----------------------------------------------------

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        text = fenced_match.group(1).strip()

    # -----------------------------------------------------
    # 2. JSON مباشر
    # -----------------------------------------------------

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # -----------------------------------------------------
    # 3. البحث عن JSON object داخل النص
    # -----------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    if end <= start:
        return None

    json_text = text[start:end + 1]

    try:
        parsed = json.loads(json_text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        return None

    return None


# =========================================================
# CALL GEMINI ONCE
# =========================================================

def _call_vision_model_once(
    image_bytes: bytes,
    mime_type: str,
) -> str:
    """
    استدعاء واحد فقط لـ Gemini Vision.

    لا يوجد Retry هنا لأن السرعة مهمة في مسار البلاغ.
    """

    if client is None:
        raise RuntimeError(
            "Gemini client is not initialized."
        )

    response = client.models.generate_content(
        model=VISION_MODEL_NAME,
        contents=[
            _VERIFICATION_PROMPT,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            # كان 256 قبل كده - قليل جدًا لموديل بيرجّع تفكير/تمهيد قبل
            # الـ JSON النهائي، فكان بيقطع الرد ناقص قبل ما يكمل الـ JSON
            # (وده سبب "رد موديل فحص الصور لم يكن بصيغة JSON صالحة" اللي
            # كان بيظهر مع كل صورة تقريبًا - رد موجود لكن مقطوع).
            max_output_tokens=1024,
        ),
    )

    text = getattr(
        response,
        "text",
        None,
    )

    if text is None:
        return ""

    return str(text).strip()


# =========================================================
# PARSE GEMINI RESPONSE
# =========================================================

def _parse_verification_response(
    raw_text: str,
):
    """
    يحول رد Gemini إلى:
        status
        confidence
        reason
    """

    parsed = _extract_json_block(raw_text)

    if not parsed:
        print(
            "[WARN] image verification: "
            "invalid JSON from Gemini. "
            f"raw_text={raw_text!r}"
        )

        return (
            None,
            0.0,
            "رد موديل فحص الصور لم يكن بصيغة JSON صالحة",
        )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    status = str(
        parsed.get(
            "status",
            "",
        )
    ).strip()

    # -----------------------------------------------------
    # REASON
    # -----------------------------------------------------

    reason = str(
        parsed.get(
            "reason",
            "",
        )
    ).strip()

    if not reason:
        reason = "بدون توضيح من الموديل"

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    try:
        confidence = float(
            parsed.get(
                "confidence",
                0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        confidence = 0.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    # -----------------------------------------------------
    # STATUS VALIDATION
    # -----------------------------------------------------

    if status not in _ALLOWED_STATUSES:
        return (
            None,
            confidence,
            (
                "الموديل رجّع تصنيف "
                f"غير معروف: {status!r}"
            ),
        )

    return (
        status,
        confidence,
        reason,
    )


# =========================================================
# MAIN IMAGE VERIFICATION FUNCTION
# =========================================================

def verify_incident_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> ImageVerificationResult:
    """
    فحص صورة بلاغ واحدة باستخدام Gemini Vision.

    النتيجة الممكنة:
        evidence_detected
        hazard_detected
        irrelevant
        needs_review

    لا يوجد false هنا إطلاقًا.

    يستخدم استدعاء Gemini واحد فقط.
    """

    # -----------------------------------------------------
    # 1. التأكد من وجود صورة
    # -----------------------------------------------------

    if not image_bytes:
        return _needs_review(
            "لم يتم استلام أي بيانات صورة صالحة للفحص"
        )

    # -----------------------------------------------------
    # 2. التأكد من Gemini Client
    # -----------------------------------------------------

    if client is None:
        return _needs_review(
            "تعذر تهيئة خدمة فحص الصور بالذكاء الاصطناعي"
        )

    # -----------------------------------------------------
    # 3. فحص الصورة
    # -----------------------------------------------------

    try:
        print(
            "[IMAGE AI] Starting image verification..."
        )

        raw_text = _call_vision_model_once(
            image_bytes,
            mime_type,
        )

        print(
            "[IMAGE AI] Gemini response received."
        )

    except Exception as e:
        print(
            "[WARN] image verification failed: "
            f"{type(e).__name__}: {e}"
        )

        return _needs_review(
            "تعذر فحص الصورة آليًا حاليًا، "
            "ويحتاج البلاغ مراجعة.",
            error=str(e),
        )

    # -----------------------------------------------------
    # 4. رد فارغ
    # -----------------------------------------------------

    if not raw_text:
        return _needs_review(
            "لم يرجع موديل فحص الصور نتيجة."
        )

    # -----------------------------------------------------
    # 5. تحليل JSON
    # -----------------------------------------------------

    (
        status,
        confidence,
        reason,
    ) = _parse_verification_response(
        raw_text
    )

    if status is None:
        return _needs_review(
            reason,
            raw=raw_text,
        )

    # -----------------------------------------------------
    # 6. LOW CONFIDENCE
    # -----------------------------------------------------

    if (
        confidence < MIN_CONFIDENCE_NEEDS_REVIEW
        and status != "needs_review"
    ):
        return ImageVerificationResult(
            status="needs_review",
            confidence=confidence,
            reason=(
                "ثقة الموديل منخفضة "
                f"({confidence:.2f}) "
                f"على تصنيف '{status}'، "
                "ويحتاج البلاغ مراجعة بشرية"
            ),
            raw_model_output=raw_text,
        )

    # -----------------------------------------------------
    # 7. النتيجة النهائية
    # -----------------------------------------------------

    result = ImageVerificationResult(
        status=status,
        confidence=confidence,
        reason=reason,
        raw_model_output=raw_text,
    )

    print(
        "[IMAGE AI] "
        f"status={status} "
        f"confidence={confidence:.2f}"
    )

    return result


# =========================================================
# IMAGE REPORT FLAG
# =========================================================

def image_report_flag(
    result: ImageVerificationResult,
) -> Dict[str, Any]:
    """
    تحويل نتيجة فحص الصورة إلى Flag
    يمكن تخزينه مع البلاغ.

    لا يوجد هنا أي حكم نهائي بأن البلاغ false.
    """

    label_map = {
        "evidence_detected":
            "Evidence detected",

        "hazard_detected":
            "Road hazard detected",

        "irrelevant":
            "Irrelevant",

        "needs_review":
            "Needs Review",
    }

    return {
        "image_status":
            result.status,

        "image_label":
            label_map.get(
                result.status,
                "Needs Review",
            ),

        "image_confidence":
            round(
                float(
                    result.confidence
                ),
                2,
            ),

        "image_reason":
            result.reason,

        "image_error":
            result.error,
    }