
# -*- coding: utf-8 -*-

"""
image_verification.py
=====================

مرحلة فحص الصورة (Image Verification) لبلاغات المواطنين
في مشروع EgyRoad IQ.

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

المتطلبات:
    pip install google-genai
"""

from __future__ import annotations

import json
import os
import re
import time

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
# VERIFICATION SETTINGS
# =========================================================

MAX_RETRIES = int(
    os.environ.get(
        "IMAGE_VERIFY_MAX_RETRIES",
        "2",
    )
)

RETRY_DELAY_SECONDS = float(
    os.environ.get(
        "IMAGE_VERIFY_RETRY_DELAY",
        "1.5",
    )
)


# أقل ثقة تجعلنا نحتاج مراجعة بشرية
MIN_CONFIDENCE_NEEDS_REVIEW = 0.40


# أقل ثقة لقبول evidence/hazard مباشرة
MIN_CONFIDENCE_CONFIRMED = 0.65


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
- يجب أن تكون الإجابة JSON فقط بدون أي كلام إضافي.

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
    # إزالة Markdown code fence
    # -----------------------------------------------------

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:

        text = fenced_match.group(1).strip()


    # -----------------------------------------------------
    # محاولة JSON مباشر
    # -----------------------------------------------------

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError:

        pass


    # -----------------------------------------------------
    # البحث عن أول JSON object
    # -----------------------------------------------------

    start = text.find("{")

    end = text.rfind("}")


    if start == -1 or end == -1:

        return None


    if end <= start:

        return None


    json_text = text[
        start:end + 1
    ]


    try:

        parsed = json.loads(
            json_text
        )

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
    استدعاء واحد فعلي لـ Gemini Vision.
    """

    from google import genai
    from google.genai import types


    client = genai.Client()


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
            # (وده بالظبط سبب "رد موديل فحص الصور لم يكن بصيغة JSON صالحة"
            # اللي بيظهر مع كل صورة تقريبًا - رد موجود لكن مقطوع).
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
# GEMINI WITH RETRY
# =========================================================

def _call_vision_model_with_retry(
    image_bytes: bytes,
    mime_type: str,
):

    """
    استدعاء Gemini مع Retry.

    يرجع:

        (raw_text, error)

    """

    last_error = None


    total_attempts = (
        MAX_RETRIES + 1
    )


    for attempt in range(
        1,
        total_attempts + 1,
    ):

        try:

            raw_text = (
                _call_vision_model_once(
                    image_bytes,
                    mime_type,
                )
            )


            if raw_text:

                return (
                    raw_text,
                    None,
                )


            last_error = (
                "رد فارغ من موديل Gemini"
            )


        except Exception as e:

            last_error = str(e)


        # -------------------------------------------------
        # Retry
        # -------------------------------------------------

        if attempt < total_attempts:

            delay = (
                RETRY_DELAY_SECONDS
                * attempt
            )

            time.sleep(
                delay
            )


    # نطبع سبب الفشل النهائي فى الـ logs (Render → Logs) عشان نعرف بالظبط
    # ليه كل المحاولات فشلت - استثناء API؟ مفتاح غير صحيح؟ Timeout؟ - بدل
    # ما نعرف بس إنها فشلت من غير أي تفاصيل.
    print(
        f"[WARN] image verification: all {total_attempts} attempts failed. "
        f"last_error={last_error!r}"
    )

    return (
        None,
        last_error,
    )


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

    parsed = _extract_json_block(
        raw_text
    )


    if not parsed:

        # نطبع رد Gemini الخام فى الـ logs (يظهر فى Render → Logs) عشان
        # لو الرد لسه بيفشل بعد رفع max_output_tokens، نقدر نشوف بالظبط
        # الموديل رجّع إيه بدل ما نخمّن السبب.
        print(
            "[WARN] image verification: invalid JSON from Gemini. "
            f"raw_text={raw_text!r}"
        )

        return (
            None,
            0.0,
            "رد موديل فحص الصور لم يكن بصيغة JSON صالحة",
        )


    status = str(
        parsed.get(
            "status",
            "",
        )
    ).strip()


    reason = str(
        parsed.get(
            "reason",
            "",
        )
    ).strip()


    if not reason:

        reason = (
            "بدون توضيح من الموديل"
        )


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
    """


    # -----------------------------------------------------
    # 1. التأكد من وجود صورة
    # -----------------------------------------------------

    if not image_bytes:

        return _needs_review(
            "لم يتم استلام أي بيانات صورة صالحة للفحص"
        )


    # -----------------------------------------------------
    # 2. التأكد من Google GenAI
    # -----------------------------------------------------

    try:

        from google import genai  # noqa: F401

    except ImportError:

        return _needs_review(

            "مكتبة google-genai غير مثبتة. "
            "شغّلي: pip install google-genai"
        )


    # -----------------------------------------------------
    # 3. الاتصال بـ Gemini
    # -----------------------------------------------------

    raw_text, error = (
        _call_vision_model_with_retry(
            image_bytes,
            mime_type,
        )
    )


    if raw_text is None:

        return _needs_review(

            "تعذر الاتصال بموديل فحص الصور "
            "بعد عدة محاولات، يحتاج البلاغ مراجعة يدويًا",

            error=error,
        )


    # -----------------------------------------------------
    # 4. تحليل JSON
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
    # 5. ثقة منخفضة
    # -----------------------------------------------------

    if (

        confidence
        < MIN_CONFIDENCE_NEEDS_REVIEW

        and status
        != "needs_review"

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
    # 6. DOUBLE CHECK
    # -----------------------------------------------------

    if (

        status
        in (
            "evidence_detected",
            "hazard_detected",
        )

        and confidence
        < MIN_CONFIDENCE_CONFIRMED

    ):

        (
            second_raw,
            second_error,
        ) = (
            _call_vision_model_with_retry(
                image_bytes,
                mime_type,
            )
        )


        # -------------------------------------------------
        # فشل الفحص الثاني
        # -------------------------------------------------

        if second_raw is None:

            return ImageVerificationResult(

                status="needs_review",

                confidence=confidence,

                reason=(

                    f"التصنيف الأولي '{status}' "
                    f"بثقة {confidence:.2f}، "
                    "لكن فحص التأكيد الثاني فشل، "
                    "لذلك يحتاج البلاغ مراجعة بشرية"

                ),

                raw_model_output=raw_text,

                error=second_error,
            )


        # -------------------------------------------------
        # تحليل الفحص الثاني
        # -------------------------------------------------

        (
            second_status,
            second_confidence,
            second_reason,
        ) = _parse_verification_response(
            second_raw
        )


        # -------------------------------------------------
        # الفحص الثاني غير صالح
        # -------------------------------------------------

        if second_status is None:

            return ImageVerificationResult(

                status="needs_review",

                confidence=min(
                    confidence,
                    second_confidence,
                ),

                reason=(

                    "تعذر تأكيد التصنيف "
                    "في الفحص الثاني"

                ),

                raw_model_output=(

                    f"{raw_text}\n"
                    "---second_pass---\n"
                    f"{second_raw}"

                ),
            )


        # -------------------------------------------------
        # هل الفحصان متفقان؟
        # -------------------------------------------------

        agrees = (

            second_status == status

            and

            second_confidence
            >= MIN_CONFIDENCE_NEEDS_REVIEW

        )


        if not agrees:

            return ImageVerificationResult(

                status="needs_review",

                confidence=min(

                    confidence,

                    second_confidence,

                ),

                reason=(

                    f"الفحص الأول رجّع "
                    f"'{status}' "
                    f"({confidence:.2f})، "

                    f"والفحص الثاني رجّع "
                    f"'{second_status}' "
                    f"({second_confidence:.2f})، "

                    "لذلك النتيجة تحتاج مراجعة بشرية"

                ),

                raw_model_output=(

                    f"{raw_text}\n"
                    "---second_pass---\n"
                    f"{second_raw}"

                ),
            )


        # -------------------------------------------------
        # الفحصان متفقان
        # -------------------------------------------------

        confidence = (
            second_confidence
        )

        reason = (
            second_reason
        )

        raw_text = (

            f"{raw_text}\n"
            "---second_pass_confirmed---\n"
            f"{second_raw}"

        )


    # -----------------------------------------------------
    # 7. النتيجة النهائية
    # -----------------------------------------------------

    return ImageVerificationResult(

        status=status,

        confidence=confidence,

        reason=reason,

        raw_model_output=raw_text,
    )


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
