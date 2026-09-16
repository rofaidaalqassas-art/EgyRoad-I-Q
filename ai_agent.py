# -*- coding: utf-8 -*-

"""
ai_agent.py
===========

ROADWISE / EgyRoad IQ AI Assistant

الوظائف:
- الإجابة على أسئلة ROADWISE باستخدام البيانات المحلية.
- استخدام Gemini فقط لصياغة الإجابات التي تحتاج AI.
- تصنيف بلاغات المواطنين.
- لا يتم استخدام بيانات ROADWISE عند تصنيف البلاغ.
- تصنيف البلاغ يتم باستدعاء Gemini واحد فقط.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from google import genai


# =========================================================
# SETTINGS
# =========================================================

DATA_FILE = "accidents_data.xlsx"

# إجابة مساعد ROADWISE العادي:
# محاولة أولى + إعادة محاولة واحدة فقط عند الحاجة.
MAX_RETRIES = 1
RETRY_DELAY_SECONDS = 0.5

NUMBER_MATCH_TOLERANCE_RATIO = 0.01

client = genai.Client()


# =========================================================
# GOVERNORATE MAP
# =========================================================

GOVERNORATE_AR_TO_EN = {
    "الإسكندرية": "Alexandria",
    "اسكندرية": "Alexandria",
    "أسوان": "Aswan",
    "اسوان": "Aswan",
    "أسيوط": "Asyut",
    "اسيوط": "Asyut",
    "البحيرة": "Beheira",
    "بني سويف": "Beni Suef",
    "القاهرة": "Cairo",
    "الدقهلية": "Dakahlia",
    "دمياط": "Damietta",
    "الفيوم": "Faiyum",
    "الغربية": "Gharbiya",
    "الجيزة": "Giza",
    "الإسماعيلية": "Ismailia",
    "الاسماعيلية": "Ismailia",
    "كفر الشيخ": "Kafr El Sheikh",
    "الأقصر": "Luxor",
    "الاقصر": "Luxor",
    "مطروح": "Matrouh",
    "المنوفية": "Menoufia",
    "المنيا": "Minya",
    "الوادي الجديد": "New Valley",
    "شمال سيناء": "North Sinai",
    "بورسعيد": "Port Said",
    "بور سعيد": "Port Said",
    "القليوبية": "Qalyubia",
    "قنا": "Qena",
    "البحر الأحمر": "Red Sea",
    "الشرقية": "Sharkia",
    "سوهاج": "Sohag",
    "جنوب سيناء": "South Sinai",
    "السويس": "Suez",
}


# =========================================================
# ROADWISE CACHE
# =========================================================

@dataclass
class RoadwiseCache:
    df: pd.DataFrame
    kpi_summary: str
    governorate_counts: pd.Series
    road_counts: pd.Series
    cause_counts: pd.Series
    precomputed_answers: dict = field(default_factory=dict)


def load_roadwise_cache(
    data_file: str = DATA_FILE,
) -> RoadwiseCache:
    """
    تحميل بيانات ROADWISE مرة واحدة عند تشغيل السيرفر.
    """

    needed_cols = (
        "Governorate_EN",
        "Highway_Name",
        "Cause_Category",
        "Fatalities_Count",
        "Injuries_Count",
        "Total_Economic_Loss_EGP",
    )

    df = pd.read_excel(
        data_file,
        usecols=lambda c: c in needed_cols,
    )

    governorate_counts = df["Governorate_EN"].value_counts()
    road_counts = df["Highway_Name"].value_counts()
    cause_counts = df["Cause_Category"].value_counts()

    # كل صف في شيت Accidents هو حادثة واحدة - نفس المنطق اللي باقي
    # المشروع كله (data_prep.py, /api/data, لوحة القيادة) بيعتمد عليه.
    # لازم نحسبه هنا قبل ما نقص df للأعمدة الأربعة بس تحت.
    total_accidents = len(df)

    total_fatalities = int(
        pd.to_numeric(
            df["Fatalities_Count"],
            errors="coerce",
        ).fillna(0).sum()
    )

    total_injuries = int(
        pd.to_numeric(
            df["Injuries_Count"],
            errors="coerce",
        ).fillna(0).sum()
    )

    total_economic_loss = float(
        pd.to_numeric(
            df["Total_Economic_Loss_EGP"],
            errors="coerce",
        ).fillna(0).sum()
    )

    # نحتفظ فقط بالأعمدة المطلوبة للأسئلة اللاحقة.
    df = df[
        [
            "Governorate_EN",
            "Highway_Name",
            "Fatalities_Count",
            "Injuries_Count",
        ]
    ]

    # كل صف فى شيت Accidents هو حادثة واحدة بالفعل - محسوبة فوق قبل ما
    # نقص df، فـ "عدد الحوادث" رقم موثوق ومتّسق مع باقي المشروع كله.

    kpi_summary = (
        "ROADWISE KPI SUMMARY\n"
        f"Total Accidents: {total_accidents}\n"
        f"Total Fatalities: {total_fatalities}\n"
        f"Total Injuries: {total_injuries}\n"
        f"Total Economic Loss: {total_economic_loss} EGP\n"
    )

    most_dangerous_gov = (
        governorate_counts.index[0]
        if len(governorate_counts)
        else None
    )

    most_dangerous_road = (
        road_counts.index[0]
        if len(road_counts)
        else None
    )

    precomputed_answers = {
        "most_dangerous_governorate": (
            f"أكثر محافظة من حيث عدد الحوادث هي "
            f"{most_dangerous_gov} بعدد "
            f"{int(governorate_counts.iloc[0])} حادثة."
        )
        if most_dangerous_gov is not None
        else "البيانات غير كافية لتحديد المحافظة الأعلى.",

        "most_dangerous_road": (
            f"أكثر طريق من حيث عدد الحوادث هو "
            f"{most_dangerous_road} بعدد "
            f"{int(road_counts.iloc[0])} حادثة."
        )
        if most_dangerous_road is not None
        else "البيانات غير كافية لتحديد الطريق الأعلى.",

        "total_accidents": (
            f"إجمالي عدد الحوادث المسجلة هو "
            f"{total_accidents} حادثة."
        ),

        "total_fatalities": (
            f"إجمالي عدد الوفيات المسجلة هو "
            f"{total_fatalities} حالة وفاة."
        ),

        "total_injuries": (
            f"إجمالي عدد الإصابات المسجلة هو "
            f"{total_injuries} إصابة."
        ),

        "total_economic_loss": (
            f"إجمالي الخسائر الاقتصادية المسجلة هو "
            f"{total_economic_loss} جنيه مصري."
        ),
    }

    return RoadwiseCache(
        df=df,
        kpi_summary=kpi_summary,
        governorate_counts=governorate_counts,
        road_counts=road_counts,
        cause_counts=cause_counts,
        precomputed_answers=precomputed_answers,
    )


# =========================================================
# PRECOMPUTED QUESTIONS
# =========================================================

_PRECOMPUTED_PATTERNS = [
    (
        (
            "أخطر محافظ",
            "اكثر محافظ",
            "أكثر محافظ",
        ),
        "most_dangerous_governorate",
    ),
    (
        (
            "أخطر طريق",
            "اكثر طريق",
            "أكثر طريق",
            "أخطر الطرق",
            "أكثر الطرق",
        ),
        "most_dangerous_road",
    ),
    (
        (
            "عدد الوفيات",
            "اجمالي الوفيات",
            "إجمالي الوفيات",
            "كام حالة وفاة",
        ),
        "total_fatalities",
    ),
    (
        (
            "عدد الإصابات",
            "عدد الاصابات",
            "اجمالي الاصابات",
            "إجمالي الإصابات",
        ),
        "total_injuries",
    ),
    (
        (
            "عدد الحوادث",
            "اجمالي الحوادث",
            "إجمالي الحوادث",
            "كام حادثة",
        ),
        "total_accidents",
    ),
    (
        (
            "خسائر اقتصادية",
            "الخساير الاقتصادية",
            "التكلفة الاقتصادية",
        ),
        "total_economic_loss",
    ),
]


def _match_precomputed(
    question: str,
    cache: RoadwiseCache,
) -> Optional[str]:

    if _find_mentioned_governorate(
        question,
        cache,
    ) is not None:
        return None

    if _find_mentioned_road(
        question,
        cache,
    ) is not None:
        return None

    for keywords, key in _PRECOMPUTED_PATTERNS:
        if any(
            keyword in question
            for keyword in keywords
        ):
            return cache.precomputed_answers.get(key)

    return None


# =========================================================
# FIND GOVERNORATE
# =========================================================

def _find_mentioned_governorate(
    question: str,
    cache: RoadwiseCache,
) -> Optional[str]:

    # العربي أولًا
    for name_ar, name_en in GOVERNORATE_AR_TO_EN.items():

        if (
            name_ar in question
            and name_en in cache.governorate_counts.index
        ):
            return name_en

    # الإنجليزي
    for governorate in cache.governorate_counts.index:

        governorate_text = str(governorate)

        if (
            governorate_text
            and governorate_text.lower()
            in question.lower()
        ):
            return governorate

    return None


# =========================================================
# FIND ROAD
# =========================================================

def _find_mentioned_road(
    question: str,
    cache: RoadwiseCache,
) -> Optional[str]:

    question_lower = question.lower()

    for road in cache.road_counts.index:

        road_text = str(road)

        if (
            road_text
            and road_text.lower()
            in question_lower
        ):
            return road

    return None


# =========================================================
# BUILD RELEVANT DATA
# =========================================================

def build_relevant_slice(
    question: str,
    cache: RoadwiseCache,
) -> str:

    parts = [
        cache.kpi_summary
    ]

    governorate = _find_mentioned_governorate(
        question,
        cache,
    )

    if governorate is not None:

        gov_df = cache.df[
            cache.df["Governorate_EN"]
            == governorate
        ]

        parts.append(
            f"\nGovernorate '{governorate}' detail:\n"
            f"Accidents: {len(gov_df)}\n"
            f"Fatalities: "
            f"{int(pd.to_numeric(gov_df['Fatalities_Count'], errors='coerce').fillna(0).sum())}\n"
            f"Injuries: "
            f"{int(pd.to_numeric(gov_df['Injuries_Count'], errors='coerce').fillna(0).sum())}\n"
        )

    road = _find_mentioned_road(
        question,
        cache,
    )

    if road is not None:

        road_df = cache.df[
            cache.df["Highway_Name"]
            == road
        ]

        parts.append(
            f"\nRoad '{road}' detail:\n"
            f"Accidents: {len(road_df)}\n"
            f"Fatalities: "
            f"{int(pd.to_numeric(road_df['Fatalities_Count'], errors='coerce').fillna(0).sum())}\n"
            f"Injuries: "
            f"{int(pd.to_numeric(road_df['Injuries_Count'], errors='coerce').fillna(0).sum())}\n"
        )

    if governorate is None and road is None:

        parts.append(
            "\nTop 5 Governorates:\n"
            + cache.governorate_counts
            .head(5)
            .to_string()
        )

        parts.append(
            "\nTop 5 Causes:\n"
            + cache.cause_counts
            .head(5)
            .to_string()
        )

    result = "\n".join(parts)

    print(
        f"[DEBUG] build_relevant_slice "
        f"question={question!r} "
        f"governorate={governorate!r} "
        f"road={road!r}"
    )

    return result


# =========================================================
# GEMINI CALL
# =========================================================

def _call_gemini_with_retry(
    prompt: str,
):
    """
    استدعاء Gemini لمساعد ROADWISE.

    يستخدم retry محدود فقط للأسئلة العادية.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 2,
    ):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )

            text = (
                response.text or ""
            ).strip()

            if text:
                return text, None

            last_error = "رد فارغ من الموديل"

        except Exception as e:

            last_error = str(e)

        if attempt <= MAX_RETRIES:

            time.sleep(
                RETRY_DELAY_SECONDS * attempt
            )

    print(
        "[WARN] ai_agent: Gemini call failed "
        f"after retries. "
        f"last_error={last_error!r}"
    )

    return None, last_error


# =========================================================
# NUMBER EXTRACTION
# =========================================================

def _extract_numbers(text: str):

    if not text:
        return []

    # يدعم:
    # 123
    # 1,234
    # 123.45
    # 1,234.56

    raw_matches = re.findall(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        text,
    )

    numbers = []

    for match in raw_matches:

        cleaned = match.replace(",", "")

        try:
            numbers.append(
                float(cleaned)
            )
        except ValueError:
            continue

    return numbers


def _find_unverified_numbers(
    answer_text: str,
    roadwise_data: str,
):

    data_numbers = _extract_numbers(
        roadwise_data
    )

    answer_numbers = _extract_numbers(
        answer_text
    )

    unverified = []

    for num in answer_numbers:

        if num < 10:
            continue

        found_match = any(
            abs(num - data_num)
            <= max(
                1.0,
                data_num
                * NUMBER_MATCH_TOLERANCE_RATIO,
            )
            for data_num in data_numbers
        )

        if not found_match:
            unverified.append(num)

    return unverified


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(
    question: str,
    cache: RoadwiseCache,
) -> str:

    # -----------------------------------------------------
    # 1. إجابات مباشرة
    # -----------------------------------------------------

    precomputed = _match_precomputed(
        question,
        cache,
    )

    if precomputed:
        return precomputed

    # -----------------------------------------------------
    # 2. البيانات المرتبطة بالسؤال فقط
    # -----------------------------------------------------

    data_slice = build_relevant_slice(
        question,
        cache,
    )

    # -----------------------------------------------------
    # 3. AI لصياغة الإجابة
    # -----------------------------------------------------

    prompt = f"""
You are ROADWISE AI Assistant.

You are an AI assistant for road safety in Egypt.

Answer the user's question using ONLY the ROADWISE
data provided below.

Do not invent statistics.

If the requested information is not available,
clearly say that it is not available.

ROADWISE DATA:

{data_slice}

USER QUESTION:

{question}

Answer in clear Arabic.
"""

    answer_text, error = _call_gemini_with_retry(
        prompt
    )

    if answer_text is None:

        return (
            "تعذر الحصول على إجابة من موديل "
            "ROADWISE AI حاليًا، برجاء إعادة المحاولة لاحقًا."
            f"\n(تفاصيل الخطأ: {error})"
        )

    # -----------------------------------------------------
    # 4. تحقق من الأرقام
    # -----------------------------------------------------

    unverified_numbers = _find_unverified_numbers(
        answer_text,
        data_slice,
    )

    if unverified_numbers:

        numbers_str = "، ".join(
            str(int(n)) if n.is_integer() else str(n)
            for n in unverified_numbers
        )

        answer_text += (
            "\n\n⚠️ تنبيه تحقق: "
            "الإجابة تحتوي على أرقام "
            f"({numbers_str}) غير واضحة "
            "المطابقة للبيانات المرفقة."
        )

    return answer_text


# =========================================================
# LEGACY DIRECT TEST
# =========================================================

def ask_gemini(
    question: str,
) -> str:

    cache = load_roadwise_cache()

    return answer_question(
        question,
        cache,
    )


# ============================================================
# FAST COMPLAINT CLASSIFICATION
# ============================================================

def classify_complaint(
    description: str,
    governorate: str = "",
    road_name: str = "",
) -> dict:
    """
    تصنيف البلاغ بشكل مستقل عن ROADWISE.

    مهم:
    - لا يتم تحميل Excel.
    - لا يتم إرسال بيانات ROADWISE.
    - لا يتم استخدام answer_question().
    - لا يتم استخدام _call_gemini_with_retry().
    - محاولة Gemini واحدة فقط.
    """

    # -----------------------------------------------------
    # تنظيف المدخلات
    # -----------------------------------------------------

    description = str(
        description or ""
    ).strip()

    governorate = str(
        governorate or ""
    ).strip()

    road_name = str(
        road_name or ""
    ).strip()

    if not description:

        return {
            "classification": "needs_review",
            "confidence": 0,
            "reason": "لم يتم إدخال وصف كافٍ للبلاغ.",
        }

    # -----------------------------------------------------
    # Prompt مختصر
    # -----------------------------------------------------

    prompt = f"""
أنت نظام تصنيف بلاغات الطرق في منصة EgyRoad IQ.

صنّف البلاغ إلى فئة واحدة فقط:

likely_valid
بلاغ واضح ومعقول ويتعلق بحادث أو خطر حقيقي على الطريق.

likely_false
البلاغ غير متعلق بالطرق أو المرور، أو عبثي، أو توجد
تناقضات واضحة جدًا في المعلومات.

needs_review
المعلومات غير كافية للحكم أو تحتاج تحقق إضافي.

مهم:
- لا تعتبر أي بلاغ كاذبًا 100%.
- إذا كانت المعلومات غير كافية استخدم needs_review.
- لا تخترع معلومات.
- أرجع النتيجة فقط بالشكل التالي:

classification: likely_valid
confidence: 85
reason: سبب مختصر بالعربية

البلاغ:

الوصف:
{description}

المحافظة:
{governorate}

الطريق:
{road_name}
"""

    try:

        print(
            "[COMPLAINT AI] Starting classification..."
        )

        # =================================================
        # استدعاء Gemini واحد فقط
        # =================================================

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        text = (
            response.text or ""
        ).strip()

        if not text:

            print(
                "[WARN] Complaint AI returned empty response."
            )

            return {
                "classification": "needs_review",
                "confidence": 0,
                "reason": (
                    "لم يرجع نموذج الذكاء الاصطناعي نتيجة."
                ),
            }

        # =================================================
        # CLASSIFICATION
        # =================================================

        classification = "needs_review"

        if re.search(
            r"\blikely[_\s-]false\b",
            text,
            re.IGNORECASE,
        ):

            classification = "likely_false"

        elif re.search(
            r"\blikely[_\s-]valid\b",
            text,
            re.IGNORECASE,
        ):

            classification = "likely_valid"

        elif re.search(
            r"\bneeds[_\s-]review\b",
            text,
            re.IGNORECASE,
        ):

            classification = "needs_review"

        # =================================================
        # CONFIDENCE
        # =================================================

        confidence = 0

        confidence_match = re.search(
            r"confidence\s*:\s*(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )

        if confidence_match:

            try:

                confidence = int(
                    float(
                        confidence_match.group(1)
                    )
                )

                confidence = max(
                    0,
                    min(
                        100,
                        confidence,
                    ),
                )

            except Exception:

                confidence = 0

        # =================================================
        # REASON
        # =================================================

        reason = ""

        reason_match = re.search(
            r"reason\s*:\s*(.*)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if reason_match:

            reason = (
                reason_match
                .group(1)
                .strip()
            )

        # لو لم نجد reason
        if not reason:

            reason = (
                "تم تحليل محتوى البلاغ."
            )

        # إزالة أي حقول زائدة من reason
        reason = re.sub(
            r"^classification\s*:.*$",
            "",
            reason,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        reason = re.sub(
            r"^confidence\s*:.*$",
            "",
            reason,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        reason = reason.strip()

        if not reason:

            reason = (
                "تم تحليل محتوى البلاغ."
            )

        # =================================================
        # FINAL RESULT
        # =================================================

        print(
            "[COMPLAINT AI] "
            f"classification={classification} "
            f"confidence={confidence}"
        )

        return {
            "classification": classification,
            "confidence": confidence,
            "reason": reason,
        }

    except Exception as e:

        print(
            "[WARN] Complaint AI failed: "
            f"{type(e).__name__}: {e}"
        )

        # الفشل التقني لا يعني أن البلاغ كاذب.
        return {
            "classification": "needs_review",
            "confidence": 0,
            "reason": (
                "تعذر فحص البلاغ آليًا حاليًا، "
                "ويحتاج إلى مراجعة."
            ),
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    roadwise_cache = load_roadwise_cache()

    questions = [
        "ما أكثر الطرق من حيث عدد الحوادث؟",
        "ما أخطر محافظة؟",
    ]

    for question in questions:

        answer = answer_question(
            question,
            roadwise_cache,
        )

        print(
            "\n===================================="
        )

        print(
            "ROADWISE AI —",
            question,
        )

        print(
            "===================================="
        )

        print(answer)