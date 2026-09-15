import re
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from google import genai

# ==============================
# SETTINGS
# ==============================

DATA_FILE = "accidents_data.xlsx"

# خريطة أسماء المحافظات عربي -> إنجليزي (Governorate_EN فى البيانات) -
# ثابتة لأن محافظات مصر الـ27 معروفة ومستقرة. من غيرها، _find_mentioned_
# governorate كان بيدوّر على الاسم الإنجليزي ("Cairo") جوه سؤال مكتوب
# بالعربي ("عدد الحوادث فى القاهرة")، فمستحيل يتطابق أبدًا.
GOVERNORATE_AR_TO_EN = {
    "الإسكندرية": "Alexandria", "اسكندرية": "Alexandria",
    "أسوان": "Aswan", "اسوان": "Aswan",
    "أسيوط": "Asyut", "اسيوط": "Asyut",
    "البحيرة": "Beheira",
    "بني سويف": "Beni Suef",
    "القاهرة": "Cairo",
    "الدقهلية": "Dakahlia",
    "دمياط": "Damietta",
    "الفيوم": "Faiyum",
    "الغربية": "Gharbiya",
    "الجيزة": "Giza",
    "الإسماعيلية": "Ismailia", "الاسماعيلية": "Ismailia",
    "كفر الشيخ": "Kafr El Sheikh",
    "الأقصر": "Luxor", "الاقصر": "Luxor",
    "مطروح": "Matrouh",
    "المنوفية": "Menoufia",
    "المنيا": "Minya",
    "الوادي الجديد": "New Valley",
    "شمال سيناء": "North Sinai",
    "بورسعيد": "Port Said", "بور سعيد": "Port Said",
    "القليوبية": "Qalyubia",
    "قنا": "Qena",
    "البحر الأحمر": "Red Sea",
    "الشرقية": "Sharkia",
    "سوهاج": "Sohag",
    "جنوب سيناء": "South Sinai",
    "السويس": "Suez",
}

# إعدادات التحقق (Verification) لإجابات متخذ القرار:
#   - MAX_RETRIES/RETRY_DELAY_SECONDS: إعادة محاولة عند بطء/فشل مؤقت في
#     الاتصال بالموديل، بدل ما السؤال يفشل تمامًا من أول عثرة.
#   - عدد الفروق المسموح بيها بين رقم رجعه الموديل ورقم موجود فعلاً في
#     البيانات (Total_Economic_Loss_EGP ممكن يترقّم/يتقرّب)، عشان الفحص
#     ميرفضش أرقام صحيحة اتقربت.
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5
NUMBER_MATCH_TOLERANCE_RATIO = 0.01  # 1% فرق مسموح به (تقريب/فواصل عشرية)

client = genai.Client()


# ==============================
# ROADWISE CACHE
# ------------------------------
# مهم جدًا: load_roadwise_cache() المفروض تتنادى مرة واحدة بس عند تشغيل
# الـ FastAPI (main.py، في startup event)، والناتج (RoadwiseCache) يتسيب
# في app.state عشان يتستخدم مع كل سؤال من غير ما نعيد قراءة الإكسيل تاني.
# ده أكبر سبب بطء في الشكل القديم (get_roadwise_summary كانت بتقرا
# الإكسيل بالكامل مع كل سؤال).
# ==============================

@dataclass
class RoadwiseCache:
    df: pd.DataFrame
    kpi_summary: str                  # ملخص صغير (أرقام إجمالية بس) - ده الافتراضي اللي بيتبعت للـ AI
    governorate_counts: pd.Series
    road_counts: pd.Series
    cause_counts: pd.Series
    precomputed_answers: dict = field(default_factory=dict)


def load_roadwise_cache(data_file: str = DATA_FILE) -> RoadwiseCache:
    """يتحمّل مرة واحدة بس عند تشغيل السيرفر - مش مع كل سؤال.

    مهم: بنقرأ الأعمدة المطلوبة بس (usecols) بدل الملف كامل بكل أعمدته،
    وبعد حساب الملخصات (kpi_summary/cause_counts) بنسيب فى cache.df بس
    الأعمدة اللي فعلاً بتتفلتر بيها لاحقًا (Governorate_EN/Highway_Name/
    Fatalities_Count/Injuries_Count فى build_relevant_slice). الكاش ده
    بيفضل محفوظ فى الذاكرة طول عمر السيرفر (بيتحمّل مرة عند startup فى
    main.py)، فكل عمود زيادة فيه بيتضاعف تأثيره - وده كان بيساهم فى تخطي
    حد الذاكرة على Render حتى لو محدش سأل الـ AI أصلاً."""

    needed_cols = (
        "Governorate_EN", "Highway_Name", "Cause_Category",
        "Fatalities_Count", "Injuries_Count", "Total_Economic_Loss_EGP",
    )
    df = pd.read_excel(data_file, usecols=lambda c: c in needed_cols)

    governorate_counts = df["Governorate_EN"].value_counts()
    road_counts = df["Highway_Name"].value_counts()
    cause_counts = df["Cause_Category"].value_counts()

    total_accidents = len(df)
    total_fatalities = int(df["Fatalities_Count"].sum())
    total_injuries = int(df["Injuries_Count"].sum())
    total_economic_loss = float(df["Total_Economic_Loss_EGP"].sum())

    # مش محتاجين نسيب Cause_Category/Total_Economic_Loss_EGP فى الـ df
    # المحفوظ - استخدامهم كان لحظة التحميل بس (فوق) لحساب cause_counts
    # والملخص. build_relevant_slice() بعد كده بيستخدم بس الأعمدة الأربعة
    # الباقية.
    df = df[["Governorate_EN", "Highway_Name", "Fatalities_Count", "Injuries_Count"]]

    kpi_summary = (
        "ROADWISE KPI SUMMARY\n"
        f"Total Accidents: {total_accidents}\n"
        f"Total Fatalities: {total_fatalities}\n"
        f"Total Injuries: {total_injuries}\n"
        f"Total Economic Loss: {total_economic_loss} EGP\n"
    )

    most_dangerous_gov = governorate_counts.index[0] if len(governorate_counts) else None
    most_dangerous_road = road_counts.index[0] if len(road_counts) else None

    # إجابات جاهزة لأسئلة متكررة - بتتحسب مرة واحدة هنا من البيانات
    # الحقيقية، وبترجع من الـ backend مباشرة من غير ما نطلب من الـ AI
    # يحللها كل مرة.
    precomputed_answers = {
        "most_dangerous_governorate": (
            f"أخطر محافظة من حيث عدد الحوادث هي {most_dangerous_gov} "
            f"بعدد {int(governorate_counts.iloc[0])} حادثة."
        ) if most_dangerous_gov is not None else "البيانات غير كافية لتحديد أخطر محافظة.",
        "most_dangerous_road": (
            f"أخطر طريق من حيث عدد الحوادث هو {most_dangerous_road} "
            f"بعدد {int(road_counts.iloc[0])} حادثة."
        ) if most_dangerous_road is not None else "البيانات غير كافية لتحديد أخطر طريق.",
        "total_fatalities": f"إجمالي عدد الوفيات المسجلة هو {total_fatalities} حالة وفاة.",
        "total_injuries": f"إجمالي عدد الإصابات المسجلة هو {total_injuries} إصابة.",
        "total_accidents": f"إجمالي عدد الحوادث المسجلة هو {total_accidents} حادثة.",
        "total_economic_loss": f"إجمالي الخسائر الاقتصادية المسجلة هو {total_economic_loss} جنيه مصري.",
    }

    return RoadwiseCache(
        df=df,
        kpi_summary=kpi_summary,
        governorate_counts=governorate_counts,
        road_counts=road_counts,
        cause_counts=cause_counts,
        precomputed_answers=precomputed_answers,
    )


# أسئلة متكررة معروفة مسبقًا -> مفتاح في precomputed_answers، من غير
# استدعاء AI خالص لها.
_PRECOMPUTED_PATTERNS = [
    (("أخطر محافظ", "اكثر محافظ", "أكثر محافظ"), "most_dangerous_governorate"),
    (("أخطر طريق", "اكثر طريق", "أكثر طريق", "أخطر الطرق", "أكثر الطرق"), "most_dangerous_road"),
    (("عدد الوفيات", "اجمالي الوفيات", "إجمالي الوفيات", "كام حالة وفاة"), "total_fatalities"),
    (("عدد الإصابات", "عدد الاصابات", "اجمالي الاصابات", "إجمالي الإصابات"), "total_injuries"),
    (("عدد الحوادث", "اجمالي الحوادث", "إجمالي الحوادث", "كام حادثة"), "total_accidents"),
    (("خسائر اقتصادية", "الخساير الاقتصادية", "التكلفة الاقتصادية"), "total_economic_loss"),
]


def _match_precomputed(question: str, cache: RoadwiseCache) -> Optional[str]:
    """الإجابات الجاهزة (precomputed_answers) كلها أرقام إجمالية عامة
    (مصر كلها)، مش لمحافظة أو طريق معيّن. لو السؤال فيه اسم محافظة أو
    طريق مذكور صراحة (زي "عدد الحوادث فى القاهرة")، لازم نتخطى الإجابة
    الجاهزة العامة ونسيب السؤال يروح لـ build_relevant_slice + Gemini،
    اللي فعليًا بيفلتر على المحافظة/الطريق المذكورة. قبل الإصلاح ده، أي
    سؤال فيه كلمة "عدد الحوادث" كان بيرجّع إجمالي مصر كله حتى لو
    المحافظة مذكورة صريح فى نفس الجملة.
    """
    if _find_mentioned_governorate(question, cache) is not None:
        return None
    if _find_mentioned_road(question, cache) is not None:
        return None
    for keywords, key in _PRECOMPUTED_PATTERNS:
        if any(kw in question for kw in keywords):
            return cache.precomputed_answers.get(key)
    return None


def _find_mentioned_governorate(question: str, cache: RoadwiseCache) -> Optional[str]:
    # الأول ندوّر بالاسم العربي (الأكثر شيوعًا فى أسئلة المستخدمين الفعلية)
    for name_ar, name_en in GOVERNORATE_AR_TO_EN.items():
        if name_ar in question and name_en in cache.governorate_counts.index:
            return name_en
    # وبعدين بالاسم الإنجليزي زي ما كان (لو حد كتبه إنجليزي فعلاً)
    for governorate in cache.governorate_counts.index:
        if str(governorate) and str(governorate) in question:
            return governorate
    return None


def _find_mentioned_road(question: str, cache: RoadwiseCache) -> Optional[str]:
    for road in cache.road_counts.index:
        if str(road) and str(road) in question:
            return road
    return None


def build_relevant_slice(question: str, cache: RoadwiseCache) -> str:
    """
    بدل ما نبعت كل بيانات ROADWISE مع كل سؤال، بنبعت بس KPIs + الجزء
    المرتبط فعليًا بالسؤال (محافظة/طريق مذكور فيه) - ده اللي بيقلل حجم
    الـ prompt وبالتالي وقت واستهلاك الاستدعاء بشكل كبير.
    """
    parts = [cache.kpi_summary]

    governorate = _find_mentioned_governorate(question, cache)
    if governorate is not None:
        gov_df = cache.df[cache.df["Governorate_EN"] == governorate]
        parts.append(
            f"\nGovernorate '{governorate}' detail:\n"
            f"Accidents: {len(gov_df)}\n"
            f"Fatalities: {int(gov_df['Fatalities_Count'].sum())}\n"
            f"Injuries: {int(gov_df['Injuries_Count'].sum())}\n"
        )

    road = _find_mentioned_road(question, cache)
    if road is not None:
        road_df = cache.df[cache.df["Highway_Name"] == road]
        parts.append(
            f"\nRoad '{road}' detail:\n"
            f"Accidents: {len(road_df)}\n"
            f"Fatalities: {int(road_df['Fatalities_Count'].sum())}\n"
            f"Injuries: {int(road_df['Injuries_Count'].sum())}\n"
        )

    if governorate is None and road is None:
        # سؤال عام - أعلى 5 محافظات وأعلى 5 أسباب بس، مش كل التفاصيل.
        parts.append("\nTop 5 Governorates:\n" + cache.governorate_counts.head(5).to_string())
        parts.append("\nTop 5 Causes:\n" + cache.cause_counts.head(5).to_string())

    result = "\n".join(parts)
    # نطبع اسم المحافظة/الطريق اللي اتكشفوا (أو None) + البيانات اللي فعليًا
    # هتتبعت لـ Gemini - عشان لو جاوب إجابة غلط، نشوف هل السبب إن الكشف
    # فشل (governorate=None رغم إن السؤال فيه اسم محافظة)، أو إن البيانات
    # وصلته صح لكنه اخترع رد مختلف عنها.
    print(
        f"[DEBUG] build_relevant_slice question={question!r} "
        f"governorate={governorate!r} road={road!r} slice={result!r}"
    )
    return result


# ==============================
# ASK GEMINI (مع تحقق)
# ==============================

def _call_gemini_with_retry(prompt: str):
    """
    بينادي الموديل مع إعادة محاولة (retry) عند بطء/فشل مؤقت في الاتصال،
    بدل ما السؤال يفشل بـ Exception غير واضح لمتخذ القرار.
    بيرجع (text, error) - text بتكون None لو كل المحاولات فشلت.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):  # أول محاولة + MAX_RETRIES إعادة
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            text = (response.text or "").strip()
            if text:
                return text, None
            last_error = "رد فارغ من الموديل"
        except Exception as e:
            last_error = str(e)

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    # نطبع سبب الفشل النهائي فى الـ logs (Render → Logs) عشان نعرف بالظبط
    # ليه مساعد Gemini فشل - نفس الثغرة اللي كانت فى image_verification.py:
    # last_error كان بيتسجل بس مايتطبعش، فمفيش أي أثر نشوفه وقت الفشل.
    print(
        f"[WARN] ai_agent: Gemini call failed after retries. "
        f"last_error={last_error!r}"
    )

    return None, last_error


def _extract_numbers(text: str):
    """يستخرج كل الأرقام (صحيحة أو عشرية، ممكن فيها فواصل آلاف) من نص."""
    raw_matches = re.findall(r"\d[\d,]*\.?\d*", text)
    numbers = []
    for match in raw_matches:
        cleaned = match.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def _find_unverified_numbers(answer_text: str, roadwise_data: str):
    """
    يتأكد إن كل رقم "كبير"/مهم ذكره الموديل في إجابته موجود (أو قريب
    كفاية) من رقم فعلي في البيانات اللي اتبعتت له فعلًا - عشان نمسك حالة
    إن الموديل اخترع إحصائية مش موجودة أصلًا.
    """
    data_numbers = _extract_numbers(roadwise_data)
    answer_numbers = _extract_numbers(answer_text)

    unverified = []
    for num in answer_numbers:
        if num < 10:  # كسور/نسب صغيرة أو ترقيم - مش هدف الفحص هنا
            continue

        found_match = any(
            abs(num - data_num) <= max(1.0, data_num * NUMBER_MATCH_TOLERANCE_RATIO)
            for data_num in data_numbers
        )
        if not found_match:
            unverified.append(num)

    return unverified


def answer_question(question: str, cache: RoadwiseCache) -> str:
    """
    نقطة الدخول اللي المفروض main.py ينادي عليها مع كل سؤال، مع الـ
    cache المحمّلة مرة واحدة بس عند تشغيل السيرفر. الـ AI هنا آخر طبقة
    بس، زي ما اتفقنا:

        Question -> تحديد المطلوب -> استخراج البيانات المرتبطة بس -> AI يصيغ الإجابة

    مش:

        Question -> AI يقرأ كل بيانات المشروع من الصفر -> يفكر -> يجيب
    """
    # 1) سؤال عن KPI متكرر ومعروف -> إجابة مباشرة من الـ backend، من غير AI خالص.
    precomputed = _match_precomputed(question, cache)
    if precomputed:
        return precomputed

    # 2) غير كده -> نجهز بس الجزء المرتبط بالسؤال (KPIs + محافظة/طريق لو مذكورة).
    data_slice = build_relevant_slice(question, cache)

    prompt = f"""
You are ROADWISE AI Assistant.

You are an AI assistant for road safety in Egypt.

Answer the user's question using the ROADWISE data provided below.

Do not invent statistics.

If the requested information is not available in the data,
clearly say that it is not available.

ROADWISE DATA:
{data_slice}

USER QUESTION:
{question}

Answer in clear Arabic.
"""

    # 3) الـ AI بيصيغ الإجابة فقط من الجزء المرتبط - مش من كل المشروع.
    answer_text, error = _call_gemini_with_retry(prompt)

    if answer_text is None:
        return (
            "تعذر الحصول على إجابة من موديل ROADWISE AI بعد عدة محاولات، "
            "برجاء إعادة المحاولة لاحقًا أو مراجعة البيانات يدويًا."
            f"\n(تفاصيل الخطأ: {error})"
        )

    unverified_numbers = _find_unverified_numbers(answer_text, data_slice)
    if unverified_numbers:
        numbers_str = "، ".join(str(n) for n in unverified_numbers)
        answer_text += (
            "\n\n⚠️ تنبيه تحقق: الإجابة دي فيها أرقام "
            f"({numbers_str}) مش واضح إنها موجودة بالظبط في البيانات "
            "المرفقة - يُنصح بمراجعتها قبل الاعتماد عليها في قرار."
        )

    return answer_text


def ask_gemini(question: str) -> str:
    """
    نسخة للتوافق مع الاستخدام القديم/الاختبار المباشر (بتحمّل البيانات
    من الإكسيل في كل نداء). في main.py الفعلي، استخدمي load_roadwise_cache()
    مرة واحدة عند بدء التشغيل، وanswer_question(question, cache) مع كل
    سؤال - عشان منرجعش نقرا الإكسيل من الأول كل مرة.
    """
    cache = load_roadwise_cache()
    return answer_question(question, cache)


# ==============================
# CLASSIFY COMPLAINT (مستقلة تمامًا عن بيانات ROADWISE)
# ------------------------------
# main.py القديم كان بيصنّف نص البلاغات عن طريق ask_gemini() - يعني كل
# بلاغ كان بيتحمّله معاه كل بيانات ROADWISE (get_roadwise_summary) جوه
# برومبت تاني بيقول "أجب عن سؤال متخذ القرار من بيانات ROADWISE"، والبرومبت
# ده لفّ حوالين برومبت التصنيف نفسه. ده هو أكبر سبب لبطء تصنيف الشكاوى
# ولاحتمال رجوع تصنيف غلط (زي بلاغ غير حقيقي يطلع "يبدو صحيحًا") - لأن
# الموديل كان بيتعامل مع برومبت مركّب ومربك بدل تعليمات تصنيف واضحة
# ومباشرة. الدالة دي بديلة: بتنادي Gemini مباشرة، بدون أي بيانات ROADWISE
# وبدون أي تعليمات تانية حوالين برومبت التصنيف.
# ==============================

def classify_complaint(description: str, governorate: str = "", road_name: str = "") -> dict:
    """
    تصنيف نص بلاغ واحد لثلاث فئات فقط (likely_valid / likely_false /
    needs_review). استدعاء مباشر ومستقل لـ Gemini - من غير أي بيانات
    ROADWISE ومن غير أي برومبت تاني ملفوف حواليه.
    """
    prompt = f"""
أنت نظام ذكي لتحليل بلاغات حوادث ومخاطر الطرق في مصر، جزء من EgyRoad IQ.

بيانات البلاغ:

وصف البلاغ:
{description}

المحافظة:
{governorate}

الطريق:
{road_name}

صنّفي البلاغ إلى واحد من ثلاثة تصنيفات فقط - ممنوع الحكم القاطع
(صحيح 100% أو كاذب 100%) بدون دليل كافٍ:

likely_valid: البلاغ واضح ومنطقي ويتعلق بحادث أو خطر حقيقي على الطريق
مثل حفرة، حادث، سيارة متعطلة، طريق مغلق، عمود ساقط، إشارة مرور تالفة،
أو خطر مشابه، والتفاصيل متسقة مع بعضها.

likely_false: البلاغ لا علاقة له بالطرق أو المرور أو الحوادث، أو
محتوى عبثي/غير منطقي، أو فيه تناقض واضح في التفاصيل.

needs_review: أي حالة تانية - المعلومات غير كافية للحكم، أو البلاغ
معقول لكن ناقص تفاصيل تؤكده.

أرجعي النتيجة بهذا الشكل فقط، بدون أي نص إضافي قبله أو بعده:

classification: [likely_valid أو likely_false أو needs_review]
confidence: [رقم من 0 إلى 100]
reason: [سبب مختصر باللغة العربية]
"""

    answer_text, error = _call_gemini_with_retry(prompt)

    if answer_text is None:
        # فشل تقني (مش وضوح غير كافي في البلاغ) -> needs_review دايمًا،
        # مش likely_false، عشان الفشل التقني ما يترجمش لرفض بلاغ حقيقي.
        return {
            "classification": "needs_review",
            "confidence": 0,
            "reason": f"تعذر تحليل البلاغ آليًا بعد عدة محاولات: {error}",
        }

    text = answer_text.strip()

    classification = "needs_review"
    if re.search(r"\blikely_false\b", text, re.IGNORECASE):
        classification = "likely_false"
    elif re.search(r"\blikely_valid\b", text, re.IGNORECASE):
        classification = "likely_valid"

    confidence = 0
    match = re.search(r"confidence\s*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        confidence = max(0, min(100, int(match.group(1))))

    reason = text
    reason_match = re.search(r"reason\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()

    return {
        "classification": classification,
        "confidence": confidence,
        "reason": reason,
    }


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    # في main.py الفعلي: ده بيحصل مرة واحدة بس عند startup الـ FastAPI.
    roadwise_cache = load_roadwise_cache()

    questions = [
        "ما أكثر الطرق من حيث عدد الحوادث؟",   # عام -> AI بجزء صغير من البيانات
        "ما أخطر محافظة؟",                       # متكرر -> إجابة فورية من الـ backend، من غير AI
    ]

    for question in questions:
        answer = answer_question(question, roadwise_cache)  # مفيش إعادة تحميل هنا
        print("\n====================================")
        print("ROADWISE AI —", question)
        print("====================================")
        print(answer)