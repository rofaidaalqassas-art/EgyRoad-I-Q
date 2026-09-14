import pandas as pd
from google import genai

# ==============================
# SETTINGS
# ==============================

DATA_FILE = "accidents_data.xlsx"

client = genai.Client()


# ==============================
# ROADWISE DATA SUMMARY
# ==============================

def get_roadwise_summary():

    df = pd.read_excel(DATA_FILE)

    summary = f"""
ROADWISE DATA SUMMARY

Total Accidents: {len(df)}

Total Fatalities: {df["Fatalities_Count"].sum()}

Total Injuries: {df["Injuries_Count"].sum()}

Severity:
{df["Severity_Level"].value_counts().to_string()}

Governorate Accident Counts:
{df["Governorate_EN"].value_counts().to_string()}

Road Accident Counts:
{df["Highway_Name"].value_counts().to_string()}

Cause Counts:
{df["Cause_Category"].value_counts().to_string()}


Collision Type Counts:
{df["Collision_Type"].value_counts().to_string()}

Weather Condition Counts:
{df["Weather_Condition"].value_counts().to_string()}

Road Surface Condition Counts:
{df["Road_Surface_Condition"].value_counts().to_string()}

Lighting Condition Counts:
{df["Lighting_Condition"].value_counts().to_string()}

Total Economic Loss:
{df["Total_Economic_Loss_EGP"].sum()} EGP
"""

    return summary


# ==============================
# ASK GEMINI
# ==============================

def ask_gemini(question):

    roadwise_data = get_roadwise_summary()

    prompt = f"""
You are ROADWISE AI Assistant.

You are an AI assistant for road safety in Egypt.

Answer the user's question using the ROADWISE data provided below.

Do not invent statistics.

If the requested information is not available in the data,
clearly say that it is not available.

ROADWISE DATA:
{roadwise_data}

USER QUESTION:
{question}

Answer in clear Arabic.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    question = "ما أكثر الطرق من حيث عدد الحوادث؟"

    answer = ask_gemini(question)

    print("\n====================================")
    print("ROADWISE AI")
    print("====================================")
    print(answer)