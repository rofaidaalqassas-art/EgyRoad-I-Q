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

Top Governorates:
{df["Governorate_EN"].value_counts().head(10).to_string()}

Top Roads:
{df["Highway_Name"].value_counts().head(10).to_string()}

Top Causes:
{df["Cause_Category"].value_counts().head(10).to_string()}

Collision Types:
{df["Collision_Type"].value_counts().head(10).to_string()}

Weather Conditions:
{df["Weather_Condition"].value_counts().head(10).to_string()}

Road Surface:
{df["Road_Surface_Condition"].value_counts().head(10).to_string()}

Lighting:
{df["Lighting_Condition"].value_counts().head(10).to_string()}

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