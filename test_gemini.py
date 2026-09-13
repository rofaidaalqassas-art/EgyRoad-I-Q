import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY is not set")
    exit()

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello to ROADWISE in one short sentence."
)

print("✅ Gemini is working!")
print(response.text)
