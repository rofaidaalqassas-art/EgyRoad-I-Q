import pandas as pd

DATA_FILE = "accidents_data.xlsx"

df = pd.read_excel(DATA_FILE)

print("====================================")
print("ROADWISE DATA SUMMARY")
print("====================================")

# 1. إجمالي الحوادث
print("\nTotal Accidents:", len(df))

# 2. إجمالي الوفيات والإصابات
print("Total Fatalities:", df["Fatalities_Count"].sum())
print("Total Injuries:", df["Injuries_Count"].sum())

# 3. شدة الحوادث
print("\nSeverity:")
print(df["Severity_Level"].value_counts())

# 4. أكثر المحافظات
print("\nTop Governorates:")
print(df["Governorate_EN"].value_counts().head(10))

# 5. أكثر الطرق
print("\nTop Roads:")
print(df["Highway_Name"].value_counts().head(10))

# 6. أسباب الحوادث
print("\nTop Causes:")
print(df["Cause_Category"].value_counts().head(10))

# 7. نوع التصادم
print("\nCollision Types:")
print(df["Collision_Type"].value_counts().head(10))

# 8. الطقس
print("\nWeather Conditions:")
print(df["Weather_Condition"].value_counts().head(10))

# 9. حالة الطريق
print("\nRoad Surface:")
print(df["Road_Surface_Condition"].value_counts().head(10))

# 10. الإضاءة
print("\nLighting:")
print(df["Lighting_Condition"].value_counts().head(10))

# 11. وقت الحوادث
print("\nTime of Day:")
print(df["Time_Of_Day"].value_counts())

# 12. إجمالي الخسائر الاقتصادية
print("\nTotal Economic Loss:")
print(df["Total_Economic_Loss_EGP"].sum())

print("\n====================================")
print("SUMMARY COMPLETE")
print("====================================")