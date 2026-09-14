import pandas as pd

DATA_FILE = "accidents_data.xlsx"

df = pd.read_excel(DATA_FILE)


def show_value_counts(title, column, top=10):
    """يطبع value_counts لعمود لو موجود فعليًا فى الملف، وبيتخطاه بأمان
    (مع رسالة واضحة) لو العمود مش موجود - بدل ما السكريبت كله يقع
    بـ KeyError ويوقف قبل ما يوصل لباقي الأعمدة."""
    print(f"\n{title}:")
    if column not in df.columns:
        print(f"  [تخطّي] العمود '{column}' غير موجود فى الملف.")
        return
    counts = df[column].value_counts()
    print(counts.head(top) if top else counts)


print("====================================")
print("ROADWISE DATA SUMMARY")
print("====================================")

# 1. إجمالي الحوادث
print("\nTotal Accidents:", len(df))

# 2. إجمالي الوفيات والإصابات
if "Fatalities_Count" in df.columns:
    print("Total Fatalities:", df["Fatalities_Count"].sum())
if "Injuries_Count" in df.columns:
    print("Total Injuries:", df["Injuries_Count"].sum())

# 3. شدة الحوادث
show_value_counts("Severity", "Severity_Level", top=None)

# 4. أكثر المحافظات
show_value_counts("Top Governorates", "Governorate_EN")

# 5. أكثر الطرق
show_value_counts("Top Roads", "Highway_Name")

# 6. أسباب الحوادث
show_value_counts("Top Causes", "Cause_Category")

# 7. نوع التصادم
show_value_counts("Collision Types", "Collision_Type")

# 8. الطقس
show_value_counts("Weather Conditions", "Weather_Condition")

# 9. حالة الطريق
show_value_counts("Road Surface", "Road_Surface_Condition")

# 10. الإضاءة
show_value_counts("Lighting", "Lighting_Condition")

# 11. وقت الحوادث
# ملحوظة: باقي الكود فى المشروع (main.py, train_model.py, data_prep.py)
# بيستخدم Hour_24 مش Time_Of_Day. لو "Time_Of_Day" مش عمود فعلي فى الإكسيل
# عندك، بنرجع نبني توزيع مبسّط من Hour_24 بدل ما نتخطى الإحصائية دي بالكامل.
if "Time_Of_Day" in df.columns:
    show_value_counts("Time of Day", "Time_Of_Day", top=None)
elif "Hour_24" in df.columns:
    print("\nTime of Day (مبني من Hour_24 لأن 'Time_Of_Day' غير موجود):")
    bins = [-1, 5, 11, 16, 20, 23]
    labels = ["Late Night (0-5)", "Morning (6-11)", "Afternoon (12-16)", "Evening (17-20)", "Night (21-23)"]
    print(pd.cut(df["Hour_24"], bins=bins, labels=labels).value_counts())
else:
    print("\nTime of Day:\n  [تخطّي] لا يوجد عمود 'Time_Of_Day' ولا 'Hour_24' فى الملف.")

# 12. إجمالي الخسائر الاقتصادية
if "Total_Economic_Loss_EGP" in df.columns:
    print("\nTotal Economic Loss:")
    print(df["Total_Economic_Loss_EGP"].sum())

print("\n====================================")
print("SUMMARY COMPLETE")
print("====================================")
