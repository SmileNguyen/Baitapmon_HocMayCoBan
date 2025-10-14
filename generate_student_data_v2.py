import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
rng = np.random.default_rng(42)

N = 1000

# ==== Đặc trưng danh mục ====
gender = rng.choice(["Male", "Female"], size=N, p=[0.52, 0.48])
family_income = rng.choice(["Low", "Medium", "High"], size=N, p=[0.3, 0.5, 0.2])
internet = rng.choice(["Yes", "No"], size=N, p=[0.9, 0.1])
extracurricular = rng.choice(["Yes", "No"], size=N, p=[0.55, 0.45])
part_time_job = rng.choice(["Yes", "No"], size=N, p=[0.4, 0.6])
parent_education = rng.choice(["Primary", "Secondary", "Bachelor", "Graduate"], size=N, p=[0.2, 0.4, 0.3, 0.1])
scholarship = rng.choice(["Yes", "No"], size=N, p=[0.15, 0.85])
housing = rng.choice(["Dorm", "Rent", "With_Family"], size=N, p=[0.35, 0.3, 0.35])

# ==== Đặc trưng số ====
age = np.clip(rng.normal(21, 2, N).round(), 18, 30).astype(int)
attendance = np.clip(rng.normal(80, 12, N), 30, 100).round(1)
avg_score_10 = np.clip(rng.normal(7, 1.5, N), 0, 10).round(2)
credits_earned = np.clip(rng.normal(100, 25, N), 0, 160).round().astype(int)
failures = np.clip(rng.poisson(1.0, N), 0, 5)
study_time_hours = np.clip(rng.lognormal(mean=np.log(9), sigma=0.5, size=N), 1, 40).round(1)
absences = np.clip(rng.poisson(6, N), 0, 40)
mental_health_score = np.clip(rng.normal(60, 15, N), 20, 100).round().astype(int)
commute_time_min = np.clip(rng.normal(25, 15, N), 0, 120).round().astype(int)
semesters_completed = np.clip(rng.normal(8, 2, N), 1, 12).round().astype(int)
gpa = np.clip((avg_score_10 / 10) * 4 + rng.normal(0, 0.2, N), 0, 4).round(2)

# ==== Mã hoá nội suy ====
parent_edu_score = pd.Series(parent_education).map({
    "Primary": 0, "Secondary": 1, "Bachelor": 2, "Graduate": 3
}).values

income_score = pd.Series(family_income).map({
    "Low": 0, "Medium": 1, "High": 2
}).values

extracurricular_bin = (pd.Series(extracurricular) == "Yes").astype(int).values
part_time_job_bin = (pd.Series(part_time_job) == "Yes").astype(int).values
scholarship_bin = (pd.Series(scholarship) == "Yes").astype(int).values
internet_bin = (pd.Series(internet) == "Yes").astype(int).values

# ==== Hàm xác suất tốt nghiệp (điều chỉnh để chỉ ~70% tốt nghiệp) ====
lin = (
    -4.5  # độ lệch tổng thể: càng cao => tỉ lệ tốt nghiệp càng lớn
    + 1.2 * gpa
    + 0.025 * attendance
    + 0.01 * credits_earned
    - 0.5 * failures
    - 0.03 * absences
    + 0.03 * study_time_hours
    - 0.005 * commute_time_min
    + 0.2 * parent_edu_score
    + 0.1 * income_score
    + 0.25 * extracurricular_bin
    - 0.2 * part_time_job_bin
    + 0.35 * scholarship_bin
    + 0.015 * mental_health_score
    + 0.1 * internet_bin
    + rng.normal(0, 0.8, N)
)
prob_graduate = expit(lin)
graduated = (rng.uniform(size=N) < prob_graduate).astype(int)

# ==== Tạo DataFrame ====
df = pd.DataFrame({
    "gender": gender,
    "age": age,
    "attendance": attendance,
    "avg_score_10": avg_score_10,
    "gpa_4": gpa,
    "credits_earned": credits_earned,
    "failures": failures,
    "study_time_hours_per_week": study_time_hours,
    "absences": absences,
    "part_time_job": part_time_job,
    "family_income": family_income,
    "internet": internet,
    "extracurricular": extracurricular,
    "parent_education": parent_education,
    "scholarship": scholarship,
    "housing": housing,
    "commute_time_min": commute_time_min,
    "semesters_completed": semesters_completed,
    "mental_health_score": mental_health_score,
    "graduated": graduated
})

# ==== Xuất CSV ====
df.to_csv("student_graduation.csv", index=False)
print("✅ Created student_graduation.csv with", len(df), "rows")
print("Graduation distribution:")
print(df['graduated'].value_counts(normalize=True))
