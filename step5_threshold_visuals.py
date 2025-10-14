import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

# === Load mô hình tốt nhất ===
model = joblib.load("best_model_RandomForest_newdata.joblib")
df = pd.read_csv("student_graduation.csv")

cat_cols = ['gender','part_time_job','family_income','internet','extracurricular',
            'parent_education','scholarship','housing']
num_cols = [c for c in df.columns if c not in (cat_cols + ['graduated'])]

from sklearn.model_selection import train_test_split
X = df.drop(columns=['graduated'])
y = df['graduated']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === Dự đoán xác suất ===
y_prob = model.predict_proba(X_test)[:,1]

# === Biểu đồ Precision–Recall để chọn threshold ===
prec, rec, thresh = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs Threshold")
plt.legend()
plt.grid()
plt.show()

# === Chọn threshold mới (ví dụ 0.35) ===
threshold = 0.35
y_pred_new = (y_prob >= threshold).astype(int)

# === Đánh giá ===
print(f"\n🔹 Threshold = {threshold}")
print(classification_report(y_test, y_pred_new, digits=4))
cm = confusion_matrix(y_test, y_pred_new)
print("Confusion matrix:\n", cm)

# === Vẽ Confusion Matrix ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (Threshold={threshold})")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
