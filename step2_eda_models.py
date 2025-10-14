# step2_eda_models.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import joblib

# 1) Đọc dữ liệu
df = pd.read_csv("student_graduation.csv")

# 2) EDA ngắn: kiểm tra target balance
print("Shape:", df.shape)
print("Graduated distribution:\n", df["graduated"].value_counts(normalize=True))

# 3) Xác định cột số & cột phân loại
cat_cols = ['gender','part_time_job','family_income','internet','extracurricular',
            'parent_education','scholarship','housing']
num_cols = [c for c in df.columns if c not in (cat_cols + ['graduated'])]

X = df.drop(columns=['graduated'])
y = df['graduated']

# 4) Tiền xử lý: scale số + one-hot cat
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
    ]
)

# 5) Khởi tạo các mô hình để so sánh
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=4, random_state=42
    ),
    "GradBoost": GradientBoostingClassifier(random_state=42),
    "SVM-RBF": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42),
}

# 6) Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 7) Train + đánh giá từng mô hình
results = []
best_model_name, best_auc = None, -1
best_clf = None

for name, clf in models.items():
    pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Dự đoán
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]  # cần cho AUC/ROC

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results.append((name, acc, prec, rec, f1, auc))
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC-AUC:", auc)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_clf = pipe

# 8) Bảng tổng hợp
summary = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1","ROC_AUC"])
print("\n*** Summary ***")
print(summary.sort_values("ROC_AUC", ascending=False))

# 9) Lưu mô hình tốt nhất
joblib.dump(best_clf, f"best_graduation_model_{best_model_name}.joblib")
print(f"\n✅ Saved best model: best_graduation_model_{best_model_name}.joblib (ROC-AUC={best_auc:.4f})")

# 10) (Tuỳ chọn) Tính tầm quan trọng đặc trưng nếu best là RandomForest/GradBoost
if best_model_name in ["RandForest", "GradBoost"]:
    # Lấy tên cột sau preprocess
    ohe = best_clf.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([num_cols, cat_feature_names])

    # Lấy importances
    importances = best_clf.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]
    print("\nTop-15 feature importances:")
    for i in top_idx:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # Vẽ barplot nhanh
    plt.figure()
    plt.bar(range(len(top_idx)), importances[top_idx])
    plt.xticks(range(len(top_idx)), feature_names[top_idx], rotation=60, ha="right")
    plt.title(f"Top-15 Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.show()

# 11) Demo dự đoán cho 1 hồ sơ giả định
sample = pd.DataFrame({
    'gender': ['Female'],
    'age': [21],
    'attendance': [88.0],
    'avg_score_10': [7.5],
    'gpa_4': [3.1],
    'credits_earned': [115],
    'failures': [0],
    'study_time_hours_per_week': [12.0],
    'absences': [4],
    'part_time_job': ['No'],
    'family_income': ['Medium'],
    'internet': ['Yes'],
    'extracurricular': ['Yes'],
    'parent_education': ['Bachelor'],
    'scholarship': ['No'],
    'housing': ['Dorm'],
    'commute_time_min': [20],
    'semesters_completed': [8],
    'mental_health_score': [68],
})
prob = best_clf.predict_proba(sample)[:,1][0]
pred = best_clf.predict(sample)[0]
print(f"\n🔮 Sample predicted prob graduate = {prob:.3f} → class={pred}")
