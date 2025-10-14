import pandas as pd
import numpy as np

# 1) Đọc dữ liệu
df = pd.read_csv("student_graduation.csv")
print(df.shape)
print(df.head())

# 2) Thông tin sơ bộ & thiếu dữ liệu
print(df.info())
print(df.isna().sum())

# 3) Mô tả thống kê nhanh cho cột số
print(df.describe())

# 4) Kiểm tra phân bố nhãn
print(df['graduated'].value_counts(normalize=True))

# 5) Mã hoá one-hot cho biến phân loại (chuẩn bị cho mô hình)
cat_cols = ['gender','part_time_job','family_income','internet','extracurricular',
            'parent_education','scholarship','housing']
df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 6) Tách X, y
X = df_ml.drop(columns=['graduated'])
y = df_ml['graduated']

# 7) Chia train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 8) Mô hình baseline: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=200, n_jobs=None)  # n_jobs cho phiên bản sklearn mới có; nếu báo lỗi thì bỏ n_jobs
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
