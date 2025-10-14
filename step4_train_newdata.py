import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

# === 1. Load dữ liệu mới ===
df = pd.read_csv("student_graduation.csv")
print("📊 Dataset shape:", df.shape)
print(df["graduated"].value_counts(normalize=True))

# === 2. Chia train/test ===
cat_cols = ['gender','part_time_job','family_income','internet','extracurricular',
            'parent_education','scholarship','housing']
num_cols = [c for c in df.columns if c not in (cat_cols + ['graduated'])]

X = df.drop(columns=['graduated'])
y = df['graduated']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === 3. Tiền xử lý ===
preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])

# === 4. Thử 2 mô hình mạnh: RandomForest và GradientBoost ===
models = {
    "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "GradBoost": GradientBoostingClassifier(random_state=42)
}

param_grid = {
    "RandomForest": {
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [8, 12, None],
        "clf__min_samples_split": [2, 4],
    },
    "GradBoost": {
        "clf__n_estimators": [200, 300],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5],
    }
}

results = []
best_auc = 0
best_pipe = None
best_name = ""

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    grid = GridSearchCV(pipe, param_grid[name], cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    print(f"\n=== {name} ===")
    print("Best params:", grid.best_params_)
    print("ROC-AUC(test):", auc_val)
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    results.append((name, auc_val))
    if auc_val > best_auc:
        best_auc = auc_val
        best_pipe = grid.best_estimator_
        best_name = name

# === 5. Kết luận & biểu đồ ROC ===
print("\nSummary AUC:")
for name, auc_val in results:
    print(f"{name}: {auc_val:.3f}")
print(f"\n🏆 Best model: {best_name} (AUC={best_auc:.3f})")

# ROC curve plot
y_prob = best_pipe.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={best_auc:.3f})", lw=2)
plt.plot([0,1],[0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model")
plt.legend()
plt.tight_layout()
plt.show()

# === 6. Lưu mô hình ===
joblib.dump(best_pipe, f"best_model_{best_name}_newdata.joblib")
print(f"\n💾 Saved best model: best_model_{best_name}_newdata.joblib")
