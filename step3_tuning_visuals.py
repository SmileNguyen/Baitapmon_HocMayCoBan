import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, RocCurveDisplay
)
import joblib

# === 1. Đọc dữ liệu ===
df = pd.read_csv("student_graduation.csv")

cat_cols = ['gender','part_time_job','family_income','internet','extracurricular',
            'parent_education','scholarship','housing']
num_cols = [c for c in df.columns if c not in (cat_cols + ['graduated'])]

X = df.drop(columns=['graduated'])
y = df['graduated']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === 2. Tiền xử lý ===
preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])

# === 3. GridSearch cho RandomForest ===
param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 8, 12, 16],
    "clf__min_samples_split": [2, 4, 6],
}
pipe = Pipeline(steps=[('prep', preprocess), ('clf', RandomForestClassifier(random_state=42))])

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print("✅ Best params:", grid.best_params_)
print("🏆 Best ROC-AUC (CV):", grid.best_score_)

# === 4. Đánh giá trên tập test ===
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

print("\n📊 Test set report:")
print(classification_report(y_test, y_pred))

# === 5. ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], color='red', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RandomForest (Tuned)')
plt.legend()
plt.tight_layout()
plt.show()

# === 6. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - RandomForest (Tuned)')
plt.show()

# === 7. Feature Importances ===
ohe = best_model.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, cat_names])
importances = best_model.named_steps["clf"].feature_importances_
idx = np.argsort(importances)[-15:][::-1]

plt.figure(figsize=(8,5))
sns.barplot(x=importances[idx], y=feature_names[idx])
plt.title("Top 15 Feature Importances (Tuned RandomForest)")
plt.tight_layout()
plt.show()

# === 8. Lưu mô hình tốt nhất ===
joblib.dump(best_model, "best_graduation_model_tuned.joblib")
print("\n💾 Model saved: best_graduation_model_tuned.joblib")
