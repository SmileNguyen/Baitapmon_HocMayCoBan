import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

df = pd.read_csv("student_graduation.csv")
cat_cols = ['gender','part_time_job','family_income','internet','extracurricular',
            'parent_education','scholarship','housing']
num_cols = [c for c in df.columns if c not in (cat_cols + ['graduated'])]

X = df.drop(columns=['graduated'])
y = df['graduated']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])

pipe = ImbPipeline(steps=[
    ('prep', preprocess),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_split": [2, 4],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)

best = gs.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:,1]

print("Best params:", gs.best_params_)
print("ROC-AUC(test):", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(best, "best_model_smote_rf.joblib")
print("\n✅ Model saved: best_model_smote_rf.joblib")
