import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("student_graduation.csv")
print("== Toàn bộ dataset ==")
print(df["graduated"].value_counts(), "\n")
print(df["graduated"].value_counts(normalize=True), "\n")

X = df.drop(columns=["graduated"])
y = df["graduated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("== Train set ==")
print(y_train.value_counts(), "\n")
print(y_train.value_counts(normalize=True), "\n")

print("== Test set ==")
print(y_test.value_counts(), "\n")
print(y_test.value_counts(normalize=True))
