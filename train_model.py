import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# Load cleaned dataset
df = pd.read_csv("data/cleaned_cardio.csv")

X = df.drop("cardio", axis=1)
y = df["cardio"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# XGBoost Model
model = XGBClassifier(
    n_estimators=700,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Cross validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

print("Cross Validation Accuracy:", cv_scores.mean())

# Save model
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(accuracy, open("model/accuracy.pkl", "wb"))

print("Model, scaler, and accuracy saved.")