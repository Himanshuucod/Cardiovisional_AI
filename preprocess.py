import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/cardio_train.csv", sep=";")

print("Original Shape:", df.shape)

# Convert age from days to years
df["age"] = df["age"] / 365

# Remove impossible blood pressure values
df = df[(df["ap_hi"] > 50) & (df["ap_hi"] < 250)]
df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 200)]

# Remove cases where systolic < diastolic
df = df[df["ap_hi"] > df["ap_lo"]]

# Feature Engineering
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

# Drop ID column
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

print("After Cleaning:", df.shape)

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_cardio.csv", index=False)

print("Clean dataset saved.")