import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("../filtered_student_dataset.csv")

df.drop(columns=[
    "How many hours do you sleep daily?",
    "Do you complete assignments on time?",
    "What is your average internal marks (out of 100)?",
    "Do you actively participate in class?",
    "Do you have any backlogs?"
], inplace=True, errors='ignore')

df["What was your previous SGPA?"] = (df["What was your previous SGPA?"] / 4) * 10
df["What is your current CGPA?"] = (df["What is your current CGPA?"] / 4) * 10

def convert_range(value):
    if isinstance(value, str) and "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2
    return value

df["Average attendance on class"] = df["Average attendance on class"].apply(convert_range)

df.replace({"Yes": 1, "No": 0}, inplace=True)
df = df.infer_objects(copy=False)

X = df[[
    "Average attendance on class",
    "How many hour do you study daily?",
    "How many times do you seat for study in a day?",
    "What was your previous SGPA?",
    "How many hour do you spent daily in social media?",
    "Do you attend in teacher consultancy for any kind of academical problems?",
    "Are you engaged with any co-curriculum activities?"
]]

y = df["What is your current CGPA?"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

reg_model.fit(X_train_scaled, y_train)

y_pred = reg_model.predict(X_test_scaled)
y_train_pred = reg_model.predict(X_train_scaled)

print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

scores = cross_val_score(reg_model, X, y, cv=5, scoring='r2')
print("Cross-validation R2:", scores)
print("Average R2:", scores.mean())

joblib.dump(reg_model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully!")