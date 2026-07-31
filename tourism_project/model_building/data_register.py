import pandas as pd
from pathlib import Path

# Dataset path
DATA_PATH = Path("/content/tourism_project/data/tourism.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Expected columns
expected_columns = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "ProdTaken"
]

# Validate columns
missing_cols = set(expected_columns) - set(df.columns)

if missing_cols:
    raise ValueError(
        f"Missing columns in dataset: {missing_cols}"
    )

print("=" * 50)
print("Dataset Registration Successful")
print("=" * 50)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print("\nColumn Names:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["ProdTaken"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())
