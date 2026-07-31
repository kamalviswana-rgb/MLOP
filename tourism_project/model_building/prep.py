import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load dataset
DATA_PATH = Path("/content/tourism_project/data/tourism.csv")

df = pd.read_csv(DATA_PATH)

# Remove unnecessary columns
columns_to_drop = [
    "CustomerID"
]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

# Save split datasets
OUTPUT_DIR = Path("/content/tourism_project/data")

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

print("Data preparation completed")
print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")

print("\nFiles created:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "test.csv")
