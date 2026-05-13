"""
Step 3: Split features.csv into train.csv (80%) and test.csv (20%).

Stratified by class label so the class ratio is preserved in both splits.
Output: data/train.csv, data/test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

CSV_IN    = Path(r"C:/Ageisnet/data/features.csv")
TRAIN_OUT = Path(r"C:/Ageisnet/data/train.csv")
TEST_OUT  = Path(r"C:/Ageisnet/data/test.csv")
SEED      = 42

if __name__ == "__main__":
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} rows from {CSV_IN}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT,   index=False)

    print(f"Train: {len(train)} rows -> {TRAIN_OUT}")
    print(f"Test:  {len(test)}  rows -> {TEST_OUT}")
    print(f"\nTrain label counts:\n{train['label'].value_counts()}")
    print(f"\nTest  label counts:\n{test['label'].value_counts()}")
