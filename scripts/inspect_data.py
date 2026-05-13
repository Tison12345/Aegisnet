import pandas as pd
from pathlib import Path

df = pd.read_csv(r"C:\Ageisnet\data\train.csv")

for label in ["alert", "drowsy"]:
    sub = df[df["label"] == label].sort_values("frame_path")
    names = [Path(p).stem for p in sub["frame_path"].head(8)]
    print(f"{label} first 8 filenames: {names}")

print()
print("Train shape:", df.shape)
print("Roll std:", round(df["roll"].std(), 2), "  -- checking for remaining gimbal issues")
print("Roll IQR:", round(df["roll"].quantile(0.25), 2), "to", round(df["roll"].quantile(0.75), 2))
print("Roll values > 90:", (df["roll"].abs() > 90).sum())
