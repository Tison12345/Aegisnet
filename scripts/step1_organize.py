"""
Step 1: Organize raw dataset frames into the project folder structure.

Source:  C:/Users/preet/Downloads/archive/Driver Drowsiness Dataset (DDD)/
         ├── Drowsy/      → data/processed/drowsy/
         └── Non Drowsy/  → data/processed/alert/

No re-encoding is done — files are copied as-is.
"""

import shutil
from pathlib import Path

SRC_DROWSY    = Path(r"C:/Users/preet/Downloads/archive/Driver Drowsiness Dataset (DDD)/Drowsy")
SRC_ALERT     = Path(r"C:/Users/preet/Downloads/archive/Driver Drowsiness Dataset (DDD)/Non Drowsy")
DST_DROWSY    = Path(r"C:/Ageisnet/data/processed/drowsy")
DST_ALERT     = Path(r"C:/Ageisnet/data/processed/alert")

def copy_class(src: Path, dst: Path, label: str) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    for i, f in enumerate(files):
        shutil.copy2(f, dst / f.name)
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] copied {i+1}/{len(files)}")
    print(f"  [{label}] done - {len(files)} files -> {dst}")
    return len(files)

if __name__ == "__main__":
    print("Organising dataset...")
    n_drowsy = copy_class(SRC_DROWSY, DST_DROWSY, "drowsy")
    n_alert  = copy_class(SRC_ALERT,  DST_ALERT,  "alert")
    print(f"\nTotal: {n_drowsy + n_alert} frames  (drowsy={n_drowsy}, alert={n_alert})")
