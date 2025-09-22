# split_dataset.py
import os, shutil, random
from glob import glob

random.seed(42)

SRC = r"Asset/Dataset"         # has Normal/ and Suspicious/
DST = r"Asset/DatasetSplit"    # will create train/ and val/
VAL_RATIO = 0.2                 # 20% for validation

classes = ["Normal", "Suspicious"]
for split in ["train", "val"]:
    for c in classes:
        os.makedirs(os.path.join(DST, split, c), exist_ok=True)

for c in classes:
    files = sum([glob(os.path.join(SRC, c, ext)) for ext in ("*.jpg","*.png","*.jpeg")], [])
    random.shuffle(files)
    val_n = int(len(files) * VAL_RATIO)
    val_files = files[:val_n]
    train_files = files[val_n:]

    for f in train_files:
        shutil.copy2(f, os.path.join(DST, "train", c, os.path.basename(f)))
    for f in val_files:
        shutil.copy2(f, os.path.join(DST, "val", c, os.path.basename(f)))

print("✅ Split complete:", DST)
