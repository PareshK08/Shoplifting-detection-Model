# train_keypoints_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

CSV = r"Asset/Dataset/dataset_keypoints.csv"
df = pd.read_csv(CSV)

# Drop rows with all-zero keypoints if any (optional but helpful)
kpt_cols = [c for c in df.columns if c.startswith("kpt_")]
df = df[df[kpt_cols].sum(axis=1) > 0]

X = df[kpt_cols].values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)  # Normal/Suspicious -> 0/1

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

print("🔎 Validation report:")
print(classification_report(y_val, clf.predict(X_val), target_names=le.classes_))

# Save classifier + label encoder
joblib.dump(clf, "keypoints_rf.pkl")
joblib.dump(le, "keypoints_label_encoder.pkl")
print("✅ Saved: keypoints_rf.pkl, keypoints_label_encoder.pkl")
