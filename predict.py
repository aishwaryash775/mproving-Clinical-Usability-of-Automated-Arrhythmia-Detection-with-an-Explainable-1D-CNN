import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH = "model-00006-0.159-0.952-0.002-0.961.h5"
ECG_FILE = "ecg_100.csv"
WINDOW_SIZE = 100

print("🔹 Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print("Model input shape:", model.input_shape)

# Step 1: Read ECG
print("🔹 Reading ECG data...")
df = pd.read_csv(ECG_FILE)
if 'MLII' in df.columns:
    ecg_signal = df['MLII'].values
else:
    ecg_signal = df.iloc[:, 0].values

print(f"✅ ECG signal length: {len(ecg_signal)} samples")

# Step 2: Normalize
ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

# Step 3: Split into 100-sample windows
segments = []
for i in range(0, len(ecg_signal) - WINDOW_SIZE, WINDOW_SIZE):
    segment = ecg_signal[i:i + WINDOW_SIZE]
    segments.append(segment)
segments = np.array(segments)

# 👇 Reshape for TimeDistributed Conv2D
segments = segments.reshape(-1, 1, 100, 1, 1)
print(f"✅ Prepared {segments.shape[0]} segments for prediction, shape: {segments.shape}")

# Step 4: Predict
print("🔹 Running predictions...")
preds = model.predict(segments, verbose=1)

predicted_classes = np.argmax(preds, axis=1)
confidence = np.max(preds, axis=1)

ZONES = {
    0: "🟢 Green Zone (Normal)",
    1: "🟡 Yellow Zone (Borderline)",
    2: "🔴 Red Zone (Arrhythmia)"
}

(unique, counts) = np.unique(predicted_classes, return_counts=True)
majority_zone = unique[np.argmax(counts)]

print("\n✅ Final Zone Classification:")
print(f"   {ZONES.get(majority_zone, 'Unknown Zone')}")
print(f"   (Model Confidence ≈ {np.mean(confidence)*100:.2f}%)")

# Save results
output = pd.DataFrame({
    "segment_id": range(len(predicted_classes)),
    "predicted_class": predicted_classes,
    "confidence": confidence
})
output.to_csv("predicted_results.csv", index=False)
print("📁 Results saved to predicted_results.csv")
