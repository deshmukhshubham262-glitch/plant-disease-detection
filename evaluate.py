import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config

# ---- 1. Load the trained model ----
model = tf.keras.models.load_model(config.MODEL_PATH)

# ---- 2. Load class names (same order used during training) ----
with open(config.CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ---- 3. Load the test dataset ----
test_ds = tf.keras.utils.image_dataset_from_directory(
    config.TEST_DIR,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical",
    shuffle=False  # keep order so predictions line up with true labels
)

# ---- 4. Get predictions ----
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---- 5. Print classification report ----
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_names))

# Save metrics as JSON
with open(config.METRICS_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"Metrics saved to {config.METRICS_PATH}")

# ---- 6. Confusion matrix ----
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(config.CONFUSION_MATRIX_PATH)
print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_PATH}")