import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import config

# ---- 1. Load datasets from folders ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    config.TRAIN_DIR,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    config.VAL_DIR,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Save class names so predict.py / app.py know the label order
os.makedirs(config.MODEL_DIR, exist_ok=True)
with open(config.CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# Speed up data loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ---- 2. Data augmentation (helps the model generalize) ----
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ---- 3. Build the model using MobileNetV2 (transfer learning) ----
base_model = tf.keras.applications.MobileNetV2(
    input_shape=config.IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze pretrained layers for now

inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---- 4. Train (with checkpointing so you can stop and resume) ----
checkpoint_path = os.path.join(config.MODEL_DIR, "checkpoint.keras")

# If a checkpoint exists from a previous run, resume from it
initial_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Found existing checkpoint at {checkpoint_path}, resuming training...")
    model = tf.keras.models.load_model(checkpoint_path)
    # Read how many epochs were already done, if we saved that info
    epoch_log_path = os.path.join(config.MODEL_DIR, "last_epoch.txt")
    if os.path.exists(epoch_log_path):
        with open(epoch_log_path, "r") as f:
            initial_epoch = int(f.read().strip())
        print(f"Resuming from epoch {initial_epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq="epoch"
)

class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(os.path.join(config.MODEL_DIR, "last_epoch.txt"), "w") as f:
            f.write(str(epoch + 1))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback, EpochLogger()]
)

# ---- 5. Save the model ----
model.save(config.MODEL_PATH)
print(f"Model saved to {config.MODEL_PATH}")

# ---- 6. Plot and save training history ----
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over epochs")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over epochs")

plt.tight_layout()
plt.savefig(config.TRAINING_HISTORY_PATH)
print(f"Training history plot saved to {config.TRAINING_HISTORY_PATH}")