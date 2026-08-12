import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# Image settings
IMG_SIZE = (224, 224)   # width, height fed into the model
BATCH_SIZE = 16

# Training settings
EPOCHS = 15
LEARNING_RATE = 0.0001