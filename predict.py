import sys
import json
import numpy as np
import tensorflow as tf

import config

CONFIDENCE_THRESHOLD = 0.40

def format_disease_only(raw_name):
    """Return only the condition/disease part, not the crop name."""
    parts = raw_name.split("___")
    condition = parts[1] if len(parts) > 1 else parts[0]
    condition = condition.replace("_", " ").replace("(", "").replace(")", "").strip()
    if condition.lower() == "healthy":
        return "Healthy — No Disease Detected"
    return condition

def predict_image(image_path):
    # Load model
    model = tf.keras.models.load_model(config.MODEL_PATH)

    # Load class names
    with open(config.CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    # Load treatments
    import os
    treatments_path = os.path.join(config.MODEL_DIR, "treatments.json")
    with open(treatments_path, "r") as f:
        treatments = json.load(f)

    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    raw_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    print("-" * 40)
    if confidence >= CONFIDENCE_THRESHOLD:
        disease_name = format_disease_only(raw_class)
        solution = treatments.get(raw_class, "No specific guidance available.")
        print(f"Diagnosis: {disease_name}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Recommended action: {solution}")
    else:
        print("No match in database.")
        print(f"(Top guess was '{format_disease_only(raw_class)}' at only {confidence:.2%} confidence \u2014 too low to trust)")
    print("-" * 40)

    return raw_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)