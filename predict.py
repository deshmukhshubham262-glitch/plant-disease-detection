import sys
import json
import numpy as np
import tensorflow as tf

import config

def predict_image(image_path):
    # Load model
    model = tf.keras.models.load_model(config.MODEL_PATH)

    # Load class names
    with open(config.CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")

    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)