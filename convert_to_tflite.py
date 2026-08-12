import tensorflow as tf
import config

# Load your trained Keras model
model = tf.keras.models.load_model(config.MODEL_PATH)

# Wrap the model in a tf.function with a fixed input shape
@tf.function(input_signature=[tf.TensorSpec(shape=[1] + list(config.IMG_SIZE) + [3], dtype=tf.float32)])
def serving_fn(x):
    return model(x, training=False)

concrete_func = serving_fn.get_concrete_function()

# Convert directly from the concrete function (avoids the buggy Keras conversion path)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = config.MODEL_PATH.replace(".keras", ".tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_path}")