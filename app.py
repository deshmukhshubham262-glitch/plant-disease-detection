import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # fallback if running locally with full TF

import config

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(config.BASE_DIR, "temp_upload")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

print("Loading TFLite model...")
tflite_path = config.MODEL_PATH.replace(".keras", ".tflite")
interpreter = tflite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(config.CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
print("Model loaded. Ready to predict.")

def format_label(raw_name):
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else ""
    return f"{crop} — {condition}"

def predict_image(filepath):
    img = Image.open(filepath).convert("RGB").resize(config.IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    top_indices = np.argsort(preds)[::-1][:3]
    return [(format_label(class_names[i]), f"{preds[i]:.1%}") for i in top_indices]

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plant Disease Detector</title>
<style>
  :root { --green: #2e7d32; --green-light: #e8f5e9; --gray-bg: #f7f8f7; --text: #1a1a1a; --muted: #6b7280; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; background: var(--gray-bg); margin: 0; padding: 40px 20px; color: var(--text); }
  .container { max-width: 640px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 32px; }
  header h1 { font-size: 28px; margin: 0; color: var(--green); }
  header p { color: var(--muted); margin-top: 6px; font-size: 14px; }
  .card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 20px; }
  #dropzone { border: 2px dashed #c8e6c9; border-radius: 12px; padding: 40px 20px; text-align: center; cursor: pointer; transition: 0.2s; }
  #dropzone:hover, #dropzone.dragover { border-color: var(--green); background: var(--green-light); }
  #dropzone p { margin: 8px 0 0; color: var(--muted); font-size: 14px; }
  #dropzone .icon { font-size: 36px; }
  #fileInput { display: none; }
  #preview { display: none; max-width: 100%; max-height: 260px; border-radius: 12px; margin-top: 16px; }
  button#submitBtn { width: 100%; margin-top: 16px; padding: 12px; background: var(--green); color: white; border: none; border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer; display: none; }
  button#submitBtn:hover { background: #276b2b; }
  button#submitBtn:disabled { background: #a5d6a7; cursor: not-allowed; }
  #loading { display: none; text-align: center; padding: 20px; color: var(--muted); }
  .spinner { width: 28px; height: 28px; border: 3px solid #d7e8d8; border-top-color: var(--green); border-radius: 50%; margin: 0 auto 10px; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .result-row { display: flex; justify-content: space-between; align-items: center; margin: 10px 0; }
  .result-label { font-size: 14px; }
  .result-label.top { font-weight: 700; font-size: 16px; }
  .bar-bg { background: #eee; border-radius: 6px; height: 8px; width: 100%; margin-top: 4px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--green); border-radius: 6px; }
  .result-item { margin-bottom: 12px; }
  .confidence-note { font-size: 12px; color: var(--muted); margin-top: 10px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf photo to check for common tomato diseases</p>
  </header>
  <div class="card">
    <form id="uploadForm" method="POST" enctype="multipart/form-data">
      <div id="dropzone">
        <div class="icon">📷</div>
        <p><strong>Click to upload</strong> or drag a photo here</p>
        <img id="preview">
      </div>
      <input type="file" name="image" id="fileInput" accept="image/*" required>
      <button type="submit" id="submitBtn">Analyze Leaf</button>
    </form>
    <div id="loading"><div class="spinner"></div>Analyzing image...</div>
  </div>
  {% if results %}
  <div class="card">
    <h3 style="margin-top:0;">Results</h3>
    {% for label, conf in results %}
    <div class="result-item">
      <div class="result-row">
        <span class="result-label {{ 'top' if loop.first else '' }}">{{ label }}</span>
        <span class="result-label">{{ conf }}</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width: {{ conf }};"></div></div>
    </div>
    {% endfor %}
    <p class="confidence-note">Model accuracy is around 90% on test data, but always confirm with an expert before treating your plants.</p>
  </div>
  {% endif %}
</div>
<script>
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const submitBtn = document.getElementById('submitBtn');
  const form = document.getElementById('uploadForm');
  const loading = document.getElementById('loading');
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    fileInput.files = e.dataTransfer.files;
    showPreview();
  });
  fileInput.addEventListener('change', showPreview);
  function showPreview() {
    const file = fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = 'block';
      submitBtn.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }
  form.addEventListener('submit', () => {
    submitBtn.disabled = true;
    submitBtn.style.display = 'none';
    loading.style.display = 'block';
  });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            results = predict_image(filepath)
    return render_template_string(PAGE_TEMPLATE, results=results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)