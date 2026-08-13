import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

import config

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(config.BASE_DIR, "temp_upload")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

CONFIDENCE_THRESHOLD = 0.40

print("Loading TFLite model...")
tflite_path = config.MODEL_PATH.replace(".keras", ".tflite")
interpreter = tflite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(config.CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
print(f"Model loaded. {len(class_names)} classes ready.")

treatments_path = os.path.join(config.MODEL_DIR, "treatments.json")
with open(treatments_path, "r") as f:
    treatments = json.load(f)

def format_disease_only(raw_name):
    parts = raw_name.split("___")
    condition = parts[1] if len(parts) > 1 else parts[0]
    condition = condition.replace("_", " ").replace("(", "").replace(")", "").strip()
    if condition.lower() == "healthy":
        return "Healthy — No Disease Detected"
    return condition

def predict_image(filepath):
    img = Image.open(filepath).convert("RGB").resize(config.IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    top_indices = np.argsort(preds)[::-1][:3]
    results = [(format_disease_only(class_names[i]), float(preds[i])) for i in top_indices]
    top_confidence = results[0][1]
    is_confident = top_confidence >= CONFIDENCE_THRESHOLD

    top_raw_class = class_names[top_indices[0]]
    solution = treatments.get(top_raw_class, "No specific guidance available for this condition.")

    return results, is_confident, solution

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LEAFSCAN // Plant Diagnostic System</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@500;700;900&display=swap');

  :root {
    --neon: #00ffb2;
    --neon-dim: #00ffb233;
    --bg: #05080a;
    --panel: #0c1418;
    --border: #1c2b2e;
    --text: #d3f5e8;
    --muted: #5c7a75;
    --danger: #ff4d4d;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Share Tech Mono', monospace;
    background:
      linear-gradient(rgba(0,255,178,0.03) 1px, transparent 1px) 0 0 / 100% 24px,
      linear-gradient(90deg, rgba(0,255,178,0.03) 1px, transparent 1px) 0 0 / 24px 100%,
      var(--bg);
    color: var(--text);
    margin: 0;
    padding: 32px 16px;
    min-height: 100vh;
  }
  .container { max-width: 620px; margin: 0 auto; }

  header { text-align: center; margin-bottom: 28px; }
  header h1 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 26px;
    letter-spacing: 4px;
    color: var(--neon);
    text-shadow: 0 0 12px var(--neon-dim), 0 0 2px var(--neon);
    margin: 0;
  }
  header p { color: var(--muted); font-size: 12px; letter-spacing: 2px; margin-top: 8px; text-transform: uppercase; }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 18px;
    position: relative;
  }
  .panel::before {
    content: "";
    position: absolute; top: -1px; left: -1px;
    width: 14px; height: 14px;
    border-top: 2px solid var(--neon);
    border-left: 2px solid var(--neon);
  }
  .panel::after {
    content: "";
    position: absolute; bottom: -1px; right: -1px;
    width: 14px; height: 14px;
    border-bottom: 2px solid var(--neon);
    border-right: 2px solid var(--neon);
  }

  #dropzone {
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 46px 20px;
    text-align: center;
    cursor: pointer;
    transition: 0.25s;
  }
  #dropzone:hover, #dropzone.dragover {
    border-color: var(--neon);
    box-shadow: inset 0 0 24px var(--neon-dim);
  }
  #dropzone .icon { font-size: 34px; opacity: 0.8; }
  #dropzone p { margin: 10px 0 0; color: var(--muted); font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
  #fileInput { display: none; }

  #previewWrap { position: relative; display: none; margin-top: 16px; }
  #preview { width: 100%; max-height: 260px; object-fit: cover; border-radius: 4px; border: 1px solid var(--border); display: block; }
  .scan-line {
    position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
    box-shadow: 0 0 8px var(--neon);
    animation: scan 1.6s linear infinite;
    display: none;
  }
  @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }

  button#submitBtn {
    width: 100%; margin-top: 16px; padding: 13px;
    background: transparent; color: var(--neon);
    border: 1px solid var(--neon); border-radius: 3px;
    font-family: 'Orbitron', sans-serif; font-size: 13px; letter-spacing: 3px;
    text-transform: uppercase; cursor: pointer; display: none;
    transition: 0.2s;
  }
  button#submitBtn:hover { background: var(--neon-dim); box-shadow: 0 0 16px var(--neon-dim); }
  button#submitBtn:disabled { opacity: 0.4; cursor: not-allowed; }

  #loading { display: none; text-align: center; padding: 18px; color: var(--neon); font-size: 12px; letter-spacing: 3px; }
  #loading .dots::after { content: ''; animation: dots 1.2s steps(4,end) infinite; }
  @keyframes dots { 0%,20%{content:'';} 40%{content:'.';} 60%{content:'..';} 80%,100%{content:'...';} }

  .status-line { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px; }
  .status-line .ok { color: var(--neon); }
  .status-line .bad { color: var(--danger); }

  .primary-result {
    text-align: center; padding: 14px 0 20px;
    border-bottom: 1px solid var(--border); margin-bottom: 16px;
  }
  .primary-result .label { font-size: 10px; color: var(--muted); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px; }
  .primary-result .disease {
    font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 20px;
    color: var(--neon); text-shadow: 0 0 10px var(--neon-dim);
  }
  .primary-result.low-confidence .disease { color: var(--danger); text-shadow: 0 0 10px rgba(255,77,77,0.3); }

  .result-row { display: flex; justify-content: space-between; align-items: center; margin: 12px 0 4px; font-size: 12px; letter-spacing: 0.5px; }
  .bar-bg { background: #0a1512; border: 1px solid var(--border); border-radius: 2px; height: 6px; width: 100%; overflow: hidden; margin-bottom: 10px; }
  .bar-fill { height: 100%; background: var(--neon); box-shadow: 0 0 6px var(--neon); }

  .footnote { font-size: 10px; color: var(--muted); letter-spacing: 0.5px; margin-top: 14px; line-height: 1.6; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>⟢ LEAFSCAN</h1>
    <p>Plant Diagnostic Imaging System</p>
  </header>

  <div class="panel">
    <div class="status-line">
      <span>SYSTEM STATUS: <span class="ok">ONLINE</span></span>
      <span>CLASSES: {{ num_classes }}</span>
    </div>
    <form id="uploadForm" method="POST" enctype="multipart/form-data">
      <div id="dropzone">
        <div class="icon">▣</div>
        <p>Click or drop leaf image to scan</p>
      </div>
      <div id="previewWrap">
        <img id="preview">
        <div class="scan-line" id="scanLine"></div>
      </div>
      <input type="file" name="image" id="fileInput" accept="image/*" required>
      <button type="submit" id="submitBtn">▶ Run Diagnostic</button>
    </form>
    <div id="loading"><span class="dots">ANALYZING SAMPLE</span></div>
  </div>

  {% if results %}
  <div class="panel">
    <div class="primary-result {{ 'low-confidence' if not is_confident else '' }}">
      <div class="label">{{ 'DIAGNOSIS' if is_confident else '⚠ UNRECOGNIZED SAMPLE' }}</div>
      <div class="disease">
        {% if is_confident %}{{ results[0][0] }}{% else %}NO MATCH IN DATABASE{% endif %}
      </div>
    </div>

    <div style="font-size:10px; color:var(--muted); letter-spacing:2px; margin-bottom:6px;">CONFIDENCE BREAKDOWN</div>
    {% for label, conf in results %}
    <div class="result-row">
      <span>{{ label }}</span>
      <span>{{ "%.1f"|format(conf * 100) }}%</span>
    </div>
    <div class="bar-bg"><div class="bar-fill" style="width: {{ (conf * 100)|round(1) }}%;"></div></div>
    {% endfor %}

    <div class="footnote">
      {% if is_confident %}
      MODEL ACCURACY ≈90% ON TEST DATA. VERIFY WITH AN EXPERT BEFORE TREATMENT.
      {% else %}
      SIGNAL CONFIDENCE BELOW THRESHOLD ({{ (threshold*100)|int }}%). THIS SAMPLE MAY NOT MATCH ANY TRAINED PLANT/DISEASE TYPE.
      {% endif %}
    </div>
  </div>

  {% if is_confident and solution %}
  <div class="panel">
    <div class="label" style="font-size:10px; color:var(--muted); letter-spacing:3px; margin-bottom:10px;">▣ RECOMMENDED ACTION</div>
    <div style="font-size:13px; line-height:1.7; color:var(--text);">{{ solution }}</div>
  </div>
  {% endif %}
  {% endif %}
</div>

<script>
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const previewWrap = document.getElementById('previewWrap');
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
      previewWrap.style.display = 'block';
      submitBtn.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  form.addEventListener('submit', () => {
    submitBtn.disabled = true;
    submitBtn.style.display = 'none';
    loading.style.display = 'block';
    document.getElementById('scanLine').style.display = 'block';
  });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    is_confident = True
    solution = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            results, is_confident, solution = predict_image(filepath)
    return render_template_string(
        PAGE_TEMPLATE,
        results=results,
        is_confident=is_confident,
        threshold=CONFIDENCE_THRESHOLD,
        num_classes=len(class_names),
        solution=solution
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)