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

CROP_ICONS = {
    "Apple": "\U0001F34E", "Blueberry": "\U0001FAD0", "Cherry": "\U0001F352", "Corn": "\U0001F33D",
    "Grape": "\U0001F347", "Orange": "\U0001F34A", "Peach": "\U0001F351", "Pepper,": "\U0001FAD1",
    "Potato": "\U0001F954", "Raspberry": "\U0001F347", "Soybean": "\U0001F331", "Squash": "\U0001F383",
    "Strawberry": "\U0001F353", "Tomato": "\U0001F345"
}

def get_crop_icon(crop_name):
    for key, icon in CROP_ICONS.items():
        if crop_name.startswith(key):
            return icon
    return "\U0001F33F"

def format_disease_only(raw_name):
    parts = raw_name.split("___")
    condition = parts[1] if len(parts) > 1 else parts[0]
    condition = condition.replace("_", " ").replace("(", "").replace(")", "").strip()
    if condition.lower() == "healthy":
        return "Healthy \u2014 No Disease Detected"
    return condition

def build_disease_library():
    library = {}
    for raw_name in class_names:
        parts = raw_name.split("___")
        crop = parts[0].replace("_", " ").strip()
        condition = format_disease_only(raw_name)
        library.setdefault(crop, {"icon": get_crop_icon(parts[0]), "conditions": []})
        library[crop]["conditions"].append(condition)
    return dict(sorted(library.items()))

DISEASE_LIBRARY = build_disease_library()

def looks_like_a_leaf(filepath, min_green_ratio=0.12):
    """Rough heuristic: checks if the image has enough green/plant-like coloring."""
    img = Image.open(filepath).convert("RGB").resize((100, 100))
    pixels = np.array(img, dtype=np.float32)

    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

    # A pixel counts as "plant-like" if green is the dominant or a strong channel,
    # OR if it's a brownish/yellowish tone (covers diseased, dry, or autumn leaves too)
    green_dominant = (g > r) & (g > b * 0.8)
    brownish = (r > 80) & (r < 200) & (g > 60) & (g < 180) & (b < 120) & (r >= g)

    plant_like = green_dominant | brownish
    ratio = np.mean(plant_like)

    return ratio >= min_green_ratio

def predict_image(filepath):
    img = Image.open(filepath).convert("RGB").resize(config.IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    top_indices = np.argsort(preds)[::-1][:3]
    results = [(format_disease_only(class_names[i]), float(preds[i])) for i in top_indices]
    top_raw_class = class_names[top_indices[0]]
    solution = treatments.get(top_raw_class, "No specific guidance available for this condition.")

    return results, solution

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeafCare - Plant Health Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

  :root {
    --green-dark: #14532d;
    --green: #16a34a;
    --green-mid: #22c55e;
    --green-light: #86efac;
    --green-pale: #ecfdf3;
    --yellow: #fbbf24;
    --orange: #fb923c;
    --cream: #fefdfb;
    --card: #ffffff;
    --text: #1a2e22;
    --muted: #6b7c72;
    --shadow: 0 8px 30px rgba(22, 163, 74, 0.10);
  }
  * { box-sizing: border-box; }
  body { font-family: 'Poppins', sans-serif; background: var(--cream); color: var(--text); margin: 0; padding: 0; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 0 20px 60px; }

  .hero {
    background: linear-gradient(135deg, #16a34a, #4ade80 50%, #86efac);
    color: white; padding: 50px 20px 90px; position: relative; overflow: hidden; text-align: center;
  }
  .hero-blob { position: absolute; border-radius: 50%; opacity: 0.25; filter: blur(2px); }
  .blob1 { width: 160px; height: 160px; background: white; top: -40px; left: -40px; }
  .blob2 { width: 100px; height: 100px; background: var(--yellow); bottom: 10px; right: 10%; opacity: 0.35; }
  .blob3 { width: 70px; height: 70px; background: white; top: 40%; right: -20px; }

  .hero-content { position: relative; max-width: 640px; margin: 0 auto; }
  .hero .badge { display: inline-block; background: rgba(255,255,255,0.25); padding: 7px 18px; border-radius: 24px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 18px; backdrop-filter: blur(4px); }
  .hero h1 { font-size: 38px; font-weight: 900; margin: 0 0 12px; text-shadow: 0 2px 12px rgba(0,0,0,0.1); }
  .hero p { font-size: 16px; opacity: 0.95; margin: 0 auto; max-width: 500px; line-height: 1.6; }

  .hero-visual { display: flex; justify-content: center; gap: 6px; margin-top: 28px; font-size: 54px; filter: drop-shadow(0 8px 16px rgba(0,0,0,0.15)); }
  .hero-visual span { display: inline-block; animation: float 3s ease-in-out infinite; }
  .hero-visual span:nth-child(2) { animation-delay: 0.3s; }
  .hero-visual span:nth-child(3) { animation-delay: 0.6s; }
  .hero-visual span:nth-child(4) { animation-delay: 0.9s; }
  .hero-visual span:nth-child(5) { animation-delay: 1.2s; }
  @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }

  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: -50px; position: relative; z-index: 2; }
  .stat-card { background: var(--card); border-radius: 16px; padding: 18px 8px; text-align: center; box-shadow: var(--shadow); }
  .stat-card .num { font-size: 18px; font-weight: 800; color: var(--green); }
  .stat-card .label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.3px; margin-top: 3px; font-weight: 600; }

  .section { margin-top: 46px; }
  .section-title { font-size: 22px; font-weight: 800; color: var(--green-dark); margin: 0 0 4px; }
  .section-sub { font-size: 13px; color: var(--muted); margin: 0 0 18px; }

  .scan-card { background: var(--card); border-radius: 24px; padding: 26px; box-shadow: var(--shadow); border: 1px solid var(--green-pale); }
  #dropzone { border: 3px dashed var(--green-light); border-radius: 18px; padding: 46px 20px; text-align: center; cursor: pointer; background: linear-gradient(160deg, var(--green-pale), #ffffff); transition: 0.25s; }
  #dropzone:hover, #dropzone.dragover { background: var(--green-pale); border-color: var(--green); transform: scale(1.01); }
  #dropzone .icon { font-size: 46px; }
  #dropzone p { margin: 10px 0 0; color: var(--green-dark); font-weight: 600; font-size: 15px; }
  #dropzone span { color: var(--muted); font-size: 12px; }
  #fileInput { display: none; }

  #previewWrap { display: none; margin-top: 16px; }
  #preview { width: 100%; max-height: 280px; object-fit: cover; border-radius: 18px; }

  button#submitBtn {
    width: 100%; margin-top: 16px; padding: 15px;
    background: linear-gradient(135deg, var(--green), var(--green-mid));
    color: white; border: none; border-radius: 14px;
    font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 15px;
    cursor: pointer; display: none; transition: 0.2s;
    box-shadow: 0 6px 16px rgba(22,163,74,0.3);
  }
  button#submitBtn:hover { transform: translateY(-2px); }
  button#submitBtn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  #loading { display: none; text-align: center; padding: 18px; color: var(--green); font-size: 14px; font-weight: 600; }

  .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }
  @media (max-width: 600px) { .result-grid { grid-template-columns: 1fr; } }

  .primary-result { text-align: center; background: linear-gradient(135deg, var(--green-pale), #ffffff); border-radius: 20px; padding: 22px; border: 1px solid var(--green-light); display: flex; flex-direction: column; justify-content: center; }
  .primary-result .label { font-size: 12px; color: var(--green); font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
  .primary-result .disease { font-size: 20px; font-weight: 800; color: var(--green-dark); }

  .chart-box { background: var(--card); border-radius: 20px; padding: 18px; border: 1px solid #eee; display: flex; align-items: center; justify-content: center; }
  .chart-box canvas { max-height: 180px; }

  .result-row { display: flex; justify-content: space-between; font-size: 13px; margin: 10px 0 4px; font-weight: 500; }
  .bar-bg { background: #eef3ee; border-radius: 8px; height: 9px; overflow: hidden; margin-bottom: 10px; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--green-mid), var(--green)); border-radius: 8px; }

  .solution-box { background: linear-gradient(135deg, #fff8ec, #fffdf7); border: 1px solid #fde8b6; border-radius: 18px; padding: 18px; margin-top: 16px; }
  .solution-box .stitle { font-size: 12px; font-weight: 800; color: var(--orange); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .solution-box p { margin: 0; font-size: 13px; line-height: 1.7; }

  .footnote { font-size: 11px; color: var(--muted); margin-top: 16px; text-align: center; }

  .category-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; margin-bottom: 18px; }
  .library-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
  .crop-card { background: var(--card); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); border: 1px solid #f0f0f0; transition: 0.2s; }
  .crop-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(22,163,74,0.15); }
  .crop-card .crop-head { display: flex; align-items: center; gap: 10px; font-weight: 700; color: var(--green-dark); margin-bottom: 10px; font-size: 15px; }
  .crop-card .crop-head .emoji { font-size: 28px; }
  .crop-card ul { margin: 0; padding-left: 18px; font-size: 12px; color: var(--muted); line-height: 1.9; }

  .tips-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
  .tip-card { background: var(--card); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); border: 1px solid #f0f0f0; }
  .tip-card .temoji { font-size: 30px; margin-bottom: 8px; }
  .tip-card h4 { margin: 0 0 4px; font-size: 14px; color: var(--green-dark); font-weight: 700; }
  .tip-card p { margin: 0; font-size: 12px; color: var(--muted); line-height: 1.6; }

  footer { text-align: center; color: var(--muted); font-size: 11px; margin-top: 56px; padding-top: 20px; border-top: 1px solid #eee; }

  @media (max-width: 480px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .hero h1 { font-size: 28px; }
    .hero-visual { font-size: 38px; }
  }
</style>
</head>
<body>
  <div class="hero">
    <div class="hero-blob blob1"></div>
    <div class="hero-blob blob2"></div>
    <div class="hero-blob blob3"></div>
    <div class="hero-content">
      <span class="badge">\U0001F33F AI-Powered Plant Health Scanner</span>
      <h1>LeafCare Dashboard</h1>
      <p>Snap a photo of a leaf and get an instant health check &mdash; plus simple steps to help it recover.</p>
      <div class="hero-visual"><span>\U0001F345</span><span>\U0001F34E</span><span>\U0001F33D</span><span>\U0001F347</span><span>\U0001F353</span></div>
    </div>
  </div>

  <div class="wrap">
    <div class="stats">
      <div class="stat-card"><div class="num">Multiple</div><div class="label">Plant Types</div></div>
      <div class="stat-card"><div class="num">Multiple</div><div class="label">Conditions</div></div>
      <div class="stat-card"><div class="num">94%</div><div class="label">Accuracy</div></div>
      <div class="stat-card"><div class="num">Free</div><div class="label">To Use</div></div>
    </div>

    <div class="section">
      <div class="section-title">\U0001F50D Scan Your Leaf</div>
      <div class="section-sub">Upload a clear, well-lit photo for the most accurate reading.</div>

      <div class="scan-card">
        <form id="uploadForm" method="POST" enctype="multipart/form-data">
          <div id="dropzone">
            <div class="icon">\U0001F331</div>
            <p>Click or drag a photo here</p>
            <span>JPG, PNG supported</span>
          </div>
          <div id="previewWrap"><img id="preview"></div>
          <input type="file" name="image" id="fileInput" accept="image/*" required>
          <button type="submit" id="submitBtn">\u2728 Analyze Leaf</button>
        </form>
        <div id="loading">\U0001F33F Analyzing your leaf...</div>
         
      {% if not_a_leaf %}
      <div class="solution-box" style="background:#fef2f2; border-color:#fecaca; margin-top:18px;">
          <div class="stitle" style="color:#dc2626;">\u26A0 No Leaf Detected</div>
          <p>This image doesn't appear to show a plant leaf. Please upload a clear photo of a single leaf for analysis.</p>
      </div>
      {% endif %}

        {% if results %}
        <div class="result-grid">
          <div class="primary-result">
            <div class="label">Diagnosis</div>
            <div class="disease">{{ results[0][0] }}</div>
          </div>
          <div class="chart-box"><canvas id="confChart"></canvas></div>
        </div>

        {% for label, conf in results %}
        <div class="result-row"><span>{{ label }}</span><span>{{ "%.1f"|format(conf * 100) }}%</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width: {{ (conf * 100)|round(1) }}%;"></div></div>
        {% endfor %}

        {% if solution %}
        <div class="solution-box">
          <div class="stitle">\U0001FA7A Recommended Action</div>
          <p>{{ solution }}</p>
        </div>
        {% endif %}

        <div class="footnote">Based on visual pattern matching &mdash; always confirm with an expert before treatment.</div>

        <script>
          new Chart(document.getElementById('confChart'), {
            type: 'doughnut',
            data: {
              labels: [{% for label, conf in results %}"{{ label }}",{% endfor %}],
              datasets: [{
                data: [{% for label, conf in results %}{{ (conf*100)|round(1) }},{% endfor %}],
                backgroundColor: ['#16a34a', '#4ade80', '#bbf7d0'],
                borderWidth: 0
              }]
            },
            options: {
              plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } },
              cutout: '65%'
            }
          });
        </script>
        {% endif %}
      </div>
    </div>

    <div class="section">
      <div class="section-title">\U0001F4DA What We Cover</div>
      <div class="section-sub">LeafCare recognizes a wide range of plant health conditions across several categories.</div>

      <div class="category-grid">
        <div class="tip-card"><div class="temoji">\U0001F344</div><h4>Fungal Diseases</h4><p>Blights, rusts, mildews, and leaf spots caused by fungal infection.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F9A0</div><h4>Bacterial Infections</h4><p>Spots, cankers, and rot caused by bacterial pathogens.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F9EC</div><h4>Viral Conditions</h4><p>Mosaic patterns, curling, and discoloration from viral infection.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F577\uFE0F</div><h4>Pest Damage</h4><p>Visible damage patterns caused by mites and other pests.</p></div>
      </div>

      <div class="library-grid">
        {% for crop, data in library.items() %}
        <div class="crop-card">
          <div class="crop-head"><span class="emoji">{{ data.icon }}</span> {{ crop }}</div>
          <ul>{% for item in data.conditions %}<li>{{ item }}</li>{% endfor %}</ul>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="section">
      <div class="section-title">\U0001F4A1 General Plant Care Tips</div>
      <div class="section-sub">Simple habits that help prevent disease before it starts.</div>
      <div class="tips-grid">
        <div class="tip-card"><div class="temoji">\U0001F4A7</div><h4>Water at the Base</h4><p>Avoid wetting leaves &mdash; most fungal diseases spread through moisture on foliage.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F32C\uFE0F</div><h4>Improve Airflow</h4><p>Space plants out and prune dense growth so leaves dry quickly.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F504</div><h4>Rotate Growing Areas</h4><p>Avoid planting the same thing in the same spot every season.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F9E4</div><h4>Clean Tools &amp; Hands</h4><p>Disinfect tools between plants to avoid spreading unseen infections.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F342</div><h4>Remove Fallen Debris</h4><p>Old leaves and fruit on the ground often harbor fungal spores.</p></div>
        <div class="tip-card"><div class="temoji">\U0001F50E</div><h4>Inspect Regularly</h4><p>Catching disease early makes treatment far more effective.</p></div>
      </div>
    </div>

    <footer>Built with a custom-trained AI model &middot; Educational use only, not a substitute for expert diagnosis</footer>
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
  });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    solution = None
    not_a_leaf = False
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            if looks_like_a_leaf(filepath):
                results, solution = predict_image(filepath)
            else:
                not_a_leaf = True

    return render_template_string(
        PAGE_TEMPLATE,
        results=results,
        solution=solution,
        library=DISEASE_LIBRARY,
        not_a_leaf=not_a_leaf
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)