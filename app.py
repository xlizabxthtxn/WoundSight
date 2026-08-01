"""
WoundSight Flask Application
"""

import uuid
import argparse
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_DIR = Path("static/uploads")
RESULT_DIR = Path("static/results")
CURVES_DIR = Path("static/curves")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
CLASS_NAMES = ["background", "normal", "diabetic", "pressure", "surgical", "venous"]

CLASS_COLORS = {
    "background": "#8B7FA8",
    "normal":     "#4ABFB0",
    "diabetic":   "#E8A842",
    "pressure":   "#C4607A",
    "surgical":   "#7B9ED9",
    "venous":     "#D4845A",
}

CLASS_LABELS = {
    "background": "Background / Non-Wound",
    "normal":     "Normal Healthy Skin",
    "diabetic":   "Diabetic Ulcer",
    "pressure":   "Pressure Wound",
    "surgical":   "Surgical Wound",
    "venous":     "Venous Ulcer",
}

SEVERITY = {
    "background": "None",
    "normal":     "None",
    "diabetic":   "High",
    "pressure":   "High",
    "surgical":   "Moderate",
    "venous":     "High",
}

FIRST_AID = {
    "background": "No wound detected in this region.",
    "normal":     "Healthy skin — no treatment needed.",
    "diabetic":   "Clean gently. Do NOT self-treat. Seek immediate medical care.",
    "pressure":   "Relieve pressure immediately. Keep clean. See wound care specialist.",
    "surgical":   "Keep sterile. Do not remove dressing. Follow surgeon instructions.",
    "venous":     "Elevate affected limb. Apply compression bandage. Seek vascular specialist.",
}

DOCTOR_RECS = {
    "background": ["No wound detected. No clinical action required."],
    "normal":     ["Skin appears healthy. Schedule routine review if patient has risk factors."],
    "diabetic":   ["Diabetic ulcer detected — urgent care required.", "Refer to diabetic specialist.", "Assess vascular status.", "Review glycaemic control.", "This AI report supports but does not replace clinical judgement."],
    "pressure":   ["Pressure wound detected — classify stage and document.", "Relieve pressure immediately — reposition every 2 hours.", "Refer to wound care specialist.", "This AI report supports but does not replace clinical judgement."],
    "surgical":   ["Post-operative wound detected.", "Assess for signs of infection.", "Ensure sterile dressing technique.", "This AI report supports but does not replace clinical judgement."],
    "venous":     ["Venous ulcer detected.", "Confirm with ABPI before applying compression.", "Initiate compression bandaging if ABPI > 0.8.", "This AI report supports but does not replace clinical judgement."],
}

PATIENT_RECS = {
    "background": ["No wound was found in this image."],
    "normal":     ["Your skin appears healthy.", "Continue good skin hygiene practices."],
    "diabetic":   ["A diabetic wound has been detected. Please see your doctor today.", "Do not attempt to self-treat.", "Keep the wound clean and dry."],
    "pressure":   ["A pressure wound has been detected. Inform your nurse or doctor immediately.", "Change your position regularly — at least every 2 hours."],
    "surgical":   ["Keep the area clean and dry as instructed by your surgeon.", "Do not remove your dressing without medical advice."],
    "venous":     ["Elevate your limb above heart level as much as possible.", "Wear your compression stockings as instructed."],
}

# ── FIX 1: Detection uses Auto_250 path (runs are saved as Auto_xxx) ─────────
DETECTION_MODELS = [
    ("AdamW 250ep ⭐", "runs/detect/runs/AdamW_250/weights/best.pt"),
    ("SGD 250ep ⭐",   "runs/detect/runs/SGD_250/weights/best.pt"),
]

# ── Metrics: Phase 1 (model size) + Phase 2 (optimiser) ─────────────────────
METRICS_RUNS = {
    # Phase 1 — model size comparison
    "Nano_SGD_250" : "runs/detect/runs/Nano_SGD_250/results.csv",
    "SGD_250"      : "runs/detect/runs/SGD_250/results.csv",
    "Large_SGD_250": "runs/detect/runs/Large_SGD_250/results.csv",
    # Phase 2 — optimiser comparison (YOLO26m)
    "AdamW_50"  : "runs/detect/runs/AdamW_50/results.csv",
    "AdamW_100" : "runs/detect/runs/AdamW_100/results.csv",
    "AdamW_150" : "runs/detect/runs/AdamW_150/results.csv",
    "AdamW_200" : "runs/detect/runs/AdamW_200/results.csv",
    "AdamW_250" : "runs/detect/runs/AdamW_250/results.csv",
    "SGD_50"    : "runs/detect/runs/SGD_50/results.csv",
    "SGD_100"   : "runs/detect/runs/SGD_100/results.csv",
    "SGD_150"   : "runs/detect/runs/SGD_150/results.csv",
    "SGD_200"   : "runs/detect/runs/SGD_200/results.csv",
    "SGD_250"   : "runs/detect/runs/SGD_250/results.csv",
}

app        = Flask(__name__)
model      = None
all_models = {}


def load_models():
    global model, all_models
    from ultralytics import YOLO

    for run_name, weights_path in DETECTION_MODELS:
        p = Path(weights_path)
        if p.exists():
            print(f"  Loading: {run_name}")
            all_models[run_name] = YOLO(str(p))
        else:
            print(f"  Skipping: {run_name} — not found")

    if not all_models:
        raise FileNotFoundError("No trained models found.")

    model = all_models.get("SGD 250ep ⭐") or list(all_models.values())[0]
    print(f"Primary model: SGD 250ep (F1: 0.850)")


def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def get_recommendations(cls, conf):
    URGENCY = {
        "None":     "No action required",
        "Low":      "Non-urgent — schedule review soon",
        "Moderate": "Prompt — review within 48 hours",
        "High":     "Urgent — seek medical attention today",
    }
    sev = SEVERITY.get(cls, "None")
    if conf >= 85:
        note = f"High confidence ({conf:.1f}%) — result is reliable."
    elif conf >= 65:
        note = f"Moderate confidence ({conf:.1f}%) — consider clinical review."
    else:
        note = f"Low confidence ({conf:.1f}%) — manual clinical review strongly advised."
    return {
        "risk":         sev,
        "urgency":      URGENCY.get(sev, "Seek medical attention"),
        "conf_note":    note,
        "doctor_recs":  DOCTOR_RECS.get(cls, ["Seek clinical review."]),
        "patient_recs": PATIENT_RECS.get(cls, ["Seek medical attention."]),
    }


def get_run_metrics(run_name):
    clean = run_name.replace(" ⭐","").replace(" ","_")
    mapping = {
        "AdamW_250ep": "AdamW_250",
        "SGD_250ep":   "SGD_250",
    }
    key = mapping.get(clean, clean)
    csv_path = METRICS_RUNS.get(key)
    if not csv_path or not Path(csv_path).exists():
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["F1"] = (2 * df["metrics/precision(B)"] * df["metrics/recall(B)"] /
                    (df["metrics/precision(B)"] + df["metrics/recall(B)"])).fillna(0)
        best = df.loc[df["metrics/mAP50(B)"].idxmax()]
        return (round(float(best["metrics/mAP50(B)"]) * 100, 1),
                round(float(best["F1"]) * 100, 1))
    except Exception as e:
        print(f"Metrics error for {run_name}: {e}")
        return None, None


def draw_boxes(img, boxes):
    import cv2
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id >= len(CLASS_NAMES):
            continue
        cls_name  = CLASS_NAMES[cls_id]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color_bgr = hex_to_bgr(CLASS_COLORS.get(cls_name, "#8B7FA8"))
        cv2.rectangle(img, (x1,y1), (x2,y2), color_bgr, 2)
        label = f"{cls_name} {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1,y1-th-8), (x1+tw+4,y1), color_bgr, -1)
        cv2.putText(img, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return img


def load_metrics():
    runs = []
    try:
        import pandas as pd
    except ImportError:
        return runs
    for run_name, csv_path in METRICS_RUNS.items():
        if not Path(csv_path).exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df["F1"] = (2 * df["metrics/precision(B)"] * df["metrics/recall(B)"] /
                        (df["metrics/precision(B)"] + df["metrics/recall(B)"])).fillna(0)
            best = df.loc[df["metrics/mAP50(B)"].idxmax()]
            p1_names = {"Nano_SGD_250", "SGD_250", "Large_SGD_250"}
            # FIX 5: correct optimizer detection
            if run_name.startswith("AdamW") or run_name.startswith("Auto"):
                opt = "AdamW"
            else:
                opt = "SGD"
            if run_name.startswith("Nano"):
                model_size = "YOLO26n (Nano)"
            elif run_name.startswith("Large"):
                model_size = "YOLO26l (Large)"
            else:
                model_size = "YOLO26m (Medium)"
            runs.append({
                "run_name":     run_name,
                "optimizer":    opt,
                "model_size":   model_size,
                "phase":        "Phase1" if run_name in p1_names else "Phase2",
                "total_epochs": int(df.iloc[-1]["epoch"]),
                "best_epoch":   int(best["epoch"]),
                "map50":        round(float(best["metrics/mAP50(B)"]),    4),
                "map5095":      round(float(best["metrics/mAP50-95(B)"]), 4),
                "precision":    round(float(best["metrics/precision(B)"]),4),
                "recall":       round(float(best["metrics/recall(B)"]),   4),
                "f1":           round(float(best["F1"]),                   4),
            })
        except Exception as e:
            print(f"Warning loading {run_name}: {e}")
    return runs


def make_curves():
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # FIX: use AdamW_ prefix for curve generation
        adamw_runs = {k:v for k,v in METRICS_RUNS.items() if k.startswith("AdamW")}
        sgd_runs   = {k:v for k,v in METRICS_RUNS.items() if k.startswith("SGD") and not k.startswith("SGD_250") or k == "SGD_250"}
        # Only Phase 2 SGD for curves (not Phase 1 nano/large)
        sgd_p2 = {k:v for k,v in METRICS_RUNS.items() if k.startswith("SGD_") and not any(x in k for x in ["Nano","Large"])}

        auto_colors = ["#E8A842","#4ABFB0","#C4607A","#7B9ED9","#D4845A","#8B7FA8"]
        sgd_colors  = ["#7B9ED9","#4ABFB0","#E8A842","#C4607A","#D4845A","#8B7FA8"]
        images = []
        BG, CARD, BORDER, TEXT, DIM = "#1A1035","#231545","#3D2F7A","#F0EBF8","#B8A8D4"

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.patch.set_facecolor(CARD)
        for ax in axes: ax.set_facecolor(BG)

        for i,(name,path) in enumerate(sorted(adamw_runs.items(), key=lambda x: int(''.join(filter(str.isdigit,x[0])) or 0))):
            if not Path(path).exists(): continue
            df = pd.read_csv(path); df.columns = df.columns.str.strip()
            axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], color=auto_colors[i%len(auto_colors)], lw=2, label=name)

        for i,(name,path) in enumerate(sorted(sgd_p2.items(), key=lambda x: int(''.join(filter(str.isdigit,x[0])) or 0))):
            if not Path(path).exists(): continue
            df = pd.read_csv(path); df.columns = df.columns.str.strip()
            axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], color=sgd_colors[i%len(sgd_colors)], lw=2, label=name)

        for ax, title in zip(axes, ["mAP50 — AdamW", "mAP50 — SGD"]):
            ax.set_xlabel("Epoch", color=DIM); ax.set_ylabel("mAP50", color=DIM)
            ax.set_title(title, color=TEXT, fontsize=12)
            ax.tick_params(colors=DIM); ax.grid(True, color=BORDER, lw=0.7)
            ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        plt.tight_layout()
        plt.savefig(CURVES_DIR/"map50.png", dpi=130, bbox_inches="tight", facecolor=CARD)
        plt.close()
        images.append({"filename":"map50.png","label":"mAP50 — AdamW vs SGD"})

        fig, axes = plt.subplots(1, 2, figsize=(18,7))
        fig.patch.set_facecolor(CARD)
        for ax in axes: ax.set_facecolor(BG)
        all_p2 = {**adamw_runs, **sgd_p2}
        all_colors = auto_colors + sgd_colors
        for i,(name,path) in enumerate(sorted(all_p2.items())):
            if not Path(path).exists(): continue
            df = pd.read_csv(path); df.columns = df.columns.str.strip()
            col = all_colors[i%len(all_colors)]
            axes[0].plot(df["epoch"], df["train/box_loss"], color=col, lw=1.5, label=name)
            axes[1].plot(df["epoch"], df["train/cls_loss"], color=col, lw=1.5, label=name)
        for ax, title in zip(axes, ["Box Loss","Class Loss"]):
            ax.set_xlabel("Epoch",color=DIM); ax.set_ylabel("Loss",color=DIM)
            ax.set_title(title,color=TEXT,fontsize=12)
            ax.tick_params(colors=DIM); ax.grid(True,color=BORDER,lw=0.7)
            ax.legend(facecolor=CARD,labelcolor=TEXT,fontsize=7,ncol=2)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout()
        plt.savefig(CURVES_DIR/"loss.png", dpi=130, bbox_inches="tight", facecolor=CARD)
        plt.close()
        images.append({"filename":"loss.png","label":"Loss Curves — AdamW vs SGD"})
        return images
    except Exception as e:
        print(f"Curve error: {e}"); return []


PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700;800;900&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1A1035;color:#F0EBF8;font-family:'Inter',sans-serif;min-height:100vh}
nav{background:#0D1B3E;border-bottom:1px solid #3D2F7A;padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:60px;position:sticky;top:0;z-index:100}
.brand{font-size:22px;font-weight:900;color:#F0EBF8;font-family:'DM Sans',sans-serif;letter-spacing:2px}.brand span{color:#E8A842}
.nav-links a{color:#B8A8D4;text-decoration:none;margin-left:1.5rem;font-size:14px}
.nav-links a:hover,.nav-links a.active{color:#E8A842}
.hero{background:linear-gradient(135deg,#0D1B3E 0%,#2D1F5E 60%,#3D2F7A 100%);padding:3rem 2rem;text-align:center;border-bottom:1px solid #3D2F7A}
.hero h1{font-size:2.2rem;font-weight:600;color:#F0EBF8;margin-bottom:.5rem;font-family:'DM Sans',sans-serif;letter-spacing:.5px}.hero h1 span{color:#E8A842}
.hero p{color:#B8A8D4;font-size:.95rem}
.container{max-width:1200px;margin:0 auto;padding:2rem 1.5rem}
.card{background:#231545;border:1px solid #3D2F7A;border-radius:12px;padding:1.25rem;margin-bottom:1.25rem}
.ct{font-size:.72rem;font-weight:600;color:#E8A842;text-transform:uppercase;letter-spacing:.1em;margin-bottom:1rem;border-bottom:1px solid #3D2F7A;padding-bottom:.6rem}
.upload{border:2px dashed #5A4A9A;border-radius:12px;padding:2.5rem;text-align:center;cursor:pointer;background:#231545;margin-bottom:1rem;transition:border-color .2s}
.upload:hover{border-color:#E8A842}
.upload h3{font-size:1rem;color:#F0EBF8;margin-bottom:.3rem}
.upload p{color:#B8A8D4;font-size:.85rem}
.upload input{display:none}
.preview{max-width:100%;max-height:220px;border-radius:8px;margin-top:1rem;object-fit:contain;display:none}
.btn{display:block;width:100%;padding:.85rem;background:#E8A842;color:#0D1B3E;border:none;border-radius:8px;font-size:1rem;font-weight:700;cursor:pointer;margin-bottom:1.5rem;letter-spacing:.5px;transition:background .15s}
.btn:hover{background:#F5CC7A}.btn:disabled{background:#3D2F7A;color:#B8A8D4;cursor:not-allowed}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:768px){.grid2{grid-template-columns:1fr}}
.spinner{width:36px;height:36px;border:3px solid #3D2F7A;border-top-color:#E8A842;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto .75rem}
@keyframes spin{to{transform:rotate(360deg)}}
.loading{display:none;text-align:center;padding:2rem;color:#B8A8D4}
#results{display:none}
.det{display:flex;gap:.75rem;padding:.85rem;background:#1A1035;border-radius:8px;margin-bottom:.6rem;border-left:4px solid #3D2F7A}
.tag{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.7rem;font-weight:600;margin-top:4px}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.75rem;font-weight:600;margin-bottom:.75rem}
.sl{font-size:.7rem;font-weight:600;color:#4ABFB0;text-transform:uppercase;letter-spacing:.1em;margin:.85rem 0 .4rem}
.arr{color:#E8A842;font-weight:700;flex-shrink:0}
.note{font-size:.78rem;color:#B8A8D4;background:#1A1035;padding:8px 12px;border-radius:6px;margin-bottom:8px;border:1px solid #3D2F7A}
img.full{width:100%;border-radius:8px}
.empty{text-align:center;padding:2rem;color:#B8A8D4;font-size:.9rem}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{text-align:left;padding:.5rem .75rem;font-size:.65rem;color:#B8A8D4;text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid #3D2F7A}
td{padding:.55rem .75rem;border-bottom:1px solid #3D2F7A;color:#F0EBF8}
tr.best{background:#2D1F5E}
tr.group-header td{background:#1A1035;color:#E8A842;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;padding:.4rem .75rem}
.disclaimer{background:#231545;border:1px solid #3D2F7A;border-left:4px solid #E8A842;border-radius:8px;padding:1rem;margin-bottom:1.25rem;font-size:.8rem;color:#B8A8D4;line-height:1.6}
.ds-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:1rem}
.ds-card{background:#1A1035;border-radius:8px;padding:.85rem;text-align:center}
.ds-val{font-size:1.6rem;font-weight:700;color:#E8A842}
.ds-lbl{font-size:.7rem;color:#B8A8D4;margin-top:.2rem}
.cls-row{display:flex;align-items:center;gap:.75rem;padding:.5rem;background:#1A1035;border-radius:6px;margin-bottom:.4rem}
.cls-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.cls-name{font-size:.8rem;font-weight:500;flex:1}
.cls-nums{display:flex;gap:.5rem;font-size:.72rem;color:#B8A8D4}
.cls-num{background:#231545;padding:2px 8px;border-radius:4px}
</style>
"""

INDEX_HTML = PAGE_STYLE + """
<nav>
  <div class="brand">Wound<span>Sight</span></div>
  <div class="nav-links">
    <a href="/" class="active">Detection</a>
    <a href="metrics">Metrics</a>
  </div>
</nav>
<div class="hero">
  <h1>Wound Detection &amp; <span>Classification</span></h1>
  <p>YOLO26m &nbsp;&middot;&nbsp; AZH Clinical Dataset &nbsp;&middot;&nbsp; 6 Wound Classes</p>
</div>
<div class="container">

  <div class="upload" id="up" onclick="document.getElementById('fi').click()"
       ondragover="event.preventDefault();this.style.borderColor='#E8A842'"
       ondragleave="this.style.borderColor='#5A4A9A'"
       ondrop="drop(event)">
    <div style="font-size:2.5rem;margin-bottom:.75rem">&#x1FA79;</div>
    <h3>Upload a wound image</h3>
    <p>JPG, PNG &nbsp;&middot;&nbsp; Click or drag and drop</p>
    <input type="file" id="fi" accept=".jpg,.jpeg,.png,.bmp" onchange="preview(this)">
    <img id="prev" class="preview">
  </div>

  <button class="btn" id="btn" disabled onclick="detect()">Detect Wound</button>

  <div id="info">
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.25rem">
      <div class="card" style="margin-bottom:0;text-align:center;padding:1.5rem">
        <div style="font-size:2rem;margin-bottom:.5rem">&#x1F9E0;</div>
        <div style="font-weight:600;color:#E8A842;margin-bottom:.3rem">YOLO26m Model</div>
        <div style="font-size:.8rem;color:#B8A8D4">Deep learning object detection trained on AZH Clinical Wound Dataset</div>
      </div>
      <div class="card" style="margin-bottom:0;text-align:center;padding:1.5rem">
        <div style="font-size:2rem;margin-bottom:.5rem">&#x1F4CA;</div>
        <div style="font-weight:600;color:#4ABFB0;margin-bottom:.3rem">85.0% F1 Score</div>
        <div style="font-size:.8rem;color:#B8A8D4">Best result — SGD optimizer, 250 epochs</div>
      </div>
      <div class="card" style="margin-bottom:0;text-align:center;padding:1.5rem">
        <div style="font-size:2rem;margin-bottom:.5rem">&#x1FA7A;</div>
        <div style="font-weight:600;color:#C4607A;margin-bottom:.3rem">6 Wound Classes</div>
        <div style="font-size:.8rem;color:#B8A8D4">Background, Normal, Diabetic, Pressure, Surgical, Venous</div>
      </div>
    </div>

    <div class="card">
      <div class="ct">Wound Classes</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem">
        <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem;background:#1A1035;border-radius:6px">
          <div style="width:10px;height:10px;border-radius:50%;background:#8B7FA8;flex-shrink:0"></div>
          <div><div style="font-size:.8rem;font-weight:500;color:#8B7FA8">Background</div><div style="font-size:.7rem;color:#B8A8D4">Non-wound region</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem;background:#1A1035;border-radius:6px">
          <div style="width:10px;height:10px;border-radius:50%;background:#4ABFB0;flex-shrink:0"></div>
          <div><div style="font-size:.8rem;font-weight:500;color:#4ABFB0">Normal</div><div style="font-size:.7rem;color:#B8A8D4">Healthy skin</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem;background:#1A1035;border-radius:6px">
          <div style="width:10px;height:10px;border-radius:50%;background:#E8A842;flex-shrink:0"></div>
          <div><div style="font-size:.8rem;font-weight:500;color:#E8A842">Diabetic</div><div style="font-size:.7rem;color:#B8A8D4">Diabetic ulcer</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem;background:#1A1035;border-radius:6px">
          <div style="width:10px;height:10px;border-radius:50%;background:#C4607A;flex-shrink:0"></div>
          <div><div style="font-size:.8rem;font-weight:500;color:#C4607A">Pressure</div><div style="font-size:.7rem;color:#B8A8D4">Pressure wound</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem;background:#1A1035;border-radius:6px">
          <div style="width:10px;height:10px;border-radius:50%;background:#7B9ED9;flex-shrink:0"></div>
          <div><div style="font-size:.8rem;font-weight:500;color:#7B9ED9">Surgical</div><div style="font-size:.7rem;color:#B8A8D4">Post-operative wound</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem;background:#1A1035;border-radius:6px">
          <div style="width:10px;height:10px;border-radius:50%;background:#D4845A;flex-shrink:0"></div>
          <div><div style="font-size:.8rem;font-weight:500;color:#D4845A">Venous</div><div style="font-size:.7rem;color:#B8A8D4">Venous ulcer</div></div>
        </div>
      </div>
    </div>

    <div class="disclaimer">
      <span style="color:#E8A842;font-weight:600">&#9888; Disclaimer: &nbsp;</span>
      This system is intended for research and educational purposes only. Results should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for wound assessment and treatment.
    </div>
  </div>

  <div class="loading" id="load">
    <div class="spinner"></div>
    <p>Running detection — AdamW 250ep vs SGD 250ep...</p>
  </div>

  <div id="results">
    <div class="card">
      <div class="ct">Detected Wounds — SGD 250 Epochs (Best Model — F1 85.0%)</div>
      <div id="dets"></div>
    </div>
    <div class="card">
      <div class="ct">Clinical Recommendations</div>
      <div id="recs"></div>
    </div>
    <div class="card">
      <div class="ct">Model Comparison — Best AdamW vs Best SGD</div>
      <p style="font-size:.78rem;color:#B8A8D4;margin-bottom:1rem">Same image detected by both best models</p>
      <div class="grid2" id="compimgs"></div>
      <div style="margin-top:1.25rem" id="comptable"></div>
      <p style="font-size:.7rem;color:#B8A8D4;margin-top:.75rem;font-style:italic">mAP50 and F1 reflect best epoch performance on validation set</p>
    </div>
    <div class="disclaimer">
      <span style="color:#E8A842;font-weight:600">&#9888; Disclaimer: &nbsp;</span>
      This system is intended for research and educational purposes only. Results should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for wound assessment and treatment.
    </div>
  </div>

</div>
<script>
const SEV={None:'background:#1A3A2A;color:#4ABFB0',Low:'background:#1A2A4A;color:#7B9ED9',Moderate:'background:#3A2A0A;color:#E8A842',High:'background:#3A1525;color:#C4607A'}
const UBG={None:'#1A3A2A',Low:'#1A2A4A',Moderate:'#3A2A0A',High:'#3A1525'}
const UTC={None:'#4ABFB0',Low:'#7B9ED9',Moderate:'#E8A842',High:'#C4607A'}

function preview(inp){
  const f=inp.files[0]; if(!f) return
  const r=new FileReader()
  r.onload=e=>{const p=document.getElementById('prev');p.src=e.target.result;p.style.display='block';document.getElementById('btn').disabled=false}
  r.readAsDataURL(f)
}
function drop(e){
  e.preventDefault()
  document.getElementById('up').style.borderColor='#5A4A9A'
  const f=e.dataTransfer.files[0]; if(!f) return
  document.getElementById('fi').files=e.dataTransfer.files
  preview(document.getElementById('fi'))
}
async function detect(){
  const f=document.getElementById('fi').files[0]; if(!f) return
  document.getElementById('btn').disabled=true
  document.getElementById('load').style.display='block'
  document.getElementById('results').style.display='none'
  const fd=new FormData(); fd.append('file',f)
  try{
    const res=await fetch('predict',{method:'POST',body:fd})
    const text=await res.text()
    let data
    try{data=JSON.parse(text)}catch(e){console.error(text);alert('Server error');return}
    if(data.error){alert(data.error);return}
    document.getElementById('info').style.display='none'
    renderDetections(data.detections)
    renderRecs(data.rec)
    renderComparison(data.model_comparison)
    document.getElementById('results').style.display='block'
  }catch(e){alert('Request failed: '+e.message)}
  finally{document.getElementById('load').style.display='none';document.getElementById('btn').disabled=false}
}

function renderComparison(comp){
  if(!comp||!comp.length) return
  document.getElementById('compimgs').innerHTML = comp.map(m=>`
    <div style="background:#1A1035;border-radius:8px;overflow:hidden;border:2px solid ${m.is_best?m.color:'#3D2F7A'}">
      <img src="static/results/${m.result_img}" style="width:100%;display:block">
      <div style="padding:.75rem">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.3rem">
          <span style="font-size:.78rem;color:#B8A8D4;font-weight:500">${m.run_name}</span>
          ${m.is_best?'<span style="background:#2D1F5E;color:#E8A842;padding:1px 8px;border-radius:8px;font-size:.65rem;font-weight:600">&#11088; BEST</span>':''}
        </div>
        <div style="font-weight:600;color:${m.color};font-size:.88rem;margin-bottom:.3rem">${m.predicted}</div>
        <div style="display:flex;flex-wrap:wrap;gap:.5rem;font-size:.72rem">
          <span style="color:#B8A8D4">Conf: <b style="color:${m.color}">${m.confidence}%</b></span>
          ${m.map50!=null?`<span style="color:#B8A8D4">mAP50: <b style="color:#E8A842">${m.map50}%</b></span>`:''}
          ${m.f1!=null?`<span style="color:#B8A8D4">F1: <b style="color:#4ABFB0">${m.f1}%</b></span>`:''}
          ${m.inf_time!=null?`<span style="color:#B8A8D4">Speed: <b style="color:#7B9ED9">${m.inf_time}ms</b></span>`:''}
        </div>
      </div>
    </div>`).join('')
  document.getElementById('comptable').innerHTML = `
    <table><thead><tr>
      <th>Model</th><th>Optimizer</th><th>Detected</th><th>Confidence</th>
      <th>mAP50</th><th>F1</th><th>Inference</th>
    </tr></thead><tbody>
      ${comp.map(m=>`
        <tr class="${m.is_best?'best':''}">
          <td style="color:${m.is_best?'#E8A842':'#F0EBF8'};font-weight:${m.is_best?600:400}">${m.run_name}</td>
          <td style="color:#B8A8D4;font-size:.75rem">${m.run_name.includes('AdamW')?'AdamW':'SGD'}</td>
          <td style="color:${m.color};font-weight:500">${m.predicted}</td>
          <td style="color:${m.color}">${m.confidence}%</td>
          <td style="color:#E8A842">${m.map50!=null?m.map50+'%':'—'}</td>
          <td style="color:#4ABFB0">${m.f1!=null?m.f1+'%':'—'}</td>
          <td style="color:#B8A8D4">${m.inf_time!=null?m.inf_time+' ms':'—'}</td>
        </tr>`).join('')}
    </tbody></table>`
}

function renderDetections(detections){
  const dl=document.getElementById('dets')
  if(!detections||!detections.length){dl.innerHTML='<div class="empty">No wounds detected above confidence threshold.</div>';return}
  dl.innerHTML=detections.map(d=>`
    <div class="det" style="border-left-color:${d.color}">
      <div>
        <div style="font-weight:600;color:${d.color}">${d.label}</div>
        <div style="font-size:.75rem;color:#B8A8D4;margin-top:2px">Confidence: ${d.confidence}%</div>
        <span class="tag" style="${SEV[d.severity]||''}">${d.severity}</span>
        <div style="font-size:.75rem;color:#B8A8D4;margin-top:5px;font-style:italic;line-height:1.5">${d.first_aid}</div>
      </div>
    </div>`).join('')
}

function renderRecs(rec){
  document.getElementById('recs').innerHTML=`
    <div class="note">${rec.conf_note}</div>
    <span class="badge" style="background:${UBG[rec.risk]||'#2D1F5E'};color:${UTC[rec.risk]||'#B8A8D4'}">${rec.urgency}</span>
    <div class="sl">For the doctor</div>
    ${rec.doctor_recs.map(r=>`<div style="display:flex;gap:.4rem;font-size:.82rem;margin-bottom:.3rem;line-height:1.5"><span class="arr">&#8594;</span>${r}</div>`).join('')}
    <div class="sl">For the patient</div>
    ${rec.patient_recs.map(r=>`<div style="display:flex;gap:.4rem;font-size:.82rem;margin-bottom:.3rem;line-height:1.5"><span class="arr">&#8594;</span>${r}</div>`).join('')}`
}
</script>
"""

METRICS_HTML = PAGE_STYLE + """
<nav>
  <div class="brand">Wound<span>Sight</span></div>
  <div class="nav-links">
    <a href=".">Detection</a>
    <a href="metrics" class="active">Metrics</a>
  </div>
</nav>
<div class="hero">
  <h1>Metrics <span>Dashboard</span></h1>
  <p>YOLO26m &nbsp;&middot;&nbsp; Model Size Comparison &nbsp;&middot;&nbsp; AdamW vs SGD &nbsp;&middot;&nbsp; Epoch Comparison</p>
</div>
<div class="container">
  <div id="content"><div class="loading" style="display:block"><div class="spinner"></div><p>Loading...</p></div></div>
</div>
<script>
function pct(v){return v==null?'—':(v>1?v:v*100).toFixed(1)+'%'}
async function load(){
  try{
    const res  = await fetch('metrics-data')
    const data = await res.json()
    if(!data.runs||!data.runs.length){
      document.getElementById('content').innerHTML='<div class="empty">No metrics found.</div>';return
    }

    const runs = data.runs
    // FIX 3: correct p1Names — only runs actually in METRICS_RUNS Phase 1
    const p1Names    = ['Nano_SGD_250','SGD_250','Large_SGD_250']
    const phase1     = runs.filter(r=>p1Names.includes(r.run_name))
    // FIX 2: filter by AdamW_ prefix (not Auto_)
    const phase2auto = runs.filter(r=>r.run_name.startsWith('AdamW'))
    const phase2sgd  = runs.filter(r=>r.run_name.startsWith('SGD_') && !r.run_name.includes('Nano') && !r.run_name.includes('Large'))

    const all       = [...phase1,...phase2auto,...phase2sgd]
    const bestMap50 = all.reduce((a,b)=>(a.map50||0)>=(b.map50||0)?a:b,{})
    const bestF1    = all.reduce((a,b)=>(a.f1||0)>=(b.f1||0)?a:b,{})
    const bestPrec  = all.reduce((a,b)=>(a.precision||0)>=(b.precision||0)?a:b,{})
    const bestRec   = all.reduce((a,b)=>(a.recall||0)>=(b.recall||0)?a:b,{})
    const bestP1    = phase1.length     ? phase1.reduce((a,b)=>(a.f1||0)>=(b.f1||0)?a:b,{})     : {}
    const bestAuto  = phase2auto.length ? phase2auto.reduce((a,b)=>(a.f1||0)>=(b.f1||0)?a:b,{}) : {}
    const bestSGD   = phase2sgd.length  ? phase2sgd.reduce((a,b)=>(a.f1||0)>=(b.f1||0)?a:b,{})  : {}

    let h = ''

    // Summary cards at top
    h+='<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem">'
    ;[['Best mAP50',bestMap50.map50,'#E8A842',bestMap50.run_name],
      ['Best F1',bestF1.f1,'#4ABFB0',bestF1.run_name],
      ['Best Precision',bestPrec.precision,'#C4607A',bestPrec.run_name],
      ['Best Recall',bestRec.recall,'#7B9ED9',bestRec.run_name]
    ].forEach(([l,v,c,rn])=>{
      h+=`<div class="card" style="margin-bottom:0"><div class="ct">${l}</div>
        <div style="font-size:1.9rem;font-weight:600;color:${c}">${pct(v)}</div>
        <div style="font-size:.7rem;color:#B8A8D4;margin-top:.2rem">${rn||'—'}</div></div>`
    })
    h+='</div>'

    // Phase 1 table
    h+='<div class="card">'
    h+='<div class="ct">Phase 1 — Model Size Comparison (SGD, 250 Epochs, 640×640)</div>'
    h+='<p style="font-size:.78rem;color:#B8A8D4;margin-bottom:1rem">YOLO26n (nano), YOLO26m (medium) and YOLO26l (large) under identical training conditions</p>'
    if(phase1.length){
      const mlabel=n=>n.includes('Nano')?'YOLO26n (Nano)':n.includes('Large')?'YOLO26l (Large)':'YOLO26m (Medium)'
      h+='<table><thead><tr>'
      ;['Run','Model','Epochs','Best Epoch','mAP50','mAP50-95','Precision','Recall','F1'].forEach(x=>h+=`<th>${x}</th>`)
      h+='</tr></thead><tbody>'
      phase1.forEach(r=>{
        const ib=r.run_name===bestP1.run_name
        h+=`<tr class="${ib?'best':''}">
          <td style="color:${ib?'#E8A842':'#F0EBF8'};font-weight:${ib?600:400}">${r.run_name}${ib?' &#11088;':''}</td>
          <td style="color:#B8A8D4;font-size:.75rem">${mlabel(r.run_name)}</td>
          <td>${r.total_epochs||'—'}</td><td>${r.best_epoch||'—'}</td>
          <td style="color:#E8A842;font-weight:500">${pct(r.map50)}</td>
          <td>${pct(r.map5095)}</td><td>${pct(r.precision)}</td>
          <td>${pct(r.recall)}</td>
          <td style="color:#4ABFB0;font-weight:500">${pct(r.f1)}</td>
        </tr>`
      })
      h+='</tbody></table>'
    } else {
      h+='<div class="empty">No Phase 1 runs found</div>'
    }
    h+='<p style="font-size:.7rem;color:#B8A8D4;margin-top:.75rem;font-style:italic">All runs use SGD optimiser, seed 42, 70:20:10 split, 640×640 resolution</p>'
    h+='</div>'

    // Phase 2 table
    h+='<div class="card">'
    h+='<div class="ct">Phase 2 — Optimiser Comparison (YOLO26m, AdamW vs SGD, 50–250 Epochs)</div>'
    h+='<p style="font-size:.78rem;color:#B8A8D4;margin-bottom:1rem">Comparing AdamW and SGD on the best-performing model size from Phase 1</p>'
    h+='<table><thead><tr>'
    ;['Run','Optimizer','Epochs','Best Epoch','mAP50','mAP50-95','Precision','Recall','F1'].forEach(x=>h+=`<th>${x}</th>`)
    h+='</tr></thead><tbody>'

    const p2row=(r,gb)=>{
      const ib=r.run_name===gb.run_name
      const opt=r.run_name.startsWith('AdamW')?'AdamW':'SGD'
      return `<tr class="${ib?'best':''}">
        <td style="color:${ib?'#E8A842':'#F0EBF8'};font-weight:${ib?600:400}">${r.run_name}${ib?' &#11088;':''}</td>
        <td style="color:#B8A8D4;font-size:.75rem">${opt}</td>
        <td>${r.total_epochs||'—'}</td><td>${r.best_epoch||'—'}</td>
        <td style="color:#E8A842;font-weight:500">${pct(r.map50)}</td>
        <td>${pct(r.map5095)}</td><td>${pct(r.precision)}</td>
        <td>${pct(r.recall)}</td>
        <td style="color:#4ABFB0;font-weight:500">${pct(r.f1)}</td>
      </tr>`
    }

    if(phase2auto.length){
      h+='<tr class="group-header"><td colspan="9">AdamW Optimizer</td></tr>'
      // FIX 4: sort AdamW by epoch
      phase2auto.sort((a,b)=>a.total_epochs-b.total_epochs).forEach(r=>{h+=p2row(r,bestAuto)})
    }
    if(phase2sgd.length){
      h+='<tr class="group-header"><td colspan="9">SGD Optimizer</td></tr>'
      phase2sgd.sort((a,b)=>a.total_epochs-b.total_epochs).forEach(r=>{h+=p2row(r,bestSGD)})
    }
    h+='</tbody></table>'
    h+='<p style="font-size:.7rem;color:#B8A8D4;margin-top:.75rem;font-style:italic">All runs use YOLO26m, seed 42, 70:20:10 split, 640×640 resolution</p>'
    h+='</div>'

    // Training curves
    if(data.curve_images&&data.curve_images.length){
      h+='<div class="card"><div class="ct">Training Curves — Phase 2 (AdamW vs SGD)</div>'
      h+='<div style="display:grid;grid-template-columns:1fr;gap:1.5rem">'
      data.curve_images.forEach(img=>{
        h+=`<div><p style="font-size:.75rem;color:#B8A8D4;margin-bottom:.4rem">${img.label}</p>
            <img src="static/curves/${img.filename}" class="full"></div>`
      })
      h+='</div></div>'
    }

    // Dataset info
    h+=`<div class="card">
      <div class="ct">Dataset Information — AZH Clinical Wound Dataset</div>
      <div class="ds-grid">
        <div class="ds-card"><div class="ds-val">930</div><div class="ds-lbl">Total Images</div></div>
        <div class="ds-card"><div class="ds-val">6</div><div class="ds-lbl">Wound Classes</div></div>
        <div class="ds-card"><div class="ds-val">70:20:10</div><div class="ds-lbl">Train:Val:Test Split</div></div>
        <div class="ds-card"><div class="ds-val">648</div><div class="ds-lbl">Training Images</div></div>
        <div class="ds-card"><div class="ds-val">184</div><div class="ds-lbl">Validation Images</div></div>
        <div class="ds-card"><div class="ds-val">98</div><div class="ds-lbl">Test Images</div></div>
      </div>
      <div class="ct" style="margin-top:.5rem">Per-Class Distribution</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
        <div class="cls-row"><div class="cls-dot" style="background:#8B7FA8"></div><span class="cls-name" style="color:#8B7FA8">Background</span><div class="cls-nums"><span class="cls-num">Train 70</span><span class="cls-num">Val 20</span><span class="cls-num">Test 10</span><span class="cls-num">Total 100</span></div></div>
        <div class="cls-row"><div class="cls-dot" style="background:#4ABFB0"></div><span class="cls-name" style="color:#4ABFB0">Normal Skin</span><div class="cls-nums"><span class="cls-num">Train 70</span><span class="cls-num">Val 20</span><span class="cls-num">Test 10</span><span class="cls-num">Total 100</span></div></div>
        <div class="cls-row"><div class="cls-dot" style="background:#E8A842"></div><span class="cls-name" style="color:#E8A842">Diabetic</span><div class="cls-nums"><span class="cls-num">Train 129</span><span class="cls-num">Val 37</span><span class="cls-num">Test 19</span><span class="cls-num">Total 185</span></div></div>
        <div class="cls-row"><div class="cls-dot" style="background:#C4607A"></div><span class="cls-name" style="color:#C4607A">Pressure</span><div class="cls-nums"><span class="cls-num">Train 93</span><span class="cls-num">Val 26</span><span class="cls-num">Test 15</span><span class="cls-num">Total 134</span></div></div>
        <div class="cls-row"><div class="cls-dot" style="background:#7B9ED9"></div><span class="cls-name" style="color:#7B9ED9">Surgical</span><div class="cls-nums"><span class="cls-num">Train 114</span><span class="cls-num">Val 32</span><span class="cls-num">Test 18</span><span class="cls-num">Total 164</span></div></div>
        <div class="cls-row"><div class="cls-dot" style="background:#D4845A"></div><span class="cls-name" style="color:#D4845A">Venous</span><div class="cls-nums"><span class="cls-num">Train 172</span><span class="cls-num">Val 49</span><span class="cls-num">Test 26</span><span class="cls-num">Total 247</span></div></div>
      </div>
    </div>`

    document.getElementById('content').innerHTML=h
  }catch(e){
    document.getElementById('content').innerHTML=`<div class="empty">Error: ${e.message}</div>`
  }
}
load()
</script>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/metrics")
def metrics_page():
    return render_template_string(METRICS_HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if not file.filename or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file."}), 400

        uid      = str(uuid.uuid4())[:8]
        filename = secure_filename(f"{uid}_{file.filename}")
        img_path = UPLOAD_DIR / filename
        file.save(img_path)

        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({"error": "Could not read image."}), 400

        results    = model(str(img_path), conf=0.25, verbose=False)
        boxes      = results[0].boxes
        detections = []

        for box in boxes:
            cls_id  = int(box.cls[0]); conf = float(box.conf[0])
            if cls_id >= len(CLASS_NAMES): continue
            cls_name = CLASS_NAMES[cls_id]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            color_bgr = hex_to_bgr(CLASS_COLORS.get(cls_name, "#8B7FA8"))
            cv2.rectangle(img,(x1,y1),(x2,y2),color_bgr,2)
            label = f"{cls_name} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
            cv2.rectangle(img,(x1,y1-th-8),(x1+tw+4,y1),color_bgr,-1)
            cv2.putText(img,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
            detections.append({
                "class_name": cls_name,
                "label":      CLASS_LABELS.get(cls_name, cls_name),
                "confidence": round(conf*100, 1),
                "severity":   SEVERITY.get(cls_name, "Unknown"),
                "first_aid":  FIRST_AID.get(cls_name, "Seek medical attention."),
                "color":      CLASS_COLORS.get(cls_name, "#8B7FA8"),
            })

        result_name = f"{uid}_result.jpg"
        cv2.imwrite(str(RESULT_DIR/result_name), img)

        rec = get_recommendations(detections[0]["class_name"], detections[0]["confidence"]) if detections else {
            "risk":"None","urgency":"No action required","conf_note":"No wounds detected.",
            "doctor_recs":["No clinical action required."],"patient_recs":["No wounds detected."]}

        import time
        model_comparison = []

        for run_name, m in all_models.items():
            try:
                res = m(str(img_path), conf=0.25, verbose=False)
                b   = res[0].boxes
                img_comp = cv2.imread(str(img_path))
                if b and len(b) > 0:
                    cls_id   = int(b.cls[0])
                    conf_val = float(b.conf[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                    img_comp = draw_boxes(img_comp, b)
                else:
                    cls_name = "none detected"
                    conf_val = 0.0

                comp_name = f"{uid}_{run_name.replace(' ','_').replace('⭐','')}_result.jpg"
                cv2.imwrite(str(RESULT_DIR/comp_name), img_comp)
                best_map50, best_f1 = get_run_metrics(run_name)

                m(str(img_path), verbose=False)
                times = []
                for _ in range(5):
                    t0 = time.time()
                    m(str(img_path), verbose=False)
                    times.append((time.time() - t0) * 1000)
                inf_time = round(sum(times)/len(times), 1)

                model_comparison.append({
                    "run_name":   run_name,
                    "predicted":  CLASS_LABELS.get(cls_name, cls_name),
                    "class_name": cls_name,
                    "confidence": round(conf_val * 100, 1),
                    "color":      CLASS_COLORS.get(cls_name, "#8B7FA8"),
                    "is_best":    "SGD" in run_name,
                    "result_img": comp_name,
                    "map50":      best_map50,
                    "f1":         best_f1,
                    "inf_time":   inf_time,
                })
            except Exception as e:
                print(f"Comparison error for {run_name}: {e}")

        return jsonify({
            "detections":       detections,
            "result_image":     result_name,
            "rec":              rec,
            "model_comparison": model_comparison,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics-data")
def metrics_data():
    try:
        return jsonify({"runs":load_metrics(),"curve_images":make_curves()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"runs":[],"curve_images":[],"error":str(e)})


@app.route("/static/results/<path:filename>")
def result_file(filename): return send_from_directory(RESULT_DIR, filename)

@app.route("/static/curves/<path:filename>")
def curve_file(filename): return send_from_directory(CURVES_DIR, filename)

@app.route("/static/uploads/<path:filename>")
def upload_file(filename): return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    print("\nWoundSight starting...")
    load_models()
    print(f"\nOpen: http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)