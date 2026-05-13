"""
Gradio Web Demo — Distracted Driver Detection
With prominent danger/safe alert banners
"""

import sys
import logging
import tempfile
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.architecture import create_model, load_checkpoint
from src.data.dataset import IDX_TO_NAME
from src.explainability.gradcam import ExplainablePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DEFINITIONS = {
    0: {"name": "Safe Driving", "risk": "safe", "color": "#27ae60", "emoji": "✅"},
    1: {
        "name": "Texting (Right Hand)",
        "risk": "high",
        "color": "#e74c3c",
        "emoji": "📱",
    },
    2: {
        "name": "Phone Call (Right Hand)",
        "risk": "high",
        "color": "#e74c3c",
        "emoji": "📞",
    },
    3: {
        "name": "Texting (Left Hand)",
        "risk": "high",
        "color": "#e74c3c",
        "emoji": "📱",
    },
    4: {
        "name": "Phone Call (Left Hand)",
        "risk": "high",
        "color": "#e74c3c",
        "emoji": "📞",
    },
    5: {"name": "Radio Adjusting", "risk": "medium", "color": "#f39c12", "emoji": "📻"},
    6: {"name": "Drinking", "risk": "medium", "color": "#f39c12", "emoji": "🥤"},
    7: {"name": "Reaching Behind", "risk": "high", "color": "#e74c3c", "emoji": "🙆"},
    8: {"name": "Hair / Makeup", "risk": "medium", "color": "#f39c12", "emoji": "💄"},
    9: {
        "name": "Talking to Passenger",
        "risk": "low",
        "color": "#3498db",
        "emoji": "💬",
    },
}


def load_predictor(model_path=None, architecture="efficientnet_b3"):
    model = create_model(architecture, pretrained=False, device=DEVICE)

    if model_path and Path(model_path).exists():
        model = load_checkpoint(model, model_path, DEVICE)
    else:
        # Download from HuggingFace if not local
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id="keerthana-25/distracted-driver-detection",
                filename="models/best_model.pth",
                repo_type="space",
            )
            model = load_checkpoint(model, downloaded, DEVICE)
            logger.info("Model loaded from HuggingFace")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using random weights.")

    return ExplainablePredictor(model, device=DEVICE)


MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models" / "best_model.pth"))
PREDICTOR = load_predictor(MODEL_PATH)


def _alert_html(label, confidence, risk):
    if risk == "safe":
        bg, border, icon = "linear-gradient(135deg,#1a5c2a,#27ae60)", "#2ecc71", "✅"
        title = "DRIVER IS SAFE"
        anim = ""
    elif risk == "high":
        bg, border, icon = "linear-gradient(135deg,#7b0000,#c0392b)", "#ff4444", "🚨"
        title = "DANGER — DRIVER IS DISTRACTED!"
        anim = "animation: flash 0.7s infinite alternate;"
    elif risk == "medium":
        bg, border, icon = "linear-gradient(135deg,#7a4800,#e67e22)", "#f39c12", "⚠️"
        title = "WARNING — Unsafe Behaviour Detected"
        anim = ""
    else:
        bg, border, icon = "linear-gradient(135deg,#003d6b,#2980b9)", "#3498db", "ℹ️"
        title = "Low Risk Detected"
        anim = ""

    return f"""
<style>
@keyframes flash {{
  from {{ box-shadow: 0 0 25px 8px #ff2222aa; border-color: #ff4444; }}
  to   {{ box-shadow: 0 0 55px 18px #ff0000cc; border-color: #ff0000; }}
}}
</style>
<div style="background:{bg}; border:3px solid {border}; border-radius:16px;
            padding:28px 32px; text-align:center; font-family:'Inter',sans-serif; {anim}">
  <div style="font-size:3.8rem; line-height:1.1;">{icon}</div>
  <div style="font-size:1.9rem; font-weight:800; color:white;
              letter-spacing:0.03em; margin:10px 0 8px;">{title}</div>
  <div style="font-size:1.1rem; color:rgba(255,255,255,0.9);">
    Detected: <b>{label}</b> &nbsp;·&nbsp; Confidence: <b>{confidence:.1%}</b>
  </div>
</div>
"""


def _empty_html():
    return """
<div style="border:2px dashed #444; border-radius:16px; padding:48px;
            text-align:center; color:#666; font-family:'Inter',sans-serif; font-size:1rem;">
  📷 &nbsp; Upload a dashcam image — the alert will appear here
</div>
"""


def predict_image(image):
    if image is None:
        return _empty_html(), None, None, ""

    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    result = PREDICTOR.predict(pil_image, top_k=5, generate_cam=True)

    pred_class = result["predicted_class"]
    pred_label = result["predicted_label"]
    confidence = result["confidence"]
    risk = CLASS_DEFINITIONS[pred_class]["risk"]

    alert = _alert_html(pred_label, confidence, risk)
    cam_overlay = result.get("cam_overlay")
    prob_fig = _prob_chart(result["all_probabilities"])

    details = "### Top-5 Predictions\n"
    for p in result["top_k_predictions"]:
        ci = CLASS_DEFINITIONS.get(p["class_idx"], {})
        details += f"\n{ci.get('emoji','•')} **{p['label']}** — {p['confidence']:.1%}"

    return alert, cam_overlay, prob_fig, details


def _prob_chart(probs):
    names = [
        CLASS_DEFINITIONS[i]["emoji"] + "  " + CLASS_DEFINITIONS[i]["name"]
        for i in range(10)
    ]
    colors = [CLASS_DEFINITIONS[i]["color"] for i in range(10)]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(
        range(10),
        [p * 100 for p in probs],
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
        height=0.65,
    )
    ax.set_yticks(range(10))
    ax.set_yticklabels(names, fontsize=9, color="white")
    ax.set_xlabel("Confidence (%)", color="white", fontsize=10)
    ax.set_title("Class Probabilities", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 108)

    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax.text(
                bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}",
                va="center",
                ha="left",
                color="white",
                fontsize=8,
            )

    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def build_interface():
    css = """
    footer { display: none !important; }
    .gr-button { font-size: 1.1rem !important; }
    """

    with gr.Blocks(
        title="🚗 Distracted Driver Detection",
        theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"),
        css=css,
    ) as demo:

        gr.HTML("""
        <div style="text-align:center; padding:24px 0 4px; font-family:'Inter',sans-serif;">
          <h1 style="font-size:2.2rem; margin:0;">🚗 Distracted Driver Detection</h1>
          <p style="color:#aaa; margin-top:6px; font-size:1rem;">
            EfficientNet-B3 &nbsp;+&nbsp; Grad-CAM &nbsp;|&nbsp; State Farm Dataset &nbsp;|&nbsp; 10 Behaviour Classes
          </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Live Detection ──
            with gr.Tab("🔍 Live Detection"):

                # Full-width alert at the top
                alert_box = gr.HTML(value=_empty_html())

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Dashcam Image",
                            type="numpy",
                            height=320,
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze Driver Behavior", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        cam_output = gr.Image(
                            label="Grad-CAM — Model Attention Map",
                            height=320,
                        )

                with gr.Row():
                    prob_chart = gr.Plot(label="Class Probability Distribution")
                    details_md = gr.Markdown(
                        value="*Upload an image to see predictions*"
                    )

                # Trigger on button click
                analyze_btn.click(
                    fn=predict_image,
                    inputs=[image_input],
                    outputs=[alert_box, cam_output, prob_chart, details_md],
                )
                # Also auto-trigger when image is uploaded
                image_input.change(
                    fn=predict_image,
                    inputs=[image_input],
                    outputs=[alert_box, cam_output, prob_chart, details_md],
                )

            # ── Tab 2: Risk Guide ──
            with gr.Tab("⚠️ Risk Guide"):
                gr.Markdown("""
## Alert Colour System

| Alert | Colour | Trigger |
|-------|--------|---------|
| 🚨 **DANGER** — flashing red | Red | Texting, phone calls, reaching behind |
| ⚠️ **WARNING** | Orange | Drinking, radio, hair/makeup |
| ℹ️ **Low Risk** | Blue | Talking to passenger |
| ✅ **SAFE** | Green | Safe driving |

---

## Class Definitions

| Class | Behaviour | Risk |
|-------|-----------|------|
| c0 | Safe Driving | ✅ Safe |
| c1 | Texting — Right Hand | 🚨 High |
| c2 | Phone Call — Right Hand | 🚨 High |
| c3 | Texting — Left Hand | 🚨 High |
| c4 | Phone Call — Left Hand | 🚨 High |
| c5 | Radio Adjusting | ⚠️ Medium |
| c6 | Drinking | ⚠️ Medium |
| c7 | Reaching Behind | 🚨 High |
| c8 | Hair / Makeup | ⚠️ Medium |
| c9 | Talking to Passenger | ℹ️ Low |

---

## Applications
- **Fleet management** — real-time driver monitoring
- **Insurance scoring** — behavioural risk assessment
- **Driver coaching** — personalised safety feedback
- **Regulatory compliance** — mandatory distraction reporting
                """)

            # ── Tab 3: Model Info ──
            with gr.Tab("📊 Model Info"):
                gr.Markdown("""
## Architecture

| Component | Details |
|-----------|---------|
| Backbone | EfficientNet-B3 (ImageNet pretrained) |
| Head | 2-layer MLP · BatchNorm · SiLU · Dropout(0.4) |
| Parameters | ~12M |
| Input size | 224×224 RGB |
| Loss | Label Smoothing Cross-Entropy (ε=0.1) |
| Explainability | Grad-CAM on last convolutional layer |

## Training Strategy

**Phase 1 (epochs 1–3):** Backbone frozen → train head only  
**Phase 2 (epoch 4+):** Unfreeze backbone with 10× lower LR

## Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | ~94% |
| F1 Macro | ~0.94 |
| AUROC | ~0.997 |
                """)

        gr.HTML("""
        <div style="text-align:center; color:#555; font-size:0.8rem; padding:12px 0;">
          Distracted Driver Detection · EfficientNet-B3 + Grad-CAM · MLflow + TensorBoard
        </div>
        """)

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
    )
