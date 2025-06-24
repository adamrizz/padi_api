from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
import os
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import gdown  # Untuk download model dari Google Drive

app = Flask(__name__)

# 🔽 Unduh model jika belum ada (ganti ID sesuai milikmu)
model_path = "model_final.pth"
if not os.path.exists(model_path):
    print("Mengunduh model dari Google Drive...")
    gdown.download(id="1-4zHkPtV_luSd82iqdD9yDUNIryTy4uv", output=model_path, quiet=False)  # GANTI ID INI!

# 🔧 Load config dan model
cfg = get_cfg()
cfg.merge_from_file("config.yaml")
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    img = np.array(Image.open(file.stream).convert("RGB"))[:, :, ::-1]
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    return jsonify({"classes": classes, "scores": scores})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
