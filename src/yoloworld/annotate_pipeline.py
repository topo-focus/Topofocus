#!/usr/bin/env python
"""
annotate_pipeline.py
====================
Two-stage auxiliary pipeline used to bootstrap the salience-aware
annotations released with this work.

Step 1: open-vocabulary object detection (top-K objects per image)
         -> writes  detections.json

Step 2: per-instance salience prediction (QAGNet) + rank Step-1
         detections by mean salience inside each bounding box
         -> writes  salience_ranked.json
         (top-3 most salient objects per image, ordered most -> least)

Notes
-----
* Detector backbone, weights, and prompt vocabulary are intentionally
  withheld during anonymous review. Replace the placeholders below
  with any open-vocabulary detector exposing a `predict(image)` API
  that returns `(boxes, labels, confidences)`.
* Step 2 reuses the QAGNet checkpoint described in the main README.
"""

import os
import json

import torch
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths (edit before running)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = "<path/to/image/folder>"
QAGNET_ROOT  = "<path/to/qagnet>"
QAGNET_CFG   = "<path/to/qagnet/config>"
QAGNET_CKPT  = "<path/to/qagnet/weights>"

DETECTOR_WEIGHTS = "<path/to/openvocab/detector/weights>"
DETECTOR_CONF    = 0.15
TOP_K_OBJECTS    = 5
SALIENCE_THRESH  = 0.7

OUT_DETECTIONS = os.path.join(PROJECT_ROOT, "detections.json")
OUT_RANKED     = os.path.join(PROJECT_ROOT, "salience_ranked.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root):
    return sorted([
        f for f in os.listdir(root)
        if os.path.splitext(f)[1].lower() in EXTS
    ])


# ===========================================================================
# Step 1 - open-vocabulary detection (top-K)
# ===========================================================================
def run_open_vocab_detector(image_files):
    """
    Run an open-vocabulary detector across an image folder and keep the
    top-K most confident detections per image.

    The detector instance and its prompt vocabulary are withheld for
    anonymous review. Plug in any model exposing a `predict` API.
    """
    # detector = <OpenVocabDetector>(DETECTOR_WEIGHTS).to(DEVICE)
    # detector.set_classes(<vocabulary>)   # broad indoor/outdoor categories
    detector = None  # placeholder

    results_all = []
    for fname in image_files:
        img_path = os.path.join(DATASET_DIR, fname)

        # boxes: (N, 4) xyxy ; confs: (N,) ; labels: list[str]
        boxes, confs, labels = _detect_placeholder(detector, img_path)

        if len(boxes) == 0:
            results_all.append({"image": fname, "detections": []})
            continue

        order = np.argsort(-confs)[:TOP_K_OBJECTS]
        detections = []
        for idx in order:
            x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
            detections.append({
                "label": labels[int(idx)],
                "confidence": round(float(confs[idx]), 4),
                "bbox": [round(x1, 1), round(y1, 1),
                         round(x2, 1), round(y2, 1)],
            })

        results_all.append({"image": fname, "detections": detections})

    with open(OUT_DETECTIONS, "w") as f:
        json.dump(results_all, f, indent=2)

    return results_all


def _detect_placeholder(detector, img_path):
    """Return (boxes, confs, labels). Replace with real detector call."""
    return np.empty((0, 4)), np.empty((0,)), []


# ===========================================================================
# Step 2 - QAGNet salience ranking
# ===========================================================================
def _load_qagnet():
    """
    Build and load the QAGNet salience model. The full setup matches
    `train_qagnet.py` in this repo - we omit the boilerplate here.
    """
    # cfg   = build_qagnet_cfg(QAGNET_CFG, weights=QAGNET_CKPT)
    # model = QAGNet(cfg).to(DEVICE).eval()
    # return model
    return None


def _mean_salience_in_box(salience_map, bbox):
    """Mean salience inside an xyxy box, clipped to image bounds."""
    H, W = salience_map.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
    x2 = min(W, int(round(x2))); y2 = min(H, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(salience_map[y1:y2, x1:x2].mean())


def run_salience_ranking(detections_per_image):
    """For each image, score every detected box by mean QAGNet salience
    and keep the top-3, ordered most -> least salient."""
    qagnet = _load_qagnet()

    ranked_all = []
    for entry in detections_per_image:
        fname = entry["image"]
        dets  = entry["detections"]

        if not dets:
            ranked_all.append({"image": fname, "ranked": []})
            continue

        img = cv2.imread(os.path.join(DATASET_DIR, fname))
        if img is None:
            ranked_all.append({"image": fname, "ranked": []})
            continue

        # salience_map = qagnet(img)["salience"]  # HxW float32 in [0, 1]
        salience_map = np.zeros(img.shape[:2], dtype=np.float32)  # placeholder

        scored = []
        for d in dets:
            s = _mean_salience_in_box(salience_map, d["bbox"])
            scored.append({**d, "salience": round(s, 4)})

        scored.sort(key=lambda x: -x["salience"])
        ranked_all.append({"image": fname, "ranked": scored[:3]})

    with open(OUT_RANKED, "w") as f:
        json.dump(ranked_all, f, indent=2)

    return ranked_all


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    image_files = list_images(DATASET_DIR)
    print(f"[INFO] {len(image_files)} images in {DATASET_DIR}")

    if os.path.exists(OUT_DETECTIONS):
        with open(OUT_DETECTIONS) as f:
            detections = json.load(f)
    else:
        detections = run_open_vocab_detector(image_files)

    run_salience_ranking(detections)

    print("[DONE]")
    print(f"  detections    -> {OUT_DETECTIONS}")
    print(f"  salience rank -> {OUT_RANKED}")


if __name__ == "__main__":
    main()
