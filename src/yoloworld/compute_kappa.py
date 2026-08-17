#!/usr/bin/env python
"""
compute_kappa.py
================
Compute Cohen's kappa between human salience rankings (from the
low-vision verification study) and the auxiliary detector + salience
pipeline rankings.

Inputs
------
  - human_rankings.json
        Per-image human-ranked top-3 salient objects (free text).
  - salience_ranked.json
        Output of `annotate_pipeline.py` Step 2 - top-3 detector boxes
        ordered by mean salience.
  - detections.json
        Output of `annotate_pipeline.py` Step 1 - used to build the
        detector-side label vocabulary.

Why this script exists
----------------------
Humans describe objects in free text ("man", "the cup"), while the
detector emits a fixed open-vocabulary label set. Direct string
comparison under-counts agreement: humans describe "a person" once,
the detector may report "person" plus parts ("hat", "shoe"). We
therefore (i) build a super-category mapping over the detector
vocabulary, (ii) project both annotators into that space, and
(iii) compute kappa over rank positions.
"""

import json
from collections import defaultdict

from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Inputs (edit before running)
# ---------------------------------------------------------------------------
HUMAN_FILE     = "human_rankings.json"
RANKED_FILE    = "salience_ranked.json"
DETECTIONS_FILE = "detections.json"


# ---------------------------------------------------------------------------
# Super-category map (subset shown; full table withheld for review)
# ---------------------------------------------------------------------------
SUPER_CATEGORIES = {
    "person":   {"person", "man", "woman", "child", "people"},
    "vehicle":  {"car", "bus", "truck", "bicycle", "motorcycle"},
    "animal":   {"dog", "cat", "bird", "horse", "cow"},
    "furniture": {"chair", "table", "couch", "bed", "shelf"},
    "kitchen":  {"cup", "bottle", "bowl", "plate", "fork", "knife", "spoon"},
    "device":   {"phone", "laptop", "tv", "monitor", "keyboard", "mouse"},
    # ... additional buckets omitted for review ...
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def detector_vocab(detections):
    vocab = set()
    for entry in detections:
        for d in entry.get("detections", []):
            vocab.add(d["label"].lower())
    return sorted(vocab, key=len, reverse=True)


def map_to_super(text, vocab):
    """Map a free-text or detector label to a super-category, or None."""
    text = text.lower().strip()
    for super_cat, members in SUPER_CATEGORIES.items():
        if text in members:
            return super_cat
    for label in vocab:
        if label in text:
            for super_cat, members in SUPER_CATEGORIES.items():
                if label in members:
                    return super_cat
    return None


def aligned_label_streams(human, ranked, vocab):
    """Build paired label streams across all images by rank position."""
    ranked_by_image = {e["image"]: e for e in ranked}
    h_stream, r_stream = [], []

    for entry in human:
        fname = entry["image"]
        if fname not in ranked_by_image:
            continue
        h_top = entry.get("ranked", [])[:3]
        r_top = ranked_by_image[fname].get("ranked", [])[:3]
        for i in range(min(len(h_top), len(r_top))):
            h = map_to_super(h_top[i], vocab)
            r = map_to_super(r_top[i]["label"], vocab)
            if h is None or r is None:
                continue
            h_stream.append(h)
            r_stream.append(r)

    return h_stream, r_stream


def main():
    human      = load_json(HUMAN_FILE)
    ranked     = load_json(RANKED_FILE)
    detections = load_json(DETECTIONS_FILE)

    vocab = detector_vocab(detections)
    h, r  = aligned_label_streams(human, ranked, vocab)

    if not h:
        print("[WARN] No aligned pairs - check inputs.")
        return

    kappa = cohen_kappa_score(h, r)
    agree = sum(1 for a, b in zip(h, r) if a == b) / len(h)

    print(f"Aligned pairs : {len(h)}")
    print(f"Raw agreement : {agree:.3f}")
    print(f"Cohen's kappa : {kappa:.3f}")


if __name__ == "__main__":
    main()
