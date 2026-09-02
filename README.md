# Focus Where It Counts: A Salience-Driven Vision-Language Model for Low Vision Assistance

This is the official anonymous repository for **“Focus Where It Counts: A Salience-Driven Vision-Language Model for Low Vision Assistance.”** It includes the salience prediction module, the salience-aware LLaVA model, and three salience-aware image-captioning datasets: **Salience COCO**, **Salience Flickr**, and **Salience VizWiz**.

---

## Table of Contents

1. [Overview](#overview)
2. [Salience-Aware Datasets](#salience-aware-datasets)
3. [Component A — QAGNet (Salience Prediction Module)](#component-a--qagnet-salience-prediction-module)
4. [Component B — Salience-LLaVA](#component-b--salience-llava)
   - [B.1 Environment Setup](#b1-environment-setup)
   - [B.2 Training](#b2-training)
   - [B.3 Inference](#b3-inference)
   - [B.4 Evaluation](#b4-evaluation)

---

## Overview

Existing vision-language models describe scenes without considering how humans naturally prioritise visual information, which is critical for low vision users who rely on verbalised descriptions to make safety-relevant decisions. We propose a **salience-driven captioning framework** that prioritises scene elements according to their importance for human-centred assistance.

The runtime architecture consists of three components:

1. **Vision Backbone Branch.** A CLIP-ViT vision encoder that tokenises the input image into patch features and projects them, via an MLP head, into the LLM input space (dimension `D = 4096` to match Vicuna).
2. **Salience Branch (QAGNet).** A Query-as-Graph Network built on top of Mask2Former, predicting per-instance salience masks and rankings. Salience features are extracted at multiple scales and injected as auxiliary visual tokens.
3. **Large Language Model.** Vicuna-v1.5 (LLaVA-v1.5 base), which conditions on (a) the projected CLIP tokens, (b) the salience tokens from QAGNet, and (c) the textual instruction, and emits an ordered, salience-aware caption.

---

## Salience-Aware Datasets

We curate three salience-aware image-captioning datasets that address three limitations of prior captioning benchmarks: (i) absence of salience-aware information, (ii) insufficient scene complexity, and (iii) the inclusion of contexts (e.g., outdoor sports) that are not representative of the daily activities of low vision users.

### Dataset Statistics

| Dataset          | #Images | Train / Test split | #Total words | #Caption sentences | Avg sentences | Avg words/sentence | Vocab size |
|------------------|--------:|--------------------|-------------:|-------------------:|--------------:|-------------------:|-----------:|
| Salience COCO    |   8,295 | 6,636 / 1,659      |      728,043 |             46,012 |             5 |                 16 |      6,657 |
| Salience Flickr  |   8,725 | 6,980 / 1,745      |      712,516 |             46,190 |             5 |                 15 |      6,882 |
| Salience VizWiz  |   1,045 | 779 / 266          |       73,689 |              4,815 |             4 |                 15 |      2,854 |

### Human Verification of Annotation Quality

To validate that the generated rankings reflect the perceptual priorities of low-vision users, four IRB-approved low-vision participants independently ranked the three most salient objects in a randomly sampled verification set. Cohen’s kappa between human and model rankings reached **0.70**, indicating substantial agreement and supporting the reliability of the released annotations.

> **Data availability.** Due to the anonymous review policy and dataset size, the Google Drive link for **Salience COCO**, **Salience Flickr**, and **Salience VizWiz**  will be provided upon acceptance.

---

## Component A — QAGNet (Salience Prediction Module)

QAGNet: it consumes an input image and outputs a salience mask along with per-instance salience scores

```bash
# Run pretrained QAGNet over an image folder to produce salience tensors
bash scripts/precompute_saliency.sh \
    --input  /path/to/images \
    --output playground/data/<dataset>/salience \
    --weights /path/to/qagnet_swinl.pth \
    --conf-threshold 0.7
```

The output directory will contain `{image_id}.npy` salience tensors and a JSON sidecar listing per-object salience scores and ranks. The pretrained QAGNet weights link is withheld during review.

---

## Component B — Salience-LLaVA

Salience-LLaVA fine-tunes LLaVA-v1.5 with the QAGNet salience branch attached and is supervised on the three salience-aware datasets above.

### B.1 Environment Setup

```bash
conda create -n salience_llava python=3.10 -y
conda activate salience_llava

conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt
pip install transformers accelerate peft deepspeed bitsandbytes
pip install sentencepiece tokenizers einops einops-exts timm
pip install pillow numpy scipy flash-attn --no-build-isolation
```

Then clone the upstream LLaVA repository and install it in editable mode (our training entry point patches LLaVA's data loader and model to accept salience tokens):

```bash
git clone <llava_repository_url>
cd <llava_repository>
pip install -e .
```

### B.2 Training

 To launch training:

```bash
bash scripts/finetune.sh
```

The script trains the salience-aware LLaVA model using saliency-enhanced inputs. A LoRA variant is also provided for GPU-constrained setups:

```bash
bash scripts/finetune_lora.sh
```

### B.3 Inference

Once trained, a single image can be captioned with:

```bash
python src/train_saliency_llava.py \
    --mode infer \
    --model-path /path/to/salience_llava_ckpt \
    --image-file ./assets/example.jpg \
    --salience-file ./assets/example_salience.npy
```



### B.4 Evaluation

We evaluate on the held-out test splits along two complementary axes: standard captioning quality (BLEU, METEOR, ROUGE-L, CIDEr) and salience-aware ordering via our proposed **SCMI** (Salience-Consistent Mention Index), which measures whether predicted captions mention objects in the same descending salience order as the ground truth.

```bash
# Generate predictions on a test split
python src/train_saliency_llava.py \
    --mode eval \
    --model-path /path/to/salience_llava_ckpt \
    --dataset salience_coco \
    --split test \
    --answers-file ./outputs/salience_coco_pred.jsonl

# Score predictions
python src/eval/run_eval.py \
    --pred ./outputs/salience_coco_pred.jsonl \
    --gt   playground/data/salience_coco/annotations.json \
    --metrics bleu meteor rouge cider scmi
```

Repeat with `--dataset salience_flickr` or `--dataset salience_vizwiz` to evaluate on the other two benchmarks. Full quantitative tables and ablations are reported in the paper.

### 5. [Dataset](https://drive.google.com/file/d/1zxZJOPXhlaRaUoEccA_0QA5ERH6jeezZ/view?usp=sharing)
