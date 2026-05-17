# FedSPA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👀 Introduction

This repository contains the code for our ICML 2026 paper `Beyond Description: Federated Adaptation via Semantic-Visual Prototype  Alignment`. 

**FedSPA** (*Federated Adaptation via Semantic-Visual Prototype Alignment*) is a a federated few-shot adaptation method for vision-language models built on [CLIP](https://github.com/openai/CLIP). FedSPA maintains client-side personalized visual prototypes and server-side learnable global semantic prototypes initialized from CLIP text features. Clients update only lightweight visual prototypes, while the server refines global semantic prototypes via regularized contrastive alignment using uploaded visual prototypes.

Supported backbones: **RN50** (`--backbone RN`) and **ViT-B/16** (`--backbone VIT`).

## ⏳ Setup

### 1. Environment

We recommend **PyTorch 2.1+** with a CUDA build that matches your GPU driver. Install PyTorch first, then CLIP and the remaining dependencies:

```bash
# Example (CUDA 12.1); adjust versions for your system
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

`info-nce-pytorch` is used for contrastive alignment on the server; see [RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch) for details.

### 2. Dataset

FedSPA follows the **CoOp** dataset layout under a single root directory (`--root_path` , default: `/path/to/DATA`). Put all datasets under that folder:

```
$DATA/
├── caltech-101/
├── dtd/
├── eurosat/
├── fgvc_aircraft/
├── food-101/
├── oxford_flowers/
├── oxford_pets/
├── stanford_cars/
├── sun397/
└── ucf101/
```

For download links, fixed train/val/test splits (`split_zhou_*.json`), and per-dataset directory structure, see [CoOp DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). FedSPA uses the 10 image classification benchmarks listed in `train.sh`.

Optional LLM/CuPL-style prompts are bundled under `gpt3_prompts/` and can be used for semantic prototype initialization with `--gpt3_prompts`.

## 📦 Usage

### Batch training (`train.sh`)

Edit `ROOT_PATH`, client count, datasets, and method configs in `train.sh`, then run:

```bash
cd FedSPA
bash train.sh
```

`train.sh` sweeps RN/ViT backbones, few-shot settings, and all benchmark datasets. Logs are written to:

```
{root_path}/output/{output_subdir}/FedSPA/{timestamp}_{RN|VIT}.txt
```

### Single run (`main.py`)

```bash
python main.py \
  --root_path /path/to/DATA \
  --datasets dtd \
  --backbone RN \
  --num_shots 8 \
  --num_clients 10 \
  --partition distribution \
  --dirichlet_alpha 0.1 \
  --local_epochs 5 \
  --global_epochs 10 \
  --local_epochs_server 100 \
  --local_epochs_last 100 \
  --local_batch_size 8 \
  --global_batch_size 8
```

Use `--datasets oxford_pets/caltech101` to run multiple datasets in one invocation (separated by `/`).

Per-dataset hyperparameters (`alpha`, `beta`, learning rates) are loaded from `configs/{dataset}.yaml`.

## 📁 Project structure

```
FedSPA/
├── main.py              # Entry point: CLIP load, data prep, federated loop
├── train.sh             # Batch experiment script
├── requirements.txt
├── configs/             # Per-dataset hyperparameters
├── datasets/            # Dataset loaders (CoOp-style)
├── gpt3_prompts/        # CuPL JSON prompts
└── utils/
    ├── client.py        # Local client prototype updates
    ├── server.py        # Global visual initialization & regularized InfoNCE semantic alignment
    └── utils.py         # Feature cache, CLIP classifier, metrics
```

## 🙏 Acknowledgements

Our codebase is adapted from [DPE-CLIP](https://github.com/zhangce01/DPE-CLIP), [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp), [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/), and [CuPL](https://github.com/sarahpratt/CuPL). We thank the authors for releasing their code.

## 📌 Citation

If you find this code useful, please cite our work (BibTeX to be added).

