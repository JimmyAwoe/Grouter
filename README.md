# Grouter: Decoupled Router for MoE Training

**Accelerating MoE Pre-training via Knowledge Distillation from Pre-trained Routers**

<p align="center">
  <a href="https://arxiv.org/abs/2603.06626"><img src="https://img.shields.io/badge/arXiv-2603.06626-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  &thinsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-LICENSE-informational" alt="License"></a>
  &thinsp;
  <a href="https://modelscope.cn/models/JimmyAwoe/Grouter-Official"><img src="https://img.shields.io/badge/ModelScope-Grouter-624AFF?style=flat" alt="ModelScope"></a>
</p>
<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README_zh.md">中文</a>
</p>

---

## Overview

**Grouter** (Global Router) is a novel approach to accelerating Mixture-of-Experts (MoE) model pre-training by leveraging routing knowledge distilled from pre-trained MoE models. Instead of learning expert routing from scratch during training, Grouter transfers the routing policy of a mature *Source Model* to a new *Target Model*, enabling faster convergence and better load balancing — all with minimal computational overhead.

### Key Idea

> *Why spend billions of tokens re-learning which experts are good at what, when a pre-trained model already knows?*

Grouter distills the routing function of a pre-trained MoE (e.g., Qwen3-30B-A3B) into a lightweight, standalone router network. This router is then used to guide expert selection during the pre-training of a new MoE model, replacing or augmenting the standard trainable router.

### Highlights

- **Significant Convergence Speedup** — Grouter-guided training converges substantially faster than standard auxiliary-loss-based routing
- **Minimal Overhead** — The Grouter network is tiny (~30M params) compared to the target model, adding negligible FLOPs  
- **Flexible Expert Scaling** — Expert Folding and Expert Expanding allow adaptation to any target expert count
- **Easy Distillation** — Partial model loading means you can distill routing from a 235B-parameter model on just 8 GPUs
- **Drop-in Integration** — Works with Megatron-LM; just add a few flags to your training script

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Grouter-Guided MoE Training                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Tokens ──┬──► Target MoE Router ──► Expert Selection    │
│                  │         (learning)            ▲               │
│                  │                               │               │
│                  └──► Grouter ────────────► Routing Guidance     │
│                       (frozen, distilled)                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Grouter Distillation:                                          │
│  Source MoE (e.g. Qwen3-30B) ─► First N Layers + Router Head   │
│                                  ─► Lightweight Grouter Network │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Environment Setup

We recommend using Docker. Our experiments are based on the `nvcr.io/nvidia/nemo:25.07` image.

```bash
export GROUTER_PATH=/path/to/Grouter
cd $GROUTER_PATH

# Build image and start container
bash env/create_image.sh
bash env/create_container.sh
```

**Alternative: Manual environment setup**

If Docker is unavailable, ensure the following packages are installed:

```
wandb
sentencepiece
tiktoken
datasets
transformers==4.51.0
accelerate
omegaconf
modelscope
```

Set `GROUTER_PATH` to point to this repository's root.

### 2. Set Up Megatron-LM

```bash
cd $GROUTER_PATH

# Clone and pin Megatron-LM to the tested version
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git reset --hard e7c55de9 && cd ..

# Create symlink for Grouter packages
ln -s ../utils_grouter Megatron-LM/utils_grouter

# Apply our patches
cp -r Megatron-LM-Patch/* Megatron-LM/
```

### 3. Download Grouter Weights (Recommended)

We provide pre-trained Grouter weights via [ModelScope](https://modelscope.cn/models/JimmyAwoe/Grouter-Official):

```bash
cd $GROUTER_PATH

# Base Grouter (distilled from Qwen3-30B-A3B, with Expert Tuning on C4, 128 experts)
modelscope download --model JimmyAwoe/Grouter-Official grouter.pth \
    --local_dir utils_grouter/grouter/grouter_model

# Expert Folding/Expanding variants
modelscope download --model JimmyAwoe/Grouter-Official grouter_ft_32.pth \
    --local_dir utils_grouter/grouter/grouter_model  # 32 experts
modelscope download --model JimmyAwoe/Grouter-Official grouter_ft_64.pth \
    --local_dir utils_grouter/grouter/grouter_model  # 64 experts
modelscope download --model JimmyAwoe/Grouter-Official grouter_ft_256.pth \
    --local_dir utils_grouter/grouter/grouter_model  # 256 experts

# Untuned base (for custom datasets or fine-tuning)
modelscope download --model JimmyAwoe/Grouter-Official grouter_base.pth \
    --local_dir utils_grouter/grouter/grouter_model
```

### 4. Data Preparation

```bash
cd $GROUTER_PATH

# Download C4 dataset (adjust --num_files based on storage)
python3 dataset/download_c4.py --num_files 256

# Preprocess for Megatron-LM (tokenize to binary format)
bash scripts/preprocess_data.sh        # Process files 0~255
bash scripts/preprocess_data.sh 10     # Process first 10 files
bash scripts/preprocess_data.sh 250 599  # Process files 250~599
```

### 5. Train with Grouter

```bash
cd $GROUTER_PATH
bash scripts/grouter_train.sh
```

This launches pre-training of a Tiny-Qwen3-style MoE model with Grouter guidance on 8 GPUs.

---

## Advanced Usage

### Distill Your Own Grouter

If you want to distill from a Source Model yourself (instead of using our pre-trained weights):

**Step-by-step distillation guide**

#### 1. Download the Source Model

```bash
cd $GROUTER_PATH
python3 model_home/download_qwen3_30b.py
```

#### 2. Convert Weights to Megatron Format

```bash
# Set up Megatron-Bridge for weight conversion
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge && git reset --hard 45ed38a9
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git reset --hard b8581a35 && cd ../..

# Convert HuggingFace → Megatron format
bash scripts/bridge_qwen3.sh
```

#### 3. Run Distillation

```bash
bash scripts/distillation_qwen3.sh
```

**Partial Loading:** Our framework only loads the first *N* transformer layers (where *N* is the router layer index, typically 1). This dramatically reduces GPU requirements:


| Mode                   | GPU Requirement (Qwen3-235B) |
| ---------------------- | ---------------------------- |
| Full model loading     | ~256× H100                   |
| Partial loading (ours) | ~8× H100                     |


Enable via `--grouter-allow-partial-load` with `--num-layers` set to the target layer count.

### Expert Folding (Reduce Expert Count)

Adapt a Grouter to fewer experts using affinity-based merging:

```bash
cd $GROUTER_PATH
bash scripts/run_construct_mapping.sh 64  # Fold to 64 experts
```

### Expert Expanding (Increase Expert Count)

Scale a Grouter to more experts:

```bash
cd $GROUTER_PATH
bash scripts/run_expand_expert_mapping.sh 256  # Expand to 256 experts
```

### Expert Tuning

Fine-tune the Grouter's load balancing for the target model's data distribution:

```bash
cd $GROUTER_PATH
bash scripts/run_finetune_grouter.sh \
    utils_grouter/grouter/grouter_model/cvt64_map_affinity.json \
    utils_grouter/grouter/grouter_model/grouter_base.pth
```

---

## Repository Structure

```
Grouter/
├── env/                          # Docker environment setup
│   ├── Dockerfile
│   ├── create_image.sh
│   └── create_container.sh
├── scripts/                      # Training & utility scripts
│   ├── grouter_train.sh          # Main training script with Grouter
│   ├── distillation_qwen3.sh    # Grouter distillation
│   ├── bridge_qwen3.sh          # Weight format conversion
│   ├── preprocess_data.sh       # Data preprocessing
│   ├── run_construct_mapping.sh  # Expert Folding
│   ├── run_expand_expert_mapping.sh  # Expert Expanding
│   └── run_finetune_grouter.sh  # Expert Tuning
├── utils_grouter/                # Core Grouter library
│   ├── grouter/                  # Grouter model & hooks
│   │   ├── general_router.py    # Grouter network definition
│   │   ├── grouter_hook.py      # Training integration hooks
│   │   ├── grouter_model/       # Model configs & weight storage
│   │   └── structure/           # Architecture variants (MHA, MLA, etc.)
│   ├── megatron_integration/    # Megatron-LM integration layer
│   └── tools/                   # Standalone tools (distillation, mapping, etc.)
├── Megatron-LM-Patch/           # Patches for Megatron-LM framework
├── dataset/                     # Data download scripts
└── model_home/                  # Model download scripts
```

---

## Pre-trained Grouter Models


| Model                | Source Model  | Experts | Dataset | Description                     |
| -------------------- | ------------- | ------- | ------- | ------------------------------- |
| `grouter.pth`        | Qwen3-30B-A3B | 128     | C4      | Default, Expert-Tuned           |
| `grouter_base.pth`   | Qwen3-30B-A3B | 128     | C4      | Untuned, for custom fine-tuning |
| `grouter_ft_32.pth`  | Qwen3-30B-A3B | 32      | C4      | Expert Folding + Tuning         |
| `grouter_ft_64.pth`  | Qwen3-30B-A3B | 64      | C4      | Expert Folding + Tuning         |
| `grouter_ft_256.pth` | Qwen3-30B-A3B | 256     | C4      | Expert Expanding + Tuning       |


All models available at: [ModelScope/JimmyAwoe/Grouter-Official](https://modelscope.cn/models/JimmyAwoe/Grouter-Official)

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{xu2026grouter,
  title={Grouter: Decoupling Routing from Representation for Accelerated MoE Training},
  author={Xu, Yuqi and Hu, Rizhen and Liu, Zihan and Sun, Mou and Yuan, Kun},
  journal={arXiv preprint arXiv:2603.06626},
  year={2026}
}
```

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge). We thank the open-source community for these excellent frameworks.