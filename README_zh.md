# Grouter: 解耦路由器以加速MoE训练

**通过预训练路由器的知识蒸馏加速MoE模型预训练**

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

## 概述

**Grouter**（Global Router，全局路由器）是一种新颖的MoE模型预训练加速方法。其核心思想是：将已有预训练MoE模型（Source Model）的路由知识蒸馏为一个轻量级的独立路由网络，然后用该网络指导新MoE模型（Target Model）的专家选择，从而实现更快的收敛速度和更好的负载均衡。

### 核心思想

> *既然已有的预训练模型已经学会了"哪些专家擅长什么"，为什么还要在新模型训练中花费数十亿token重新学习？*

### 核心优势

- **显著加速收敛** — Grouter引导的训练收敛速度远超传统辅助损失路由方法
- **极低计算开销** — Grouter网络仅约30M参数，相对目标模型几乎无额外FLOPs
- **灵活的专家数量适配** — Expert Folding与Expert Expanding支持任意目标专家数量
- **低成本蒸馏** — 部分模型加载技术使得从235B参数模型蒸馏路由仅需8张H100
- **即插即用** — 与Megatron-LM无缝集成，只需在训练脚本中添加几个参数即可启用

---

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Grouter引导的MoE训练                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入Tokens ──┬──► 目标MoE路由器 ──► 专家选择                    │
│                │      （学习中）          ▲                       │
│                │                         │                       │
│                └──► Grouter ──────► 路由指导                      │
│                    （冻结，蒸馏而来）                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Grouter蒸馏流程:                                                │
│  Source MoE（如Qwen3-30B）─► 前N层Transformer + 路由头           │
│                               ─► 轻量级Grouter网络               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境配置

建议使用Docker构建实验环境。我们的实验均基于 `nvcr.io/nvidia/nemo:25.07` 镜像。

```bash
export GROUTER_PATH=/path/to/Grouter
cd $GROUTER_PATH

# 构建镜像并启动容器
bash env/create_image.sh
bash env/create_container.sh
```

> `create_container.sh` 会自动设置 `GROUTER_PATH=/workspace/Grouter`。若不使用此脚本启动容器，请确保手动设置该环境变量。

**替代方案：手动环境配置**

若无法使用Docker，请确保安装以下依赖：

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

我们的训练基于Megatron-LM，你也可以参考 [Megatron-LM官方](https://github.com/NVIDIA/Megatron-LM) 提供的环境配置方案。经测试，`nvcr.io/nvidia/pytorch:25.03-py3` 及更新版本的镜像同样可以正常运行训练。

### 2. 配置Megatron-LM

```bash
cd $GROUTER_PATH

# 克隆并切换到测试过的版本
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git reset --hard e7c55de9 && cd ..

# 创建软链接，使Megatron-LM能找到Grouter相关包
ln -s ../utils_grouter Megatron-LM/utils_grouter

# 应用我们的补丁
cp -r Megatron-LM-Patch/* Megatron-LM/
```

### 3. 下载Grouter权重（推荐）

我们通过 [ModelScope](https://modelscope.cn/models/JimmyAwoe/Grouter-Official) 提供预训练好的Grouter权重：

```bash
cd $GROUTER_PATH

# 默认Grouter（蒸馏自Qwen3-30B-A3B，已在C4数据集进行Expert Tuning，128专家）
modelscope download --model JimmyAwoe/Grouter-Official grouter.pth \
    --local_dir utils_grouter/grouter/grouter_model

# Expert Folding/Expanding变体
modelscope download --model JimmyAwoe/Grouter-Official grouter_ft_32.pth \
    --local_dir utils_grouter/grouter/grouter_model  # 32专家
modelscope download --model JimmyAwoe/Grouter-Official grouter_ft_64.pth \
    --local_dir utils_grouter/grouter/grouter_model  # 64专家
modelscope download --model JimmyAwoe/Grouter-Official grouter_ft_256.pth \
    --local_dir utils_grouter/grouter/grouter_model  # 256专家

# 未微调的Base版本（适用于自定义数据集或微调场景）
modelscope download --model JimmyAwoe/Grouter-Official grouter_base.pth \
    --local_dir utils_grouter/grouter/grouter_model
```

### 4. 数据准备

```bash
cd $GROUTER_PATH

# 下载C4数据集（根据存储容量调整--num_files）
python3 dataset/download_c4.py --num_files 256

# Megatron-LM预处理（tokenize为二进制格式）
bash scripts/preprocess_data.sh          # 处理文件0~255
bash scripts/preprocess_data.sh 10       # 处理前10个文件
bash scripts/preprocess_data.sh 250 599  # 处理文件250~599
```

> C4数据集包含1024个文件，每个约320MB。Megatron-LM支持数据不足时自动启动多epoch训练，因此存储有限时可适当减少数据量。

### 5. 使用Grouter进行训练

```bash
cd $GROUTER_PATH
bash scripts/grouter_train.sh
```

此脚本将在8张GPU上启动Tiny-Qwen3风格MoE模型的Grouter引导预训练。

---

## 进阶使用

### 自行蒸馏Grouter

若希望从Source Model自行蒸馏（而非使用我们提供的权重）：

**完整蒸馏流程**

#### 1. 下载Source Model

```bash
cd $GROUTER_PATH
python3 model_home/download_qwen3_30b.py
```

你可以使用其他MoE模型作为Source Model。根据我们的经验，不同Source Model蒸馏得到的Grouter均能有效加速收敛，但最终效果有所差异。

#### 2. 转换权重格式

```bash
# 配置Megatron-Bridge用于权重转换
cd $GROUTER_PATH
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge && git reset --hard 45ed38a9
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git reset --hard b8581a35 && cd ../..

# HuggingFace → Megatron格式转换
bash scripts/bridge_qwen3.sh
```

转换完成后输出：

```
✅ Successfully imported model to: model_home/qwen3-30b-a3b-converted
```

#### 3. 启动蒸馏

```bash
bash scripts/distillation_qwen3.sh
```

**部分加载技术：** 由于第N层Router的输出仅依赖前N层Transformer，我们的框架支持仅加载前N层，在无任何损失的情况下极大节省计算量：


| 加载方式        | GPU需求（Qwen3-235B） |
| ----------- | ----------------- |
| 全量加载        | ~256张 H100        |
| 部分加载（我们的方法） | ~8张 H100          |


通过 `--grouter-allow-partial-load` 和 `--num-layers` 参数启用。

### Expert Folding（减少专家数量）

使用基于专家亲和度的合并，将Grouter适配到更少的专家数量：

```bash
cd $GROUTER_PATH
bash scripts/run_construct_mapping.sh 64  # 折叠到64专家
```

### Expert Expanding（增加专家数量）

将Grouter扩展到更多专家：

```bash
cd $GROUTER_PATH
bash scripts/run_expand_expert_mapping.sh 256  # 扩展到256专家
```

### Expert Tuning

对Grouter进行微调以适配目标模型的数据分布，解决负载不均衡问题：

```bash
cd $GROUTER_PATH
bash scripts/run_finetune_grouter.sh \
    utils_grouter/grouter/grouter_model/cvt64_map_affinity.json \
    utils_grouter/grouter/grouter_model/grouter_base.pth
```

微调完成后自动保存至 `utils_grouter/grouter/grouter_model/checkpoint_step_400.pth`。

---

## 仓库结构

```
Grouter/
├── env/                          # Docker环境配置
│   ├── Dockerfile
│   ├── create_image.sh
│   └── create_container.sh
├── scripts/                      # 训练与工具脚本
│   ├── grouter_train.sh          # Grouter引导训练主脚本
│   ├── distillation_qwen3.sh    # Grouter蒸馏
│   ├── bridge_qwen3.sh          # 权重格式转换
│   ├── preprocess_data.sh       # 数据预处理
│   ├── run_construct_mapping.sh  # Expert Folding
│   ├── run_expand_expert_mapping.sh  # Expert Expanding
│   └── run_finetune_grouter.sh  # Expert Tuning
├── utils_grouter/                # Grouter核心库
│   ├── grouter/                  # Grouter模型与钩子
│   │   ├── general_router.py    # Grouter网络定义
│   │   ├── grouter_hook.py      # 训练集成钩子
│   │   ├── grouter_model/       # 模型配置与权重存放
│   │   └── structure/           # 架构变体（MHA, MLA等）
│   ├── megatron_integration/    # Megatron-LM集成层
│   └── tools/                   # 独立工具（蒸馏、映射等）
├── Megatron-LM-Patch/           # Megatron-LM框架补丁
├── dataset/                     # 数据下载脚本
└── model_home/                  # 模型下载脚本
```

---

## 预训练Grouter模型


| 模型                   | Source Model  | 专家数 | 数据集 | 说明                        |
| -------------------- | ------------- | --- | --- | ------------------------- |
| `grouter.pth`        | Qwen3-30B-A3B | 128 | C4  | 默认版本，已Expert Tuning       |
| `grouter_base.pth`   | Qwen3-30B-A3B | 128 | C4  | 未微调，适合自定义场景               |
| `grouter_ft_32.pth`  | Qwen3-30B-A3B | 32  | C4  | Expert Folding + Tuning   |
| `grouter_ft_64.pth`  | Qwen3-30B-A3B | 64  | C4  | Expert Folding + Tuning   |
| `grouter_ft_256.pth` | Qwen3-30B-A3B | 256 | C4  | Expert Expanding + Tuning |


所有模型可从 [ModelScope/JimmyAwoe/Grouter-Official](https://modelscope.cn/models/JimmyAwoe/Grouter-Official) 下载。

---

## 引用

如果本工作对您有帮助，请引用我们的论文：

```bibtex
@article{xu2026grouter,
  title={Grouter: Decoupling Routing from Representation for Accelerated MoE Training},
  author={Xu, Yuqi and Hu, Rizhen and Liu, Zihan and Sun, Mou and Yuan, Kun},
  journal={arXiv preprint arXiv:2603.06626},
  year={2026}
}
```

---

## 许可证

本项目基于 Apache License 2.0 开源 — 详见 [LICENSE](LICENSE)。

## 致谢

本项目基于 [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 构建。感谢开源社区提供的优秀框架。