# Grouter Megatron Integration

这个模块提供了将预处理的Grouter数据集集成到Megatron-LM框架中的功能，用于EP通信优化。

## 功能特性

- **直接数据加载**: 从预处理的Grouter数据集中直接加载数据，跳过Megatron的标准数据处理流程
- **节点感知**: 根据当前节点和GPU ID自动加载对应的数据集
- **批次优化**: 支持预优化的micro batch数据加载
- **无缝集成**: 与现有Megatron训练脚本兼容

## 文件结构

```
megatron_integration/
├── __init__.py
├── grouter_dataset_adapter.py      # Grouter数据集适配器
├── grouter_megatron_integration.py # Megatron集成函数
├── grouter_pretrain_gpt.py         # 修改后的训练脚本
├── add_argument.py                 # 命令行参数扩展
└── README.md                       # 本文档
```

## 使用方法

### 1. 准备数据

首先使用`run_integration.py`处理你的数据集：

```bash
python run_integration.py \
    --config /path/to/optimization_config.json \
    --dataset-path /path/to/clustered/data \
    --predispatch-path /path/to/predispatch/data \
    --output-dir /path/to/megatron/processed/data \
    --data-prefix node_0_gpu_0_data node_0_gpu_1_data
```

### 2. 运行训练

使用提供的训练脚本：

```bash
bash scripts/run_grouter_megatron_training.sh
```

或者直接使用torchrun：

```bash
torchrun --nproc-per-node 8 --nnodes 2 \
    Megatron-LM/pretrain_gpt.py \
    --use-grouter-dataset \
    --grouter-data-path /path/to/processed/data \
    --use-grouter-weight \
    --grouter-checkpoint-path /path/to/grouter/checkpoint \
    --num-experts 64 \
    --moe-router-topk 6 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --seq-length 4096 \
    --train-iters 10000
```

### 3. 命令行参数

新增的Grouter相关参数：

- `--use-grouter-dataset`: 启用Grouter数据集模式
- `--grouter-data-path`: Grouter预处理数据路径
- `--grouter-node-id`: 节点ID（可选，自动检测）
- `--grouter-gpu-id`: GPU ID（可选，自动检测）
- `--use-grouter-weight`: 使用Grouter权重计算
- `--grouter-checkpoint-path`: Grouter检查点路径

## 数据格式要求

Grouter数据集应该按以下结构组织：

```
grouter_data_path/
├── node_0_gpu_0_data.bin
├── node_0_gpu_0_data.idx
├── node_0_gpu_0_data_predispatch.bin
├── node_0_gpu_0_data_predispatch.idx
├── node_0_gpu_0_assignments.json
├── node_0_gpu_1_data.bin
├── node_0_gpu_1_data.idx
├── ...
└── node_N_gpu_M_data.bin
```

每个GPU的数据集包含：
- `*.bin` 和 `*.idx`: 主要的token数据
- `*_predispatch.bin` 和 `*_predispatch.idx`: 预计算的dispatch信息
- `*_assignments.json`: 样本分配信息

## 核心组件

### GrouterDataset

继承自MegatronDataset，提供：
- 直接从预处理数据加载
- 支持predispatch数据
- 自动批次填充和GPU传输

### GrouterBatchLoader

替换`get_batch_on_this_tp_rank`，提供：
- 优化的批次加载
- 节点感知的数据选择
- 与现有训练循环兼容

### 集成函数

- `grouter_train_valid_test_datasets_provider`: 替换标准数据集提供者
- `grouter_get_batch`: 替换标准批次获取函数
- `patch_megatron_for_grouter`: 自动修补Megatron函数

## 性能优化

1. **预计算**: 所有数据处理在训练前完成
2. **直接加载**: 跳过Megatron的数据预处理步骤
3. **内存优化**: 只加载当前GPU需要的数据
4. **通信优化**: 利用预优化的样本分配

## 故障排除

### 常见问题

1. **数据集未找到**
   - 检查`--grouter-data-path`是否正确
   - 确认数据文件命名格式正确

2. **节点/GPU ID错误**
   - 检查环境变量`RANK`和`WORLD_SIZE`
   - 手动指定`--grouter-node-id`和`--grouter-gpu-id`

3. **内存不足**
   - 减少`--micro-batch-size`
   - 检查数据预处理是否正确

### 调试模式

启用详细日志：

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
```

## 扩展开发

要添加新功能：

1. 在`grouter_dataset_adapter.py`中扩展`GrouterDataset`类
2. 在`grouter_megatron_integration.py`中添加新的集成函数
3. 在`add_argument.py`中添加新的命令行参数

## 注意事项

- 确保所有节点都能访问相同的数据路径
- 数据预处理必须与训练配置匹配
- 建议在单节点上先测试集成