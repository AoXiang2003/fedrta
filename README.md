# FedSVD LoRA Dynamics Experiment

这是一个单文件实验，用于研究两个问题：

1. LoRA 是否优先形成完整训练终点更新矩阵中能量更高的 SVD 方向；
2. 六个独立客户端在训练过程中，LoRA 的 `A`、`B` 和 `BA` 相似度如何变化，以及 `B` 和 `BA` 对实际共享效用的预测能力是否发生分化。

客户端由两个 SST-2、两个 QNLI 和两个 50:50 混合客户端组成。各客户端从相同 LoRA 初始化开始独立训练，不执行联邦聚合。

## 文件

- `fed_svd_dynamics_experiment.py`：训练、checkpoint、SVD 分析、客户端匹配、统计检验和绘图的完整实现。
- `requirements.txt`：Python 依赖。PyTorch 应根据云端 CUDA 版本单独安装。

## 推荐运行顺序

首先运行不下载模型的数学自测试：

```bash
python fed_svd_dynamics_experiment.py --self-test-only
```

然后使用 RoBERTa-base 检查完整流水线：

```bash
python fed_svd_dynamics_experiment.py \
  --smoke-test \
  --output-dir outputs/fed_svd_smoke_v5
```

在正式付费训练前，使用一个客户端和一个优化步骤检查 RoBERTa-large 显存：

```bash
python fed_svd_dynamics_experiment.py \
  --model-name FacebookAI/roberta-large \
  --epochs 1 \
  --train-per-client 32 \
  --reference-per-client 8 \
  --compatibility-per-client 8 \
  --batch-size 32 \
  --eval-batch-size 8 \
  --standard-clients sst2_0 \
  --skip-compatibility \
  --output-dir outputs/roberta_large_preflight_v5
```

默认的单种子六客户端探索实验：

```bash
python fed_svd_dynamics_experiment.py \
  --output-dir outputs/fed_svd_roberta_large_r8_seed42
```

默认只运行 `standard` 组。Freeze-A 因果对照和额外谱截断性能评估需要显式启用，以避免探索阶段产生不必要的训练和前向开销。

## 断点恢复

```bash
python fed_svd_dynamics_experiment.py \
  --resume \
  --output-dir outputs/fed_svd_roberta_large_r8_seed42
```

训练 checkpoint 使用独立的训练协议指纹。改变数据编码、提示模板、训练过程或 checkpoint 计划时必须使用新的训练 schema；只修改统计或绘图代码时，可以在验证 checkpoint 后复用已有训练轨迹。

## 主要输出

- `protocol.json`：科学协议、训练指纹、分析指纹和资源估计；
- `data_manifest.json`：数据划分、内容哈希和 token 截断审计；
- `checkpoints/`：各客户端独立 LoRA 轨迹；
- `seed_*/metrics/`：逐层相似度、谱形成、稳定性、共享效用和 QAP 结果；
- `seed_*/figures/`：A/B/BA 逐层曲线、15 对热图、谱形成与兼容性图；
- `summary.json`：跨 seed 汇总和质量警告。
