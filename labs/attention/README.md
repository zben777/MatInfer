# Attention 教学实现与正确性检查

当前包含 [GQA 参考实现](gqa.py)及[正确性测试](test_gqa.py)，对应[原理与逐行解释](../../docs/01-model/attention/gqa-from-scratch.md)。

实现默认使用因果 Mask，支持统一长度 Batch 的完整 Prefill、逐 Token 与分块缓存输入，也支持无缓存的双向模式。缓存保存真实 KV 头，内部显式重复仅用于教学；没有 RoPE、Padding、分页或融合 Kernel。

## 运行

依赖 Python 与 PyTorch。从仓库根目录运行：

```bash
python -m unittest discover -s labs/attention -v
```

测试使用 CPU float64、小尺寸随机数据，与 PyTorch SDPA 逐 Q 头对照，覆盖输出、因果可见性、缓存等价、梯度和参数检查。它不是模型精度评测或性能 Benchmark。

## 验证记录

验证日期：2026-10-05。macOS，Python 3.9.6，PyTorch 2.8.0，CPU float64。5 项测试全部通过，涵盖不同头配置的多个子用例；未测量 RTX 4090 或其他 GPU 性能。环境没有 NumPy，PyTorch 启动时提示 NumPy 不可用；这些测试未使用 NumPy。

## 后续实验

- 增加 RoPE，验证 Prefill 与 Decode 的位置一致性。
- 用预分配/分页缓存代替反复拼接，区分正确性参考与生产实现。
- 在 GPU 上比较显式 repeat 和支持原生 GQA 的融合 Backend，记录版本、Shape、dtype、累加精度和真实显存流量。

[返回实验导航](../README.md)
