# 手写 GQA：头映射、因果 Mask 与 KV Cache

> 状态：教学实现与正确性验证。基于用户提供的显式 repeat 版本逐步补充。这里不包含 RoPE、Padding、Dropout、分页缓存或 GPU 性能优化，不等同于完整 LLM Attention。

## 问题与核心结论

用户代码正确表达了 GQA 的核心头映射：Q 头数多于 KV 头数时，同组 Q 头使用相同 K/V。不过原类漏掉了 `super().__init__()`，创建 Linear 子模块时会报错；补上初始化后，其无 Mask、无缓存的数据流对应完整序列的双向自注意力。

用于 Decoder 自回归模型时，至少要加入因果 Mask；做增量推理时，还要保存未扩展的历史 KV，并处理 Query 相对历史 Key 的位置偏移。

## 头数与分组

定义 Batch B、当前输入长度 S、隐藏维度 D、Query 头数 $`H_{q}`$、KV 头数 $`H_{kv}`$，以及每头维度 $`d_{h}`$。这里 Q/K/V 每头维度相同，且 $`H_{q}`$ 能被 $`H_{kv}`$ 整除：

```math
g=H_q/H_{kv},\qquad
j(i)=\left\lfloor i/g\right\rfloor,\quad
i=0,\ldots,H_q-1.
```

g 是每个 KV 头对应的 Q 头数。连续分组映射下：

```math
H_i=\mathrm{softmax}_{\mathrm{row}}
\left(\frac{Q_iK_{j(i)}^T}{\sqrt{d_h}}+M\right)V_{j(i)}.
```

例如 $`H_{q}`$ = 8、$`H_{kv}`$ = 2，g = 4：Q 头 0–3 对应 KV 头 0，Q 头 4–7 对应 KV 头 1。它们共享 Key/Value，但由于 Query 不同，Attention 权重与输出通常不同。

$`H_{kv}`$ = $`H_{q}`$ 时退化为 MHA；$`H_{kv}`$ = 1 时为 MQA。原始设计与质量/速度研究参见 [GQA 论文](https://aclanthology.org/2023.emnlp-main.298/)。共享输入不要求各 Q 头串行执行，具体依赖边界见 [MHA 各头是否独立](mha-head-independence.md)。

## 逐步对应用户代码

| 操作 | 输出 Shape | 含义 |
| --- | --- | --- |
| `w_q(x)` | `[B, S, Hq*dh]` | 每个 Q 头有自己的投影特征 |
| `w_k(x)` / `w_v(x)` | `[B, S, Hkv*dh]` | 只生成 Hkv 个 KV 头 |
| `view` + `transpose(1, 2)` | Q：`[B, Hq, S, dh]`；KV：`[B, Hkv, S, dh]` | 分头并将头维移到序列维之前 |
| KV `repeat_interleave(g, dim=1)` | `[B, Hq, S, dh]` | 将同一 KV 头连续重复 g 次，便于普通 batched matmul |
| `q @ k.transpose(-2, -1)` | `[B, Hq, S, S]` | 每个 Q 头与对应 KV 头生成分数 |
| 除以 `sqrt(dh)` + Mask + Softmax | `[B, Hq, S, S]` | 沿最后的 Key 维归一化 |
| `attn @ v` | `[B, Hq, S, dh]` | 加权求和 |
| transpose + contiguous + view | `[B, S, Hq*dh]` | 按 Token 合并头特征 |
| `w_o` | `[B, S, D]` | 输出投影与跨头特征混合 |

`hidden_dim` 不必在数学上等于 `num_q_heads * head_dim`：输入和输出 Linear 可以完成维度映射。常见模型常选择二者相等，但不需要无条件加这条断言。

### 原版本的正确边界

原来的核心代码是：

```python
k = k.repeat_interleave(self.group_size, dim=1)
v = v.repeat_interleave(self.group_size, dim=1)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, v)
```

这段分组和矩阵乘逻辑正确，能解释 GQA。需要补充四个条件：

1. 原类初始化漏掉了 `super().__init__()`，需要在创建子模块之前补上，否则无法正常实例化。头数和维度还必须为正，$`H_{q}`$ 必须能被 $`H_{kv}`$ 整除。
2. 没有因果 Mask 时，序列位置可以读取未来 Token；这在双向注意力中合法，在自回归模型中不合适。
3. 没有 `past_key_value` 时，不是利用缓存的增量 Decode。
4. 显式 repeat 增加临时 KV 数据，不是高效 GQA Backend 的必要步骤。

## 自回归 Mask 怎样构造？

没有历史缓存时，位置 q 只能看到 Key 位置 k ≤ q。用负无穷屏蔽未来分数，且必须在 Softmax 前进行。

有长度 $`T_{p}`$ 的历史缓存、当前输入包含 S 个新 Token 时：

```math
\begin{aligned}
T_k&=T_p+S,\qquad
Q\in\mathbb{R}^{B\times H_q\times S\times d_h},\\
K,V&\in\mathbb{R}^{B\times H_{kv}\times T_k\times d_h},\\
\mathrm{allowed}(q,k)&\iff k\leq T_p+q,
\quad q=0,\ldots,S-1.
\end{aligned}
```

例如已有 3 个 Token，再输入 2 个 Token：第一个新 Query 位于全局位置 3，能看 Key 0–3；第二个位于位置 4，能看 Key 0–4。Mask 是 2 × 5，不能不加偏移地套普通左上对齐的下三角。

特别是单 Token Decode，若缓存已经包含历史和当前 Token，它应该能看全部有效 Key。错误使用 1 × T 的左上对齐 causal Mask，可能只允许它看第一个 Key。

[PyTorch SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)说明 `is_causal=True` 在非方阵情况下采用左上对齐。因此调用库接口时，应确认对应版本和对齐语义，必要时传显式偏移 Mask；不能一律照搬完整 Prefill 的调用。

## KV Cache 怎样保持 GQA 的存储优势？

历史缓存保留 `[B, Hkv, T, dh]`，而不是 repeat 后的 `[B, Hq, T, dh]`。对于 K/V dtype 相同、每个元素 b 字节的单层缓存：

```math
\mathrm{KVBytes}=2BTH_{kv}d_hb.
```

固定 B、T、$`d_{h}`$、dtype 和 Q 头数时，相比 MHA 的理想缓存大小比例为 $`H_{kv}`$/$`H_{q}`$。比如 8 个 Q 头、2 个 KV 头，是相应 MHA 的 1/4；不包含分页元数据、分配粒度、量化元信息或其他 Buffer。

用户原代码的 repeat 是为了清楚呈现数学对应。高效实现可以直接索引/广播共享 KV，并在合适的工作划分中复用它，不必先物化全部重复副本。保留 Q 头数也意味着分数和输出头数不自动减少：KV 缓存缩小四倍，不代表 Attention FLOPs 或端到端延迟缩小四倍。

## 可运行教学代码与调用方式

完整实现放在 [labs/attention/gqa.py](../../../labs/attention/gqa.py)，默认启用因果 Mask，并支持返回紧凑 KV Cache。保留显式 repeat，便于与原始笔记逐行对照。

```python
import sys
import torch

sys.path.insert(0, "labs/attention")  # 从仓库根目录运行
from gqa import GQA

model = GQA(hidden_dim=32, num_q_heads=8, num_kv_heads=2, head_dim=4).eval()
x = torch.randn(1, 6, 32)
with torch.no_grad():
    # 完整因果序列计算
    full = model(x)
    # 前缀 Prefill，缓存仍然只有 2 个 KV 头
    prefix, cache = model(x[:, :5], use_cache=True)
    # 输入最后一个 Token，追加历史缓存
    last, cache = model(x[:, 5:], past_key_value=cache, use_cache=True)
    torch.testing.assert_close(last, full[:, 5:])
    # 双向教学计算，不能用作上述自回归等价对照
    bidirectional = model(x, is_causal=False)
```

这个最小实现只接受 Batch 内统一长度、无 Padding 的输入。不同请求的有效长度、空可见行、滑动窗口、RoPE 位置、缓存生命周期等需要后续独立实现和测试。缓存拼接使用 `torch.cat`，会复制历史数据；生产缓存通常采用预分配或分页，不能把此代码当性能基准。

## 正确性检查与实际限制

运行入口：

```bash
python -m unittest discover -s labs/attention -v
```

[测试代码](../../../labs/attention/test_gqa.py)检查：

- 与逐 Q 头调用 PyTorch SDPA 的结果一致，覆盖双向/因果与 MHA/GQA/MQA。
- 改变未来 Token 不影响因果前缀输出。
- 完整因果计算与逐 Token、分块缓存执行一致；检查缓存保留 Hkv 头数。
- 输入梯度和参数梯度与 SDPA 对照一致。
- 非法头数与零维度被拒绝。

本机验证环境与结果记录在 [实验说明](../../../labs/attention/README.md)。正确性检查不说明 GPU 速度；FP16/BF16 数值误差、融合 Backend 和真实模型推理另需验证。

## 面试与自测

1. **为什么使用 `repeat_interleave` 而不是随意 `repeat`？** 连续分组约定下，需要得到 KV0、KV0、…、KV1、KV1、…，保持 Q 到 KV 的组映射；不同复制顺序可能改变映射。
2. **GQA 主要减少什么？** KV 投影输出与历史 KV 存储、相关流量；不自动消除每个 Q 头的 Attention 计算。
3. **为什么 `.transpose(...).contiguous().view(...)`？** 转置后布局通常不满足直接 view 的连续性条件；这里显式整理布局再合并头维，也可用 reshape 并理解它可能复制。
4. **缓存应该在 repeat 前还是后保存？** 前；保留真实 Hkv 个 KV 头。
5. **为什么 Decode 的 Mask 容易错？** Query 的局部位置从 0 开始，但其全局位置包含历史长度，需要偏移。
6. **这份代码是完整 LLM Attention 吗？** 不是；尚未覆盖位置编码、变长请求、生产缓存和高性能 Backend。

[Attention 专题导航](README.md) · [Linear Attention](linear-attention.md)
