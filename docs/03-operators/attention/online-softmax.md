# Online Softmax：Decode 怎样分块计算 Attention？

> **核心问题：当前 token 需要关注整段历史，但一个 KV 块只包含部分历史。怎样边读取、边计算，最后仍然得到整行 Softmax 的结果？**
>
> 从单 token decode 的一个注意力头出发，先理解三个状态 `m、ℓ、U`，再看它们怎样支撑 FlashAttention。这里讨论标准 Attention 的前向计算，不含 dropout。

[Attention 与 FFN](../../01-model/transformer/attention-and-ffn.md) · [FlashAttention 版本演进](../../04-backends/attention/flashattention-evolution.md) · [返回核心算子](../README.md)

## 1. Attention 里面具体做什么？

> 1. **取出当前 Query。** 当前输入 token 投影得到 Q；它的 K、V 加入本层的 KV Cache。这里的当前输入 token，还不是本次最后预测出来的下一个 token。
> 2. **读取可见的 Key，计算相关性。** Q 与每个 Key 做点积，并除以头维度的平方根，得到一行分数 S。
> 3. **沿历史位置做 Softmax。** 将分数变成非负、总和为 1 的权重 P。Padding、未来位置等不可见位置在这一步之前屏蔽。
> 4. **用权重聚合 Value。** 每个权重乘对应的 V，再求和，得到当前头的输出 O。

```math
S=\frac{QK^{\top}}{\sqrt{d_h}}+M,\qquad P=\mathrm{softmax}(S),\qquad O=PV
```

| 单头张量 | 单 token decode 的 Shape | 含义 |
| --- | --- | --- |
| Q | `1 × d_h` | 当前 token 的 Query |
| K、V | `T × d_h` | 本层全部可见 KV，包含当前位置；本例取 K、V 维度相同 |
| S、P | `1 × T` | 当前 Query 对 T 个位置的分数与权重 |
| O | `1 × d_h` | 聚合后的头输出 |

> **Decode 与 prefill 的边界：** decode 只有一个 Query；长度 N 的完整 prefill 有 N 个 Query，因而 S、P 的逻辑 Shape 是 `N × N`。后面分块公式可以逐行应用到 prefill，但不能把 decode 也讲成构造完整的 `T × T` 矩阵。

## 2. 为什么一个块一个块做 Softmax，再拼起来不行？

> 1. **每一行只有一个全局分母。** 当前 Query 对全部可见 Key 的指数分数之和，是所有权重共同使用的分母。
> 2. **块内 Softmax 的分母不同。** 如果两个块分别归一化，每块权重之和都会变成 1；直接拼接后总和变成 2，也丢失了两个块之间的相对权重。
> 3. **不仅要保存块内输出，还要保存它的尺度。** 只留下一个归一化的局部输出，无法判断这个块在全局结果里该占多大比重。

稳定 Softmax 先减去整行最大值，再计算指数：

```math
m=\max_t s_t,\qquad \ell=\sum_t e^{s_t-m},\qquad p_t=\frac{e^{s_t-m}}{\ell}
```

> 减去最大值不会改变 Softmax 结果，因为分子和分母同乘了一个因子。它让最大的指数项变成 1，避免直接计算很大的正指数。Online Softmax 的难点是：**后面读到的块可能有更大的分数，之前的统计量必须随之换尺度。**

## 3. 需要保存哪三个状态？

> 这里把未归一化的输出累积叫 **U**，把最终输出叫 **O**，避免同一个 `O_i` 在不同伪代码里含义变化。

| 状态 | 一个 Query | 一个含 B_r 行的 Query 块 | 保存什么 |
| --- | --- | --- | --- |
| m | 标量 | `B_r` | 已处理 Key 的逐行最大分数 |
| ℓ | 标量 | `B_r` | 以当前 m 为基准的指数和 |
| U | `d_h` 向量 | `B_r × d_h` | 以当前 m 为基准的加权 Value 分子 |

处理过的 Key 集合记为 A，状态满足：

```math
m=\max_{t\in A}s_t,\qquad
\ell=\sum_{t\in A}e^{s_t-m},\qquad
U=\sum_{t\in A}e^{s_t-m}V_t
```

> 1. **m 管数值尺度。** 新最大值出现时，以它为基准重新表示旧统计量。
> 2. **ℓ 管分母。** 所有块的权重最终必须共同归一化。
> 3. **U 管 Attention 输出的分子。** 只维护 m 和 ℓ，能得到 Softmax 的归一化信息，却不能凭空得到 Value 的加权和。
> 4. **最后再除一次。** 扫完全部可见 KV 后，逐行计算 `O = U / ℓ`。

## 4. 新块到来，旧状态怎样修正？

### 4.1 先计算新块的局部状态

> 当前块的分数叫 S_b，Value 叫 V_b。先得到它自己的最大值 m_b、指数矩阵 E_b 和指数和 ℓ_b。**E_b 尚未归一化，不能直接把它叫作最终概率 P。**

```math
m_b=\mathrm{rowmax}(S_b),\qquad
E_b=e^{S_b-m_b},\qquad
\ell_b=\mathrm{rowsum}(E_b),\qquad U_b=E_bV_b
```

### 4.2 统一到新的最大值

> 旧状态和新块各自使用不同的最大值，不能直接相加。先取共同基准 m_new，再分别乘修正系数 α、β。

```math
m_{\mathrm{new}}=\max(m,m_b),\qquad
\alpha=e^{m-m_{\mathrm{new}}},\qquad
\beta=e^{m_b-m_{\mathrm{new}}}
```

修正系数来自下面的恒等式，既适用于分母，也适用于输出分子：

```math
e^{s_t-m_{\mathrm{new}}}=e^{s_t-m}\,e^{m-m_{\mathrm{new}}}
```

### 4.3 合并分母与输出分子

```math
\ell_{\mathrm{new}}=\alpha\ell+\beta\ell_b,\qquad
U_{\mathrm{new}}=\alpha U+\beta U_b,\qquad
O_{\mathrm{final}}=\frac{U_{\mathrm{final}}}{\ell_{\mathrm{final}}}
```

> 1. **旧结果不是重新计算。** 将旧 ℓ、U 乘 α，就把之前所有 Key 的贡献转换到新尺度。
> 2. **新块使用同一尺度。** 将块内 ℓ_b、U_b 乘 β，然后才能与旧结果相加。
> 3. **按行广播。** 多 Query 时，α、β 是 `B_r` 个标量；乘 `B_r × d_h` 的 U 必须写成 `alpha[:, None]`、`beta[:, None]`。
> 4. **初始化要考虑 Mask。** 初始 `m = -∞、ℓ = 0、U = 0`。如果某行的新块全被屏蔽，该行应跳过更新；不要直接计算 `-∞ - (-∞)`，否则会产生 NaN。最终完全没有可见 Key 的行，要明确规定输出策略。

> 这些状态也可以并行计算后再合并，而不必只能串行扫描。实数运算下结果相同；浮点运算改变求和顺序，通常会有舍入差异，不能要求不同 kernel 逐位一致。[Online Softmax 原论文](https://arxiv.org/abs/1805.02867)讨论了在线维护归一化统计量的方法；这里进一步加入 Value 的加权分子。

## 5. 用三个分数手算一遍

> 设一行分数为 `[0, ln 2, ln 4]`，三个 Value 简化成标量 `[1, 3, 5]`。完整 Softmax 的权重是 `[1/7, 2/7, 4/7]`，最终输出为 `27/7 ≈ 3.857143`。
>
> 现在把前两个位置作为第一块，最后一个位置作为第二块。

| 步骤 | m | ℓ | U | 解释 |
| --- | --- | --- | --- | --- |
| 第一块 `[0, ln 2]` | ln 2 | `1/2 + 1 = 1.5` | `1/2 × 1 + 1 × 3 = 3.5` | 指数以 ln 2 为基准 |
| 第二块局部状态 | ln 4 | 1 | 5 | 块内只有一个位置 |
| 将第一块换到 ln 4 基准 | ln 4 | `1.5 × 1/2 = 0.75` | `3.5 × 1/2 = 1.75` | α = 1/2，β = 1 |
| 合并 | ln 4 | `0.75 + 1 = 1.75` | `1.75 + 5 = 6.75` | 分子、分母同尺度 |
| 最终归一化 | — | — | `6.75 / 1.75 = 27/7` | 与完整计算一致 |

> **为什么不能平均局部输出？** 第一块输出是 `3.5 / 1.5 = 7/3`，第二块输出是 5；简单平均得到 `11/3`，与 `27/7` 不同。每个块的全局权重需要由最大值与指数和恢复，不能默认各占一半。

## 6. 分块参考代码：把广播与 Mask 写清楚

> 以下是用于核对数学的 PyTorch 参考实现，支持一个或多个 Query 行。它会创建当前块的分数和指数张量，使用普通 PyTorch 算子执行；**它不是融合 GPU kernel，也没有 FlashAttention 的性能特征。** 输入要求 Q、K、V 在同一设备、同一浮点 dtype，元素有限；`valid` 为同设备的布尔矩阵，True 表示可见。

```python
import math
import torch


def online_attention(q, k, v, block_size, valid=None):
    # q: [B_r, d_h], k: [T, d_h], v: [T, d_v]
    if block_size <= 0 or k.shape[0] == 0:
        raise ValueError("block_size and context length must be positive")
    rows = q.shape[0]
    m = torch.full((rows,), -torch.inf, dtype=q.dtype, device=q.device)
    ell = torch.zeros_like(m)
    u = torch.zeros((rows, v.shape[1]), dtype=q.dtype, device=q.device)

    for start in range(0, k.shape[0], block_size):
        end = min(start + block_size, k.shape[0])
        scores = (q @ k[start:end].T) / math.sqrt(q.shape[1])
        if valid is not None:
            scores = scores.masked_fill(~valid[:, start:end], -torch.inf)

        block_max = scores.amax(dim=1)
        m_new = torch.maximum(m, block_max)
        # 尚未遇到有效 Key 的行，使用 0 作临时指数基准。
        # 被屏蔽分数仍为 -inf，其指数为 0，状态保持空。
        base = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
        alpha = torch.exp(m - base)  # 空状态下 exp(-inf) = 0
        e = torch.exp(scores - base[:, None])
        ell = alpha * ell + e.sum(dim=1)
        u = alpha[:, None] * u + e @ v[start:end]
        m = m_new

    if torch.any(ell == 0):
        raise ValueError("each query must have at least one visible key")
    return u / ell[:, None]
```

> 1. **这份代码直接使用新的全局基准。** `e = exp(scores - m_new)` 已包含前面公式中的 β，因此不用额外再乘 β。
> 2. **旧状态仍然要缩放。** `alpha * ell` 和 `alpha[:, None] * u` 缺一不可。
> 3. **块大小不是数学条件。** 最后一个 KV 块可以短于 `block_size`；只要所有可见位置都处理一次，结果就成立。
> 4. **低精度实现另有要求。** 实际 FP16/BF16 kernel 通常需要更高精度的归约与累积，还涉及指数近似、Tensor Core 布局和寄存器资源；本代码用于验证递推关系，适合先用 FP64 核对。

## 7. 为什么它能支撑 FlashAttention？

> 1. **当前块可以算完就丢。** S_b、E_b 的贡献已归入 m、ℓ、U，无需保存此前所有分数与概率。
> 2. **融合后可以省掉中间张量的显存往返。** 分块 QK、指数归约和乘 V 在 kernel 内衔接，避免把完整 S、P 写入显存后再读回来。
> 3. **算法状态不等于 Shared Memory 分配。** m、ℓ、U 可以映射到寄存器和累加器；K/V tile 等可能放 Shared Memory。具体存放取决于实现，而不是所有中间量都放进一个“SRAM 数组”。
> 4. **Online Softmax 不等于完整 FlashAttention。** 它解决分块后的全局归一化；高性能实现还需要 tiling、并行分工、layout、流水和同步。

### 7.1 4096 × 4096，到底占多少空间？

> 以下只统计一个 batch 中的一个头，假设 S、P 各自完整物化，且都用同一种 dtype；忽略 allocator、其他张量与复用。N = 4096、d_h = 64 时，prefill 的 S、P 均为 `4096 × 4096`，矩阵大小由两个序列维决定，d_h 不直接参与这项存储计算。

| 场景 | S 或 P 的 Shape | 每张矩阵 FP32 | 两张合计 FP32 | 两张合计 FP16/BF16 |
| --- | --- | --- | --- | --- |
| 完整 prefill | `4096 × 4096` | 64 MiB | 128 MiB | 64 MiB |
| 单 token decode，T = 4096 | `1 × 4096` | 16 KiB | 32 KiB | 16 KiB |

> 这里使用二进制单位：`1 MiB = 2²⁰ bytes`。128 MiB 不是无条件成立的峰值：如果 S、P dtype 不同，或复用同一缓冲区，实际占用会改变。它说明完整物化的代价，不是说普通 Attention 会先尝试把两张完整矩阵塞进 Shared Memory。
>
> Shared Memory 按 SM、CTA 的资源限制分配，不能把所有 SM 的容量合成一个任意 kernel 都能访问的共享池。分块的目的，是让局部工作集适配片上资源，并减少完整中间张量的读写。数据中心 GPU 常说 HBM；RTX 4090 使用 GDDR6X，本文涉及它时应理解为对应的显存访问。

### 7.2 是否把计算复杂度变成线性？

> 1. **完整 dense prefill 的主计算仍是 O(N² d_h)。** FlashAttention 改变数据流和中间存储，不会免除所有可见 Q/K 配对的点积。
> 2. **单 token decode 的主计算是 O(T d_h)。** 一个 Query 扫描历史，本来就随历史长度线性增长。
> 3. **KV Cache 仍存在。** Online Softmax 的小状态只是本次 Attention 的计算状态，下一步 decode 仍需要历史 K/V；它不同于用固定递归状态替代逐 token KV 的 Linear Attention。

## 8. 面试时怎样讲？

### 为什么分块后还能得到完整 Softmax 的结果？

> 因为每个块保留最大值、指数和与加权 Value 分子。合并时先把新旧状态转换到同一个最大值基准，再分别加总分母和分子，最后归一化。这个过程保留了所有位置之间的相对权重。

### 为什么分母和输出累积都要 rescale？

> 它们都依赖之前的最大值。新的最大值出现后，旧的指数项统一乘 `exp(m_old - m_new)`；分母是指数项的和，输出分子是指数项乘 V 的和，所以二者都必须乘这个系数。

### Online Softmax 会不会不再读取 KV Cache？

> 会继续读取。它避免保留完整分数与权重，不会替代历史 K/V。单 token decode 仍要处理可见历史，长上下文时 KV 读取常常是关键开销。

[继续：FlashAttention v1 / v2 / v3 怎样改变执行过程](../../04-backends/attention/flashattention-evolution.md)
