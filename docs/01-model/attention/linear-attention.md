# Linear Attention：怎样用固定状态代替逐 Token KV Cache？

> 状态：基础原理与形状推导。讨论可分解核特征映射的基础 Linear Attention，不将所有线性注意力或带门控、衰减的变体视为相同算法。未进行模型质量或 GPU 性能实测。

## 问题与核心结论

Linear Attention 中文是“线性注意力”，“线程注意力”容易与 GPU Thread 混淆。

它的基础思路是：使用可分解的相似度函数，将历史 K/V 聚合为固定形状状态，再用 Query 查询这个状态。**这是改变相似度定义或近似原有核函数，不是把标准 Softmax Attention 原封不动改一下矩阵乘顺序。**

“线性”指固定特征维度时，对序列长度 N 的计算复杂度呈线性增长；不表示整个运算是一个线性函数。

## 从 4096 × 4096 分数矩阵说起

单头、忽略 Batch，令 N = 4096，$`d_{k}`$ = $`d_{v}`$ = 128：

```math
\begin{aligned}
Q,K&\in\mathbb{R}^{4096\times128},\quad
V\in\mathbb{R}^{4096\times128},\\
QK^T&:\ (4096\times128)(128\times4096)
\longrightarrow4096\times4096,\\
O&=\mathrm{softmax}_{\mathrm{row}}
\left(QK^T/\sqrt{128}+M\right)V.
\end{aligned}
```

所以不是“两个 4096 × 128 直接相乘”，而是第二个矩阵先转置。完整分数矩阵有 16,777,216 个元素；若以 FP16 显式保存，需要 32 MiB。这只是一个单头分数矩阵，不包含其他头、Batch、概率矩阵和缓存。

密集标准 Attention 的交互计算约为 O(N²($`d_{k}`$ + $`d_{v}`$))。朴素实现会保存 N × N 中间矩阵，但精确 Attention 不必在外部显存完整保存它：[FlashAttention](https://arxiv.org/abs/2205.14135)通过分块和在线归一化减少 IO，基础密集算法的交互计算仍然是二次规模。

## 为什么不能直接写成 Q(KᵀV)？

没有 Softmax 时，矩阵乘法结合律成立：

```math
(QK^T)V=Q(K^TV).
```

但标准 Attention 是对整个分数矩阵逐行取 Softmax，通常不满足：

```math
\mathrm{softmax}_{\mathrm{row}}(QK^T)V
\ne Q(K^TV).
```

Softmax 包含指数变换和行归一化，不能直接跨过矩阵乘法移到另一侧。简单删掉 Softmax，得到的是另一种计算，不是标准 Attention 的等价实现。

## 特征映射与归一化

下面从一般归一化相似度自行展开形状。令 $`q_{i}`$、$`k_{j}`$ 为列向量，$`v_{j}`$ 为 $`d_{v}`$ 维列向量，映射 φ 将 Q/K 特征变为 r 维。假设产生非负相似度且分母为正：

```math
\begin{aligned}
\mathrm{sim}(q_i,k_j)
&=\phi(q_i)^T\phi(k_j),\qquad
\phi(q_i),\phi(k_j)\in\mathbb{R}^{r},\\
o_i&=\frac{\sum_j\phi(q_i)^T\phi(k_j)v_j}
{\sum_j\phi(q_i)^T\phi(k_j)}.
\end{aligned}
```

实现中通常会对分母加稳定项或作保护，但具体策略会影响结果，不能省略说明。映射可以定义不同核函数，也可以近似某种核；不能宣称任意有限维 φ 都精确复现 Softmax。

令映射后的 Q/K 按行排列，形状为 N × r，则分子的重新结合是：

```math
(\Phi_Q\Phi_K^T)V
=\Phi_Q(\Phi_K^TV),\qquad
\Phi_K^TV\in\mathbb{R}^{r\times d_v}.
```

这次可重排的是特征映射点积，行归一化分母另行维护。基础核分解和递归计算来源见 [Transformers are RNNs §3](https://arxiv.org/html/2006.16236v3#S3)。下面进一步区分非因果与因果数据流。

## 非因果聚合与因果递推

非因果情况可以将所有位置一次聚合；自回归模型的位置 t 只能看到 j ≤ t，必须使用历史前缀状态：

```math
\begin{aligned}
S_t&=\sum_{j\leq t}\phi(k_j)v_j^T
\in\mathbb{R}^{r\times d_v},\\
z_t&=\sum_{j\leq t}\phi(k_j)
\in\mathbb{R}^{r},\\
S_t&=S_{t-1}+\phi(k_t)v_t^T,\quad
z_t=z_{t-1}+\phi(k_t),\\
o_t&=\frac{S_t^T\phi(q_t)}
{z_t^T\phi(q_t)}\in\mathbb{R}^{d_v}.
\end{aligned}
```

$`S_{0}`$ 和 $`z_{0}`$ 初始化为零。每个 Batch、每层、每个相应头都需要自己的状态；“固定大小”不是整张模型只保留一个矩阵。Prefill 若直接用全序列总状态计算所有位置，就会读到未来信息，必须计算各位置的前缀状态或使用等价的因果扫描。

## 复杂度与状态大小

以下省略 Batch、头数和层数，不包括输入/输出 Linear，忽略特征映射自身的成本，令 r、$`d_{k}`$、$`d_{v}`$ 固定：

| 项目 | 标准密集因果 Attention | 基础核 Linear Attention |
| --- | --- | --- |
| 全序列交互计算 | O(N²($`d_{k}`$ + $`d_{v}`$)) | O(N r $`d_{v}`$) |
| 历史长度 t 的单 Token Decode | O(t($`d_{k}`$ + $`d_{v}`$)) | O(r $`d_{v}`$) |
| Decode 历史状态 | O(t($`d_{k}`$ + $`d_{v}`$)) | O(r $`d_{v}`$ + r) |
| 是否保存每个历史 Token 的 K/V | 是，可能压缩或量化 | 基础递推不需要 |

例如 r = $`d_{v}`$ = 128，S 有 16384 个元素，z 有 128 个元素；若状态用 FP32，合计约 64.5 KiB/头。标准单头长度 4096、K/V 都为 128 维、FP16 缓存约 2 MiB。这里是两种明确 dtype 下的存储量对比，不是质量相同或性能提升比例的证明。

如果 r 随精度要求或序列长度增长，或者变体还保存局部窗口缓存，就不能照搬“状态与序列长度完全无关”的结论。训练或批量 Prefill 的中间激活、输出、扫描实现也会占据额外内存。

## 固定状态的收益与边界

标准 Decode 可以针对当前 Query 重新读取历史 K/V；基础 Linear Attention 则把历史折叠到 S、z 中。它避免每一步扫描全部历史，但聚合之后不能像原始 KV 列表一样逐条访问历史表示，表达方式因此改变。

这种表示差异可能影响精确检索和长距离任务，程度需由具体模型、变体与评测证明。不能用“线性注意力一定不适合长距离”概括全部方法，也不能将已训练的标准 Softmax 模型无条件替换后假设质量不变。

固定状态仍可能存放在显存中，每一步要读写它。能否减少实际外部显存流量取决于状态大小、精度、缓存复用和 Kernel；理论上不扫描历史不等于已经测得显著加速。

## 与其他 Attention 优化的区别

- **GQA/MQA**：减少 KV 头数，通常保留 Softmax Attention，历史缓存仍随长度增长。
- **FlashAttention**：重排精确 Attention 的计算与存储，避免完整中间矩阵写回外部显存。
- **PagedAttention**：重点改善 KV Cache 的分页组织和管理，不把历史变成固定递归状态。
- **KV Quantization**：减少每个缓存元素的存储量，需要考虑量化误差及解码成本。
- **Linear Attention**：改变/近似相似度计算，并利用可聚合状态避免逐 Token 历史扫描。

这里说明机制差异，不对当前全部 LLM 的采用比例作未验证的概括。

## 面试与自测

1. **为什么不能把 Softmax Attention 直接改成 Q(KᵀV)？** Softmax 非线性和行归一化不能跨过乘法；需要核分解，并单独维护分母。
2. **固定状态是什么？** 对基础形式是 $`S_{t}`$ 和 $`z_{t}`$，大小由 r、$`d_{v}`$ 决定，按层、头、请求维护。
3. **标准 Decode 每步也是 O(N²) 吗？** 不是。缓存已有 K/V 后，单个 Query 对历史的交互是 O(N)；全序列交互才是 O(N²)。
4. **FlashAttention 是否将密集 Attention 计算变成 O(N)？** 没有；减少中间存储与 IO，不等于改变密集交互的阶数。
5. **状态固定为什么仍需验证性能？** 状态流量、精度、更新成本、并行粒度与模型质量都会影响收益。

[Attention 专题导航](README.md) · [手写 GQA](gqa-from-scratch.md)
