# FlashAttention v1 / v2 / v3：数据流、并行与异步流水

> **核心问题：同样计算标准 Attention，为什么先减少显存读写还不够，后来还要调整线程分工，并重叠 Softmax 与矩阵计算？**
>
> 先读 [Online Softmax 的推导](../../03-operators/attention/online-softmax.md)。本文聚焦三个版本的前向设计；伪代码表达算法与依赖，不代表任意版本、任意 Shape 的源码都使用同一个 grid 或指令序列。

[返回高性能 Backend](../README.md) · [MMA 与内存流水线](../../05-kernel-engineering/pipeline/mma-memory-pipeline.md) · [算子融合](../../03-operators/fusion/operator-fusion.md)

## 1. 先确定场景：Decode 与 prefill 看见的瓶颈不同

> 1. **单 token decode：Q 只有一行。** 每个头的 S、P 是 `1 × T`。重点是读历史 KV、减少启动开销，以及在 batch/head 不足时增加并行任务。
> 2. **Prefill：Q 有多行。** 完整 S、P 是 `N × N`，中间矩阵落显存的代价明显；Query 行块也提供了大量互不依赖的任务。
> 3. **同一个分块归一化方法可以用于两者。** 但用很多 Query 行提高并行度的优化，在单 token decode 中不能原样获得同样收益。
> 4. **下面的循环对比主要用 prefill 展示数据流。** 回到 decode 时，要单独看 Query 数量与 KV 长度，不以版本编号推断延迟高低。

## 2. Naive Attention 的访存代价在哪里？

> 典型拆分实现分别执行 QK、Softmax、PV。QK 将 S 写入显存；Softmax 读取 S、写出 P；PV 读取 P，再得到 O。长 prefill 时，两张中间矩阵可能比 Q/K/V 大很多。
>
> [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)用 IO-aware 分块避免完整 S、P 的显存往返。关键是局部计算后立刻合并输出状态，而不是把完整矩阵搬进 Shared Memory。

| 要保存的量 | 完整物化方案 | 分块融合方案 |
| --- | --- | --- |
| S、P | 可能以完整矩阵写入显存 | 处理当前 tile，随后复用局部空间 |
| m、ℓ、输出累积 | Softmax 与输出计算分别处理 | 每行维护可合并状态 |
| Q、K、V 与最终 O | 仍需读写 | 仍需读写，按 tile 安排 |

> **“减少 O(N²) 中间存储”与“消除 O(N²) 计算”是两件事。** Dense prefill 的主要计算仍是 O(N² d_h)。此外，本页只讲前向；论文中的反向重计算是另一条减少存储的途径。

## 3. v1：先解决完整 S、P 落显存的问题

### 3.1 KV 外层、Q 内层意味着什么？

> 原始论文的算法描述先固定一个 K/V 块，再遍历各 Q 块：当前 K/V 块供多个 Q 块使用，O、ℓ、m 则随块更新。按这个数据流，当前 Q 块的状态需要在下一轮 KV 到来时重新读入。
>
> 下面是无 Mask 的概念伪代码，`load/store` 表示显式存取边界；初始化和末尾的短块处理略去。O 保存**已经归一化的部分输出**，因此不能使用下一节 U 的更新式直接替换。

```python
# 算法示意：Q/K/V 已分块，初始 O = 0, ell = 0, m = -inf。
# 描述原论文的循环和状态读写，不是可运行的 CUDA 程序。
for j in kv_blocks:
    K_j, V_j = load_kv(j)
    for i in query_blocks:
        Q_i = load_q(i)
        O_i, ell_i, m_i = load_state(i)

        S = (Q_i @ K_j.T) / sqrt(d_h)
        m_b = rowmax(S)
        E_b = exp(S - m_b[:, None])
        ell_b = rowsum(E_b)
        m_new = maximum(m_i, m_b)
        alpha = exp(m_i - m_new)
        beta = exp(m_b - m_new)
        ell_new = alpha * ell_i + beta * ell_b

        numerator = (alpha * ell_i)[:, None] * O_i
        numerator += beta[:, None] * (E_b @ V_j)
        O_new = numerator / ell_new[:, None]
        store_state(i, O_new, ell_new, m_new)
```

> 1. **先恢复旧分子。** O_i 已经除过旧分母，合并时要乘回 ℓ_i，再乘最大值修正系数 α。
> 2. **新分母必须先计算。** `ell_new` 由新旧块的指数和合并得到；缺少这一步，代码既不能执行，也无法正确归一化。
> 3. **每一轮更新都会归一化。** 这包含对输出元素的逐行除法，后续还有减少它的空间。
> 4. **S、P 的巨大物化已经被避免。** 状态 O/ℓ/m 的读写依然存在，但规模不同，不能据此说 v1 没有减少显存 IO。

### 3.2 算法循环与实际并行怎样对应？

> FA2 论文回顾的早期前向实现按 batch/head 分配 CTA。教学上可用 `grid ≈ (batch, heads)` 表示；每个 CTA 内仍有大量线程合作完成分块计算。“一个 CTA”不等于一个线程。
>
> 这是解释论文版本演进的背景，不能把它扩展成所有名为 FA1 的后续实现都只能使用这个 grid；具体 dispatch、Shape 特化与修订需查对应源码。

## 4. v2：让已有计算更容易跑满 GPU

> v2 的三个主方向是：**少做非矩阵运算、增加 Query 块级并行、减少 warp 间输出归约。** 版本区别依据 [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)；以下伪代码和数值式用来解释这些设计。

### 4.1 累积未归一化分子，最后才除分母

> 将状态改成 U，使它在 KV 循环中保持未归一化。每轮只修正旧 U 并加上新贡献；扫完全部 KV 后，再逐行除以 ℓ。

```python
# 无 Mask 的算法示意；一个任务处理一个 Q 行块。
Q_i = load_q(i)
U = zeros(B_r, d_h)
ell = zeros(B_r)
m = full(B_r, -inf)

for j in kv_blocks:
    K_j, V_j = load_kv(j)
    S = (Q_i @ K_j.T) / sqrt(d_h)
    m_new = maximum(m, rowmax(S))
    alpha = exp(m - m_new)
    E = exp(S - m_new[:, None])

    U = alpha[:, None] * U + E @ V_j
    ell = alpha * ell + rowsum(E)
    m = m_new

store_output(i, U / ell[:, None])
```

> 1. **循环内的 U 不是最终 O。** 数值还没除分母，将它直接输出会出错。
> 2. **减少的是重复归一化。** 点积和 Value 聚合仍然存在；最大值变化时的 rescale 也不能省掉。
> 3. **状态随 Q 块驻留片上。** 在没有 spill 的理想实现里，U、ℓ、m 不必每轮写回显存。寄存器不足时的 spill 会破坏这种收益，需要看编译结果与 profile。

### 4.2 Q 外层、KV 内层，增加 CTA 并行

> 1. **不同 Q 行的输出独立。** 一个 CTA 可以负责一个 Q 行块，再顺序扫它需要的 KV 块。
> 2. **可并行的任务数增加。** 教学上写作 `grid ≈ (batch, heads, query_tiles)`，不规定实际 CUDA grid 各轴的排列。
> 3. **成本也随之改变。** 不同 Q 块可能重复加载 K/V；缓存、tile 尺寸与 occupancy 决定实际代价，不能只看循环层数。
> 4. **单 token decode 没有很多 Q 行块。** 填充到一个 tile 也不会创造新的有效 Query，所以这条并行收益主要在多 Query 场景体现。

### 4.3 Sliced-K → Sliced-Q：减少 CTA 内输出通信

> 1. **Sliced-K：多个 warp 分担 Key 方向。** 它们对同一组 Query 产生部分贡献，最后需要合并输出，伴随 Shared Memory 访问与同步。
> 2. **Sliced-Q：多个 warp 分担 Query 行。** K/V 供它们共同使用，各 warp 生成不同输出行，减少为合并同一行部分输出而产生的通信。
> 3. **减少特定归约不代表 kernel 完全无同步。** K/V 的加载、共享与计算安排仍有协调要求。

### 4.4 exp2 是实现选择，不是版本定义

> 为了用 `exp2`，可以把分数转换成以 2 为底的指数域。Mask 仍以负无穷屏蔽；最大值、ℓ 和 U 的递推必须一致使用该域。

```math
e^x=2^{x\log_2 e},\qquad
S_2=\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)\log_2 e
```

> 可把比例系数折进 Q 或 score，但浮点舍入路径会改变。不能把“v1 用 exp、v2 用 exp2”当成根本区别，也不能仅凭函数名断言一个 kernel 更快：编译器可能将自然指数转换为底 2 指数指令。

### 4.5 Mask 与异步加载需要另外说明

> 1. **整块全部不可见时可以跳过。** Prefill 的因果 Mask 往往可以跳过未来区域的整块；边界块还需逐元素屏蔽。
> 2. **全屏蔽行需要专门处理。** 不允许让 `-inf - (-inf)` 污染累积状态；数学参考实现见 Online Softmax 专题。
> 3. **cp.async 与 TMA 不是通用同义词。** 加载方式取决于目标 GPU 与实现。TMA 是 Hopper 等架构支持的能力，不能因为写了 FA2 就默认可用 TMA。

## 5. v3：Hopper 上怎样重叠搬运、矩阵乘和 Softmax？

> [FA3 作者的技术讲解](https://tridao.me/blog/2024/flash3/)介绍了 TMA/WGMMA 异步能力、GEMM 与 Softmax 的重叠，以及 FP8 低精度处理。在线归一化关系仍然保留，改变的是计算的组织方式。

### 5.1 分别利用什么硬件？

| 工作 | 主要执行资源或机制 | 必须等待什么 |
| --- | --- | --- |
| K/V tile 搬入 Shared Memory | TMA 异步搬运 | 数据传输完成、正确的 barrier/phase |
| QK、PV 矩阵乘 | Tensor Core，WGMMA | 相关矩阵运算完成、结果可读 |
| max、指数、求和、rescale | 常规计算路径与指数功能单元 | 所依赖的 score 或输出累加器就绪 |

> Producer 主要负责数据搬运和可用状态；consumer warpgroup 完成矩阵计算及相关归一化处理。WGMMA 的参与单位是符合要求的 warpgroup，不能把 QK、PV 随意描述成各由一个独立 warp 发射。

### 5.2 为什么重叠 Softmax 与 GEMM？

> 1. **同一块仍有顺序依赖。** QK 完成后才能读取 score；得到该块的指数权重后，才能计算对应 PV。
> 2. **不同任务可以错开。** 准备下一块 K/V，可以与当前块计算并行；不同 consumer warpgroup 可以交错执行矩阵乘与 Softmax。
> 3. **同一 warpgroup 也可以安排跨块流水。** 在满足数据依赖和累加器访问约束时，部分矩阵运算执行期间可处理另一阶段的 Softmax 工作。
> 4. **异步不等于无等待。** 完成通知决定什么时候能读取 score、访问输出累加器，以及重用输入缓冲区。

### 5.3 先看依赖，再谈流水级数

> 以下是**一个 KV 块的依赖清单**，不是完整 v3 kernel，也不是可编译的指令序列。箭头只规定先后条件；实际 kernel 让不同块、不同工作组错开执行。

```text
K_j 搬运完成 + Q 就绪
          ↓
发射 QK_j 的异步矩阵计算
          ↓ 等待对应 QK 结果可读
得到 S_j，计算 m_new、修正系数与 E_j
          ↓
使 E_j 对后续矩阵计算可见 + V_j 搬运完成
          ↓
按依赖修正旧 U，发射 E_j × V_j 并累加
          ↓ 等待相关矩阵计算完成
输出状态可安全访问；确认最后一个读者结束
          ↓
释放 K/V/E 对应阶段，允许后续覆盖
```

> 1. **Score 不会在发射 WGMMA 后立即就绪。** `S = wgmma(...)` 下一行直接 `rowmax(S)`，隐藏了关键的完成等待。
> 2. **不能提前释放缓冲区。** 某个异步矩阵计算仍在读取 V 或 E 时，producer 不得将该阶段覆盖成下一块。
> 3. **U 的 rescale 与累加有顺序约束。** 数学上要形成 `U_new = alpha × U_old + E × V`；普通 MMA 不提供任意的 `rescale=` 参数。实际实现需安排缩放与累加，避免访问尚未完成的累加器。
> 4. **两级流水不是固定定义。** 增加阶段可能隐藏更多延迟，也消耗更多 Shared Memory、寄存器与同步状态，必须结合 tile、shape 和硬件调优。

> 关于 TMA 的 full/empty 状态、WGMMA 的 commit/wait，以及为什么“发射完成”不能作为“缓冲区可覆盖”的依据，继续读 [MMA 与内存流水线](../../05-kernel-engineering/pipeline/mma-memory-pipeline.md)。

### 5.4 FP8 是另一条优化线

> v3 还利用 Hopper 的 FP8 矩阵吞吐，并通过 incoherent processing 等方式改善量化误差。它引入精度与数据转换的取舍；不能把 FP8 结果宣称为与 FP16 逐位相同，也不能把所有性能提升归结为 TMA。
>
> RTX 4090 属于 Ada、计算能力 8.9，不能执行 Hopper 的 WGMMA/TMA 路径。学习 v3 的设计有帮助，但本地验证需要选择硬件支持的实现。

## 6. 回到 Decode：为什么还会出现 Flash-Decoding？

> **一个 Query，历史很长，但 batch/head 很少**时，仅沿 Q 行切分可能提供不了足够的 CTA。此时可以沿 KV 长度切成多份，每份独立计算局部 m、ℓ、U，再合并状态。
>
> [Flash-Decoding 的原始技术说明](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)介绍了通过 KV 分段增加 decode 并行度的思路。典型方案包含局部计算和额外归并，因此不能统一说成“所有 Attention 都在单个 kernel 内完成”。

```math
m=\max_r m_r,\qquad
\ell=\sum_r e^{m_r-m}\ell_r,\qquad
U=\sum_r e^{m_r-m}U_r,\qquad O=\frac{U}{\ell}
```

> 1. **这里合并的是不同 KV 分段。** 它们使用同一个 Query，覆盖不同的 Key/Value 位置。
> 2. **如果保存的是归一化局部 O_r，要先恢复 U_r。** 使用 `U_r = ell_r × O_r` 后再套合并式，不能直接平均 O_r。
> 3. **空分段贡献为零。** 若某段没有可见 Key，令 ℓ_r、U_r 为零并跳过该段的指数修正；整行没有可见 Key 时需明确处理策略。
> 4. **增加并行也会增加成本。** 局部状态的写入、归并 kernel 的启动与归约可能抵消收益。长历史、小 batch 常值得考虑，短历史或原本任务充足时未必有利。
> 5. **KV 分段不减少 KV Cache 的容量。** 它改变一次 Attention 的任务划分，不是压缩历史信息。

## 7. 三个版本怎样放在一起记？

| 版本或设计 | 主要解决的问题 | 关键变化 | 容易误解的地方 |
| --- | --- | --- | --- |
| v1 | 完整分数与概率的显存 IO | Tiling + 在线归一化 | 不能说完全没有显存读写 |
| v2 | 非矩阵开销、任务不足、warp 间通信 | 延迟归一化、Q 块并行、Sliced-Q | 不只是调换两层循环 |
| v3 | Hopper 异步能力与执行资源利用 | TMA/WGMMA、GEMM/Softmax 重叠、FP8 路径 | 不是添加一个 TMA 调用就完成升级 |
| Flash-Decoding | 少 Query、长 KV 时并行不足 | KV 分段、局部 Attention 与合并 | 不保证只启动一个 kernel |

## 8. 面试问答

### 为什么 FlashAttention 快？

> 它将标准 Attention 分块，通过在线维护最大值、指数和与加权输出分子，使局部 score/probability 不必完整写入显存。它主要改变数据流，再结合线程分工和硬件执行方式提高效率；不是把 dense prefill 的平方计算量变成线性。

### v2 为什么最后才归一化？

> 保存未归一化分子 U 与分母 ℓ，可以避免每处理一个 KV 块就把整个输出逐元素除一次。最大值变化时仍需修正旧状态，最后统一计算 `O = U / ℓ`。

### v3 为什么不直接让每个 warp 各管一个阶段？

> 因为矩阵指令的参与单位和数据依赖都有要求。Hopper WGMMA 使用 warpgroup；数据搬运、score 就绪、概率可见和输入阶段复用也需要完成通知。角色分工必须满足这些条件，再考虑跨块、跨工作组的重叠。

### 同一套方案在 prefill 和 decode 上会一样快吗？

> 不会。Prefill 的多 Query 提供行块并行，且完整中间矩阵很大；单 token decode 主要扫一行历史，可能受 KV 读取和任务数限制。应根据 Query 数、KV 长度、batch、head、dtype 与目标 GPU，选择并测量具体实现。
