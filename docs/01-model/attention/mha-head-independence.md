# MHA 各个注意力头是否完全独立？

> 状态：原理整理与计算图推导。适用于标准 MHA；不包含额外跨头混合的变体。GPU 调度描述为实现原则，未进行性能实测。

## 问题与核心结论

“每个头独立计算”是理解 MHA 的好入口，但“数据、梯度、资源完全隔离，互不影响”范围过大。

**给定本层各头的 Q、K、V，标准 MHA 的分数计算、逐行 Softmax 和加权 V 不读取其他头的中间结果；随后输出投影可以线性混合各头结果。** 输入、损失、参数更新和硬件资源仍有关联。

这里的“独立”是计算图局部依赖的判断，不是统计独立、不同模型独立训练或独享 GPU 资源。

## 前置知识与形状

考虑自注意力，暂时省略 Batch 维度。定义序列长度 N、输入隐藏维度 D、头数 H、每头 Q/K 维度 d_k、每头 V 维度 d_v。常见实现令 d_k = d_v = d_h，但并非数学要求。

```math
\begin{aligned}
X&\in\mathbb{R}^{N\times D},\\
W_Q,W_K&\in\mathbb{R}^{D\times Hd_k},\qquad
W_V\in\mathbb{R}^{D\times Hd_v},\\
Q_{\mathrm{all}}&=XW_Q,\quad
K_{\mathrm{all}}=XW_K,\quad
V_{\mathrm{all}}=XW_V.
\end{aligned}
```

沿投影后的特征维拆头，而不是把输入 Token 或输入 X 的特征事先切成各头专属部分：每个头的投影一般都能读取 X 的全部 D 个特征。三个投影也可以由一个融合 Linear 一次生成，再拆出 Q/K/V。

```math
\begin{aligned}
Q_i,K_i&\in\mathbb{R}^{N\times d_k},\qquad
V_i\in\mathbb{R}^{N\times d_v},\\
P_i&=\operatorname{softmax}_{\mathrm{row}}
\left(\frac{Q_iK_i^T}{\sqrt{d_k}}+M\right),\\
H_i&=P_iV_i\in\mathbb{R}^{N\times d_v}.
\end{aligned}
```

M 是加性 Mask：允许的位置为 0，禁止的位置为负无穷；这里假设每个 Query 至少有一个可见 Key。Softmax 沿 Key/Token 轴归一化，不沿头轴归一化。标准多头定义见 [Attention Is All You Need §3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2)。

## 哪一段没有跨头依赖？

固定各头输入时，改变头 1 的 Q/K/V，不会改变头 2 的 H_2。头 2 的计算没有使用 H_1 或 P_1，因此可以并行计算。

但若改变的是共同输入 X，多个头的投影都可能改变。上一层已经融合的隐藏状态还会成为这一层共同输入，所以“一个头永远只接触自己那条信息”不成立。

一个具体例子：N = 3，D = 8，H = 2，d_k = d_v = 4。两个头分别产生 3 × 3 的分数矩阵、3 × 4 的输出。头 1 的 Softmax 不需要头 2 的分数；两个头却都由同一个 3 × 8 的 X 投影得到。

## Concat 与输出投影分别做什么？

```math
\begin{aligned}
H_{\mathrm{cat}}&=[H_1\ \cdots\ H_H]
\in\mathbb{R}^{N\times Hd_v},\\
Y&=H_{\mathrm{cat}}W_O,\qquad
W_O\in\mathbb{R}^{Hd_v\times D}.
\end{aligned}
```

Concat 在数学上只是把特征排列在一起，不做加权融合。工程上也不一定是显式内存拷贝：可能已写入合适布局，或通过 reshape/view 表达；布局不合适时可能需要 transpose、contiguous 或复制。

把 W_O 按输入行分为 H 个块，每块大小 d_v × D，可以直接得到：

```math
Y=\sum_{i=1}^{H}H_iW_O^{(i)}.
```

这说明每个输出维度都可以接收多个头的贡献。W_O 不是“给每个头用同一套权重”：不同头对应不同的行块，它们共同构成一个输出投影。

在这个标准 MHA 子层内，W_O 是显式跨头特征混合位置；放到整个 Transformer 中，后续残差、归一化、MLP 和下一层也会继续处理这些特征，因此不能宣称整个模型只有此处产生联系。

## 为什么不能说梯度完全独立？

令 G = ∂L/∂Y，其中 L 是最终共享损失。通过上面的分块形式求导：

```math
\frac{\partial L}{\partial H_i}
=G\left(W_O^{(i)}\right)^T.
```

每个头内部的反向计算可以分别进行，但 G 来自共同输出和后续网络。其他头改变输出后，G 也可能改变，因此某个头的参数梯度会间接受到其他头影响。所有头经过输入投影产生的梯度还需要汇总到共同输入 X。

对通常未绑定参数的 MHA，各头投影对应不同参数切片；这表示没有强制共享同一切片，不表示共享损失下的学习过程互不影响。这里仅用反向边界说明“独立”的含义，训练不是本仓库主线。

## 推理中的 KV Cache 与 GPU 并行

MHA 每个头有自己的逻辑 K/V 内容，缓存可以统一存为 `[B, H, T, d_h]`，也可以使用其他布局或分页存储。各头逻辑切片不同，不要求分别申请一块物理显存。

新增 Token 时，各头可写入互不重叠的缓存区域。但读取必须满足数据已生成、写入已完成等依赖；框架的分页管理、元数据和存储池也可能共享。

头维可作为 Kernel 的并行维度。**并行不要求“一个头一个 CUDA Stream”或“一个头一个 Kernel”。** 一个 Kernel 可以同时覆盖 Batch、头和 Query Tile，具体工作划分取决于实现。拆成很多小 Kernel 反而可能增加 launch 开销，减少数据复用。

多头之间没有相互读取分数矩阵，仍会竞争 SM、带宽、Cache 和调度资源；并行潜力不等于无限并行吞吐。

## 与 GQA 的边界对比

MHA 中 Q 头与 KV 头一一对应。GQA 中多个 Q 头读取同一个 KV 头，却仍各自形成分数、Softmax 和输出。**共享只读输入不会自动产生头间串行依赖。**

共享 K/V 会约束表示自由度；在反向传播中，多个 Q 头对共享 K/V 的梯度需要相加。这与“推理时必须等待另一个 Q 头先算完”是两件事。

因此不能直接说 MHA 的并行度上限一定更高。GQA 保留 Q 头数并减少 KV Cache，可能获得更好的访存复用；实际速度取决于 Shape、Batch 和 Kernel。参见 [手写 GQA](gqa-from-scratch.md)。

## 面试与自测

1. **MHA 各头独立吗？** 给定各头 Q/K/V，Attention 子计算没有跨头中间结果依赖；整体模型共享输入、输出和损失。
2. **每个头只看输入特征的一部分吗？** 一般不是；各头投影通常读取完整隐藏向量，再映射到不同子空间。
3. **Concat 是否融合信息、是否一定拷贝？** 数学上只排列特征；是否拷贝由布局和实现决定。W_O 执行线性融合。
4. **GQA 共享 K/V 是否影响头并行？** 不引入 Q 头之间的先后计算依赖，但会改变数据复用、资源需求和梯度汇总。
5. **为什么一个头一个 Kernel 未必快？** 数学可分解不代表这种调度最优，需要考虑 launch、粒度、复用与占用率。

[Attention 专题导航](README.md) · [Linear Attention](linear-attention.md)
