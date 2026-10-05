# 算子融合：Decode 中怎样减少访存与启动开销

> 面对“做过算子融合相关工作吗”这类问题，需要讲清楚四件事：**为什么融合、具体融合了什么、怎样实现，以及如何验证收益。** 这篇以普通自回归 decode 为主线，先解释原理，再给出按真实经历组织回答的方法。

> 单 token decode 指每个请求本步只有一个当前 token；多请求可以组成批次 B。Attention 读取历史 KV，投影与 dense FFN 主要处理 `[B, d]` 的当前表示。小批次下，启动开销、权重读取、归约效率和历史 KV 访问都可能影响性能，不能只按算子名字判断瓶颈。

## 1. 核心动机：融合省掉的是什么

### 1.1 减少中间张量的读写

> 分离执行时，前一个 kernel 往全局内存写出中间结果，后一个 kernel 再把它读回来。融合让消费者在合适的执行范围内直接使用生产者的结果，可以减少中间张量物化与读取。

> 常见的数据传递位置有：
>
> 1. **寄存器。** 同一线程持有的结果可以直接用于后续逐元素计算，不需要先写入全局内存。
> 2. **Shared Memory。** 同一 CTA 内的线程需要交换或共同使用数据时，可以显式存入共享内存，再通过适当同步使用。
> 3. **分块重算或多阶段处理。** 数据依赖不能在一个线程或 CTA 内完成时，需要重新安排分块与归约；并非所有中间结果都能永久留在片上。

> Shared Memory 是显式管理的片上存储，不等于自动缓存；寄存器也不是无限容量。减少逻辑全局内存读写不一定等量减少 HBM 流量，因为原来的中间数据可能命中 L2，寄存器溢出也可能产生额外访存。

### 1.2 用 BiasAdd + Activation 算一笔账

> 假设 `y = activation(x + bias)`，有 N 个元素，每个元素占 s 字节。只统计 x、中间张量和 y，暂不计 bias 读取与其他开销：
>
> 1. **分离执行。** BiasAdd 读 x、写中间张量，共 2Ns；Activation 再读中间张量、写 y，共 2Ns，总计 4Ns。
> 2. **融合执行。** 每个元素读 x，在寄存器中加 bias、计算激活，再写 y，总计 2Ns。
> 3. **省掉的内容。** 中间张量的一次写入与一次读取，共 2Ns；同时从两个 kernel 变成一个。

```math
\mathrm{Bytes}_{\mathrm{separate}}=4Ns,\qquad
\mathrm{Bytes}_{\mathrm{fused}}=2Ns,\qquad
\Delta\mathrm{Bytes}=2Ns.
```

> 例如单请求 `[1, 4096]` 的 FP16 向量，Ns＝8 KiB，分离版本的这部分逻辑读写量是 32 KiB，融合版本是 16 KiB。**逻辑读写减半，不代表实际延迟一定减半。** bias、缓存、启动、指令与调度都参与总时间。

### 1.3 减少 Kernel Launch 开销

> 小 shape 的计算可能很短，主机提交与 GPU 调度开销在总时间中占比明显。融合减少独立启动次数，也可以减少框架调度与临时张量管理。启动开销没有对所有机器、框架和测量方式都成立的固定微秒数，应在实际负载下测量。

> 还要区分两个不同手段：**Fusion 改变计算组合与数据传递；CUDA Graph 改变重复工作的提交方式。** Graph replay 可以降低多 kernel 的主机提交开销，但本身不会把这些 kernel 融合，也不自动消除它们之间的中间张量。两者可以同时使用。[^cuda-graphs]

## 2. 常见融合场景：从轻量操作到 Attention

> 可以先区分两类组织方式：**纵向融合**连接生产者与消费者，例如 Bias → Activation；**横向合并**把共享输入的并列计算组合起来，例如 Q/K/V 投影。它们减少开销的机制不同，不能只用“多个算子变成一个”概括。

### 2.1 Element-wise 与归约融合

> 常见组合包括：
>
> 1. **Bias + ReLU/GELU/SiLU。** 当前投影输出先加偏置再激活，两步逐元素计算可以连起来做。具体模型没有 bias 时，不应凭模板添加 BiasAdd。
> 2. **SwiGLU 的 SiLU + 逐元素乘法。** 两路投影结果已经准备好后，激活门控分支并与另一分支相乘。SwiGLU 不是对一路张量简单应用一个激活函数，还涉及另一条投影路径。
> 3. **Residual Add + LayerNorm/RMSNorm。** 先计算残差相加的向量，再对同一行做归一化，可以合并输入读取与中间结果处理；它包含归约，比纯逐元素组合更复杂。

> 逐元素组合的数据依赖通常比较直接，编译器常能处理；归约、不同布局、多消费者与跨 CTA 通信会限制融合。不能保证所有表达式在 JIT 后自动变成一个 kernel。

### 2.2 Residual + RMSNorm：中间结果是否还要保留

> 以 `u = x + residual`、`y = RMSNorm(u)` 为例。每行 N 个元素、每元素 s 字节，仅统计 x、residual、u、y，忽略 γ、统计量与额外重读：
>
> 1. **分离版本：5Ns。** Add 读两个输入、写 u，计 3Ns；Norm 读 u、写 y，计 2Ns。
> 2. **融合且只需要 y：3Ns。** 读 x 和 residual 后，在片上完成相加与统计，直接写 y，可以省掉 u 的全局写回与读取。
> 3. **融合且还需要输出 u：4Ns。** 读两个输入，写 u 和 y。虽然 u 仍需写回，但 Norm 可以直接复用本 kernel 中算出的 u，不必再从全局内存读取它。

> 在 Pre-Norm Transformer 中，相加后的 u 往往还要走残差支路。**融合必须保留实际需要的输出，不能为了宣称少一次写回，把模型仍需使用的 u 丢掉。** 对 `[1, 4096]` FP16 行，上述三个理想计数分别为 40、24、32 KiB。实际 kernel 是否重读输入、怎样存储 u，需要结合实现分析。

### 2.3 GEMM + Bias + Activation：Epilogue 融合

> Epilogue 是 GEMM 累积之后、最终结果写出过程中的后处理阶段。它可以对已经算好的输出元素进行缩放、加 bias、激活或残差相加，避免先写出裸 GEMM 结果，再启动独立逐元素 kernel 读取它。[^cutlass]

> 可以按三步解释：
>
> 1. **Mainloop 完成矩阵乘加。** 每个输出 tile 的累加器形成 GEMM 结果。
> 2. **Epilogue 使用累加结果。** 结合所需 bias、残差等数据，执行支持的后处理。
> 3. **写出需要的最终结果。** 避免可省略的中间结果物化；具体 epilogue 可能涉及寄存器、Shared Memory 和布局变换，不一定全程只在寄存器里操作。

> TensorRT-LLM 等推理框架可能通过后端提供相关融合路径，具体是否采用某种 epilogue 由模型、配置与实际执行实现决定。CUTLASS 提供 epilogue 组合与定制能力，包括特定架构、调度支持下的 Epilogue Visitor Tree（EVT）。支持范围随接口与硬件变化，不能把某个 Hopper 示例的能力直接视为所有 GPU 都支持。[^cutlass-evt]

### 2.4 QKV Projection：合并并列投影

> Q、K、V 都读取同一当前输入，可以沿权重的输出列拼接，使用一次更宽的矩阵乘法：

```math
[Q,K,V]=X[W_Q,W_K,W_V].
```

> 本步 X 为 `[B, d]`。标准 MHA 的合并权重可为 `[d, 3d]`，输出为 `[B, 3d]`，再按头布局使用；GQA 的 K/V 宽度较小，合并宽度不必等于 3d。

> 主要收益与边界是：
>
> 1. **减少启动与重复输入读取。** 三个并列投影可能合成一个调用，也改变了矩阵形状与调度空间。
> 2. **不减少投影的主要乘加量与权重数量。** 相同精度与结构下，算术工作量基本不变，三个权重块仍然需要使用。
> 3. **不保证 Tensor Core 吞吐一定提升。** decode 的 B 可能很小，瓶颈可能主要是权重读取；合并后的布局、量化方式与并行切分也会影响收益。
> 4. **避免在热路径临时拼接权重。** 通常在加载或准备阶段安排 packed 权重；若每步重新分配、复制权重，可能抵消收益。

### 2.5 Attention 融合：QK → Softmax → PV

> 未融合实现可能把分数和概率作为中间张量写出，再分别运行 Softmax 与 PV。FlashAttention 将计算按 tile 重组，维护在线 Softmax 状态与输出累积，避免在显存中物化完整的分数、概率矩阵。它的关键是 IO-aware 分块与计算安排，不只是把几个函数放进同一个源文件。[^flashattention]

> 复杂度必须区分场景：
>
> 1. **Prefill 的稠密全局 Attention。** S 个 Query 读取 S 个 Key，直接物化的分数矩阵有 S² 个元素。FlashAttention 避免存储这个完整中间矩阵，但主要稠密 Attention 计算仍按 S² 增长；它不是 Linear Attention。
> 2. **单 token decode。** 每头 Query 长度为 1，逻辑分数行是 `[1, T]`。这里本来就不是 T×T 分数矩阵，不能用“把二次空间降成线性”解释本步的收益。
> 3. **历史 KV Cache。** 融合 Attention 不会删除所有历史 K/V；标准全局 Attention 的缓存仍随上下文长度增长。融合减少中间物化，不等于不再需要缓存。

> Decode 还要解决并行度不足。Flash-Decoding 的经典方案沿历史 KV 长度切分：
>
> 1. **分片 Attention。** 多个片段并行计算局部输出，同时保存用于正确归并的 log-sum-exp 统计量。
> 2. **归并片段。** 再执行归约，根据统计量重标定各片段的贡献，得到整段 Attention 输出。
> 3. **理解执行边界。** 原始公开方案包含分片计算和归并两个 kernel；融合局部 QK、Softmax、PV 与最终使用多个阶段并不矛盾。不能把 FlashAttention 与所有 decode 实现统一说成“整个过程必然只有一个 kernel”。[^flashdecoding]

> FMHA（Fused Multi-Head Attention）是融合多头注意力实现的一类称呼，具体支持的 shape、dtype、Mask、硬件及调度取决于对应 backend，不是一个固定 shape、固定 kernel 数量的统一算法。

## 3. 实现手段：编译器融合与手写 Kernel

### 3.1 自动化与编译器层面

| 工具 / 机制 | 在融合中承担什么角色 | 需要分清什么 |
| --- | --- | --- |
| TorchDynamo | 捕获 Python 程序中的可编译图 | 捕获图不代表整张图最终成为一个 kernel |
| TorchInductor | 优化并生成代码，GPU 路径可生成 Triton kernel，也可调用外部计算库 | 生成与融合策略依赖图、布局、后端和版本 |
| CUDA Graph | 捕获、重放已有 GPU 工作，降低重复提交开销 | Graph replay 本身不做算子融合 |
| TVM / XLA | 编译图或中间表示，按依赖、模式与代价安排融合 | 融合规则不是只有 Pattern Matching，也不能保证所有模式都支持 |

> PyTorch 的图捕获、代码生成与 CUDA Graph 是不同层次，可以组合使用。不能描述成“Dynamo 利用 CUDA Graph 自动把全部操作融合成 Triton/CUDA 代码”。具体采用了什么，需要查看编译结果与实际启动的 kernel。[^compiler]

### 3.2 手写与特化 Kernel

> 可以按所需控制能力选择工具：
>
> 1. **Triton。** 用块级张量程序表达加载、归约、广播与逐元素计算，适合实现 Residual + RMSNorm 等组合。源代码写一个块，并不等于用户手动指定了所有数据都存放在 Shared Memory；实际布局与资源分配要看编译结果。
> 2. **CUDA C++。** 显式组织线程、warp、Shared Memory 与同步，适合更细的映射、位操作及特殊布局。资源控制更直接，也需要承担正确性和性能调优工作。
> 3. **CUTLASS。** 利用 GEMM mainloop 与 epilogue 体系，在合适接口中组合后处理；EVT 等机制有具体架构与调度约束。
> 4. **已有 backend 集成。** 集成 FlashAttention 或框架的 fused norm，属于选择、接入与验证已有实现；应与亲自编写 kernel 区分描述。

### 3.3 为什么不能把所有算子都融合

> 融合需要同时满足正确性与性能条件：
>
> 1. **数据依赖能在合理执行范围内完成。** 相邻算子的线程映射、归约范围和布局可能不同，跨 CTA 数据交换通常不能直接当作 CTA 内同步处理。
> 2. **片上资源容纳得下。** 更长的数据生命周期可能增加寄存器和 Shared Memory 使用，降低 occupancy，甚至产生 spill。
> 3. **没有破坏原有并行与复用。** 强行融合可能串行化工作，或为了让每个消费者局部计算而重复执行生产者。
> 4. **保留所有语义与输出。** 多消费者、中间输出、别名写入、Mask 和精度转换顺序都要正确处理。
> 5. **增加的指令与同步值得付出。** 减少访存和启动，可能换来更长的临界路径；最终要用实际负载验证。

## 4. 工程案例怎样记录与验证

> 一个可核验的案例，应包含以下六项，而不只是“我用了 Triton，延迟下降了 X%”：
>
> 1. **负载。** GPU、dtype、输入 shape、batch；Attention 还需上下文长度、Q/KV 头数等。明确是否为单 token decode。
> 2. **基线。** 哪个版本、哪些操作分开执行、实际 kernel 数量；若已有编译器融合或 CUDA Graph，要记录是否启用。
> 3. **融合边界。** 哪些中间张量被消除，哪些仍然需要输出，减少了哪些启动、逻辑读写或实际访存。
> 4. **正确性。** 与相同语义参考比较，说明 dtype、误差阈值、Mask、边界形状与精度转换顺序。
> 5. **性能测量。** 排除编译与首次初始化，预热后重复测量；使用合适的 GPU 计时与同步，明确单算子、融合子图、单层或整模型计时范围。
> 6. **定位证据与端到端结果。** 用 Nsight Systems 看启动与时间线，用 Nsight Compute 看实际访存、资源和瓶颈；再测完整 decode step 或 TPOT，避免把局部收益直接当作整模型收益。

> 若被优化区域原先占总时间比例 p，区域自身加速 r 倍，假设其余时间不变且按串行时间相加，整体加速上限模型为：

```math
\mathrm{Speedup}=\frac{1}{(1-p)+p/r}.
```

> 例如 p＝0.1、r＝2，整体约加速 1.053 倍，总时间降低约 5%，不是降低 50%。存在计算重叠与临界路径变化时，不能直接套用这个串行估计，需要看实际时间线。

## 5. 面试回答：按真实经历组织

### 已经亲自实现并验证过

> “做过，我主要针对单 token decode 中的【真实算子组合】做过融合。原先【基线实现】要启动【实测数量】个 kernel，并读写【具体中间张量】。我用【真实工具】把【具体步骤】放进同一个执行范围，保留了【仍需输出的数据】，减少了【经验证的读写或启动】。
>
> 正确性与【参考实现、精度、阈值】比较；性能在【GPU、shape、dtype】下测量。融合子图延迟从【实测值】变成【实测值】，完整 decode 的【指标】变化是【实测结果】。在【边界负载】下收益较小或出现回退，主要原因是【证据支持的瓶颈】。”

> 方括号必须用真实记录填写。没有端到端测量就明确只报告局部结果，不能把模板中的数字或别人的 benchmark 当成自己的经历。

### 集成过已有融合实现

> “我主要做过已有融合 backend 的接入与验证，暂时没有独立实现完整 Attention kernel。我验证了它在【实际负载】下的正确性、支持范围和性能，并分析它减少了哪些中间物化或提交开销。”

### 目前只学习过原理

> “目前还没有完成可报告的融合项目。我理解它主要减少中间结果物化和独立启动，但也会受到寄存器、Shared Memory、布局与同步约束。若开展实践，我会先选择 Residual + RMSNorm 或 SiLU + Multiply，明确输出语义、建立基线并验证局部与端到端收益。”

## 面试追问

**融合能把 memory-bound 算子变成 compute-bound 吗？**

> 有可能，但不能保证。融合可能减少字节数、提高算术强度，也可能让启动、归约、资源或依赖延迟成为新瓶颈。需要结合实际硬件与负载验证，不能只看表达式数量。

**Residual + RMSNorm 融合一定能省掉中间向量的写回吗？**

> 不一定。若残差支路还需要相加后的向量，它仍需输出；融合可能主要省掉 Norm 对该向量的重新读取。先列清输出与消费者，再计算收益。

**Flash-Decoding 为什么有归并 kernel，仍可视为融合优化？**

> 分片内的 QK、在线 Softmax、PV 已被组合，减少局部中间物化；跨分片还需正确合并结果。融合不要求整个问题只有一次 kernel 启动。

**启用 CUDA Graph 后，还值得做 fusion 吗？**

> 如果融合还能减少中间张量、访存或 GPU 调度成本，仍可能有收益；若主要收益原先来自主机提交，Graph replay 后收益可能缩小。比较时应在一致的 Graph 配置下测量。

## 参考与关联专题

[^flashattention]: [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)：IO-aware 分块与精确稠密 Attention 的中间存储。
[^flashdecoding]: [PyTorch：Flash-Decoding](https://pytorch.org/blog/flash-decoding/)：沿 KV 长度分片、log-sum-exp 归并与公开的两个 kernel 实现。
[^cuda-graphs]: [PyTorch：Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)：捕获、重放与提交开销。
[^compiler]: [PyTorch 编译器文档](https://docs.pytorch.org/docs/stable/torch.compiler.html)：编译工具链与后端职责；实际行为以使用版本为准。
[^cutlass]: [CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)：mainloop 与 epilogue 分工。
[^cutlass-evt]: [CUTLASS 官方 Example 49](https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu)：Hopper collective builder 与 EVT 示例，包含对应约束。

[返回融合专题](README.md) · [Attention 与 FFN 的 decode 数据流](../../01-model/transformer/attention-and-ffn.md) · [LayerNorm 与 RMSNorm](../../01-model/normalization/layernorm-rmsnorm.md) · [高性能 Backend](../../04-backends/README.md) · [Kernel 编程](../../05-kernel-engineering/README.md)
