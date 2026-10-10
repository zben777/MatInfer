# 从 Ampere 到 Blackwell：矩阵计算与数据搬运怎样重叠

> 多阶段流水线不是简单地“把同步指令换成异步指令”。真正需要管理的是：**下一块数据什么时候搬进来，当前计算什么时候能使用它，以及计算结束后什么时候允许覆盖这个缓冲区。**

> 本文比较 A100（Ampere）、H100（Hopper）与 B200（Blackwell）的代表性矩阵计算路径，连接 `cp.async`、TMA、MMA、barrier 和 stage 复用。代码是解释依赖关系的教学伪代码，不是可直接编译的 CUDA / PTX 实现；不同指令形状、类型和目标架构需要查对应文档。

## 1. 从 decode 的投影开始：为什么需要流水线

> 单 token decode 中，每个请求只有一行当前表示，多个请求可组成 `[B, d]`。QKV 投影和 FFN 投影可以表示为矩阵乘法；一个 kernel 内部会按 tile 组织工作，并沿 K 维逐块累加：

```math
C=AB,\qquad C_{\mathrm{tile}}=\sum_j A_jB_j.
```

> 对一个输出 tile，可以把工作拆成三部分：
>
> 1. **搬运。** 从全局内存读取 A、B 的当前 K 块，准备适合 Tensor Core 使用的数据。
> 2. **计算。** 用当前块更新输出 tile 的累加结果。
> 3. **推进。** 继续下一 K 块，最后执行 epilogue 并写出输出。

> 不做流水线时，可能是“加载块 0 → 计算块 0 → 加载块 1 → 计算块 1”。多阶段流水线希望提前搬运后续块，让**计算块 j 与加载块 j＋1 在时间上重叠**。它通常不改变矩阵乘法的数学结果，而是改变工作安排。[^cutlass]

> 不过，decode 的 B 很小时，矩阵可能很窄，kernel 的并行度、权重读取与 tile 利用率会限制收益。这里学习的是流水线机制，不意味着把较大 GEMM 的配置搬到 B＝1 就一定高效。

## 2. 先把“同步”和“异步”拆开

### 2.1 四个不同的问题

> 看到指令名称里的 sync / async，先问：
>
> 1. **参与者是否必须一起执行？** 一个 warp、warpgroup 或 CTA 内哪些线程需要协作，条件分支是否一致？
> 2. **发射后操作是否已完成？** 指令可以启动后台操作，也可能具有特定的完成与依赖语义。
> 3. **结果什么时候可以读取？** 数据或累加器仍在被更新时，普通线程不能提前使用它。
> 4. **输入什么时候可以覆盖？** 即使指令已发射，后台引擎仍可能在读取输入，缓冲区未必可复用。

> 参与线程的同步要求与计算完成的异步机制可以同时存在。例如 WGMMA 的指令名包含 `mma_async.sync.aligned`：async 描述矩阵操作的异步完成机制，sync / aligned 描述协作执行要求，二者并不矛盾。[^ptx]

### 2.2 `mma.sync` 该怎样理解

> Ampere 上常见的 `mma.sync` 是 warp 级矩阵乘加，操作数片段和累加结果通常通过寄存器提供。它不采用 Hopper WGMMA 那种显式的异步 group 提交、等待模型。[^ampere]

> 但不能从 `.sync` 推出“发射后整个 warp 在原地等 Tensor Core 结果，期间一条其他指令也不能发射”。该限定符涉及同一 warp 的协作执行；结果依赖、指令延迟、scoreboard 与调度是另外的层次。依赖累加器的后续操作要满足结果就绪条件，独立工作和其他 warp 则可以提供延迟隐藏。[^ptx]

> 所以准确的对比是：**Ampere 常见路径使用寄存器片段和普通 MMA 依赖；Hopper 增加了显式异步 WGMMA group 的管理。** 不能进一步推导为“Ampere 完全无法把搬运和计算重叠”。

## 3. Ampere：异步拷贝配合寄存器 MMA

### 3.1 每个组件做什么

> 以常见 FP16/BF16 tile 路径为例：
>
> 1. **`cp.async`：全局内存 → Shared Memory。** 可以发起异步拷贝，并在合适位置等待完成，避免先把这份拷贝数据读进通用寄存器再存入 Shared Memory。
> 2. **`ldmatrix` 等加载：Shared Memory → 寄存器片段。** 将计算输入组织成 warp MMA 所需的片段；具体加载方式由布局、类型与指令决定，不是所有 MMA 路径都必须用同一个 `ldmatrix` 形式。
> 3. **`mma.sync`：执行矩阵乘加。** 使用片段更新寄存器中的累加器。
> 4. **同步：发布数据与保护复用。** 拷贝完成等待和线程协作同步解决不同问题；若其他线程负责加载的数据会被本线程使用，还需要正确的跨线程可见性与顺序。[^ampere]

> `__syncthreads()` 是 CTA 范围的同步，不是整个 GPU 的“全局同步”；CTA 也不是固定 1024 个线程。是否需要每次迭代都调用它，取决于具体数据分工与同步设计。

### 3.2 一个保守的双缓冲教学流程

> 以下每个操作名是逻辑步骤，不是 CUDA API。假设 CTA 共同加载和消费 A/B tile，两个 Shared Memory slot 已正确分配，尾块由加载逻辑处理：

```text
if num_tiles == 0:
    return

所有加载参与线程：发起 tile 0 → slot 0 的 cp.async，并提交

for tile in [0, num_tiles):
    slot = tile % 2

    各加载线程：等待自身已提交的相关 cp.async 完成
    CTA 同步：当前 tile 数据准备好，并对消费者可见

    if tile + 1 < num_tiles:
        发起下一 tile → 另一个 slot 的 cp.async，并提交

    从当前 slot 加载 A/B 寄存器片段
    用 mma.sync 更新累加器

    CTA 同步：本轮所有 Shared Memory 消费者结束
    # 另一个 slot 的异步加载可与本轮计算重叠
```

> 当前 slot 与下一 slot 分开，下一块拷贝发射后不必立即等待，因此有机会与当前计算重叠。最后一次迭代不再加载越界的下一 tile；尾部还要按实现需要完成剩余等待与输出。

> `cp.async.wait_group N` 中 N 关联异步拷贝 group 的等待规则，不是“等待编号为 N 的 Shared Memory slot”。拷贝 group 与 tile / stage 是否一一对应，是 kernel 自己的组织选择，不能仅凭 `wait_group<1>` 就认定读取缓冲区已经就绪。

### 3.3 代价与边界

> 这一代流水线要权衡：
>
> 1. **输入片段寄存器。** A/B 片段与累加器都占资源，更多 stage 或更大 tile 可能增加压力。
> 2. **Shared Memory 加载与布局。** `ldmatrix` 对应片上数据读取，不应称为额外 HBM 读取；其布局和带宽会影响效率。
> 3. **同步与发射。** 同步可能产生等待，拷贝和地址计算也需要线程参与，但它们并非“致命瓶颈”的固定组合。
> 4. **已有隐藏能力。** 异步加载、寄存器预取、独立指令与多个驻留 warp 都可以帮助隐藏延迟；需要根据负载衡量限制。[^cutlass]

## 4. Hopper：TMA、mbarrier 与异步 WGMMA

### 4.1 搬运和计算分别使用什么机制

> 代表性路径分成三个部分：
>
> 1. **TMA 搬运。** 利用 tensor map 描述布局，由选定线程发起张量搬运，硬件完成主要传输工作，减少大量线程逐元素参与地址计算与拷贝的需求。发射和描述符准备仍有成本，不能叫“零开销”。
> 2. **`mbarrier` 跟踪就绪。** 把参与者到达、预期事务字节数与传输完成关联到 barrier phase，消费者等待所需阶段满足条件。
> 3. **WGMMA 异步计算。** 用 warpgroup 协作启动矩阵乘加，按照 group 的完成规则决定什么时候可以读取累加器、复用输入。[^hopper]

> WGMMA 常见形式可通过 Shared Memory 描述符读取输入，部分形式的 A 输入也可来自寄存器；累加器仍占寄存器。省掉某些输入的显式搬到寄存器步骤，不等于“零寄存器开销”，也不等于 Shared Memory 不再有读取成本。

### 4.2 WGMMA 的参与者与等待

> 需要明确四件事：
>
> 1. **Warpgroup 是对齐的四个连续 warp。** 起始 warp rank 必须是 4 的倍数；warp 0～3、4～7 是这样的组，warp 1～4 不是。
> 2. **发射不等于完成。** 指令受参与者一致性、资源与依赖约束，不是无条件“立即返回”。
> 3. **Commit 管理操作组。** `wgmma.commit_group` 把之前未提交的异步 MMA 纳入一组，不保证这组已完成。
> 4. **Wait 控制完成边界。** `wgmma.wait_group 0` 等待所有此前已提交组完成；允许保留 N 个最新 pending group 时，需要知道哪些较早组及输入阶段已安全退休。[^ptx]

> 此外还要满足寄存器访问顺序与 Shared Memory 的代理内存顺序。`wgmma.fence` 与 async proxy fence 各有用途，不能把任意一个 barrier 当作全部 fence 的替代品。完整 PTX 形式、目标架构与合法布局以对应文档为准。

### 4.3 用 full / empty 管理 stage 生命周期

> 一个环形缓冲区 slot 可按以下逻辑状态推进：
>
> 1. **Empty：允许写。** 上一轮所有消费者都已经不再读取它。
> 2. **Loading：正在搬。** Producer 发起 A/B 所需数据传输，并正确建立事务完成跟踪。
> 3. **Full：允许读。** 当前 phase 的到达与事务完成条件已经满足。
> 4. **Reading：异步计算可能仍在读。** Consumer 发起 WGMMA，输入仍需保持有效。
> 5. **Released：可以复用。** 确认相关读者结束后，才通知 Producer 进入下一轮 Empty。

> **Full 不等于计算完成，Commit 不等于输入已经读完。** 原稿中在 commit 后立刻 `arrive(empty_bar)` 的顺序不安全：Producer 可能据此覆盖 WGMMA 尚在读取的输入。

### 4.4 一个保守的 Producer–Consumer 伪代码

> 假设一个合法对齐的 consumer warpgroup，S 个 Shared Memory slot。每个 slot 的 full / empty barrier 已初始化，首次 empty 状态允许 Producer 获取；事务字节数、到达数、内存顺序和 phase 管理由对应抽象正确实现。代码中的函数是生命周期说明，不是完整 API：

```text
Producer（选定的发射线程）:
    for tile in [0, num_tiles):
        slot = tile % S
        epoch = tile // S

        等待 empty(slot, epoch)：本轮 slot 可写
        设置 full(slot, epoch) 的 A/B 事务与到达条件
        发起 A 和 B 的 TMA 搬运
        # 传输完成机制参与推进对应 full phase

Consumer（一个合法对齐的 warpgroup）:
    for tile in [0, num_tiles):
        slot = tile % S
        epoch = tile // S

        等待 full(slot, epoch)：A/B 均已就绪
        满足所需寄存器 / Shared Memory fence 与内存顺序
        发射使用本 slot 的 WGMMA
        提交 WGMMA group
        等待该 group 完成（教学版采用 wait_group 0）

        完成消费者协作：所有读者均不再使用本 slot
        通知 empty(slot, epoch + 1)：允许下一轮覆盖

    确保剩余异步工作完成，再读取结果并输出
```

> 这个教学版本每轮先等待计算完成再释放 slot，便于证明正确性；Producer 仍可以在其他 slot 提前搬数据。性能实现可以保留多个未完成计算组，但必须建立 group、输入 slot 和完成事件之间的准确对应关系，不能只删除 wait。

> `epoch` 是抽象的复用轮次。实际 mbarrier 常用 phase / parity 区分相邻轮次；反复使用同一个 slot 时，Producer 和 Consumer 必须一致跟踪，避免把上一轮的信号当成本轮完成。若有多个 consumer warpgroup，释放条件必须覆盖所有读者。

### 4.5 Hopper 改善了什么，没保证什么

> 主要机会在于减少线程参与搬运、减少部分输入寄存器中转，并让 Producer 与 Consumer 更精细地协作；但：
>
> 1. **不保证没有 `__syncthreads()`。** 初始化、布局处理与其他阶段仍可能需要 CTA 同步。
> 2. **不保证没有 `ldmatrix`。** 不同输入形式与其他计算路径仍可能用到相应加载。
> 3. **不保证气泡消失。** 内存带宽、stage 不足、资源限制、分工不平衡与等待依赖都可能造成停顿。
> 4. **不保证两种引擎总能充分重叠。** 必须有足够独立工作与合理调度；硬件支持异步只是条件之一。

## 5. Blackwell B200：tcgen05 与 Tensor Memory

### 5.1 不把它描述成“第二代 WGMMA”

> 对 B200 的代表性 Tensor Core 路径，应单独理解 `tcgen05` 指令体系与 Tensor Memory（TMEM）。TMEM 是面向 Tensor Core 数据的专用片上存储，不是普通线程寄存器的另一种名字。矩阵结果放在哪里、怎样取出以及什么时候完成，都需要按新的机制组织。[^blackwell]

> 这部分重点分四项：
>
> 1. **矩阵计算。** `tcgen05.mma` 发起受支持类型、形状与 CTA 协作规则约束的矩阵操作。
> 2. **输入与累加存储。** 根据指令形式使用 Shared Memory 描述符或 TMEM 相关操作数，累加结果使用 TMEM，不能沿用所有 Hopper 累加器假设。
> 3. **完成跟踪。** 用 `tcgen05.commit` 关联异步操作的完成与 mbarrier，再遵循所需等待及 fence 规则读取结果、释放资源。
> 4. **协作范围。** 支持的路径可涉及单 CTA 或双 CTA 组织，但不等于任意 SM 的内存可以被任意矩阵指令直接读取。[^ptx]

### 5.2 Cluster / DSM 不是 Blackwell 首创

> Hopper 已引入 Thread Block Cluster 与 Distributed Shared Memory。它们允许同一 cluster 内 CTA 按规则协作访问共享内存，不是自由指定一个“邻居 SM 编号”就能访问任意 Shared Memory。[^hopper]

> Blackwell 可以进一步结合新的计算与协作能力，但需要区分：
>
> 1. **Cluster 协作**与**跨 GPU 通信**是不同层次。
> 2. **低精度矩阵指令的输入格式与缩放**，不等于 TMA 能任意执行 FP4 / FP6 → FP8 / FP16 转换。
> 3. **数据通信能力**，不等于存在一个通用的“将 MMA 累加器直接推到 NVLink”的伪 API。

> 原稿中的 `tma_load_dequant_push_async`、`blackwell_wgmma_async`、`blackwell_nvlink_push_async` 不作为真实接口保留。若后续讨论具体格式转换、数据传输或通信融合，应对照实际 PTX、库接口与可验证代码单独展开。

### 5.3 和本地 RTX 4090 的关系

> RTX 4090 属于 Ada、计算能力 8.9，可以学习适用于该目标的 warp MMA 与异步拷贝路径；不能在它上面直接运行 H100 的 WGMMA 或 B200 的 tcgen05 路径。架构名称相似或更新，不意味着具备同一套指令；不同 Blackwell 产品的能力也不能只按家族名推断。

## 6. 三代代表性路径对照

| 维度 | Ampere A100 | Hopper H100 | Blackwell B200 |
| --- | --- | --- | --- |
| 本文讨论的计算路径 | Warp `mma.sync` | Warpgroup `wgmma.mma_async` | `tcgen05.mma` |
| 输入组织 | 常见为寄存器片段 | 支持 Shared Memory 描述符，部分 A 形式可用寄存器 | 按具体指令使用 Shared Memory / TMEM 相关操作数 |
| 代表性累加位置 | 寄存器 | 寄存器 | TMEM |
| 搬运 | `cp.async` 等 | TMA 等 | TMA 与对应架构的搬运能力 |
| 完成与协作 | 拷贝等待、MMA 数据依赖、CTA / warp 同步 | full / empty、mbarrier、WGMMA group 与 fence | mbarrier、tcgen05 commit 与对应 fence / 资源管理 |
| 优化机会 | 提前搬运、片段预取、分块复用 | 搬运角色特化、异步计算、减少部分输入中转 | 新累加存储、计算与 CTA 协作组织 |

> 表中是代表性机制，不是每代硬件全部能力清单。比较的是表达与调度方式，而不是“旧架构全部阻塞，新架构全部零开销”。

## 7. Stage 数量越多越好吗

> 增加 stage 可以让搬运更早开始，但会占用更多资源。若一个 stage 保存 A 的 Mₜ×Kₜ 元素和 B 的 Kₜ×Nₜ 元素，每元素 s 字节，S 个 stage 的基本 Shared Memory 需求约为：

```math
\mathrm{SMEM}_{\mathrm{tiles}}
\approx S(M_tK_t+K_tN_t)s.
```

> 例如 Mₜ＝Nₜ＝64、Kₜ＝32、FP16，每 stage 为 `(64×32 + 32×64)×2 = 8192` 字节，即 8 KiB。两个 stage 是 16 KiB，三个是 24 KiB；还未计 padding、barrier、其他临时空间与 epilogue 缓冲区。

> 选择时看四点：
>
> 1. **提前量是否足够。** 计算一个 tile 的时间能否覆盖下一 tile 的搬运延迟？若长期搬运吞吐不足，多加缓冲也不能创造带宽。
> 2. **驻留能力是否下降。** Shared Memory 和寄存器增多，可能减少同时驻留 CTA / warp，影响延迟隐藏。
> 3. **是否有足够 tile。** K 块很少时，预填充与排空占比较大，深流水线未必划算。
> 4. **是否适合 decode shape。** 小 B 的窄矩阵、长上下文 Attention 与较大批次 GEMM，不应共用未经验证的 stage 最优配置。

> 核对性能时，需要同时看 kernel 延迟、访存、同步等待、Tensor Core 利用率、寄存器 / Shared Memory 使用与实际占用率。本文提供语义与资源估算，没有在 A100、H100、B200 上运行 benchmark。

## 面试与追问

**mma.sync 是同步的，为什么 Ampere 还能做多阶段流水线？**

> 因为搬运与计算是不同工作。下一 tile 的异步拷贝可以与当前 tile 的 MMA 重叠，寄存器预取和多个 warp 也能帮助隐藏延迟。没有 WGMMA 式显式异步计算组，不等于没有流水线。

**wgmma.commit_group 之后能立即释放输入 slot 吗？**

> 不能。Commit 是组管理，不是输入读取完成通知；必须证明使用该 slot 的相关异步操作已经结束读取。本文采用等待对应计算组完成后再释放的保守规则。

**wgmma.wait_group 1 是等待 stage 1 吗？**

> 不是。它允许最多一个最新提交的组仍 pending，并等待较早组完成。group 与 Shared Memory stage 的对应关系由 kernel 组织决定，不能把数字当成 slot 索引。

**把 Consumer 设置成 warp 1～4，有什么问题？**

> WGMMA warpgroup 的四个连续 warp 必须从 rank 为 4 的倍数的位置开始。1～4 跨越两个对齐组，不能直接作为合法 WGMMA warpgroup 使用。

**TMA 和 mbarrier 能保证流水线没有气泡吗？**

> 不能。它们提供搬运与完成跟踪能力，是否重叠取决于 stage、资源、负载、Producer / Consumer 工作量和调度。等待本身也不是零成本。

**为什么 B200 不能照搬 H100 的累加器与 wait 逻辑？**

> B200 代表性 tcgen05 路径涉及 TMEM 和新的完成跟踪机制，存储位置、协作范围、fence 与资源生命周期不同。相似的 Producer–Consumer 思想可以迁移，具体指令协议不能直接替换名称。

## 参考与关联专题

[^ptx]: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：Warp MMA、Warpgroup、WGMMA 的 fence / commit / wait，以及 tcgen05 的操作数、完成跟踪与同步规则。本文使用语义层解释，不提供完整内联 PTX。
[^ampere]: [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)：异步 Global → Shared 拷贝、Tensor Core 与资源优化。
[^hopper]: [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：TMA、异步事务屏障、Cluster 与 DSM。
[^blackwell]: [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)：对应架构能力与资源；具体指令以目标与 PTX 支持为准。
[^cutlass]: [CUTLASS：Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：分块、软件流水与 warp 特化思想。

[返回流水线专题](README.md) · [Kernel 编程总览](../README.md) · [算子融合](../../03-operators/fusion/operator-fusion.md) · [Attention 与 FFN 的分界线](../../01-model/transformer/attention-and-ffn.md) · [RTX 4090 参数](../../06-gpu-performance/hardware/rtx-4090.md)
