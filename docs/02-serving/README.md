# 推理框架与服务系统

> 状态：内容草案。已整理原稿的技术主题、公式、流程、图布局和题目；完整教程、源码版本与实测结果将随学习补齐。示意代码不等同于已验证实现。

**核心问题：多个请求怎样被调度、组织、执行和返回？**

覆盖请求生命周期、Scheduler、Continuous Batching、KV 分页与前缀缓存、ModelRunner、CUDA Graph、Chunked Prefill、投机解码和分布式推理。

建议学习顺序：先追踪一次请求，再理解调度和显存管理，最后比较并行、PD 分离与端到端指标。

[返回总导航](../../导航.md) · [学习路线](../../roadmap/README.md) · [面试题索引](../../interview/README.md)

## 内容索引

- [整张图的核心问题](#section-01)
- [一个 Request 的完整生命周期](#section-02)
- [中间最大的区域：Scheduler](#section-03)
- [Continuous Batching](#section-04)
- [Prefill 和 Decode](#section-05)
- [KV Cache Manager](#section-06)
- [Prefix Cache / Radix Cache](#section-07)
- [ModelRunner / Executor](#section-08)
- [CUDA Graph](#section-09)
- [Chunked Prefill](#section-10)
- [Speculative Decoding](#section-11)
- [TP / PP / EP / PD](#section-12)
- [PD 分离](#section-13)
- [框架对比：vLLM / SGLang / TensorRT-LLM](#section-14)
- [系统层指标](#section-15)
- [面试与自测问题](#section-16)
- [知识图布局草案](#section-17)
- [模型层与框架层的连接](#section-18)

## 适用条件与补充

终止条件包括 EOS、长度上限、stop 条件和取消；流式返回会发生在 Decode 过程中。释放缓存时也要区分请求引用释放与前缀缓存是否保留。PD 是阶段解耦架构，不是与 TP / PP / EP 同性质的参数切分；DP / Replica 可补充为服务副本扩展专题。

详见[技术口径与待核对事项](../00-overview/technical-notes.md)。

### 从一个 Request 进入系统，到不断生成 Token

+ 第一张回答：**一个 Token 在模型里面怎么算**
+ 第二张回答：**很多 Request 在服务系统里面怎么被组织起来算**

**vLLM / SGLang / Scheduler / KV Cache / Continuous Batching / CUDA Graph / TP / EP / PD**

---

<a id="section-01"></a>

## 整张图的核心问题

顶部副标题直接写：

> **本层重点是“什么时候算、谁一起算、数据放哪里、在哪张卡上算”；框架也参与 Backend 选择与计算集成。**

---

<a id="section-02"></a>

## 一个 Request 的完整生命周期

```text
用户 Prompt
    ↓
API / Request
    ↓
Tokenizer
    ↓
Waiting Queue
    ↓
Scheduler
    ↓
资源检查
├─ KV Cache 是否有空间？
├─ Token Budget 是否足够？
└─ 当前 Batch 是否能加入？
    ↓
进入 Running Set
    ↓
┌─────────────────────┐
│ Prefill             │
│ 处理 Prompt Tokens  │
└─────────────────────┘
    ↓
建立 KV Cache
    ↓
Sampling
    ↓
┌─────────────────────┐
│ Decode Loop         │
│ 每轮生成新 Token    │
└─────────────────────┘
    ↓
Scheduler 再调度
    ↓
ModelRunner
    ↓
Sampling
    ↓
是否 EOS？
 ├─ No → 下一轮 Decode
 └─ Yes
      ↓
释放 KV Cache
      ↓
返回结果
```

> “一次请求进来以后发生什么？”

---

<a id="section-03"></a>

## 中间最大的区域：Scheduler

**Scheduler**
因为 Serving Framework 和普通 PyTorch `model.forward()` 最大的区别之一，就是这里。

```text
                     Waiting Requests
                           │
                           ↓
                    ┌──────────────┐
                    │  Scheduler   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
         Request A     Request B     Request C
         Decode        Prefill       Decode
              │            │            │
              └────────────┼────────────┘
                           ↓
                  当前 Iteration Batch
                           ↓
                     ModelRunner
```

然后 Scheduler 旁边放几个决策：

```text
Scheduler 每轮决定：

请求层面：
谁运行？
谁等待？
谁抢占？

Token 层面：
这个请求本轮跑几个 Token？

资源层面：
KV Cache 是否足够？

执行层面：
Prefill 还是 Decode？
能不能混合执行？
```

---

<a id="section-04"></a>

## Continuous Batching

### Static Batching
```text
Time ─────────────────────────→

A ████████████████████
B ███████
C ███████████

B 完成以后：
固定批次无法补入新请求；已完成请求的槽位不能继续利用
```

↓

### Continuous Batching
```text
Time ─────────────────────────→

A ████████████████████
B ███████
          D ██████████
C ███████████
              E █████████
```

> **请求完成后可在后续调度迭代补入新请求；是否准入取决于资源和调度策略。**

这时候可以解释它为什么有效：

```text
Static Batch

有效 batch size
会随着请求完成不断下降

          ↓

Continuous Batch

动态补入请求

          ↓

维持更高 GPU utilization
```

---

<a id="section-05"></a>

## Prefill 和 Decode

这个模块必须放，因为它是后面很多框架设计的根源。

画成左右对比。

### Prefill
```text
Prompt:
Token 1
Token 2
Token 3
...
Token N

一次处理大量 Token

特点：
• GEMM 大
• Attention 大
• Compute 更密集
• TTFT 重要
```

### Decode
```text
历史 Tokens
↓↓↓↓↓↓↓↓↓↓↓

        新 Token Q
             ↓

读取全部历史 KV

特点：
• 每请求通常每轮 1 Token
• KV Cache 读取多
• 小 M GEMM
• Memory / Launch 更敏感
• TPOT / ITL 重要
```

中间放：

```text
为什么框架要区分 Prefill / Decode？

因为两者：

计算特征不同
访存特征不同
Batch Shape 不同
最优调度策略不同
```

---

<a id="section-06"></a>

## KV Cache Manager

```text
Request A
Logical KV blocks

[0][1][2][3][4]
 │  │  │  │  │
 ↓  ↓  ↓  ↓  ↓

Block Table

[17][8][31][4][52]

 │  │   │  │  │
 ↓  ↓   ↓  ↓  ↓

Physical KV Cache

GPU Memory
```

> **逻辑连续，不要求物理连续。**

然后解释为什么 Page：

```text
如果每个请求申请连续大内存：

Request A: 2048 tokens
Request B: 127 tokens
Request C: 4096 tokens

长度动态变化
        ↓
难以连续分配
        ↓
碎片 / 扩容 / 搬迁问题
```

Page 后：

```text
固定大小 Block
      ↓
按需分配
      ↓
结束立即回收
      ↓
更适合动态请求
```

---

<a id="section-07"></a>

## Prefix Cache / Radix Cache

```text
Request A:
"你是一名 CUDA 专家..."
                  ↓
           Prefix KV Cache

Request B:
"你是一名 CUDA 专家..."
        + 新问题
```

共同 Prefix：

```text
[P0][P1][P2][P3]
```

不重新 Prefill：

```text
直接复用已有 KV
```

```text
Prefix Cache

核心：
相同前缀
→ 复用 KV Block
```

```text
Radix Cache

把 Prefix 组织成
Radix Tree / Trie 类结构

用于高效匹配
不同长度公共 Prefix
```

> **它解决的是重复 Prompt 的 Prefill 计算问题。**

---

<a id="section-08"></a>

## ModelRunner / Executor

```text
Scheduler
    ↓
构建本轮 Batch
    ↓
ModelRunner
    ↓
准备 Tensor / Metadata
    ↓
执行模型
    ↓
Transformer Blocks
    ↓
Attention / MoE / GEMM
    ↓
Output Hidden State
    ↓
LM Head
    ↓
Logits
```

```text
Framework

“决定怎么执行”

──────────────

Model / Kernel

“真正完成计算”
```

> 推理框架开发和算子开发到底什么区别？

最直观的答案。

---

<a id="section-09"></a>

## CUDA Graph

> 降低 CPU Launch 开销。

```text
CPU
 ↓ launch
Kernel 1
 ↓
CPU
 ↓ launch
Kernel 2
 ↓
CPU
 ↓ launch
Kernel 3
 ↓
...
```

一轮执行可能包含大量小 Kernel。

Decode 特别容易受到：

```text
CPU launch latency
+
framework overhead
```

影响。

CUDA Graph：

```text
Capture

Kernel1
 ↓
Kernel2
 ↓
Kernel3
 ↓
Kernel4

     ↓

Graph

     ↓

一次 Replay
```

> **把重复的 GPU 执行序列预先捕获，减少每轮 CPU launch / dispatch overhead。**

```text
特别适合：

Decode
+
Shape 相对稳定
+
反复执行相似计算图
```

---

<a id="section-10"></a>

## Chunked Prefill

假设：

```text
Request A:
Prompt = 8192 tokens
```

如果一次 Prefill：

```text
█████████████████████████

占用 GPU 很长时间
```

Decode Request：

```text
B B B B B B B
```

可能长时间等。

Chunked Prefill：

```text
A1 A2 A3 A4
```

拆成：

```text
[A1] → Decode B
[A2] → Decode B
[A3] → Decode B
[A4]
```

> **把超长 Prefill 拆成多个 Chunk，让 Decode 请求有机会穿插执行。**

核心 trade-off：

```text
吞吐
vs
Decode latency
vs
TTFT
```

---

<a id="section-11"></a>

## Speculative Decoding

可以放一个较小知识块。

```text
Draft Model

猜：
t1 t2 t3 t4
 ↓  ↓  ↓  ↓

Target Model
一次验证多个 Token
```

如果接受：

```text
✓ ✓ ✓
```

一轮生成多个 Token。

如果拒绝：

```text
✓ ✓ ✗
```

从拒绝位置重新走。

> **核心不是 Draft Model 有多快，而是：Draft 开销 + Acceptance Rate + Target Verification Cost。**

---

<a id="section-12"></a>

## TP / PP / EP / PD

这部分不要讲太深，先做“地图式分类”。

```text
多 GPU 推理
│
├─ TP
│  Tensor Parallel
│  一层模型横向切
│
├─ PP
│  Pipeline Parallel
│  不同 Layer 切到不同 GPU
│
├─ EP
│  Expert Parallel
│  不同 MoE Expert 分布不同 GPU
│
└─ PD
   Prefill / Decode Disaggregation
   Prefill 与 Decode 分开部署
```

### TP
```text
Linear / GEMM
        ↓
Column Parallel
Row Parallel
        ↓
AllReduce / AllGather
```

### EP
```text
Token
 ↓
Router
 ↓
Expert 所在 GPU
 ↓
All-to-All
```

通信细节随源码阅读与实验逐步补充。

---

<a id="section-13"></a>

## PD 分离

PD 值得独立一个小图。

```text
普通部署：

GPU
├─ Prefill
└─ Decode

相互干扰
```

变成：

```text
Prefill Pool
GPU GPU GPU
     │
     │ KV transfer
     ↓
Decode Pool
GPU GPU GPU
```

为什么：

```text
Prefill
更 Compute-heavy

Decode
更 Memory / latency-sensitive
```

> 可以针对不同 workload 独立扩缩容和调度。

> “PD 一定更快。”

> **PD 引入了 KV 传输和系统复杂度，收益取决于负载与硬件配置。**

---

<a id="section-14"></a>

## 框架对比：vLLM / SGLang / TensorRT-LLM

这一块只做定位，不做谁好谁坏。

```text
vLLM
│
├─ Serving Engine
├─ Paged KV
├─ Continuous Batching
└─ 通用模型生态
```

```text
SGLang
│
├─ Serving Runtime
├─ Radix / Prefix-oriented caching
├─ Scheduler
└─ 大规模 / 多种 serving 优化
```

```text
TensorRT-LLM
│
├─ NVIDIA Runtime
├─ TensorRT / CUDA 深度集成
├─ Kernel / Quantization
└─ NVIDIA GPU 优化
```

---

<a id="section-15"></a>

## 系统层指标

因为框架优化不能只看 Kernel ms。

```text
Serving Metrics
```

主要指标：

```text
TTFT
Time To First Token

TPOT
Time Per Output Token

Throughput
tokens/s

Concurrency
同时服务请求数
```

```text
GPU Memory Usage
KV Cache Utilization
```

> **框架优化最终看 End-to-End，而不是只看某个 Kernel。**

---

<a id="section-16"></a>

## 面试与自测问题

放 10 个：

1. vLLM 一次 Request 的生命周期是什么？
2. Continuous Batching 为什么可以提高吞吐？
3. Prefill 和 Decode 的性能特征有什么区别？
4. 为什么 KV Cache 要采用 Page / Block 管理？
5. Prefix Cache 和普通 KV Cache 有什么区别？
6. Radix Cache 解决什么问题？
7. Chunked Prefill 为什么能降低 Decode interference？
8. CUDA Graph 为什么特别适合 Decode？
9. TP / PP / EP 分别切什么？
10. PD 分离的收益和代价是什么？

---

<a id="section-17"></a>

## 知识图布局草案

```text
┌────────────────────────────────────────────────────────────────┐
│     大模型推理框架：从 Request 到 Next Token                   │
│     vLLM / SGLang / TensorRT-LLM                               │
├─────────────────┬────────────────────────┬─────────────────────┤
│                 │                        │ Scheduler           │
│ Request         │ Request Lifecycle      ├─────────────────────┤
│ Lifecycle       │                        │ Continuous Batching │
│                 │ Scheduler              ├─────────────────────┤
│ API             │    ↓                   │ Prefill vs Decode   │
│ ↓               │ KV Allocation          │                     │
│ Queue           │    ↓                   │                     │
│ ↓               │ ModelRunner            │                     │
│ Scheduler       │    ↓                   │                     │
│ ↓               │ Sampling               │                     │
│ Prefill         │                        │                     │
│ ↓               ├────────────────────────┼─────────────────────┤
│ Decode          │ KV Cache Manager       │ Prefix/Radix Cache  │
│ ↓               │                        │                     │
│ Output          │ Logical→Physical Page  │ Chunked Prefill     │
│                 │                        │                     │
├─────────────────┼────────────────────────┼─────────────────────┤
│ CUDA Graph      │ TP / PP / EP           │ PD Disaggregation   │
├─────────────────┴────────────────────────┴─────────────────────┤
│ Serving Metrics：TTFT / TPOT / Throughput / Concurrency        │
├────────────────────────────────────────────────────────────────┤
│ 高频面试问题                                                   │
└────────────────────────────────────────────────────────────────┘
```

---

<a id="section-18"></a>

## 模型层与框架层的连接

第一张：

```text
模型层
Transformer
   ↓
MLA / MoE
```

底部：

```text
↓ 由推理框架调度执行
```

第二张顶部：

```text
Request
↓
Scheduler
↓
ModelRunner
```

中间：

```text
ModelRunner
↓
调用 Transformer
↓
进入模型层
```

第二张底部：

```text
Attention / GEMM / RMSNorm
↓
进入算子层
```

```text
① 模型结构

        ↓

② 推理框架

        ↓

③ 核心算子

        ↓

④ 高性能 Backend

        ↓

⑤ Kernel 编程

        ↓

⑥ GPU 硬件与性能
```
