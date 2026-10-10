# 图片与图源

三张学习图已保存到 `images/` 并在[仓库首页](../README.md)展示：推理技术栈、Transformer 原理、DeepSeek 全栈面试地图。当前使用用户提供的原图，图片内部的公式与术语尚待修订。

原稿中的外部图片链接继续保留在[总览](../docs/00-overview/README.md)。

后续为知识图保留可编辑图源，并从所属专题链接到导出的图片。每张图说明范围、简化条件和来源；需要重新使用外部图片时核对作者与授权。六层 ASCII 图目前作为布局草案保存。

## 专题图源

`images/normalization/` 由 `tools/generate_norm_figures.py` 生成。

`images/attention/` 由 `tools/generate_attention_figures.py` 生成，供 [Online Softmax 专题](../docs/03-operators/attention/online-softmax.md)使用：

| 文件 | 内容 |
| --- | --- |
| `01-memory-access.svg` | Safe Softmax 的三遍读取与 online normalizer 的一遍读取 |
| `02-rebase-scaling.svg` | 换基准时每个指数项乘同一个系数，比值不变 |
| `03-block-merge.svg` | 三块合并的局部状态、α / β 缩放对象，以及漏缩放的后果 |
| `04-associativity.svg` | 顺序合并与结合律允许的分组并行合并 |
| `05-online-algorithm-shape.svg` | Softmax 与 Welford 在线方差的修正项同构 |
| `06-merge-animation.svg` | 五帧动画（CSS 关键帧循环播放，需由页面内联或 `<img>` 加载） |

`images/moe/` 由 `tools/generate_moe_figures.py` 生成，供 [DeepSeekMoE 专题](../docs/01-model/moe/deepseek-moe.md)使用：

| 文件 | 内容 |
| --- | --- |
| `01-ledger.svg` | Dense / 常规 MoE / DeepSeekMoE 三条路的参数与激活算力账本 |
| `02-shared-expert.svg` | 通用知识被 15 个专家各存一份，与共享专家只存一份的对比 |
| `03-scoring-functions.svg` | Sigmoid 与 Sqrt(Softplus) 的曲线及饱和导致的"分辨率"差异 |
| `04-bias-routing.svg` | 偏置项只进 Top-K 排序、不进门控权重，因而零梯度 |
| `05-forward-animation.svg` | 五帧动画：1×6 一次完整前向（打分 → 加偏置 → 归一化 → 前向 → 合并） |
| `06-node-limited-routing.svg` | Node-Limited Routing 把跨节点通信从 8t 压到 4t，以及它砍掉的路由自由度 |

`images/attnffn/` 由 `tools/generate_attnffn_figures.py` 生成，供 [Attention 与 FFN 专题](../docs/01-model/transformer/attention-and-ffn.md)使用：

| 文件 | 内容 |
| --- | --- |
| `01-reduction-axis.svg` | 层内六个运算各自的归约轴长度，只有 softmax 的分母随输入变长 |
| `02-scale.svg` | 有效位置数对上下文长度的曲线，以及 T = 4096 时三个刻度的对照 |
| `03-spread-animation.svg` | 四帧动画：上下文变长时注意力权重的分布（不缩放 vs 除以 sqrt(d_k)） |
| `04-two-journeys.svg` | 同一行的两段旅程：横向读历史、纵向过权重，写回都是一行 |
| `05-width.svg` | 无激活时两矩阵合成一个（2,097,152 等价于 262,144），有激活时宽度才兑现 |
| `06-drift.svg` | 参数占比从 2/3 漂到 80.77%：GQA 砍 K/V，SwiGLU 仍占 3dm ≈ 8d² |

三个生成脚本均只依赖 Python 标准库，可重新生成：`python3 tools/generate_attention_figures.py`。

图内数字与专题正文一致（O = 5、ℓ = 2.5、U = 12.5、41/9、3.25、修正项 154.7 占 95.7%；MoE 侧为 108 / 864 / 432 与 120 / 240 / 120、w = 1/7, 9/14, 3/14、y = 2.0338、激活比例 8.90% / 5.51% / 4.58% / 3.06%；Attention/FFN 侧为 4d² 与 8d²、2/3 与 80.77%、1.05 / 2.10 / 41.94 / 176.16 M 参数、有效位置数 1.10 / 3.88 / 2531.70、方差实测 63.818）。改动正文数字时须同步改脚本并重新生成。`generate_attnffn_figures.py` 里图 02、03 的曲线数据是固定种子下的蒙特卡洛结果，已固化在脚本内；需要重算时加 `ATTNFFN_RECOMPUTE=1` 运行。

## 阅读页

`.md` 里的插图用相对路径引用 SVG，GitHub 和多数编辑器可以直接显示，但部分预览器会拦掉本地 SVG，动画也停在第一帧。因此每个配图专题额外产出一份自包含的 HTML 阅读页：

| 阅读页 | 对应正文 |
| --- | --- |
| `docs/normalization.html` | `docs/01-model/normalization/layernorm-rmsnorm.md` |
| `docs/03-operators/attention/online-softmax.html` | `docs/03-operators/attention/online-softmax.md` |
| `docs/01-model/moe/deepseek-moe.html` | `docs/01-model/moe/deepseek-moe.md` |
| `docs/01-model/transformer/attention-and-ffn.html` | `docs/01-model/transformer/attention-and-ffn.md` |

阅读页由 `tools/build_reading_page.mjs` 生成，产物内联全部插图（base64 SVG）并用 MathJax 把公式渲染成内联 SVG，**不依赖任何外部资源，离线可看，插图动画正常播放**。正文改动后重新生成：

```
node tools/build_reading_page.mjs docs/03-operators/attention/online-softmax.md
```

依赖 `mathjax-full`，装在隔离的 Node 工作区，不污染本机。图源仍是 `images/` 下的 SVG —— 阅读页只是呈现层，改图请改脚本。
