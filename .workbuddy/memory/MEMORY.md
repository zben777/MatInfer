# MatInfer 项目长期约定

## 仓库性质
- `zben777/MatInfer` 是 **public** 仓库。本地路径 `/Users/benzhang/Desktop/MatInfer`，remote 配置为 HTTPS（`https://github.com/zben777/MatInfer.git`）。
- 分支 `main`。仓库内所有已提交内容均已同步至远端。

## 推送方式（2026-10-07 打通，此后固定用这条）
remote 虽是 HTTPS，但**本机没有任何 HTTPS 凭据**（无 credential.helper / .git-credentials / .netrc，钥匙串无 github 条目，未装 gh），所以不要走 `git push origin main`。改用专用 deploy key：

```bash
cd /Users/benzhang/Desktop/MatInfer
git add <显式路径>            # 不要 -A，避免带上 Live模型/ 与 .workbuddy/
git commit -m "..."
GIT_SSH_COMMAND='ssh -F /dev/null -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519_matinfer' \
  git push git@github.com:zben777/MatInfer.git main
```

- **端口 22 直连即可**（2026-10-07 12:51 实测成功，无需代理）。若哪天内网封 22，再退到 `-p 443` + `git@ssh.github.com`。
- 因为推的是显式 URL 而不是 `origin`，推完 `origin/main` 会**滞后**。要同步：`git fetch git@github.com:zben777/MatInfer.git main:refs/remotes/origin/main`。
- 读远端（`ls-remote` / `fetch`）需借 Clash 代理：`git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 <cmd>`。沙箱注入的 64985 口对 GitHub 返回 502。
- key `~/.ssh/id_ed25519_matinfer`（deploy key，已勾 Allow write access）。测绑定时**必须 `-F /dev/null`**，否则会静默回落到只读的 `id_ed25519`、返回假的 `Hi zben777/zben777.github.io!`。**`ls-remote` 成功 ≠ 有写权限**（deploy key 默认只读，读操作照样过），写权限只能靠真推一次来判定。
- 推送后仍做三重实证：本地 HEAD / 远端真实 ref / 未推送提交数，并确认 `Live模型/`、`.workbuddy/` 未泄漏。

## 专题生产流水线（2026-10-07 用户确认"后续就这样干"）
一篇专题 = `md` + `assets/images/<topic>/NN-*.svg` + `tools/generate_<topic>_figures.py` + 同目录 `html` 阅读页。
五步：写正文 → 生成图源 → 构建阅读页 → 双层验证（数字脚本 + Playwright 真实渲染）→ 推送并用 `https://benzhang.top/MatInfer/<路径>` 核验。
**机械步骤与全部命令已固化为用户级 skill `matinfer-topic-pipeline`**，下次直接调用即可，不必重新推演。

## 定位
- **纯知识仓**，不承载项目正文。用户已另开独立 GitHub 仓库承载项目（warp-candidate-filter、vLLM-TQ 等）；本仓库对项目最多保留登记链接与结论摘要。
- 主线：六层知识（模型原理 / 推理框架 / 核心算子 / 高性能 Backend / Kernel 编程 / GPU 与性能分析）+ 面试追问。

## 不纳入版本控制
- `Live模型/`：语雀导出的讨论记录与知识库素材（约 1.6M）。**用户明确要求不提交**。属本地素材，与 `docs/` 定位重叠，勿将其内容复制进 `docs/`。
- `.workbuddy/`：本地记忆与会话数据，同样不提交。
- **两者的处理方式（2026-10-07 用户拍板）：不写进 `.gitignore`**，就让它们以 untracked 状态挂在 `git status` 里。防误提交靠 `git add` 时**只写显式路径、永不 `-A`**，外加推送后的泄漏复核（`git ls-tree -r --name-only <ref> | grep -E "^(Live|\.workbuddy)"` 必须为空）。不要为了「让 status 干净」去擅自改 `.gitignore`。

## 协作约定
- 未经用户明确同意，不修改或新增项目内文件；推送（push）前必须先行确认，且推送后需做三重实证（本地 HEAD / 远端真实 ref / 未推送提交数）。

## 写作与验证规范

### 风格基准（2026-10-06 由用户指定）
**基准：苏剑林《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》（kexue.fm/archives/10091）。**
用户原话："你学学这样的"。核心区别：**要写"一次思考的记录"，不要写"知识的结构化展示"。**
反例是此前两版 `online-softmax.md`——12 个表格 + 7 个清单 + 结论前置加粗句，读起来像速查卡，不像有人在讲。

具体可迁移的规则（按重要性排序）：
1. **零表格、零项目符号（正文部分）。** 推理过程写在连贯散文里。表格只保留给"真正需要横向对比数字"的场合（如并列多种合并顺序的结果）。清单式罗列是最大忌讳。
2. **开头用事件或时间点切入，不用定义。** 例：先讲 2018 年某篇四页论文做了什么、为什么当时无人注意，再交代本文要讲什么。最后一段才说"本文想讲的是……"。
3. **章节标题用一个词或短名词词组**（`瓶颈` / `一遍扫描` / `为什么恰好是三个状态`），**不用 `1.` `2.` 编号**，也不用"1.2 为什么……"这种目录式命名。
4. **每一步回答"当事人当时为什么这么做"**，而不是"这门技术是什么"。写演变时要有时间视角，允许"事后看来……"这类评述。
5. **要有立场，也要承认不确定。** 敢写"笔者认为这个角度不贴近本质"，也敢写"当时笔者也没能提出有效的建议"。
6. **主动替读者提出反问，然后反驳它。** 例："也许会有人问：为什么这个例子非要切成三块，两块不行吗？因为……"
7. **制造悬念。** 在抛出结论前先写"一切似乎都很完美……不过别急，再深入想一步就会发现"。
8. **术语首现给英文全称 + 粗体首字母缩写**；数字要当论据用，并解释"它为什么恰好是这个数"（如 g=8 不是随手取的，因为一台机 8 张卡）。
9. **小结只写一段**，不铺陈。

### 结构与内容规则（延续，仍有效）
- **引擎前置。** 驱动整篇的那一条原理（一条恒等式、一个约束、一个观察）必须在开头交代清楚，并明说"其余内容都是它的推论"。不要藏在推导中途。
- **最高的洞察必须独立成节，不能写成"补充说明"。** 判断标准：如果某个性质解释了"为什么这件事能被工程化 / 并行化 / 推广"，它就是正文级内容。
- **参考代码、工程细节、面试问答一律放文末附录**，用 `## 附录 A/B/C` 隔开，不要打断正文的散文流。
- **示例必须覆盖公式的全部分支。** 公式里出现的每个系数、情形或分支，正文至少要亲自演示一次；只演示退化情形的例子视为未完成。

### 验证铁律
- **每个手算数字、每句"输出应为 X"的承诺，落笔前必须用脚本实测。** 数字矛盾是最高优先级的错误。反面算例（错误的做法会得到什么）同样要算出来。
- 示例中的分数优先写成 `ln` 形式（`exp(ln a) = a`），使手算落回整数或简单分数；最终输出尽量设计为整数，便于读者自查。
- 参考代码若声称"可以复现本文例子"，必须实际跑通并给出预期输出；正文公式与代码行之间应给出对照。
- 引用论文结论（加速比、访存次数等）要标明来源与口径，不用未核实的数字。

### 配图规范（2026-10-06 由用户指示"加图或加视频都可以"后确立）
- **图源是可编辑的脚本，不是手画文件。** 每篇专题的图由 `tools/generate_<topic>_figures.py` 生成到 `assets/images/<topic>/NN-*.svg`，只依赖 Python 标准库。改数字时改脚本再重新生成，不要在 SVG 里手改。
- **视觉语言必须复制现有脚本**（`tools/generate_norm_figures.py` 是模板）：viewBox `0 0 1000 H`、白底、页眉 `MATINFER / <TOPIC> / NN`（13px 绿 600）、标题 28px 600、副标题 17px muted、页脚一行 muted；颜色固定 `ink #24334a` / `muted #6d7c90` / `green #218575` / `blue #4967b0` / `amber #c88735` / `rule #d8e0e9` / `faint #b3becc`；字体 `PingFang SC,Microsoft YaHei,Arial,sans-serif`；文本一律经 `escape()`。
- 文中引用路径从 `docs/<层>/<专题>/` 出发是 `../../../assets/images/<topic>/NN-*.svg`。图后跟一行 `*图 N　说明。*` 图注。
- **动画用 CSS `@keyframes` 的 SVG**（多帧 `<g class="f1..fN">` 互斥显示 + `prefers-reduced-motion` 降级到最后一帧）。CSS 必须放在顶层 `base()` 的 `head=` 参数里，不要塞进 `<g>`。
- **动画的播放条件（2026-10-07 实测更正）**：`<img>` 引用（含 `data:` URI）时 CSS 动画**照常播放**——此前记的"img 里只显示第一帧"是错的。真正会停住的只有栅格化场景（截图、导出 PNG、把 SVG 当静态图处理的渲染器）。**不必为此额外做 GIF。**
- **图内只放图该有的东西。** 文档说明（"怎么播放""五步依次是…"）属于正文图注，不要印在图里——它容易随实现变化而过时。
- **`assets/README.md` 的「专题图源」段必须同步登记新图**，否则半年后没人知道图从哪来。

### 阅读页规范（2026-10-07 确立）
- **`.md` 为主，HTML 阅读页为辅，两者并存。** `.md` 里插图用相对路径引用 SVG，GitHub 与多数编辑器可正常显示；**部分预览器会拦掉本地 SVG**（用户就撞到过"也没图啊"）。因此每个配图专题额外产出一份**自包含** HTML 阅读页。
- **首选命令是 `tools/build_all.mjs`**（全量重建 + 链接检查，见下节）；单篇构建仍可用
  `tools/build_reading_page.mjs docs/<层>/<专题>/<名>.md`。
  产出路径与导航信息来自 `tools/topics.json`，**不在脚本里写死**（此前写死成 Attention，是隐患）。
  产出约 400 KiB。插图 base64 内联、公式 MathJax 渲染成内联 SVG，**零外部依赖、离线可看、动画正常播放**。
- 正文改动后必须重新生成阅读页，否则两者会漂移。图源永远在 `assets/images/` 的 SVG，阅读页只是呈现层。

## 工具链与专题登记表（2026-10-07 建立）
- **`tools/topics.json`：专题登记表**，每个专题记录 `md` / `out` / `managed` / `navLabel` / `navTitle` / `crumb` / `eyebrow` / `sideLinks`。
  - `managed: true` → HTML 由构建器完全接管，md 改后会被重新生成。
  - `managed: false` → 该页含**手工维护内容**（额外 CSS、侧栏项、不同页脚），构建器**只做链接检查、不覆盖**。
  - 新增专题只需在 `topics` 里追加一条；`out` 省略时默认输出到与 md 同目录的 `.html`。
- **`tools/build_all.mjs`：全量重建 + 链接检查。**
  ```
  cd /Users/benzhang/Desktop/MatInfer
  NODE_PATH=/Users/benzhang/.workbuddy/binaries/node/workspace/node_modules \
    /Users/benzhang/.workbuddy/binaries/node/versions/24.19.0/bin/node tools/build_all.mjs
  ```
  三阶段：① 跑 `tools/generate_*_figures.py` 并按哈希报告图源变化；② 重建 `managed` 的阅读页（报告"未变化/已更新/新建"）；③ 链接检查（放大图 / 图片 / 相对链接 / md 插图），有问题时退出码非 0。
  `--check-only` 只做检查，`--skip-figures` 跳过图源。
- **构建器的两个已修缺陷（2026-10-07）**：
  1. **GitHub 风格行内公式 `` $`tex`$ `` 会让 MathJax 崩溃**——原实现「先抽代码、后抽公式」，代码占位符（含 `\u0000`）被当 TeX 喂进去。改为**一次扫描的三选一正则**：`` `code` `` / `` $`tex`$ `` / `$tex$`，谁先出现谁生效。**此前 8 个文件（gqa-from-scratch / mha-head-independence / 05-kernel README 等）全都构建不了。**
  2. **不支持 `<details>` 折叠块**——裸 HTML 被当段落转义。现按块识别 `<details>…</details>`，外壳透传、内部仍走 Markdown 解析（`renderBlocks` 递归），并补了 `article details/summary` 的 CSS。layernorm-rmsnorm.md 有 4 个、gqa-head-sharing.md 有 1 个。
- **当前七个阅读页的所在地（历史遗留，位置不统一）**：
  | HTML | 源 md | managed |
  | --- | --- | --- |
  | `docs/index.html` | `docs/01-model/attention/gqa-head-sharing.md` | false |
  | `docs/normalization.html` | `docs/01-model/normalization/layernorm-rmsnorm.md` | false |
  | `docs/01-model/moe/deepseek-moe.html` | 同名 md | **true**（2026-10-07 新增） |
  | `docs/01-model/transformer/attention-and-ffn.html` | 同名 md | **true**（2026-10-07 重写该专题时补建） |
  | `docs/01-model/ffn/swiglu.html` | 同名 md | **true**（2026-10-11） |
  | `docs/05-kernel-engineering/deepgemm/cta-tiling.html` | 同名 md | **true**（2026-10-11，新主线） |
  | `docs/03-operators/attention/online-softmax.html` | 同名 md | **true**（新约定：与 md 同目录） |
  前两篇在 `docs/` 根下，是早期手工放置的。**不要因为"统一位置"就迁移**：URL 已上线，且它们各有手工内容。新专题一律按新约定放 md 同目录。
- ⚠️ **块级公式只能写 ```math 围栏**（2026-10-11 踩坑）：仓库**不支持 `$$...$$`**，现有篇目里一处都没有。行内用 `$tex$` 或 GitHub 风格 `` $`tex`$ ``，块级一律 ```math。**开工前先 grep 现有篇目确认方言**，别凭印象写。
- **改写已有 md 时（而非新建专题）要额外做两件事**：① 标题变了就同步改所有引用它的链接文字（`grep -rn "<文件名>"`，含层 README 与各专题页脚）；② 该文件若在 `topics.json` 里没有条目，补一条（`managed: true`），否则永远没有阅读页。2026-10-07 重写 `attention-and-ffn.md` 时两件都做了。
- **新增一个专题要动四处**：① `<专题>/<名>.md` 正文；② `<专题>/README.md`（表格：专题 | 核心问题，照 `attention/README.md`）；③ 层 README 的「内容索引」加一行链到 `<专题>/README.md`；④ `tools/topics.json` 追加一条。漏掉 ②③，新专题在仓库里没有入口。
- 链接检查器的自检方法：临时把某个 `href` 改回错误层级，跑 `build_all --check-only`，应报「放大图断链」。
- **新主线：`docs/05-kernel-engineering/deepgemm/`「DeepSeek Kernel 源码导读」**（2026-10-11 开，用户拍板）。学法是顺着 `deepseek-ai` 开源仓库的源码一层层走，第一层 CTA Tiling。**44 个 deepseek-ai 仓库在本机直连可读**（`api.github.com/repos/<r>/git/trees/main?recursive=1` 拉树 + `raw.githubusercontent.com` 取单文件）。手头相关：DeepGEMM（GEMM FP8/FP4/BF16 + MoE Grouped）、DeepGEMM-Ascend、FlashMLA（含 `csrc/ascend_kernels`）、DeepEP、DeepSelect、TileKernels、DeepSpec（投机解码）。**口径**：DeepGEMM 主线面向 SM90/SM100，用户的 4090 是 SM89(Ada)，Hopper 的 `wgmma`/TMA 都没有——「读源码理解设计」成立，「编译跑通 + 量性能」需要 Hopper/Blackwell 机器。
- ⚠️ **GitHub Code 页面不渲染 `.html`，只会显示源码**（2026-10-07 实测；我一度误判"点开就能渲染"，被用户当场发现）。第三方预览服务也全部不可用——实测 `raw.githack.com` 对浏览器返回落地页、`cdn.jsdelivr.net` 以 `text/plain` 下发、`htmlpreview.github.io` 不渲染，**不要推荐**。

### GitHub Pages（2026-10-07 已开启并验证）
配置：Source = `Deploy from a branch`，Branch = `main`，目录 = `/ (root)`。**不需要 `.nojekyll`**（仓库内无任何带 front matter 的 md/html，`{{`/`{%` 也为 0，Jekyll 只做静态拷贝）。

地址规则 —— 用户站 `zben777.github.io` 已绑自定义域名，**项目站会继承**，路径就是仓库内路径：

```
https://benzhang.top/MatInfer/<仓库内路径>
例：https://benzhang.top/MatInfer/docs/03-operators/attention/online-softmax.html
```

- **根地址 `https://benzhang.top/MatInfer/` 返回 200**，由 GitHub Pages 默认插件 `jekyll-readme-index` 把 `README.md` 渲染成首页。（我此前预测"根地址会 404"是错的——漏了这个默认插件。）
- 但那个首页是**无样式默认渲染，且不含指向各阅读页的入口** → 若要改善，需另加 `index.html` 落地页（**未做，等用户确认**）。
- 线上验证方式：`curl -x http://127.0.0.1:7897 <url>`；Playwright 可直接打开 https URL 做真实渲染检查。
- **推送后立刻验证会撞 CDN 旧缓存（2026-10-10 实测）**：刚 push 完 curl 新页，可能拿到 **Jekyll 生成的旧版**（特征：HTML 里含 `Begin Jekyll SEO tag`、`last-modified` 是旧日期、响应头 `x-cache: HIT` / `age: <小数字>`、`cache-control: max-age=600`）。**这不是构建失败**，等几分钟或重取即为新版。判定「真上线」必须做 `shasum -a 256` 与本地逐字节比对，别只看 200。
- **Jekyll 其实会处理仓库里的 `.md`**（此前记的"只做静态拷贝"不准确）：`<名>.md` 会被渲染成 `<名>.html`。因此**同目录 `md` + 手工 `html` 同名时**，静态 html **优先**、md 生成的那份被让位——当前五个专题都是这个格局，实测不报错、线上拿到的是自包含静态页。若哪天线上某页冒出 Jekyll SEO 头，先查是不是该路径缺了静态 html。
- 已知残留问题：`docs/normalization.html` 的 4 个「点击放大图」链接写的是 `../../../assets/...`，但该文件位于 `docs/` 下（应为 `../assets/`），**线上实测 4 个全 404**（图本身是 base64 内联，页面显示不受影响）。**未改，等用户确认。**

### 视觉验证（2026-10-07 起改用真实渲染）
**本机可以真正渲染并截图了**，不必再用估算式版面检查：
- 浏览器内核：`~/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell`（另有一份完整版 `chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app`）。
- 驱动：`playwright-core@1.63.0`，位于 `/Users/benzhang/.workbuddy/binaries/node/workspace/node_modules`；
  `NODE_PATH=... node x.mjs`，脚本内 `chromium.launch({ executablePath: <上面的路径> })`。
- **系统 Chrome 的 `--headless` 会被沙箱 SIGKILL（exit 137），Playwright 的 headless-shell 不会。**
- 用法：`page.setContent()` 注入 SVG 后 `screenshot()`，再用 Read 工具**肉眼看**；`img.screenshot()` 隔几秒重复采样并比对哈希，可验证动画是否真的在播。
- 旧的估算式检查器（按字符宽度估占位、查越界/重叠）逻辑仍可用作无浏览器时的兜底，但**优先用真实渲染**。

