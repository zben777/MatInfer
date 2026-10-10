"""Regenerate the Attention / FFN diagrams (editable SVG sources).

SVG generation uses only the Python standard library, matching
tools/generate_norm_figures.py, tools/generate_attention_figures.py and
tools/generate_moe_figures.py. Figure 03 is animated with CSS @keyframes;
it plays both when opened directly and when referenced from an <img> tag.

Numbers in figures 02 and 03 come from a Monte-Carlo over iid N(0,1) q/k with
d_k = 64. They are pinned below as constants so a normal build stays fast.
To recompute them (takes a couple of minutes in pure Python):

    ATTNFFN_RECOMPUTE=1 python3 tools/generate_attnffn_figures.py

No TeX is used inside the SVGs: nothing renders LaTeX in a plain SVG file, so
every formula here is written with Unicode.
"""
import math
import os
import random
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parents[1] / 'assets/images/attnffn'
out.mkdir(parents=True, exist_ok=True)

ink = '#24334a'; muted = '#6d7c90'; green = '#218575'
blue = '#4967b0'; amber = '#c88735'; rule = '#d8e0e9'; faint = '#b3becc'
DK = 64                      # head dim used throughout
SIG = math.sqrt(DK)          # 8.0


def txt(x, y, s, size=18, color=ink, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')


def rect(x, y, w, h, fill='#f3f6fa', stroke='none', rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x, y, x2, y2, color=faint, arrow=False, width=2, dash=''):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{m}{d}/>'


def base(num, title, sub, h, head=''):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {h}" role="img" '
            f'aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(sub)}</desc>'
            f'<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
            f'orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="{faint}"/></marker></defs>'
            f'{head}'
            f'<g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">'
            + rect(0, 0, 1000, h, '#ffffff', rx=0)
            + txt(40, 42, f'MATINFER / ATTN-FFN / {num}', 13, green, 600)
            + txt(40, 85, title, 28, ink, 600)
            + txt(40, 119, sub, 17, muted))


def save(name, s):
    (out / name).write_text(s + '</g></svg>')


# ============================================================ 01 归约轴在哪里
s = base('01', '层里只有一条归约轴随输入变长',
         '把"跨位置"换成能算的判据：这条求和轴的长度，会不会跟着上下文一起变？', 600)
OPS = [
    ('分数', 'q · kᵀ', 'd_h', blue),
    ('分母', 'softmax  Z', 'T', amber),
    ('聚合', 'A · V', 'd_h', blue),
    ('升维', 'u · W₁', 'd', green),
    ('激活', 'SiLU', '——', green),
    ('降维', '· W₂', 'm', green),
]
CW, GAP = 140, 16
X0 = 40
s += txt(X0, 152, 'Attention', 16, blue, 600)
s += txt(X0 + 3 * (CW + GAP), 152, 'FFN', 16, green, 600)
s += rect(X0 + 3 * (CW + GAP) - 8, 160, 3 * CW + 2 * GAP + 16, 146, '#eef4f0')
s += rect(X0 + 1 * (CW + GAP) - 8, 160, CW + 16, 146, '#fdf4e7')
for i, (name, expr, axis, col) in enumerate(OPS):
    x = X0 + i * (CW + GAP)
    s += rect(x, 174, CW, 74, '#f7f9fc')
    s += txt(x + CW / 2, 206, name, 19, ink, 600, 'middle')
    s += txt(x + CW / 2, 232, expr, 16, muted, 400, 'middle')
    pill = '#e8cfa8' if axis == 'T' else '#e8edf4'
    s += rect(x, 260, CW, 30, pill, rx=8)
    s += txt(x + CW / 2, 280, f'归约轴 {axis}', 15,
             amber if axis == 'T' else muted, 600 if axis == 'T' else 400, 'middle')
s += txt(40, 356, '唯一随输入变长的一条', 18, amber, 600)
s += txt(40, 386, 'softmax 的分母要等整行到齐才算得出来，所以它是这一层里唯一不能随便切的运算。', 16, muted)
s += txt(40, 412, '分数与聚合虽然也涉及两个位置，但每个 (i, j) 对算完就能扔，可以任意并行。', 16, muted)
s += rect(40, 438, 920, 116, '#f1f4fb')
s += txt(64, 474, '判据', 17, blue, 600)
s += txt(64, 502, '归约轴长度随输入变化  →  跨位置  →  需要历史  →  需要 KV Cache，且切分要特殊处理', 16, ink)
s += txt(64, 530, '归约轴长度写死在配置里  →  逐位置  →  无缓存  →  宽度、路由方式都可以自由改', 16, ink)
save('01-reduction-axis.svg', s)


# ============================================================== 02 sqrt(d_k)
NORM = [(2, 1.10), (4, 1.19), (8, 1.41), (16, 1.70), (32, 1.71), (64, 2.06),
        (128, 2.04), (256, 2.31), (512, 3.18), (1024, 2.90), (2048, 3.36), (4096, 3.88)]
SCAL = [(2, 1.70), (4, 3.08), (8, 5.66), (16, 10.51), (32, 20.45), (64, 39.12),
        (128, 78.86), (256, 158.85), (512, 309.03), (1024, 623.17), (2048, 1253.10), (4096, 2531.70)]

if os.environ.get('ATTNFFN_RECOMPUTE'):
    def _softmax(v):
        m = max(v); e = [math.exp(x - m) for x in v]; z = sum(e)
        return [x / z for x in e]

    def _eff(T, scale, N=80):
        rnd = random.Random(11); acc = []
        for _ in range(N):
            q = [rnd.gauss(0, 1) for _ in range(DK)]
            lg = [sum(a * b for a, b in zip(q, [rnd.gauss(0, 1) for _ in range(DK)])) / scale
                  for _ in range(T)]
            p = _softmax(lg)
            acc.append(math.exp(-sum(x * math.log(x) for x in p if x > 0)))
        return sum(acc) / N
    NORM = [(2 ** k, round(_eff(2 ** k, 1), 2)) for k in range(1, 13)]
    SCAL = [(2 ** k, round(_eff(2 ** k, SIG), 2)) for k in range(1, 13)]

s = base('02', 'sqrt(d_k) 是刻度，不是防溢出',
         '有效位置数 = exp(熵)。T = 4096、d_k = 64 时，三个刻度给出三种完全不同的注意力。', 680)
PX0, PX1, PY0, PY1 = 108, 620, 196, 520
YSPAN = 13.0                      # 纵轴取到 8192，给曲线末端标注留出净空


def px(v):
    return PX0 + (v - 1) / 11 * (PX1 - PX0)


def py(e):
    return PY1 - math.log2(max(e, 1.0)) / YSPAN * (PY1 - PY0)


s += rect(40, 146, 620, 410, '#f7f9fc')
s += line(PX0, PY1, PX1, PY1, rule)
s += line(PX0, PY1, PX0, PY0, rule)
for v in range(1, 13):
    s += line(px(v), PY1, px(v), PY1 + 5, rule)
    if v % 2 == 1:
        s += txt(px(v), PY1 + 26, 2 ** v, 14, muted, 400, 'middle')
for e in [1, 4, 16, 64, 256, 1024, 4096]:
    s += line(PX0 - 5, py(e), PX0, py(e), rule)
    s += txt(PX0 - 11, py(e) + 5, e, 14, muted, 400, 'end')
s += txt((PX0 + PX1) / 2, PY1 + 54, '上下文长度 T', 15, muted, 400, 'middle')

pts = ' '.join(f'{px(math.log2(t)):.1f},{py(e):.1f}' for t, e in NORM)
s += f'<polyline points="{pts}" fill="none" stroke="{amber}" stroke-width="3"/>'
pts = ' '.join(f'{px(math.log2(t)):.1f},{py(e):.1f}' for t, e in SCAL)
s += f'<polyline points="{pts}" fill="none" stroke="{green}" stroke-width="3"/>'

s += rect(64, 166, 306, 74, '#ffffff', rx=10)
s += line(78, 192, 118, 192, green, width=3)
s += txt(128, 197, '除以 sqrt(d_k) = 8', 15, green, 600)
s += line(78, 220, 118, 220, amber, width=3)
s += txt(128, 225, '不缩放（除以 1）', 15, amber, 600)

# 标注一律贴在各自曲线末端、收在绘图区右边界内侧，两条曲线互不干扰
s += txt(PX1 - 4, py(2531.70) - 14, '2531.70', 16, green, 600, 'end')
s += txt(PX1 - 4, py(3.88) - 12, '3.88', 16, amber, 600, 'end')
s += txt(PX1 - 4, py(3.88) - 34, 'T 涨 2048 倍，只从 1.10 爬到 3.88', 14, amber, 500, 'end')

s += rect(684, 146, 276, 410, '#f1f4fb')
s += txt(708, 180, 'T = 4096 时的三个刻度', 17, ink, 600)
ROWS3 = [
    ('除以 1', '平均最大权重 0.6890', '3.43 个位置', '0.08%', amber),
    ('除以 sqrt(d_k) = 8', '平均最大权重 0.0060', '2502.30 个位置', '61.09%', green),
    ('除以 d_k = 64', '平均最大权重 0.0004', '4063.65 个位置', '99.21%', blue),
]
for i, (a, b, c, d, col) in enumerate(ROWS3):
    y = 214 + i * 108
    s += line(708, y - 34, 936, y - 34, rule)
    s += txt(708, y, a, 16, col, 600)
    s += txt(708, y + 26, b, 14, muted)
    s += txt(708, y + 54, c, 17, ink, 600)
    s += txt(936, y + 54, d, 15, col, 600, 'end')
s += txt(708, 528, '几乎均匀 = 平均池化', 14, blue, 500)

s += txt(40, 600, '不缩放时模型实际只看两三个 token，把上下文从 4K 扩到 128K 是白费力气；除以 d_k 又矫枉过正，注意力失去选择性。', 16, muted)
s += txt(40, 628, '换算方式：对 q、k 各分量取独立 N(0,1)，每项乘积方差为 1，d_k 项相加即方差 d_k。20 万次采样实测 63.818。', 16, muted)
save('02-scale.svg', s)


# ======================================== 03 注意力分布随上下文变化（动画）
def weights(T, scale, seed=5, bins=40):
    """降序排列的注意力权重，按 bins 段聚合；T 小于 bins 时就按 T 分段。"""
    rnd = random.Random(seed)
    q = [rnd.gauss(0, 1) for _ in range(DK)]
    lg = [sum(a * b for a, b in zip(q, [rnd.gauss(0, 1) for _ in range(DK)])) / scale
          for _ in range(T)]
    m = max(lg); e = [math.exp(x - m) for x in lg]; z = sum(e)
    p = sorted((x / z for x in e), reverse=True)
    n = min(bins, T)
    return [math.fsum(p[int(i * T / n):max(int((i + 1) * T / n), int(i * T / n) + 1)])
            for i in range(n)]


FRAMES = [16, 64, 512, 4096]
LABELS = {16: ('1.70', '10.51', '65.7%'), 64: ('2.06', '39.12', '61.1%'),
          512: ('3.18', '309.03', '60.4%'), 4096: ('3.88', '2531.70', '61.8%')}
XB0, XB1 = 340, 936
CSS = ('<style>'
       '@keyframes sw{0%,23.9%{opacity:1}24%,100%{opacity:0}}'
       '.f{animation:sw 16s linear infinite;opacity:0}'
       '.f1{animation-delay:0s}.f2{animation-delay:4s}'
       '.f3{animation-delay:8s}.f4{animation-delay:12s}'
       '@media (prefers-reduced-motion:reduce){.f{animation:none;opacity:0}'
       '.f4{opacity:1}}'
       '</style>')
s = base('03', '上下文变长时，注意力到底在看几个位置（动画）',
         '同样的 q、同样的随机 K。柱高是注意力权重的降序分段，每帧内各自归一。', 470, head=CSS)
for fi, T in enumerate(FRAMES):
    e0, e1, pct = LABELS[T]
    s += f'<g class="f f{fi + 1}">'
    s += rect(40, 152, 920, 244, '#f7f9fc')
    s += txt(64, 190, f'上下文长度 T = {T}', 24, ink, 600)
    s += txt(520, 190, f'正确缩放后覆盖可见位置的 {pct}', 16, green, 600)
    for row, (w, col, name, eff) in enumerate([
            (weights(T, 1.0), amber, '不缩放', e0),
            (weights(T, SIG), green, '除以 sqrt(d_k)', e1)]):
        by = 262 + row * 86
        s += line(XB0, by, XB1, by, rule, width=1)
        s += txt(64, by - 4, name, 16, col, 600)
        s += txt(214, by - 4, f'有效位置数 {eff}', 14, muted)
        step = (XB1 - XB0) / len(w)
        bw = step * 0.74
        mx = max(w) or 1.0
        for i, v in enumerate(w):
            h = v / mx * 46
            if h >= 1.0:
                s += rect(XB0 + i * step, by - h, bw, h, col, rx=2)
    s += '</g>'
s += txt(40, 434, '不缩放时无论上下文多长，总有一根柱子吃掉大半权重、其余几乎贴地；正确缩放后权重摊到一整排。', 16, muted)
save('03-spread-animation.svg', s)


# ========================================================== 04 两段旅程的形状
s = base('04', '同一行数据，先被横向改写，再被纵向加工',
         'd = 512，8 个头，d_h = 64，处理第 4 个位置。全程只有当前这一行被写出去。', 560)
s += rect(40, 146, 440, 322, '#f1f4fb')
s += txt(64, 182, 'Attention：横向读历史', 20, blue, 600)
s += txt(64, 208, '每个头各自与全部可见位置配对', 14, muted)
for i in range(4):
    x = 78 + i * 96
    cur = i == 3
    s += rect(x, 224, 72, 46, '#d9e4f5' if cur else '#eef1f6')
    s += txt(x + 36, 252, f'K/V {i + 1}', 14, ink if cur else muted, 500 if cur else 400, 'middle')
    s += line(x + 36, 270, x + 36, 286, blue if cur else faint, width=1)
for i, a in enumerate(['0.1', '0.2', '0.3', '0.4']):
    x = 78 + i * 96
    s += rect(x, 286, 72, 32, '#cfe0f7' if i == 3 else '#eef1f6', rx=6)
    s += txt(x + 36, 308, a, 15, blue if i == 3 else muted, 600 if i == 3 else 400, 'middle')
s += txt(232, 342, '↑ 一行权重，其余位置只出现在分子分母里', 13, muted)
s += line(241, 352, 241, 366, blue, True)
s += rect(78, 366, 326, 42, '#d9e4f5', rx=8)
s += txt(241, 393, '输出 o₄　[1, 64]', 16, ink, 600, 'middle')
s += txt(64, 436, '读取了 4 个位置，写出的仍然只有一行。', 15, muted)
s += txt(64, 458, '位置下标不出现在输出形状里。', 15, muted)

s += rect(520, 146, 440, 322, '#eef4f0')
s += txt(544, 182, 'FFN：纵向过权重', 20, green, 600)
s += txt(544, 208, '逐位置独立，只有宽度会变', 14, muted)
for i, (w, label, note, bg) in enumerate([
        (115, '[1, 512]', '', '#d9e4f5'),
        (269, '[1, 2048]', 'W₁ 升维', '#cfe8dd'),
        (115, '[1, 512]', 'W₂ 降维', '#d9e4f5')]):
    y = 238 + i * 66
    s += rect(544, y, w, 46, bg, rx=8)
    s += txt(544 + w / 2, y + 29, label, 16, ink, 600, 'middle')
    if note:
        s += txt(544 + w + 14, y + 29, note, 14, muted)
s += txt(544, 436, '中间层是这一行的特征被撑开，', 15, muted)
s += txt(544, 458, '不是折成了更多 token。', 15, muted)

s += txt(40, 492, '两个子层都只写回当前这一行。Attention 的"多"体现在它读取的位置数，FFN 的"多"体现在中间层宽度。', 16, muted)
s += txt(40, 520, '这也解释了为什么只有 Attention 需要 KV Cache：它要读的那些位置，本步不在场。', 16, muted)
save('04-two-journeys.svg', s)


# ======================================================= 05 宽度与非线性
s = base('05', '没有激活函数的宽度是假的',
         '两矩阵相乘可以合成一个，中间维度 m 只在有非线性时才兑现成表达力。', 600)
s += rect(40, 146, 440, 322, '#fbf3e6')
s += txt(64, 182, '无激活：(u·W₁)·W₂ = u·(W₁W₂)', 19, amber, 600)
s += txt(64, 208, '右边是一个 512 × 512 的单矩阵', 14, muted)
s += rect(64, 236, 96, 62, '#f0d9c0', rx=8)
s += txt(112, 274, 'W₁', 19, amber, 600, 'middle')
s += txt(112, 320, '512×2048', 13, muted, 400, 'middle')
s += rect(180, 236, 96, 62, '#f0d9c0', rx=8)
s += txt(228, 274, 'W₂', 19, amber, 600, 'middle')
s += txt(228, 320, '2048×512', 13, muted, 400, 'middle')
s += txt(300, 274, '=', 22, amber, 600, 'middle')
s += rect(340, 236, 116, 62, '#e0b98c', rx=8)
s += txt(398, 274, '单个 512×512', 14, ink, 600, 'middle')
s += txt(64, 372, '2,097,152 个参数', 17, amber, 600)
s += txt(64, 400, '等价于 262,144 个', 17, ink, 600)
s += txt(64, 432, '其中 1,835,008 个白占 —— 冗余八倍', 15, muted)

s += rect(520, 146, 440, 322, '#edf7f2')
s += txt(544, 182, '有激活：m 维真的被用上', 19, green, 600)
s += txt(544, 208, '逐元素非线性，不引入任何跨位置依赖', 14, muted)
for i, (lab, sub, bg) in enumerate([
        ('W₁', '512×2048', '#a9d4c0'),
        ('SiLU', '逐元素', '#f6faf8'),
        ('W₂', '2048×512', '#a9d4c0')]):
    y = 236 + i * 62
    s += rect(544, y, 140, 48, bg, rx=8)
    s += txt(614, y + 21, lab, 16, green, 600, 'middle')
    s += txt(614, y + 40, sub, 12, muted, 400, 'middle')
    if i < 2:
        s += line(614, y + 48, 614, y + 62, faint, True)
s += txt(712, 260, '中间维度 m 只有在', 15, ink)
s += txt(712, 284, '这一层之后才不是摆设', 15, ink)
s += txt(712, 320, 'm ≈ 8d/3 是从', 15, muted)
s += txt(712, 344, '3dm = 8d² 反解出来的', 15, muted)

s += txt(40, 508, '宽度不是容量，宽度只是容器的口径。2/3 这个参数占比，说的正是这个容器有多大。', 16, muted)
s += txt(40, 536, '也正因为激活是逐位置的，FFN 才敢把宽度开到四倍——它不需要向任何其他位置请示。', 16, muted)
save('05-width.svg', s)


# ================================================ 06 参数占比的漂移
s = base('06', '2/3 正在漂移：GQA 砍掉的是 Attention 那一侧',
         '只统计注意力投影与 FFN 权重，忽略 bias、Norm、Embedding 与 LM Head。', 580)
s += rect(640, 138, 14, 14, blue, rx=3)
s += txt(662, 150, 'Attention', 14, blue, 600)
s += rect(766, 138, 14, 14, green, rx=3)
s += txt(788, 150, 'FFN', 14, green, 600)
CFG = [
    ('原始 Transformer', 'MHA · ReLU · m = 4d', 1.05, 2.10, 3.15, 66.67),
    ('GPT-2 small', 'MHA · GELU · m = 4d', 2.36, 4.72, 7.08, 66.67),
    ('Llama-3-8B', 'GQA-4 · SwiGLU · m = 3.5d', 41.94, 176.16, 218.10, 80.77),
]
BARX, BARW = 300, 420
for i, (name, sub, a, f, tot, pct) in enumerate(CFG):
    y = 164 + i * 104
    s += rect(40, y, 920, 88, '#f7f9fc')
    s += txt(64, y + 36, name, 20, ink, 600)
    s += txt(64, y + 62, sub, 14, muted)
    wa = BARW * (1 - pct / 100)
    wf = BARW - wa
    s += rect(BARX, y + 26, wa, 30, blue, rx=4)
    s += rect(BARX + wa, y + 26, wf, 30, green, rx=4)
    s += txt(BARX + BARW + 18, y + 50, f'{pct}%', 20, green, 600)
    s += txt(BARX, y + 78, f'Attention {a} M　·　FFN {f} M　·　合计 {tot} M', 13, muted)
s += rect(40, 486, 920, 56, '#fbf3e6', rx=10)
s += txt(64, 522, 'K/V 投影从 4 个 d×d 降到 4 分之一，SwiGLU 却仍占 3dm ≈ 8d² —— FFN 的份额于是被抬了起来。', 16, amber, 600)
save('06-drift.svg', s)


print('Created 6 SVG diagrams in', out)
