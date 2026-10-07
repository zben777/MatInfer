"""Regenerate the online-softmax diagrams (editable SVG sources).

SVG generation uses only the Python standard library, matching
tools/generate_norm_figures.py. The animated SVG (06) plays when the file is
opened directly in a browser; inside an <img> tag it shows its first frame.
"""
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parents[1] / 'assets/images/attention'
out.mkdir(parents=True, exist_ok=True)

ink = '#24334a'; muted = '#6d7c90'; green = '#218575'
blue = '#4967b0'; amber = '#c88735'; rule = '#d8e0e9'; faint = '#b3becc'


def txt(x, y, s, size=18, color=ink, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')


def rect(x, y, w, h, fill='#f3f6fa', stroke='none', rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x, y, x2, y2, color=faint, arrow=False, width=2):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    return f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>'


def bar(x, baseline, w, h, fill):
    return rect(x, baseline - h, w, h, fill, rx=4)


def base(num, title, sub, h, head=''):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {h}" role="img" '
            f'aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(sub)}</desc>'
            f'<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
            f'orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="{faint}"/></marker></defs>'
            f'{head}'
            f'<g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">'
            + rect(0, 0, 1000, h, '#ffffff', rx=0)
            + txt(40, 42, f'MATINFER / ATTENTION / {num}', 13, green, 600)
            + txt(40, 85, title, 28, ink, 600)
            + txt(40, 119, sub, 17, muted))


def save(name, s):
    (out / name).write_text(s + '</g></svg>')


# ---------------------------------------------------------------- 01 访存次数
s = base('01', '为什么要少读一遍：Safe Softmax 要读三次',
         '同一行分数被完整读取三遍，才能给出答案；online normalizer 把三遍压成一遍。', 560)
s += rect(40, 150, 440, 340, '#f7f9fc')
s += txt(64, 188, 'Safe Softmax（框架默认）', 20, ink, 600)
s += txt(64, 214, '三遍都建立在"先知道全局 maxima"之上', 15, muted)
for i, (a, b) in enumerate([('第 1 遍　读 S → 求最大值 m', '1 读'),
                            ('第 2 遍　读 S → 求指数和 ℓ', '1 读'),
                            ('第 3 遍　读 S → 写出权重 P', '1 读 + 1 写')]):
    y = 232 + i * 62
    s += rect(64, y, 392, 54, '#eef1f6', rx=8)
    s += txt(82, y + 33, a, 18, ink)
    s += txt(438, y + 33, b, 16, blue, 500, 'end')
s += rect(64, 424, 392, 46, '#f1f4fb', rx=8)
s += txt(82, 453, '每元素 4 次访存（3 读 + 1 写）', 18, blue, 600)

s += rect(520, 150, 440, 340, '#f7f9fc')
s += txt(544, 188, 'Online Normalizer（NVIDIA, 2018）', 19, ink, 600)
s += txt(544, 214, '一趟读完，边走边修正', 15, muted)
s += rect(544, 232, 392, 178, '#edf7f2', rx=8)
s += txt(564, 270, '读一块 S，立即更新两个状态', 18, green, 600)
s += txt(564, 306, 'm ← max(m, 块内最大值)', 17)
s += txt(564, 336, 'ℓ ← exp(m旧 − m新) · ℓ + Σ exp(s − m新)', 17)
s += txt(564, 372, '块内数据用完即弃，不回读', 16, muted)
s += txt(564, 398, '原始分数不必保存', 16, muted)
s += rect(544, 424, 392, 46, '#f1f4fb', rx=8)
s += txt(562, 453, '每元素 3 次访存（2 读 + 1 写）', 18, green, 600)

s += txt(40, 530, '论文在 V100 上实测：Softmax 最多 1.3×，与 TopK 融合最多 5×。省下的只是访存，不是计算。', 17, muted)
save('01-memory-access.svg', s)


# ------------------------------------------------------- 02 换基准只是乘常数
s = base('02', '换基准只是整体乘一个常数',
         '块 1 = [ln 2, ln 4]。基准从 ln 4 抬到 ln 8，每个指数项乘同一个系数，比值不变。', 560)
s += rect(40, 140, 390, 380, '#f7f9fc')
s += txt(64, 174, '以 m = ln 4 为基准', 19, ink, 600)
s += txt(64, 200, '块 1 的两个指数项', 15, muted)
s += line(64, 400, 406, 400, rule)
s += bar(72, 400, 80, 45, blue) + txt(112, 341, '½', 18, blue, 500, 'middle')
s += bar(192, 400, 80, 90, blue) + txt(232, 296, '1', 18, blue, 500, 'middle')
s += txt(112, 426, 'ln 2', 15, muted, 400, 'middle')
s += txt(232, 426, 'ln 4', 15, muted, 400, 'middle')
s += txt(64, 462, 'ℓ = ½ + 1 = 1.5', 19, blue, 600)
s += txt(64, 492, 'U = ½·1 + 1·3 = 3.5', 19, green, 600)

s += rect(570, 140, 390, 380, '#f7f9fc')
s += txt(594, 174, '以 m = ln 8 为基准', 19, ink, 600)
s += txt(594, 200, '同一块数据，没变', 15, muted)
s += line(594, 400, 936, 400, rule)
s += bar(602, 400, 80, 22.5, blue) + txt(642, 365, '¼', 18, blue, 500, 'middle')
s += bar(722, 400, 80, 45, blue) + txt(762, 343, '½', 18, blue, 500, 'middle')
s += txt(642, 426, 'ln 2', 15, muted, 400, 'middle')
s += txt(762, 426, 'ln 4', 15, muted, 400, 'middle')
s += txt(594, 462, 'ℓ = ¼ + ½ = 0.75', 19, blue, 600)
s += txt(594, 492, '0.75 = 1.5 × ½', 19, green, 600)

s += line(447, 380, 563, 380, blue, True)
s += txt(505, 352, '× ½', 24, blue, 600, 'middle')
s += txt(505, 414, '= e^(ln4 − ln8)', 15, muted, 400, 'middle')
s += txt(40, 545, '比值不变（½ : 1 = ¼ : ½），所以 ℓ 与 U 乘同一个系数就能整体搬到新基准；原始分数不必保留。', 17, muted)
save('02-rebase-scaling.svg', s)


# ------------------------------------------------------------ 03 三块合并过程
s = base('03', '三块合并：ℓ 和 U 怎么走',
         '分数 [ln 2, ln 4, ln 8, ln 4, ln 2]、Value [1, 3, 5, 7, 9]，按 2 + 2 + 1 分块，最终应为 O = 5。', 610)
cols = [
    (40, '块 1', '[ln 2, ln 4]', 'm = ln 4', 'ℓ = 1.5', 'U = 3.5'),
    (375, '块 2', '[ln 8, ln 4]', 'm = ln 8（更高）', 'ℓ = 1.5', 'U = 8.5'),
    (710, '块 3', '[ln 2]', 'm = ln 2（更低）', 'ℓ = 1', 'U = 9'),
]
for x, name, score, m, l, u in cols:
    s += rect(x, 150, 250, 192, '#f1f4fb', rx=10)
    s += txt(x + 20, 182, name, 20, ink, 600)
    s += txt(x + 20, 210, score, 17, muted)
    s += txt(x + 20, 248, m, 17)
    s += txt(x + 20, 280, l, 18, blue, 500)
    s += txt(x + 20, 310, u, 18, green, 500)

s += line(292, 246, 371, 246, amber, True)
s += txt(331, 226, 'α = ½', 17, amber, 600, 'middle')
s += txt(331, 274, '缩旧状态', 14, muted, 400, 'middle')
s += line(627, 246, 706, 246, amber, True)
s += txt(666, 226, 'β = ¼', 17, amber, 600, 'middle')
s += txt(666, 274, '缩新块', 14, muted, 400, 'middle')

s += txt(40, 372, '合并前两块后可以自查一次：O = 10.25 / 2.25 = 41 / 9 ≈ 4.556', 16, muted)
s += rect(40, 390, 920, 78, '#eef7f2', rx=10)
s += txt(500, 438, 'm = ln 8　·　ℓ = 2.5　·　U = 12.5　→　O = 12.5 / 2.5 = 5', 24, ink, 600, 'middle')

s += rect(40, 494, 920, 76, '#fbf3e6', rx=10)
s += txt(64, 524, '漏掉 β 会怎样', 18, amber, 600)
s += txt(64, 552, '直接把 ℓ₃ = 1 加到 2.25 上 → ℓ = 3.25 → 结果错；而且不报错、形状完全正确，只能靠数值对比发现。', 17)
save('03-block-merge.svg', s)


# --------------------------------------------------------- 04 串行 vs 树形合并
s = base('04', '结合律：从"只能顺序推进"到"可以分组并行"',
         '合并运算满足 (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)，所以四块可以两两先算，最后规约一次。', 650)
s += rect(40, 140, 920, 170, '#f7f9fc')
s += txt(64, 176, '顺序合并', 19, ink, 600)
s += txt(64, 202, '一次只推进一块，后面必须等前面', 15, muted)
for i, x in enumerate([64, 286, 508, 730]):
    s += rect(x, 226, 160, 50, '#eef1f6', rx=8)
    s += txt(x + 80, 257, f'块 {i + 1}', 18, ink, 500, 'middle')
    if i < 3:
        s += line(x + 164, 251, x + 218, 251, faint, True)

s += rect(40, 336, 920, 268, '#edf7f2')
s += txt(64, 372, '结合律允许的走法', 19, ink, 600)
s += txt(64, 398, '两组的中间状态互不依赖，谁先算完都不影响结果', 15, muted)
for i, x in enumerate([140, 340, 540, 740]):
    s += rect(x, 418, 160, 46, '#dceee6', rx=8)
    s += txt(x + 80, 447, f'块 {i + 1}', 17, ink, 500, 'middle')
s += rect(110, 496, 220, 46, '#cfe8dd', rx=8)
s += txt(220, 525, '状态 A = 1 ⊕ 2', 17, green, 600, 'middle')
s += rect(710, 496, 220, 46, '#cfe8dd', rx=8)
s += txt(820, 525, '状态 B = 3 ⊕ 4', 17, green, 600, 'middle')
s += rect(290, 562, 420, 46, '#b9dccb', rx=8)
s += txt(500, 591, 'A ⊕ B → (m, ℓ, U) → O', 18, ink, 600, 'middle')
s += line(220, 466, 220, 494, faint, True)
s += line(400, 466, 250, 494, faint, True)
s += line(820, 466, 820, 494, faint, True)
s += line(640, 466, 790, 494, faint, True)
s += line(230, 544, 450, 560, faint, True)
s += line(810, 544, 550, 560, faint, True)
s += txt(40, 634, '规约一次只需要 1 次 max 和 2 次乘加——切分几乎不需要犹豫，分块策略的自由度就是这么来的。', 17, muted)
save('04-associativity.svg', s)


# ------------------------------------------------------- 05 softmax 与 welford
s = base('05', '这类算法有同一个形状：基准 + 累积量 + 修正项',
         '修正项是否存在、有多大，只由两个基准之差决定；差值归零，修正项随之消失。', 620)
s += rect(40, 140, 450, 350, '#f1f4fb')
s += txt(64, 178, 'Softmax 的归一化', 21, blue, 600)
s += txt(64, 204, 'online normalizer / FlashAttention', 15, muted)
rows_l = [('缺的全局量', '整行最大值'),
          ('状态', '(m, ℓ)'),
          ('合并', 'ℓ ← exp(m₁ − m)·ℓ₁ + exp(m₂ − m)·ℓ₂'),
          ('修正项', 'exp(m₁ − m)'),
          ('何时消失', '两个基准相同（实际几乎不会）')]
for i, (a, b) in enumerate(rows_l):
    y = 250 + i * 48
    s += txt(64, y, a, 15, muted)
    s += txt(180, y, b, 16, ink)

s += rect(510, 140, 450, 350, '#edf7f2')
s += txt(534, 178, '在线计算方差（Welford）', 21, green, 600)
s += txt(534, 204, '论文自承的灵感来源', 15, muted)
rows_r = [('缺的全局量', '整组数据的均值'),
          ('状态', '(n, x̄, M₂)'),
          ('合并', 'M₂ ← M₂ₐ + M₂ᵦ + (nₐnᵦ/n)·δ²'),
          ('修正项', '(nₐnᵦ/n)·δ²，δ = x̄ᵦ − x̄ₐ'),
          ('何时消失', 'δ = 0，两组均值相同')]
for i, (a, b) in enumerate(rows_r):
    y = 250 + i * 48
    s += txt(534, y, a, 15, muted)
    s += txt(650, y, b, 16, ink)

s += rect(40, 512, 920, 78, '#f7f9fc')
s += txt(64, 544, '两个算法的修正项，都由两个基准之差唯一决定', 19, ink, 600)
s += txt(64, 572, '实测：两组数均值相差 9.5 时，修正项占最终 M₂ 的 95.7%——差得越远，越不能省掉这一步。', 17, muted)
save('05-online-algorithm-shape.svg', s)


# ------------------------------------------------------------ 06 合并过程动画
STEPS = [
    ('① 读入块 1', 'm = ln 4　ℓ = 1.5　U = 3.5', '块内最大值成为初始基准', '#f1f4fb', blue,
     [('ℓ', 1.5, 90), ('U', 3.5, 90)]),
    ('② 遇到块 2，它更大', 'm 从 ln 4 抬到 ln 8', '旧状态必须整体乘 α = ½，否则基准不一致', '#fbf3e6', amber,
     [('ℓ', 0.75, 45), ('U', 1.75, 45)]),
    ('③ 并入块 2', 'ℓ = 0.75 + 1.5 = 2.25　U = 1.75 + 8.5 = 10.25', '此时 O = 41 / 9 ≈ 4.556', '#f1f4fb', blue,
     [('ℓ', 2.25, 90), ('U', 10.25, 90)]),
    ('④ 遇到块 3，它更小', 'm 保持 ln 8', '这次要缩的是新块，β = ¼；漏掉这一步不会报错', '#fbf3e6', amber,
     [('ℓ', 2.25, 90), ('U', 10.25, 90)]),
    ('⑤ 合并完成', 'ℓ = 2.5　U = 12.5　O = 5', '与一次性计算完全一致', '#eef7f2', green,
     [('ℓ', 2.5, 90), ('U', 12.5, 90)]),
]
N = len(STEPS)
CSS = ('<style>'
       '@keyframes sw{0%,19.9%{opacity:1}20%,100%{opacity:0}}'
       '.f{animation:sw 10s linear infinite;opacity:0}'
       '.f1{animation-delay:0s}.f2{animation-delay:2s}.f3{animation-delay:4s}'
       '.f4{animation-delay:6s}.f5{animation-delay:8s}'
       '@media (prefers-reduced-motion:reduce){.f{animation:none;opacity:0}'
       '.f5{opacity:1}}'
       '</style>')
s = base('06', '合并过程（动画）：每一步都在为"基准差"买单',
         '五步循环播放：建立基准 → 抬基准缩旧状态 → 并入 → 缩新块 → 得到 O = 5。', 390, head=CSS)
for i, (title, state, note, bg, col, bars) in enumerate(STEPS):
    s += f'<g class="f f{i + 1}">'
    s += rect(40, 150, 920, 200, bg, rx=12)
    s += txt(64, 190, title, 24, col, 600)
    s += txt(64, 226, state, 20, ink)
    s += txt(64, 258, note, 17, muted)
    for j, (name, val, h) in enumerate(bars):
        bx = 600 + j * 170
        s += txt(bx, 306, name, 16, muted)
        s += bar(bx, 322, 110, h, col)
        s += txt(bx + 55, 322 - h - 10, f'{val:g}', 17, col, 500, 'middle')
    s += '</g>'
save('06-merge-animation.svg', s)

print('Created 6 SVG diagrams in', out)
