"""Regenerate the SwiGLU / gated-FFN diagrams (editable SVG sources).

SVG generation uses only the Python standard library, matching
tools/generate_norm_figures.py and tools/generate_attnffn_figures.py.
Figure 03 is animated with CSS @keyframes (four frames: gate branch ->
content branch -> elementwise product -> projection back).

No TeX is used inside the SVGs: nothing renders LaTeX in a plain SVG file,
so every formula here is written with Unicode.
"""
import math
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parents[1] / 'assets/images/swiglu'
out.mkdir(parents=True, exist_ok=True)

ink = '#24334a'; muted = '#6d7c90'; green = '#218575'
blue = '#4967b0'; amber = '#c88735'; rule = '#d8e0e9'; faint = '#b3becc'
GRAY = '#c8d1dc'


def txt(x, y, s, size=18, color=ink, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')


def rect(x, y, w, h, fill='#f3f6fa', stroke='none', rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x, y, x2, y2, color=faint, arrow=False, width=2, dash=''):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{m}{d}/>'


def path(d, color=faint, width=2, arrow=True):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>'


def curve(points, color, width=3):
    pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def base(num, title, sub, h, head=''):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {h}" role="img" '
            f'aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(sub)}</desc>'
            f'<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
            f'orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="{faint}"/></marker></defs>'
            f'{head}'
            f'<g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">'
            + rect(0, 0, 1000, h, '#ffffff', rx=0)
            + txt(40, 42, f'MATINFER / SWIGLU / {num}', 13, green, 600)
            + txt(40, 85, title, 28, ink, 600)
            + txt(40, 119, sub, 17, muted))


def save(name, s):
    (out / name).write_text(s + '</g></svg>')


# ------------------------------------------------------------ math helpers
def relu(z): return max(0.0, z)
def gelu(z): return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))
def silu(z): return z / (1.0 + math.exp(-z))
def phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
def sigm(z): return 1.0 / (1.0 + math.exp(-z))


# =================================================== 01 同一个模板 z * g(z)
X0, X1 = 110, 880
P1T, P1B = 186, 372          # activations panel
P2T, P2B = 470, 620          # gate panel
VLO, VHI = -0.7, 4.4
GLO, GHI = -0.03, 1.06


def px(z): return X0 + (z + 4) / 8 * (X1 - X0)
def py1(v): return P1B - (v - VLO) / (VHI - VLO) * (P1B - P1T)
def py2(g): return P2B - (g - GLO) / (GHI - GLO) * (P2B - P2T)


s = base('01', 'ReLU、GELU、SiLU 都是 z 乘上一个通过量',
         '门函数 g 是曲线里唯一变化的部分，而它始终只以 z 为自变量。', 726)
s += rect(40, 146, 920, 250, '#f7f9fc')
s += rect(40, 424, 920, 216, '#fbf6ee')

# ---- activation panel
s += line(X0, py1(0), X1, py1(0), rule)
s += line(px(0), P1B, px(0), P1T, rule)
for z in (-4, -2, 0, 2, 4):
    s += txt(px(z), py1(0) + 22, z, 14, muted, 400, 'middle')
s += txt((X0 + X1) / 2, P1B + 44, 'z', 15, muted, 400, 'middle')
for fn, col in ((relu, amber), (gelu, blue), (silu, green)):
    s += curve([(px(x / 60), py1(fn(x / 60))) for x in range(-240, 241)], col, 3)
s += txt(64, 176, '激活本身', 16, ink, 600)
s += rect(566, 156, 358, 34, '#ffffff', rx=10)
s += line(580, 173, 612, 173, amber, width=3)
s += txt(620, 178, 'ReLU', 15, amber, 600)
s += line(700, 173, 732, 173, blue, width=3)
s += txt(740, 178, 'GELU', 15, blue, 600)
s += line(820, 173, 852, 173, green, width=3)
s += txt(860, 178, 'SiLU', 15, green, 600)

# ---- gate panel
s += line(X0, py2(0), X1, py2(0), rule)
s += line(px(0), P2B, px(0), P2T, rule)
for z in (-4, -2, 0, 2, 4):
    s += txt(px(z), py2(0) + 22, z, 14, muted, 400, 'middle')
s += txt((X0 + X1) / 2, P2B + 44, 'z', 15, muted, 400, 'middle')
s += txt(X0 - 12, py2(1) + 5, '1', 14, muted, 400, 'end')
s += txt(X0 - 12, py2(0) + 5, '0', 14, muted, 400, 'end')
s += txt(64, 452, '门函数 g', 16, ink, 600)
s += (f'<polyline points="{px(-4):.1f},{py2(0):.1f} {px(0):.1f},{py2(0):.1f} '
      f'{px(0):.1f},{py2(1):.1f} {px(4):.1f},{py2(1):.1f}" fill="none" '
      f'stroke="{amber}" stroke-width="3"/>')
s += curve([(px(x / 60), py2(phi(x / 60))) for x in range(-240, 241)], blue, 3)
s += curve([(px(x / 60), py2(sigm(x / 60))) for x in range(-240, 241)], green, 3)
# 图例落在左上角空白区：x<470 处三条门曲线都贴近 0
s += rect(132, 436, 336, 108, '#ffffff', rx=10)
s += line(150, 464, 182, 464, amber, width=3)
s += txt(192, 469, 'ReLU：门 = 1{z>0}（阶跃）', 14, amber, 600)
s += line(150, 496, 182, 496, blue, width=3)
s += txt(192, 501, 'GELU：门 = Φ(z)', 14, blue, 600)
s += line(150, 528, 182, 528, green, width=3)
s += txt(192, 533, 'SiLU：门 = σ(z)', 14, green, 600)
s += txt(40, 682, '三条曲线共用同一个模板 z·g(z)：z 一律乘上一个不超过 1 的因子，区别只在这个因子从哪种形状里取出来。', 16, muted)
s += txt(40, 710, 'z = 1 处三个门依次是 1、0.8413、0.7311 —— 平滑激活放行的比例更小，这正是它比 ReLU "软"的地方。', 16, muted)
save('01-one-template.svg', s)


# ================================================= 02 门的两个来源
s = base('02', '门是从值里读出来的，还是另算出来的',
         '左边一条投影，门搭在中间值上；右边两条投影，门由自己那块权重生产。', 648)
s += rect(40, 146, 440, 400, '#fdf6ec')
s += rect(520, 146, 440, 400, '#eef4f0')
s += txt(64, 182, '普通 FFN', 21, amber, 600)
s += txt(64, 208, '一块投影，门只能读值', 14, muted)
cx = 260
s += rect(cx - 46, 232, 92, 46, '#ffffff', rx=8)
s += txt(cx, 261, 'x', 17, ink, 600, 'middle')
s += line(cx, 278, cx, 302, faint, True)
s += rect(cx - 74, 304, 148, 46, '#f0d9c0', rx=8)
s += txt(cx, 333, 'W₁', 17, amber, 600, 'middle')
s += line(cx, 350, cx, 374, faint, True)
s += rect(cx - 58, 376, 116, 46, '#ffffff', rx=8)
s += txt(cx, 405, 'z = W₁x', 15, ink, 600, 'middle')
s += line(cx, 422, cx, 446, faint, True)
s += rect(cx - 88, 448, 176, 46, '#f0d9c0', rx=8)
s += txt(cx, 477, 'g(z)   →   W₂', 15, amber, 600, 'middle')
s += txt(cx, 522, '门 = g(z)　一个数管两件事', 14, amber, 600, 'middle')

cx = 740
s += txt(544, 182, 'GLU', 21, green, 600)
s += txt(544, 208, '两条投影，门自己算', 14, muted)
s += rect(cx - 46, 232, 92, 46, '#ffffff', rx=8)
s += txt(cx, 261, 'x', 17, ink, 600, 'middle')
s += path('M716 278 L716 300 L600 300 L600 314', faint)
s += path('M764 278 L764 300 L880 300 L880 314', faint)
s += rect(544, 316, 148, 46, '#a9d4c0', rx=8)
s += txt(618, 345, 'W_up x', 15, ink, 600, 'middle')
s += rect(788, 316, 148, 46, '#a9d4c0', rx=8)
s += txt(862, 345, 'W_gate x', 15, ink, 600, 'middle')
s += line(618, 362, 618, 390, faint, True)
s += line(862, 362, 862, 390, faint, True)
s += rect(544, 392, 148, 44, '#ffffff', rx=8)
s += txt(618, 420, '内容 a', 15, ink, 600, 'middle')
s += rect(788, 392, 148, 44, '#cfe8dd', rx=8)
s += txt(862, 420, 'SiLU(b)', 15, green, 600, 'middle')
s += path('M618 436 L618 470 L660 470', faint)
s += path('M862 436 L862 470 L820 470', faint)
s += rect(660, 448, 160, 44, '#a9d4c0', rx=8)
s += txt(740, 476, '⊙  →  W_down', 15, ink, 600, 'middle')
s += txt(740, 522, '门 = σ(W_gate x)　独立于 a', 14, green, 600, 'middle')
s += txt(40, 600, '两条路都逐位置，FFN 的位置独立性没有被碰过。差别只在门的出身：一个搭在 z 上，一个由 W_gate 专门生产。', 16, muted)
s += txt(40, 628, '右边那块多出来的权重，就是"让门能独立于值去学"所付的全部代价。', 16, muted)
save('02-two-sources.svg', s)


# ============================== 03 SwiGLU 数据流（四帧动画）
CSS = ('<style>'
       '@keyframes sw{0%,23.9%{opacity:1}24%,100%{opacity:0}}'
       '.f{animation:sw 16s linear infinite;opacity:0}'
       '.f1{animation-delay:0s}.f2{animation-delay:4s}'
       '.f3{animation-delay:8s}.f4{animation-delay:12s}'
       '@media (prefers-reduced-motion:reduce){.f{animation:none;opacity:0}'
       '.f4{opacity:1}}'
       '</style>')


def flow(active):
    def c(name, vivid):
        return vivid if name in active else GRAY
    e = ''
    # 节点
    e += rect(430, 152, 140, 52, '#f3f6fa', rx=10)
    e += txt(500, 185, 'x', 19, ink, 600, 'middle')
    e += rect(230, 252, 200, 56, '#f3f6fa', rx=10)
    e += txt(330, 287, 'W_gate x', 17, ink, 600, 'middle')
    e += rect(570, 252, 200, 56, '#f3f6fa', rx=10)
    e += txt(670, 287, 'W_up x', 17, ink, 600, 'middle')
    e += rect(430, 448, 140, 56, '#f3f6fa', rx=10)
    e += txt(500, 484, '⊙', 22, ink, 600, 'middle')
    e += rect(400, 548, 200, 54, '#f3f6fa', rx=10)
    e += txt(500, 582, 'W_down', 17, ink, 600, 'middle')
    e += rect(430, 646, 140, 50, '#f3f6fa', rx=10)
    e += txt(500, 679, 'y', 19, ink, 600, 'middle')
    # 高亮链上的节点
    if 'gate' in active:
        e += rect(230, 252, 200, 56, '#cfe8dd', rx=10)
        e += txt(330, 287, 'W_gate x', 17, ink, 600, 'middle')
        e += rect(230, 356, 200, 54, '#cfe8dd', rx=10)
        e += txt(330, 390, 'SiLU  →  开度', 16, ink, 600, 'middle')
    else:
        e += rect(230, 356, 200, 54, '#f3f6fa', rx=10)
        e += txt(330, 390, 'SiLU  →  开度', 16, ink, 600, 'middle')
    if 'up' in active:
        e += rect(570, 252, 200, 56, '#d9e4f5', rx=10)
        e += txt(670, 287, 'W_up x', 17, ink, 600, 'middle')
    if 'mul' in active:
        e += rect(430, 448, 140, 56, '#f0d9c0', rx=10)
        e += txt(500, 484, '⊙', 22, ink, 600, 'middle')
    if 'out' in active:
        e += rect(400, 548, 200, 54, '#cfe8dd', rx=10)
        e += txt(500, 582, 'W_down', 17, ink, 600, 'middle')
        e += rect(430, 646, 140, 50, '#cfe8dd', rx=10)
        e += txt(500, 679, 'y', 19, ink, 600, 'middle')
    # 连线
    e += path('M455 204 L455 228 L350 228 L350 250', c('gate', green))
    e += path('M545 204 L545 228 L650 228 L650 250', c('up', blue))
    e += path('M330 308 L330 354', c('gate', green))
    e += path('M330 410 L330 476 L428 476', c('mul', amber))
    e += path('M670 308 L670 476 L572 476', c('mul', amber))
    e += path('M500 504 L500 546', c('out', green))
    e += path('M500 602 L500 644', c('out', green))
    # 侧标签
    e += txt(198, 287, '门支路', 14, c('gate', green), 600, 'end')
    e += txt(802, 287, '内容支路', 14, c('up', blue), 600)
    return e


s = base('03', 'SwiGLU 的前向路径（动画）',
         '两条投影、一次逐元素相乘、一次降维。四帧依次强调：门支路 → 内容支路 → 相乘 → 投回。', 786, head=CSS)
FRAMES = [('gate', '门支路：W_gate x 过 SiLU，得到每个通道的开度', green),
          ('up', '内容支路：W_up x 给出每个通道的取值', blue),
          ('mul', '逐元素相乘：开度决定内容通过多少', amber),
          ('out', '投回原维度：W_down 把 m 维压回 d 维', green)]
for i, (key, label, col) in enumerate(FRAMES):
    s += f'<g class="f f{i + 1}">'
    s += rect(40, 146, 920, 578, '#f7f9fc')
    s += txt(64, 178, label, 16, col, 600)
    s += txt(936, 178, f'{i + 1} / 4', 14, muted, 400, 'end')
    s += flow({key})
    s += '</g>'
s += txt(40, 756, '门和内容走的是两块互不约束的权重。让哪条支路去当门是个设计选择，而"门必须另起一条支路"才是这套结构的前提。', 16, muted)
save('03-swiglu-flow-animation.svg', s)


# ============================================= 04 参数账
s = base('04', '多出来的那条投影要付钱',
         'd = 4096。同样 8d² 的参数预算，普通 FFN 用两块矩阵花掉，SwiGLU 要用三块。', 626)
REFX, REFW = 300, 330          # 8d^2 -> 330 px（留出右侧数值标注的净空）
SEGS = ['#e3b177', '#c88735', '#a86f28']
CFG = [
    ('普通 FFN', 'm = 4d = 16384', 8.0, 2, '8d² = 134.2 M', green),
    ('SwiGLU 硬撑 4d', 'm = 4d = 16384', 12.0, 3, '12d² = 201.3 M', amber),
    ('SwiGLU 缩到 8d/3', 'm = 10922.67（LLaMA 取 11008）', 8.0, 3, '3dm = 8d² = 134.2 M', green),
]
for i, (name, sub, val, nseg, note, col) in enumerate(CFG):
    y = 168 + i * 116
    s += rect(40, y, 920, 100, '#f7f9fc')
    s += txt(64, y + 34, name, 20, ink, 600)
    s += txt(64, y + 60, sub, 14, muted)
    total = REFW * val / 8.0
    segw = total / nseg
    for k in range(nseg):
        fill = col if nseg == 2 else SEGS[k]
        s += rect(REFX + k * segw + (2 if k else 0), y + 30, segw - 4, 32, fill, rx=4)
    s += txt(936, y + 52, note, 17, amber if val > 8 else green, 600, 'end')
s += line(REFX + REFW, 154, REFX + REFW, 512, amber, width=2, dash='6 5')
s += txt(REFX + REFW, 148, '8d² 预算线', 14, amber, 600, 'middle')
s += rect(40, 522, 920, 66, '#fbf6ee', rx=10)
s += txt(64, 552, 'm = 8d/3 不是抄来的常数，它是从 3dm = 8d² 反解出来的：三块矩阵的参数量，要等于原来两块。', 16, amber, 600)
s += txt(64, 578, '三块分别是门投影、内容投影与降维——前两块等大，最后一块是它们的镜像。', 16, muted)
save('04-parameter-budget.svg', s)


# ============================================= 05 闸与路
s = base('05', '一个在通道维调闸，一个在专家维择路',
         '两种"自适应"作用在不同的轴上，互不冲突，今天的模型常常一起用。', 574)
s += rect(40, 146, 440, 306, '#eef4f0')
s += rect(520, 146, 440, 306, '#f1f4fb')
s += txt(64, 182, 'SwiGLU：调闸', 20, green, 600)
s += txt(64, 208, 'm 个通道，各自一个连续开度', 14, muted)
GATES = [.92, .48, .74, .30, 1.0, .44, .86, .58, .34, .96, .52, .70]
BW, BG = 24, 7
for i, g in enumerate(GATES):
    x = 70 + i * (BW + BG)
    h = g * 150
    s += rect(x, 400 - h, BW, h, green, rx=4)
s += line(64, 400, 468, 400, rule, width=2)
s += txt(64, 430, '每个 token 都把 m 个通道算满，变的只是通过多少。', 15, muted)
s += txt(544, 182, 'MoE：择路', 20, blue, 600)
s += txt(544, 208, 'N 个专家，每个 token 只走少数几个', 14, muted)
for i in range(8):
    r, c = divmod(i, 4)
    x = 566 + c * 96
    y = 234 + r * 78
    on = i in (1, 6)
    s += rect(x, y, 80, 62, '#cfe0f7' if on else '#ffffff', rx=8)
    s += txt(x + 40, y + 30, f'E{i}', 16, blue if on else muted, 600 if on else 400, 'middle')
    s += txt(x + 40, y + 50, '激活' if on else '闲置', 12, blue if on else faint, 500, 'middle')
s += txt(544, 430, '被选中的专家才算，其余的这一步不动。', 15, muted)
s += txt(40, 492, 'SwiGLU 与 MoE 不互相替代：一个是连续的通过量，一个是离散的选择。', 16, muted)
s += txt(40, 518, '今天的主流配置是两者叠用——FFN 内部是 SwiGLU，FFN 整体被切成 MoE。', 16, muted)
s += txt(40, 544, '两者都成立，靠的是同一条性质：FFN 逐位置，改它的宽度和路由方式不会牵动别的位置。', 16, muted)
save('05-gate-vs-route.svg', s)

print('Created 5 SVG diagrams in', out)
