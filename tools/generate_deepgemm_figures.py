"""Regenerate the DeepGEMM CTA-tiling diagrams (editable SVG sources).

Standard library only, matching tools/generate_norm_figures.py and
tools/generate_swiglu_figures.py.  Figure 03 is animated with CSS
@keyframes (four frames: the K loop accumulating into the register
accumulator).

No TeX inside the SVGs -- a plain SVG file renders no LaTeX, so every
formula here is written with Unicode.
"""
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parents[1] / 'assets/images/deepgemm'
out.mkdir(parents=True, exist_ok=True)

ink = '#24334a'; muted = '#6d7c90'; green = '#218575'
blue = '#4967b0'; amber = '#c88735'; rule = '#d8e0e9'; faint = '#b3becc'
GRAY = '#c8d1dc'


def txt(x, y, s, size=18, color=ink, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')


def rect(x, y, w, h, fill='#f3f6fa', stroke='none', rx=12):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}"/>')


def line(x, y, x2, y2, color=faint, arrow=False, width=2, dash=''):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{m}{d}/>'


def path(d, color=faint, width=2, arrow=True):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>'


def base(num, title, sub, h, head=''):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {h}" role="img" '
            f'aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(sub)}</desc>'
            f'<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
            f'orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="{faint}"/></marker></defs>'
            f'{head}'
            f'<g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">'
            + rect(0, 0, 1000, h, '#ffffff', rx=0)
            + txt(40, 42, f'MATINFER / DEEPGEMM / {num}', 13, green, 600)
            + txt(40, 85, title, 28, ink, 600)
            + txt(40, 119, sub, 17, muted))


def save(name, s):
    (out / name).write_text(s + '</g></svg>')


# ======================================= 01 一份数据服务几个输出
s = base('01', '同一份数据，能服务几个输出元素',
         '切块只做一件事：让搬进来的数据被更多输出元素共用。', 646)
s += rect(40, 146, 430, 292, '#fbf1ec')
s += rect(530, 146, 430, 292, '#eef4f0')

s += txt(64, 182, '不切块', 21, amber, 600)
s += txt(64, 208, '搬一次，服务一个元素', 14, muted)
s += rect(96, 252, 116, 52, '#f0d9c0', rx=8)
s += txt(154, 284, 'A[m][k]', 15, ink, 600, 'middle')
s += path('M216 278 L282 278', faint)
s += rect(288, 252, 116, 52, '#ffffff', stroke=rule, rx=8)
s += txt(346, 284, 'C[m][n]', 15, ink, 600, 'middle')
s += txt(64, 400, 'A 的第 m 行被读了 N 遍', 16, amber, 600)
s += txt(64, 424, '——每个输出列各读一次，谁也没占便宜。', 14, muted)

s += txt(554, 182, '按 64×128 切块', 21, green, 600)
s += txt(554, 208, '搬一次，服务 128 个元素', 14, muted)
s += rect(554, 252, 116, 52, '#a9d4c0', rx=8)
s += txt(612, 284, 'A[m][k]', 15, ink, 600, 'middle')
s += path('M674 278 L722 278', green)
for i in range(16):
    for j in range(7):
        s += rect(736 + i * 13, 240 + j * 13, 9, 9, green, rx=2)
s += txt(554, 400, '一次搬进片上，整块共用', 16, green, 600)
s += txt(554, 424, '——同一行只搬 N ÷ B_N = 4 遍。', 14, muted)

s += rect(40, 464, 920, 150, '#f7f9fc')
s += txt(64, 502, '同是 256×512×256 的一道乘法：', 17, ink, 600)
s += txt(64, 542, '朴素 67,108,864 次元素读', 19, amber, 600)
s += txt(360, 542, '→', 19, muted, 400, 'middle')
s += txt(430, 542, '切块 786,432 次', 19, green, 600)
s += txt(64, 578, '差 85.33 倍。这个倍数恰好是切块后算术强度的两倍——下面会看到为什么。', 15, muted)
save('01-reuse.svg', s)


# ======================================= 02 C 的 4×4 切分
s = base('02', 'C 被切成 16 块，交给 16 个 CTA',
         '256×512 的输出，按 64×128 切分；每个 CTA 拿一块。', 632)
s += txt(300, 168, 'N = 512', 15, muted, 600, 'middle')
s += txt(58, 380, 'M = 256', 15, muted, 600, 'middle')
for r in range(4):
    for c in range(4):
        x = 110 + c * 205
        y = 200 + r * 78
        hot = (r == 1 and c == 2)
        s += rect(x, y, 195, 70, '#a9d4c0' if hot else '#eef3f9', rx=8)
        s += txt(x + 97, y + 34, f'CTA {r},{c}', 15, ink if hot else muted, 600 if hot else 400, 'middle')
        s += txt(x + 97, y + 55, '64 × 128', 12, muted if hot else faint, 400, 'middle')
s += line(90, 196, 90, 512, rule, width=2)
s += line(106, 512, 930, 512, rule, width=2)
s += txt(64, 562, '高亮的 CTA(1,2) 负责 C 的第 64–127 行、第 256–383 列。', 16, muted)
s += txt(64, 594, 'Grid = ⌈512/128⌉ × ⌈256/64⌉ = 4 × 4 = 16。一个 CTA 负责一块结果，K 方向还一个都没碰。', 16, muted)
save('02-cta-grid.svg', s)


# ======================================= 03 K 循环（四帧动画）
CSS = ('<style>'
       '@keyframes kf{0%,23.9%{opacity:1}24%,100%{opacity:0}}'
       '.f{animation:kf 16s linear infinite;opacity:0}'
       '.f1{animation-delay:0s}.f2{animation-delay:4s}'
       '.f3{animation-delay:8s}.f4{animation-delay:12s}'
       '@media (prefers-reduced-motion:reduce){.f{animation:none;opacity:0}'
       '.f4{opacity:1}}'
       '</style>')

AX, AW = 80, 250            # A tile: 64 x 256, split into 8 columns
AY, AH = 262, 96
BX, BW = 420, 120           # B tile: 256 x 128, split into 8 rows
BY, BH = 186, 250
CX, CW = 620, 250           # accumulator: 64 x 128
CY, CH = 262, 96


def kloop(done, cur):
    """done: how many K blocks finished; cur: index of the block in flight."""
    e = ''
    # A: 8 vertical slices
    seg = AW / 8
    for i in range(8):
        on = (i == cur)
        e += rect(AX + i * seg, AY, seg - 3, AH, '#a9d4c0' if on else '#e8eef6', rx=3)
    e += txt(AX + AW / 2, AY + AH + 30, 'A  64 × 256　（沿 K 切 8 份，每份 64×32）', 14, muted, 400, 'middle')
    # B: 8 horizontal slices
    seg = BH / 8
    for i in range(8):
        on = (i == cur)
        e += rect(BX, BY + i * seg, BW, seg - 3, '#a9d4c0' if on else '#e8eef6', rx=3)
    e += txt(BX + BW / 2, BY + BH + 30, 'B  256 × 128　（每份 32×128）', 14, muted, 400, 'middle')
    # accumulate arrow
    e += path(f'M{AX + AW + 12} {AY + AH / 2} L{BX - 12} {AY + AH / 2}', faint, arrow=False)
    e += txt((AX + AW + BX) / 2, AY + AH / 2 - 12, '×', 20, muted, 600, 'middle')
    e += path(f'M{BX + BW + 14} {AY + AH / 2} L{CX - 14} {AY + AH / 2}', green)
    e += txt((BX + BW + CX) / 2, AY + AH / 2 - 12, '=', 20, muted, 600, 'middle')
    # accumulator
    e += rect(CX, CY, CW, CH, '#dff0e8', rx=8)
    e += txt(CX + CW / 2, CY + 40, 'C 累加器', 16, ink, 600, 'middle')
    e += txt(CX + CW / 2, CY + 66, '64 × 128　FP32', 13, muted, 400, 'middle')
    for i in range(done):
        s2 = CW / 8
        e += rect(CX + i * s2, CY + CH - 12, s2 - 2, 8, green, rx=2)
    e += txt(CX + CW / 2, CY + CH + 30, f'已累加 {done} / 8', 14, green if done else muted, 600, 'middle')
    return e


FRAMES = [('第 0 块', 0), ('第 2 块', 2), ('第 4 块', 4), ('第 6 块', 6)]
s = base('03', 'K 方向切 8 份，依次累加（动画）',
         '每一轮搬进 64×32 与 32×128 两片、共 12 KB；累加器始终在寄存器里。', 620, head=CSS)
for idx, (label, cur) in enumerate(FRAMES):
    s += f'<g class="f f{idx + 1}">'
    s += rect(40, 146, 920, 356, '#f7f9fc')
    s += txt(64, 178, f'{label}进入流水线', 16, green, 600)
    s += txt(936, 178, f'{idx + 1} / 4', 14, muted, 400, 'end')
    s += kloop(2 * idx + 1, cur)
    s += '</g>'
s += txt(40, 540, 'K 循环跑满 8 次后，C 才被写回一次；整个循环里它一直待在寄存器里，没有回过显存。', 16, muted)
s += txt(40, 572, '换一个 B_K 只是改循环次数与每片大小，不改变读入总量——这正是下一张图要说明的事。', 16, muted)
save('03-k-loop.svg', s)


# ======================================= 04 算术强度
s = base('04', '算术强度只由块的两个边决定',
         '沿 K 切几刀、每刀多宽，都不改变这些值。', 632)
BASE_Y = 512
FULL = 180.0            # 满刻度 180 FLOP/B
HSCALE = 340 / FULL
BARS = [('64×64', 32.0, green, ''),
        ('64×128', 128 / 3, green, '材料里的取法'),
        ('128×128', 64.0, green, ''),
        ('128×256', 256 / 3, green, ''),
        ('256×512', 512 / 3, faint, '片上放不下')]
for i, (name, v, col, note) in enumerate(BARS):
    x = 80 + i * 180
    h = v * HSCALE
    s += rect(x, BASE_Y - h, 120, h, col if col != faint else '#e4e9f0', rx=4)
    s += txt(x + 60, BASE_Y - h - 14, f'{v:.2f}'.rstrip('0').rstrip('.'), 16,
             ink if col != faint else muted, 600, 'middle')
    s += txt(x + 60, BASE_Y + 28, name, 14, ink, 600, 'middle')
    if note:
        s += txt(x + 60, BASE_Y + 50, note, 12, muted if col != faint else faint, 400, 'middle')
s += line(60, BASE_Y, 950, BASE_Y, rule, width=2)
s += txt(60, BASE_Y - 356, 'FLOP/Byte', 13, muted, 400)
s += line(60, 156, 80, 156, rule, width=2)
s += rect(40, 552, 920, 62, '#fbf6ee', rx=10)
s += txt(64, 580, '朴素写法是 0.5 FLOP/Byte，在这张图的刻度下贴着基线——比最矮的柱子还低 64 倍。', 16, amber, 600)
s += txt(64, 604, '五根柱子逐一代入 B_K = 16 / 32 / 64 / 128，得到的仍是同一个数。', 15, muted)
save('04-intensity.svg', s)


# ======================================= 05 四笔账
s = base('05', '块变大要付的四笔账',
         '每一笔在源码里都对应一行检查；任何一笔踩到头，这个尺寸就出局。', 664)

# --- 1 shared memory
s += rect(40, 146, 440, 226, '#fbf6ee')
s += txt(64, 180, 'Shared Memory', 19, amber, 600)
s += txt(64, 206, '每阶段要装下 A、B 两片', 14, muted)
for i, (w, lab) in enumerate([(150, '64×128·BK'), (230, '128×128·BK'), (330, '128×256·BK')]):
    y = 240 + i * 36
    s += rect(64, y, w, 24, '#e3b177' if i else '#f0d9c0', rx=4)
    s += txt(64 + w + 12, y + 18, lab, 13, muted)
s += txt(320, 356, '越大 → 级数越少', 14, amber, 600, 'end')

# --- 2 registers
s += rect(520, 146, 440, 226, '#eef4f0')
s += txt(544, 180, '寄存器', 19, green, 600)
s += txt(544, 206, 'BM×BN 个 FP32 累加器必须驻留', 14, muted)
for i in range(14):
    for j in range(6):
        s += rect(544 + i * 13, 236 + j * 13, 9, 9, green, rx=2)
s += txt(760, 290, '64×128 → 8192 个', 15, ink, 600)
s += txt(760, 314, '128×256 → 32768 个', 15, ink, 600)
s += txt(544, 356, '一个 SM 只有 256 KB 寄存器文件', 14, green, 600)

# --- 3 CTA count
s += rect(40, 396, 440, 226, '#f1f4fb')
s += txt(64, 430, 'CTA 数量', 19, blue, 600)
s += txt(64, 456, '块越大，切出来的 CTA 越少', 14, muted)
for i in range(16):
    for j in range(8):
        on = i < 2
        s += rect(64 + i * 13, 470 + j * 13, 10, 10,
                  blue if on else '#ffffff', stroke=rule if not on else 'none', rx=2)
s += txt(64, 608, '16 个 CTA 铺在 128 个 SM 上，一轮只用掉八分之一', 14, blue, 600)

# --- 4 tail waste
s += rect(520, 396, 440, 226, '#f7f9fc')
s += txt(544, 430, '尾部浪费', 19, ink, 600)
s += txt(544, 456, 'M、N 不是块尺寸整数倍时', 14, muted)
s += rect(544, 478, 336, 96, '#dbe4ef', rx=6)
s += rect(544, 478, 216, 96, '#a9c4e0', rx=6)
s += rect(760, 478, 120, 96, '#f3d9d3', rx=6)
s += txt(652, 532, '有效', 15, ink, 600, 'middle')
s += txt(820, 532, '白算', 15, amber, 600, 'middle')
s += txt(544, 604, '源码里由 last_wave_util 盯着', 14, muted)
save('05-limits.svg', s)


# ======================================= 06 三个数字的来路
s = base('06', '三个数字，三条来路',
         '它们不是搜出来的，是从各自的约束里解出来的。', 646)
COL = [(40, 293), (353, 293), (666, 294)]
for x, w in COL:
    s += rect(x, 146, w, 420, '#f7f9fc')

# --- BK
x = 40
s += txt(x + 24, 186, 'B_K', 22, green, 600)
s += txt(x + 24, 214, '由数据类型定死', 14, muted)
s += rect(x + 24, 236, 245, 52, '#eef4f0', rx=8)
s += txt(x + 146, 270, '128 × 8 ÷ 位宽', 18, ink, 600, 'middle')
for i, (name, bits, bk) in enumerate([('BF16', 16, 64), ('FP8 / FP4', 8, 128), ('MXF4', 4, 256)]):
    y = 312 + i * 56
    s += txt(x + 24, y + 22, name, 15, muted)
    s += txt(x + 146, y + 24, f'{bits} bit', 14, faint, 400, 'middle')
    s += txt(x + 269, y + 22, str(bk), 19, green, 600, 'end')
s += txt(x + 24, 500, '一行恰好 128 字节', 15, green, 600)
s += txt(x + 24, 526, '和 shared memory 的', 14, muted)
s += txt(x + 24, 546, 'swizzle 原子同宽', 14, muted)

# --- stages
x = 353
s += txt(x + 24, 186, '流水线级数', 22, blue, 600)
s += txt(x + 24, 214, '由片上容量反推', 14, muted)
s += rect(x + 24, 232, 245, 60, '#f1f4fb', rx=8)
s += txt(x + 146, 260, '（容量 − 额外）÷ 每阶段', 14, ink, 600, 'middle')
s += txt(x + 146, 281, '上限 16', 12, muted, 400, 'middle')
for i, (label, note) in enumerate([('容量 227 KB', 'Hopper 一块 SM'),
                                   ('每阶段 12 KB', 'BK=32、FP16 时'),
                                   ('级数 ≥ 3', '否则直接淘汰')]):
    y = 322 + i * 58
    s += rect(x + 24, y - 20, 245, 44, '#ffffff', stroke=rule, rx=8)
    s += txt(x + 40, y + 6, label, 15, ink, 600)
    s += txt(x + 253, y + 6, note, 13, muted, 400, 'end')
s += txt(x + 24, 520, '块越大，级数越少', 15, blue, 600)

# --- block size
x = 666
s += txt(x + 24, 186, 'B_M、B_N', 22, amber, 600)
s += txt(x + 24, 214, '被寄存器卡住', 14, muted)
s += rect(x + 24, 236, 246, 52, '#fbf6ee', rx=8)
s += txt(x + 147, 270, '至少一维 ≤ 128', 18, ink, 600, 'middle')
for i, (label, note, bad) in enumerate([('64 × 128', '8192 个累加器', False),
                                        ('128 × 128', '16384 个', False),
                                        ('128 × 256', '32768 个 · 淘汰', True)]):
    y = 318 + i * 56
    s += rect(x + 24, y - 18, 246, 42, '#ffffff', stroke=rule, rx=8)
    s += txt(x + 40, y + 8, label, 15, amber if bad else ink, 600)
    s += txt(x + 254, y + 8, note, 13, amber if bad else muted, 400, 'end')
s += txt(x + 24, 500, '两维都超就出局', 15, amber, 600)
s += txt(x + 24, 526, '线程数按 B_M 翻倍', 14, muted)
s += txt(x + 24, 546, '把每人份数压回 B_N/2', 14, muted)
save('06-source.svg', s)

print('Created 6 SVG diagrams in', out)
