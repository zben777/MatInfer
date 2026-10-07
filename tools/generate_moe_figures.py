"""Regenerate the DeepSeekMoE diagrams (editable SVG sources).

SVG generation uses only the Python standard library, matching
tools/generate_norm_figures.py and tools/generate_attention_figures.py.
Figure 05 is animated with CSS @keyframes; it plays when the file is opened
directly in a browser, and also when referenced from an <img> tag.
"""
import math
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parents[1] / 'assets/images/moe'
out.mkdir(parents=True, exist_ok=True)

ink = '#24334a'; muted = '#6d7c90'; green = '#218575'
blue = '#4967b0'; amber = '#c88735'; rule = '#d8e0e9'; faint = '#b3becc'


def txt(x, y, s, size=18, color=ink, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')


def rect(x, y, w, h, fill='#f3f6fa', stroke='none', rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x, y, x2, y2, color=faint, arrow=False, width=2, dash=''):
    m = ' marker-end="url(#arrow)"' if arrow else ''
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{m}{d}/>'


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
            + txt(40, 42, f'MATINFER / MOE / {num}', 13, green, 600)
            + txt(40, 85, title, 28, ink, 600)
            + txt(40, 119, sub, 17, muted))


def save(name, s):
    (out / name).write_text(s + '</g></svg>')


# ------------------------------------------------------------- 01 三条路的账本
s = base('01', '一次前向的账本：容量和算力被切开了',
         '同样的玩具模型（d_model = 3，大专家 d_ff = 12）。参数按 864 归一，乘加按 240 归一。', 620)
s += txt(400, 148, '挂上的参数', 15, muted, 500)
s += txt(700, 148, '一次前向的乘加', 15, muted, 500)

ROWS = [
    ('普通 FFN（Dense）', '一个大专家，d_ff = 12', 108, 120, blue),
    ('常规 MoE（8 选 2）', '8 个大专家，每 token 选 2 个', 864, 240, amber),
    ('DeepSeekMoE 式', '16 个小专家（d_ff = 3），1 共享 + 3 路由', 432, 120, green),
]
for i, (name, sub, p, c, col) in enumerate(ROWS):
    y = 166 + i * 132
    s += rect(40, y, 920, 116, '#f7f9fc')
    s += txt(64, y + 42, name, 20, ink, 600)
    s += txt(64, y + 72, sub, 14, muted)
    pw = p / 864 * 180
    s += bar(400, y + 92, max(pw, 6), 20, col)
    s += txt(400 + pw + 12, y + 108, f'{p}', 19, col, 600)
    cw = c / 240 * 180
    s += bar(700, y + 92, max(cw, 6), 20, col)
    s += txt(700 + cw + 12, y + 108, f'{c}', 19, col, 600)

s += rect(40, 570, 920, 40, '#eef7f2', rx=10)
s += txt(64, 597, 'DeepSeekMoE 的 120 与 Dense 的 120 相等，因为激活的 4 个小专家合计 d_ff = 4 × 3 = 12。', 17, green, 600)
save('01-ledger.svg', s)


# ------------------------------------------------------------ 02 共享专家隔离
s = base('02', '通用知识：学一遍，还是被 N 个专家各学一遍',
         '路由专家池 15 个。左边没有共享专家，右边把通用知识拎出来单独放。', 616)
for x0, col, t1, t2 in [(40, amber, '纯路由专家池', '每个专家都私藏一份"通用知识"'),
                        (520, green, '1 共享 + 15 路由', '通用知识只此一份，路由专家专心做专门化')]:
    s += rect(x0, 146, 440, 364, '#f7f9fc' if col == amber else '#edf7f2')
    s += txt(x0 + 24, 182, t1, 20, ink, 600)
    s += txt(x0 + 24, 208, t2, 14, muted)

s += rect(64, 218, 392, 32, '#fbf3e6', rx=8)
s += txt(84, 240, '15 个专家，每个都含一份通用知识', 15, amber, 600)
s += rect(544, 218, 392, 32, '#cfe8dd', rx=8)
s += txt(564, 240, '共享专家：通用知识，永远激活', 15, green, 600)

for i in range(5):
    for j, x in enumerate([64, 200, 336]):
        y = 262 + i * 40
        s += rect(x, y, 120, 30, '#eef1f6', rx=6)
        s += rect(x + 4, y + 4, 50, 22, '#f3ddd0', rx=4)
        s += txt(x + 29, y + 20, '通用', 12, amber, 500, 'middle')
        s += txt(x + 114, y + 20, '专有', 12, muted, 400, 'end')
    for j, x in enumerate([544, 680, 816]):
        y = 262 + i * 40
        s += rect(x, y, 120, 30, '#eaf4ee', rx=6)
        s += txt(x + 60, y + 20, '专有', 13, muted, 400, 'middle')

s += txt(64, 484, '通用知识存了 15 份，真正用得上的只有 1 份', 16, amber, 500)
s += txt(544, 484, '通用知识只存 1 份，省下的容量全给专有知识', 16, green, 500)

s += txt(40, 548, '共享专家是常驻的，占一份固定算力：给多了吃掉稀疏性，给少了装不下通用知识。', 17, muted)
s += txt(40, 576, '所以 V2 用 2 个共享专家，V3 减到 1 个 —— 路由池从 160 涨到 256 之后，常驻容量不再需要那么多。', 17, muted)
save('02-shared-expert.svg', s)


# --------------------------------------------------------- 03 三个打分函数
s = base('03', 'Sigmoid 在两端饱和，Sqrt(Softplus) 不会',
         'z 从 6 变到 8：Sigmoid 只动 0.21%，Sqrt(Softplus) 动了 15.4% —— 这就是"分辨率"。', 660)
X0, X1, Z0, Z1 = 100, 640, -4.0, 10.0
YB, YT, VMAX = 500, 200, 3.4


def px(z):
    return X0 + (z - Z0) / (Z1 - Z0) * (X1 - X0)


def py(v):
    return YB - min(v, VMAX) / VMAX * (YB - YT)


s += rect(40, 140, 640, 400, '#f7f9fc')
s += line(X0, YB, X1, YB, rule)
s += line(X0, YB, X0, YT, rule)
for z in range(-4, 11, 2):
    s += line(px(z), YB, px(z), YB + 6, rule)
    s += txt(px(z), YB + 26, z, 14, muted, 400, 'middle')
for v in [0, 1, 2, 3]:
    s += line(X0 - 6, py(v), X0, py(v), rule)
    s += txt(X0 - 12, py(v) + 5, v, 14, muted, 400, 'end')

pts = ' '.join(f'{px(-4 + i * 0.1):.1f},{py(1 / (1 + math.exp(-(-4 + i * 0.1)))):.1f}' for i in range(141))
s += f'<polyline points="{pts}" fill="none" stroke="{blue}" stroke-width="3"/>'
pts = ' '.join(f'{px(-4 + i * 0.1):.1f},{py(math.sqrt(math.log(1 + math.exp(-4 + i * 0.1)))):.1f}' for i in range(141))
s += f'<polyline points="{pts}" fill="none" stroke="{green}" stroke-width="3"/>'

s += line(X0, py(1.0), X1, py(1.0), amber, width=2, dash='7 6')
s += txt(X0 + 14, py(1.0) - 12, 'Sigmoid 的天花板 = 1', 15, amber, 600)
s += txt(X1 - 6, py(1.0) - 14, 'Sigmoid', 16, blue, 600, 'end')
s += txt(X1 - 6, py(math.sqrt(math.log(1 + math.exp(Z1)))) - 14, 'Sqrt(Softplus)', 16, green, 600, 'end')
s += line(px(6), YB, px(6), py(3.0), faint, width=1, dash='4 5')
for v, col in [(1 / (1 + math.exp(-6)), blue), (math.sqrt(math.log(1 + math.exp(6))), green)]:
    s += f'<circle cx="{px(6):.1f}" cy="{py(v):.1f}" r="4.5" fill="{col}"/>'
s += txt(370, 556, 'router logit  z', 15, muted, 400, 'middle')

s += rect(700, 140, 260, 400, '#f1f4fb')
s += txt(724, 176, '同一个 z 区间上的变化', 17, ink, 600)
s += txt(724, 204, 'z：6 → 8', 16, muted)
s += txt(724, 250, 'Sigmoid', 17, blue, 600)
s += txt(724, 276, '0.99753 → 0.99966', 15, ink)
s += txt(724, 300, '相对变化 0.21%', 16, blue, 500)
s += line(724, 322, 936, 322, rule)
s += txt(724, 354, 'Sqrt(Softplus)', 17, green, 600)
s += txt(724, 380, '2.44999 → 2.82849', 15, ink)
s += txt(724, 404, '相对变化 15.4%', 16, green, 500)
s += line(724, 426, 936, 426, rule)
s += txt(724, 456, '归一化后的权重几乎一样', 15, muted)
s += txt(724, 480, '0.305 / 0.380 / 0.315', 15, ink)
s += txt(724, 504, '对 0.300 / 0.387 / 0.313', 15, ink)

s += txt(40, 600, 'V4 报告只有一句话交待这次更换，没有论证。上面是从函数形状倒推它能解决什么：换掉的不是分布形状，', 17, muted)
s += txt(40, 628, '而是"分数撞到上限"这件事本身 —— 专家数涨到几百之后，几百个 logit 里总有偏大的。', 17, muted)
save('03-scoring-functions.svg', s)


# ------------------------------------------------------- 04 偏置只进排序不进权重
s = base('04', '偏置项只进排序，不进权重',
         'b 出现在 Top-K 的比较里，却不出现在 g 的算式里 —— 这一进一出就是"零梯度"的全部含义。', 580)
s += txt(64, 172, '输入 x', 18, ink, 600)
s += line(160, 166, 226, 166, faint, True)
s += rect(232, 144, 150, 44, '#eef1f6', rx=8)
s += txt(307, 172, 'router 打分', 17, ink, 500, 'middle')
s += line(388, 166, 454, 166, faint, True)
s += rect(460, 144, 130, 44, '#eef1f6', rx=8)
s += txt(525, 172, 's = σ(z)', 17, ink, 500, 'middle')

s += rect(620, 130, 150, 72, '#fbf3e6', rx=8)
s += txt(695, 160, '偏置 b', 18, amber, 600, 'middle')
s += txt(695, 184, '不参与优化', 14, muted, 400, 'middle')
s += line(770, 166, 822, 166, amber, True)
s += rect(828, 144, 132, 44, '#fbf3e6', rx=8)
s += txt(894, 172, 's + b 排序', 17, amber, 600, 'middle')

s += line(610, 188, 610, 246, amber)
s += line(610, 246, 826, 246, amber)
s += line(826, 246, 826, 208, amber, True)
s += txt(716, 272, 'b 只走到这里：决定"谁被选中"，然后停下', 16, amber, 600, 'middle')

s += rect(40, 306, 920, 116, '#f1f4fb')
s += txt(64, 342, 'Top-K 选出 3 个专家', 18, ink, 600)
s += txt(64, 372, '选中集合来自 s + b 的排序', 16, muted)
s += line(330, 352, 396, 352, faint, True)
s += rect(402, 330, 220, 44, '#eef1f6', rx=8)
s += txt(512, 358, 'g = s / Σs（原始分）', 17, ink, 500, 'middle')
s += line(628, 352, 694, 352, faint, True)
s += rect(700, 330, 230, 44, '#eef1f6', rx=8)
s += txt(815, 358, 'g × 各专家输出求和', 17, ink, 500, 'middle')
s += txt(64, 404, '权重算式里没有 b —— 梯度回传时，b 这条路是断的', 16, blue, 500)

s += rect(40, 442, 440, 106, '#f7f9fc')
s += txt(64, 476, '若用辅助损失', 18, amber, 600)
s += txt(64, 504, 'L = L_main + α · L_balance', 16, ink)
s += txt(64, 532, '均衡项进梯度，与主目标争夺优化方向', 15, muted)
s += rect(520, 442, 440, 106, '#edf7f2')
s += txt(544, 476, '若用偏置项', 18, green, 600)
s += txt(544, 504, 'b ← b ± γ（按负载，与梯度无关）', 16, ink)
s += txt(544, 532, '闭环控制器，改的是"谁被选中"，不是"模型学什么"', 15, muted)
save('04-bias-routing.svg', s)


# ------------------------------------------------------- 05 完整前向（动画）
FRAMES = [
    ('① 路由打分', 's = [ 1/2, 4/5, 1/5, 9/10, 1/10, 3/5, 3/10 ]',
     'logit 写成 ln 形式，Sigmoid 的输出就是干净分数', '#f1f4fb', blue,
     [('E3', '4/5', 120), ('E5', '9/10', 135), ('E7', '3/5', 90)]),
    ('② 加偏置后排序', 's′ = s + b　→　Top-3 = E4, E5, E8',
     'E3 从 4/5 被压到 3/10 而出局，E4 从 1/5 被抬到 4/5 上位', '#fbf3e6', amber,
     [('E4', '4/5', 120), ('E5', '4/5', 120), ('E8', '7/10', 105)]),
    ('③ 归一化取权重', 'g = s / Σs = 1/7, 9/14, 3/14（不是 softmax）',
     'Σs = 1/5 + 9/10 + 3/10 = 7/5；三个权重加起来恰好是 1', '#edf7f2', green,
     [('E4', '1/7', 43), ('E5', '9/14', 145), ('E8', '3/14', 48)]),
    ('④ 四个专家各自前向', 'y = 0.5811（共享 E1）, 1.1622, 1.7433, 0.7748',
     'c = Swish(0.6) = 0.387394；每个专家 y = 18c · u · δ', '#f1f4fb', blue,
     [('E1', '0.5811', 58), ('E4', '1.1622', 116), ('E5', '1.7433', 174), ('E8', '0.7748', 77)]),
    ('⑤ 合并输出', 'y = 0.5811 + 0.1660 + 1.1207 + 0.1660 = 2.0338',
     '精确形式是 (21/4)·c；E4 与 E8 的贡献相同并非巧合', '#eef7f2', green,
     [('E1', '0.5811', 58), ('E4', '0.1660', 17), ('E5', '1.1207', 112), ('E8', '0.1660', 17)]),
]
CSS = ('<style>'
       '@keyframes sw{0%,19.9%{opacity:1}20%,100%{opacity:0}}'
       '.f{animation:sw 12s linear infinite;opacity:0}'
       '.f1{animation-delay:0s}.f2{animation-delay:2.4s}.f3{animation-delay:4.8s}'
       '.f4{animation-delay:7.2s}.f5{animation-delay:9.6s}'
       '@media (prefers-reduced-motion:reduce){.f{animation:none;opacity:0}'
       '.f5{opacity:1}}'
       '</style>')
s = base('05', '1×6 一次完整前向（动画）',
         '五个步骤循环播放。图中数值与正文可逐项对照，全部经脚本核过。', 420, head=CSS)
for i, (title, state, note, bg, col, bars) in enumerate(FRAMES):
    s += f'<g class="f f{i + 1}">'
    s += rect(40, 152, 920, 216, bg, rx=12)
    s += txt(64, 194, title, 24, col, 600)
    s += txt(64, 230, state, 19, ink)
    s += txt(64, 262, note, 16, muted)
    for j, (name, val, h) in enumerate(bars):
        bx = 528 + j * 108
        s += txt(bx, 300, name, 15, muted)
        s += bar(bx, 328, 72, h, col)
        s += txt(bx + 36, 328 - max(h, 30) - 10, val, 15, col, 600, 'middle')
    s += '</g>'
s += txt(40, 400, '图中柱高按各自帧内的最大值归一，只用于比较同一帧内各专家的大小。', 15, muted)
save('05-forward-animation.svg', s)


# --------------------------------------------------------- 06 节点限制路由
s = base('06', 'Node-Limited Routing 砍掉的是什么',
         '256 个路由专家分成 8 组、每组 32 个，一组占一个节点。约束的是"触达几个节点"，不是"选几个专家"。', 620)
s += rect(40, 146, 440, 330, '#f7f9fc')
s += txt(64, 180, '无约束：8 个专家触达 8 个节点', 19, ink, 600)
s += txt(64, 206, 'M = 8，每个专家的 token 各走一次跨节点通信', 14, muted)
for i in range(8):
    y = 220 + i * 25
    s += rect(64, y, 392, 21, '#fbf3e6', rx=5)
    s += txt(78, y + 16, f'节点 {i + 1}', 13, muted)
    s += rect(240, y + 4, 202, 13, '#e8cfa8', rx=3)
s += txt(64, 448, 'IB 通信量 8t', 17, amber, 600)

s += rect(520, 146, 440, 330, '#edf7f2')
s += txt(544, 180, 'Node-Limited：最多触达 4 个节点', 19, ink, 600)
s += txt(544, 206, '先挑累计亲和分最高的 4 个节点，再在这 4×32 个专家里选', 14, muted)
for i in range(8):
    y = 220 + i * 25
    on = i < 4
    s += rect(544, y, 392, 21, '#dceee6' if on else '#eef1f6', rx=5)
    s += txt(558, y + 16, f'节点 {i + 1}', 13, ink if on else faint)
    if on:
        s += rect(720, y + 4, 202, 13, '#a9d4c0', rx=3)
    else:
        s += txt(922, y + 16, '不触达', 13, faint, 400, 'end')
s += txt(544, 448, 'IB 通信量最多 4t，叠加 NVLink 转发', 16, green, 600)
s += txt(544, 472, '有效通信用时接近减半', 16, green, 600)

s += rect(40, 496, 920, 104, '#fbf3e6', rx=10)
s += txt(64, 530, '这是用模型质量补贴通信开销', 18, amber, 600)
s += txt(64, 558, '路由的自由度被砍了：本来能在全部 256 个专家里挑最好的 8 个，现在只能在 4×32 个里挑。', 16)
s += txt(64, 584, 'V4 把这条约束去掉了，改用通信-计算 overlap 接住这笔账 —— 交换方向的反转本身就是信号。', 16, muted)
save('06-node-limited-routing.svg', s)


print('Created 6 SVG diagrams in', out)
