"""Regenerate the four editable Norm diagrams using only Python stdlib."""
from pathlib import Path
from html import escape
import math
out=Path(__file__).resolve().parents[1] / 'assets/images/normalization'
out.mkdir(parents=True, exist_ok=True)
ink='#24334a'; muted='#6d7c90'; green='#218575'; blue='#4967b0'; amber='#c88735'
def txt(x,y,s,size=18,color=ink,weight=400,anchor='start'):
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>'
def rect(x,y,w,h,fill='#f3f6fa',stroke='none',rx=12):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
def line(x,y,x2,y2,color='#b3becc',arrow=False):
 return f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="2"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>'
def base(num,title,sub,h):
 return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(sub)}</desc><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#b3becc"/></marker></defs><g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">'+rect(0,0,1000,h,'#ffffff',rx=0)+txt(40,42,f'MATINFER / NORMALIZATION / {num}',13,green,600)+txt(40,85,title,28,ink,600)+txt(40,119,sub,17,muted)
def save(name,s):
 (out/name).write_text(s+'</g></svg>')
s=base('01','归一化的范围：一个 token 的隐藏向量','典型 Transformer：输入 [B, S, D]，统计量沿最后一个维度 D 计算。',570)
s+=txt(40,170,'B：请求 / 批次',17,muted)+txt(40,201,'S：token 位置',17,muted)+txt(260,160,'D：隐藏维度 →',17,green,600)
for r in range(4):
 s+=txt(60,241+r*46,f'token {r+1}',16,muted)
 for c in range(8):s+=rect(170+c*37,215+r*46,31,34,'#e0f1eb' if r==1 else '#eef1f6',green if r==1 else 'none',5)
s+=txt(170,431,'每一行独立统计；不跨 token 或 batch 求均值。',16,muted)
s+=line(475,278,555,278,arrow=True)+rect(580,181,370,215,'#f5f8fb')+txt(604,215,'一行：x = [x₁, …, xD]',21,ink,600)+txt(604,260,'归约得到一个标量统计量',18)+txt(604,301,'用它缩放这一行的每个元素',18)+txt(604,348,'γ（以及 LN 的 β）：D 个参数',17,green)
s+=rect(40,473,910,61,'#f7f9fc')+txt(60,498,'Pre-Norm 子层',15,muted)+txt(260,511,'x → Norm → Attention / FFN → 与 x 相加',20)+txt(40,559,'Norm 约束的是子层输入；残差相加后的 hidden state 不保证具有固定尺度。',15,muted)
save('01-normalization-axis.svg',s)
s=base('02','两条计算路径：中心化与尺度缩放','LayerNorm 减去均值后除以标准差；RMSNorm 直接除以均方根。',640)
s+=rect(350,151,300,52,'#f0f3f8')+txt(500,184,'同一个输入向量 x',21,ink,600,'middle')
for x,col,label in [(40,blue,'LayerNorm'),(530,green,'RMSNorm')]:
 s+=txt(x,245,label,24,col,600)
 stages=[('均值 μ = mean(x)','中心化 c = x − μ'),('方差 v = mean(c²)','r = 1 / √(v + ε)'),('输出 y = γ ⊙ c · r + β','中心 + 尺度，再学习缩放与平移')] if x==40 else [('平方均值 q = mean(x²)','不减去均值'),('r = 1 / √(q + ε)','直接计算缩放系数'),('输出 y = γ ⊙ x · r','尺度归一化，再学习缩放')]
 for i,(a,b) in enumerate(stages):
  yy=266+i*98;s+=rect(x,yy,430,77,'#f1f4fb' if x==40 else '#edf7f2')+txt(x+20,yy+31,a,19,col,500)+txt(x+20,yy+59,b,16,muted)
  if i<2:s+=line(x+215,yy+78,x+215,yy+94,arrow=True)
s+=line(500,203,255,222)+line(500,203,745,222)+txt(40,613,'标准 RMSNorm 无 β；γ 是逐特征参数，不是整行共享的一个标量。',17,muted)
save('02-computation-paths.svg',s)
s=base('03','同一个向量：输出究竟改变了什么？','x = [1, 2, 3, 4, 5, 6, 7, 8]；手算设 ε = 0、γ = 1、β = 0。',900)
vals=[list(range(1,9)),[(i-4.5)/math.sqrt(5.25) for i in range(1,9)],[i/math.sqrt(25.5) for i in range(1,9)]]
labels=[('原始输入',muted,'均值 4.5'),('LayerNorm',blue,'均值 0 · 方差 1'),('RMSNorm',green,'RMS 1 · 均值 ≈ 0.891')]
for j,(values,(lab,col,stat)) in enumerate(zip(vals,labels)):
 top=158+j*215;s+=txt(40,top+24,lab,21,col,600)+txt(950,top+10,stat,17,col,400,'end')
 baseline=top+108;s+=line(60,baseline,944,baseline,'#d8e0e9')
 for i,v in enumerate(values):
  cx=110+i*112;scale=9 if j==0 else 30;hh=abs(v)*scale
  s+=rect(cx-22,baseline-hh if v>=0 else baseline,44,hh,col,rx=4)
  s+=txt(cx,baseline-hh-9 if v>=0 else baseline+hh+21,f'{v:.3f}' if j else str(v),15,col,500,'middle')
  s+=txt(cx,top+191,f'x{i+1}',14,muted,400,'middle')
s+=rect(40,829,910,45,'#f5f7fa')+txt(58,857,'每幅使用独立纵向尺度；用于观察符号与形状，精确数值以标签为准。',16,muted)
save('03-numerical-example.svg',s)
s=base('04','RMSNorm：从一行数据到算子执行','数学结构是 reduction + rsqrt + elementwise；物理读写次数由实现决定。',555)
blocks=[(40,'读取 x','一行 D 个元素',muted),(280,'归约平方和','mean(x²)',blue),(520,'计算系数','rsqrt(q + ε)',amber),(760,'逐元素输出','x · r · γ',green)]
for i,(x,a,b,col) in enumerate(blocks):
 s+=rect(x,173,200,110,'#f4f7fb')+txt(x+100,214,a,21,col,600,'middle')+txt(x+100,252,b,17,muted,400,'middle')
 if i<3:s+=line(x+205,226,x+235,226,arrow=True)
s+=rect(40,325,440,140,'#f1f4fb')+txt(62,359,'数值精度',20,blue,600)+txt(62,398,'低精度输入：可用 FP32 累积平方和',17)+txt(62,435,'平方在提升精度之后进行，避免先溢出',17,muted)
s+=rect(510,325,450,140,'#edf7f2')+txt(532,359,'融合与访存',20,green,600)+txt(532,398,'融合可减少中间张量与 kernel 启动',17)+txt(532,435,'是否受带宽限制，仍取决于形状和实现',17,muted)
s+=txt(40,516,'图示表达数据依赖，不表示必须启动四个 kernel，也不代表实测性能。',17,muted)
save('04-kernel-dataflow.svg',s)
print('Created four SVG diagrams')
