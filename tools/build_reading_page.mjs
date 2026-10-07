#!/usr/bin/env node
/**
 * 把专题 Markdown 构建成自包含的 HTML 阅读页。
 *
 * 产物不依赖任何外部资源：公式用 MathJax 在本地渲染成内联 SVG，
 * 插图读成 base64 内联，所以离线 / 任何机器上打开都一样。
 * 相比 .md 的好处是插图不会被预览器拦掉，SVG 里的动画也能真正播放。
 *
 * 用法：
 *   node tools/build_reading_page.mjs <input.md> [output.html]
 *
 * 依赖（装在隔离的 node workspace，不污染本机）：
 *   npm install mathjax-full
 */

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const REPO = 'https://github.com/zben777/MatInfer/blob/main';
const BODY_WIDTH = 740;

/* ------------------------------- 专题登记表 ------------------------------- */
// tools/topics.json 记录每个专题的 md → 阅读页输出路径，以及侧边栏/面包屑信息。
// 登记表里没有的专题会按路径推导出默认值，所以「只写 md、直接构建」也能跑。

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REGISTRY_PATH = path.join(REPO_ROOT, 'tools', 'topics.json');

function loadRegistry() {
  if (!fs.existsSync(REGISTRY_PATH)) return { topics: [], layerLabels: {} };
  try {
    return JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
  } catch (e) {
    console.warn(`  ! tools/topics.json 解析失败，改用路径推导：${e.message}`);
    return { topics: [], layerLabels: {} };
  }
}

/** 由路径推导导航信息：docs/<层>/<专题>/<名>.md */
function deriveMeta(mdRel, layerLabels = {}) {
  const parts = mdRel.split('/');
  const layer = layerLabels[parts[1]] || parts[1] || '';
  const topic = parts.length >= 4 ? parts[2].replace(/-/g, ' ') : '';
  return {
    navLabel: layer,
    navTitle: topic,
    crumb: [layer, topic].filter(Boolean),
    eyebrow: `图解专题 · ${topic || layer}`,
    sideLinks: [],
  };
}

/* ---------------------------------- 公式 ---------------------------------- */

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const texPackages = new TeX({ packages: AllPackages });
const svgOutput = new SVG({ fontCache: 'local' });
const mathDoc = mathjax.document('', { InputJax: texPackages, OutputJax: svgOutput });

function renderTex(latex, display) {
  const src = latex.trim();
  try {
    const node = mathDoc.convert(src, {
      display,
      em: 16,
      ex: 8,
      containerWidth: BODY_WIDTH,
    });
    return adaptor.outerHTML(node);
  } catch (e) {
    // MathJax 的报错完全不提是哪个公式，这里补上，否则只能靠逐个二分
    throw new Error(
      `公式渲染失败：${JSON.stringify(src.slice(0, 160))}\n  ${e.message}`
    );
  }
}

/* --------------------------------- 工具函数 -------------------------------- */

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** 行内：先摘出 `code` 与 $公式$，再处理强调和链接，最后放回。 */
function inline(text, baseDir) {
  const stash = [];
  const keep = (html) => `\u0000${stash.push(html) - 1}\u0000`;

  let s = text;

  // 行内代码与行内公式必须**一次扫描**：谁先出现谁生效。
  // 分两步会出事——先把 `code` 抽成占位符后，公式里的占位符会被喂给 MathJax 直接崩。
  // 支持三种写法：`code`、GitHub 风格 $`tex`$、普通 $tex$。
  s = s.replace(/`([^`]+)`|\$`([^$]+?)`\$|\$([^$\n]+?)\$/g, (_, code, gfm, tex) => {
    if (code !== undefined) return keep(`<code>${esc(code)}</code>`);
    return keep(renderTex(gfm !== undefined ? gfm : tex, false));
  });

  s = esc(s);

  // 链接
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    const url = /^https?:/.test(href) ? href : `${REPO}/${path.posix.normalize(path.posix.join(baseDir, href))}`;
    return `<a href="${url}">${label}</a>`;
  });
  // 粗体
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[+i]);
}

/* --------------------------------- 块解析 --------------------------------- */

function parseBlocks(src) {
  const lines = src.split('\n');
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i++; continue; }

    // 代码 / 公式围栏
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const lang = fence[1];
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++; // 收尾
      blocks.push({ type: lang === 'math' ? 'math' : 'code', lang, text: buf.join('\n') });
      continue;
    }

    // 裸 HTML：<details>…</details> 折叠块。外壳透传，内部仍按 Markdown 解析。
    if (/^<details[\s>]/i.test(line)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*<\/details>\s*$/i.test(lines[i])) buf.push(lines[i++]);
      i++; // 跳过 </details>
      const inner = buf.join('\n');
      const m = /<summary>[\s\S]*?<\/summary>/i.exec(inner);
      blocks.push({
        type: 'details',
        summary: m ? m[0] : '',
        children: parseBlocks(m ? inner.replace(m[0], '') : inner),
      });
      continue;
    }

    // 标题
    let m;
    if ((m = line.match(/^#\s+(.*)$/))) { blocks.push({ type: 'h1', text: m[1] }); i++; continue; }
    if ((m = line.match(/^##\s+(.*)$/))) { blocks.push({ type: 'h2', text: m[1] }); i++; continue; }
    if ((m = line.match(/^###\s+(.*)$/))) { blocks.push({ type: 'h3', text: m[1] }); i++; continue; }

    // 分隔线
    if (/^---+\s*$/.test(line)) { blocks.push({ type: 'hr' }); i++; continue; }

    // 图片（后面紧跟的斜体行算图注）
    if ((m = line.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/))) {
      blocks.push({ type: 'figure', alt: m[1], src: m[2] });
      i++;
      continue;
    }

    // 图注：整行斜体且不含公式
    if ((m = line.match(/^\*([^*].*?)\*\s*$/)) && !line.startsWith('**')) {
      blocks.push({ type: 'caption', text: m[1] });
      i++;
      continue;
    }

    // 引用
    if (/^>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
      blocks.push({ type: 'quote', text: buf.join('\n') });
      continue;
    }

    // 表格
    if (/^\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++]);
      blocks.push({ type: 'table', rows });
      continue;
    }

    // 段落
    const buf = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^```/.test(lines[i]) &&
      !/^#{1,3}\s/.test(lines[i]) &&
      !/^!\[/.test(lines[i]) &&
      !/^\|/.test(lines[i]) &&
      !/^>\s?/.test(lines[i]) &&
      !/^---+\s*$/.test(lines[i])
    ) buf.push(lines[i++]);
    blocks.push({ type: 'p', text: buf.join(' ') });
  }

  return blocks;
}

/* --------------------------------- 渲染块 --------------------------------- */

function renderTable(rows, baseDir) {
  const cells = (row) =>
    row.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
  const head = cells(rows[0]);
  const body = rows.slice(2).map(cells);

  let html = '<table><thead><tr>';
  html += head.map((c) => `<th>${inline(c, baseDir)}</th>`).join('');
  html += '</tr></thead><tbody>';
  for (const r of body) {
    html += '<tr>' + r.map((c) => `<td>${inline(c, baseDir)}</td>`).join('') + '</tr>';
  }
  return html + '</tbody></table>';
}

/** 图片内联成 base64 SVG；找不到文件就退化成相对路径引用并给出告警。 */
function renderFigure({ alt, src }, mdDir, outDir) {
  const abs = path.resolve(mdDir, src);
  const relFromOut = path.posix.normalize(
    path.relative(outDir, abs).split(path.sep).join('/')
  );
  let dataUri = relFromOut;
  if (fs.existsSync(abs)) {
    const b64 = fs.readFileSync(abs).toString('base64');
    dataUri = `data:image/svg+xml;base64,${b64}`;
  } else {
    console.warn(`  ! 找不到插图：${src}`);
  }
  return (
    `<p><a class="diagram-link" href="${relFromOut}" aria-label="打开大图：${alt}">` +
    `<img class="diagram" src="${dataUri}" alt="${esc(alt)}"></a></p>`
  );
}

/** 渲染一组块。toc 传 null 时（如 <details> 内部）不登记目录。 */
function renderBlocks(list, ctx, toc) {
  const out = [];
  for (const b of list) {
    switch (b.type) {
      case 'h1':
        break; // 标题单独放在 article 外面
      case 'h2': {
        ctx.secNo++;
        const id = `section-${ctx.secNo}`;
        if (toc) toc.push({ id, text: b.text });
        out.push(`<h2 id="${id}">${inline(b.text, ctx.baseDir)}</h2>`);
        break;
      }
      case 'h3':
        out.push(`<h3>${inline(b.text, ctx.baseDir)}</h3>`);
        break;
      case 'p':
        out.push(`<p>${inline(b.text, ctx.baseDir)}</p>`);
        break;
      case 'quote':
        out.push(`<blockquote><p>${inline(b.text, ctx.baseDir)}</p></blockquote>`);
        break;
      case 'math':
        out.push(`<div class="equation">${renderTex(b.text, true)}</div>`);
        break;
      case 'code':
        out.push(
          `<pre><code class="language-${b.lang || 'text'}">${esc(b.text)}</code></pre>`
        );
        break;
      case 'table':
        out.push(renderTable(b.rows, ctx.baseDir));
        break;
      case 'figure':
        out.push(renderFigure(b, ctx.mdDir, ctx.outDir));
        break;
      case 'caption':
        out.push(`<p><em>${inline(b.text, ctx.baseDir)}</em></p>`);
        break;
      case 'details':
        out.push(`<details>${b.summary}\n${renderBlocks(b.children, ctx, null)}</details>`);
        break;
      case 'hr':
        out.push('<hr>');
        break;
    }
  }
  return out.join('\n');
}

/* ---------------------------------- 主流程 --------------------------------- */

function build(mdPath, outPath, meta) {
  const mdDir = path.dirname(path.resolve(mdPath));
  const outDir = path.dirname(path.resolve(outPath));
  // 仓库根由脚本位置推出：链接一律先还原成相对仓库根的路径，再拼 GitHub 前缀，
  // 这样 .md 里的相对链接在阅读页上也能点。
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const baseDir = path.relative(repoRoot, mdDir).split(path.sep).join('/');
  const mdRelToRepo = path.relative(repoRoot, path.resolve(mdPath)).split(path.sep).join('/');
  const src = fs.readFileSync(mdPath, 'utf8');
  const blocks = parseBlocks(src);

  const title = (blocks.find((b) => b.type === 'h1') || {}).text || path.basename(mdPath);
  const sections = blocks.filter((b) => b.type === 'h2');

  const body = [];
  const toc = [];
  const ctx = { baseDir, mdDir, outDir, secNo: 0 };
  body.push(renderBlocks(blocks, ctx, toc));

  const tocHtml = toc
    .map((t) => `<a href="#${t.id}">${esc(t.text)}</a>`)
    .join('');
  const sideLinks = (meta.sideLinks || [])
    .map((l) => `<a class="side-link" href="${l.href}">${l.label}</a>`)
    .join('');

  const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(title)} · MatInfer</title>
<style>
:root{--ink:#272d34;--muted:#7b828b;--line:#edf0f2;--accent:#23836e;--paper:#fff;--nav:#f7f9f8;--font-size:18px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:100px}body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:230px;padding:34px 24px;background:var(--nav);border-right:1px solid var(--line)}.brand{font-size:22px;letter-spacing:-.6px;font-weight:650;color:var(--ink)}.tagline{font-size:12px;color:var(--muted);margin:8px 0 54px}.nav-label{font-size:11px;letter-spacing:1.3px;color:var(--muted);margin:28px 0 12px}.nav-title{font-size:14px;font-weight:600;margin:18px 0 12px}.side-link{display:block;padding:10px 12px;margin-left:-12px;border-radius:6px;color:#67736e;font-size:13px;line-height:1.5}.side-link.active{background:#e9f2ee;color:#226c59;font-weight:550}.side-footer{position:absolute;bottom:24px;font-size:12px;color:var(--muted)}
.main{margin-left:230px}.topbar{height:70px;display:flex;align-items:center;justify-content:space-between;padding:0 44px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}.crumb span{margin:0 12px;color:#bac1c6}.reading-controls{display:flex;gap:8px;align-items:center}.reading-controls button{border:1px solid #e6eaed;background:white;color:#666f78;border-radius:5px;min-width:30px;height:29px;font-size:14px}.reading-controls small{font-size:11px;margin-right:8px}
.layout{display:grid;grid-template-columns:minmax(0,740px) 170px;gap:64px;max-width:1120px;margin:0 auto;padding:68px 40px 100px}.eyebrow{font-size:12px;color:var(--muted);letter-spacing:.3px;margin-bottom:16px}h1{font-size:32px;line-height:1.45;letter-spacing:-.8px;font-weight:650;margin:0 0 40px}
article{font-size:var(--font-size);line-height:1.95;overflow-wrap:break-word}article p{margin:0 0 24px}article strong{font-weight:600}article h2{font-size:23px;line-height:1.6;font-weight:620;letter-spacing:-.3px;margin:52px 0 22px}article h3{font-size:19px;line-height:1.7;margin:34px 0 18px;font-weight:600}article a{border-bottom:1px solid #d4e7de}article a:hover{text-decoration:none;border-color:var(--accent)}article code{font: .84em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f4f6f5;padding:3px 6px;border-radius:4px}
article table{width:100%;border-collapse:collapse;font-size:16px;margin:26px 0 30px;line-height:1.8}article th{text-align:left;font-size:13px;font-weight:550;color:#737c84;background:#fafbfa}article th,article td{padding:13px 15px;border-bottom:1px solid #e9eeeb}article tbody tr:last-child td{border-bottom:1px solid #dce4df}
article mjx-container{font-size:105%!important}article mjx-container svg{max-width:100%;height:auto}article mjx-container[display="true"]{margin:0!important}.equation{margin:28px 0 32px;text-align:center;overflow-x:auto;padding:10px 0}
article blockquote{margin:20px 0 26px;padding:0 0 0 20px;border-left:3px solid #d8dfe5;color:var(--ink)}article blockquote p:last-child{margin-bottom:0}
article details{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:19px 0;margin:38px 0 28px;font-size:16px}article summary{cursor:pointer;color:#5c6f66;font-size:15px}article details[open] summary{margin-bottom:22px}article details p{margin-bottom:18px}
article .diagram{display:block;width:100%;height:auto;border:1px solid #edf0f2;border-radius:12px;margin:30px 0 14px}article .diagram-link{display:block;border:none;cursor:zoom-in}article p:has(.diagram-link){margin-bottom:10px}article p>em:only-child{display:block;font-size:14px;line-height:1.8;color:#78828f;margin-bottom:32px}
article pre{background:#f5f7f9;border:1px solid #e9eef2;border-radius:8px;padding:20px;overflow-x:auto;font-size:14px;line-height:1.8}article pre code{background:none;padding:0;white-space:pre}article hr{border:none;border-top:1px solid var(--line);margin:44px 0}
.article-footer{border-top:1px solid var(--line);margin-top:42px;padding-top:18px;color:var(--muted);font-size:12px;line-height:1.8}
.toc{align-self:start;position:sticky;top:36px;padding-top:5px;font-size:12px}.toc-label{color:#969da4;font-size:11px;margin-bottom:16px;letter-spacing:.5px}.toc a{display:block;color:#848d95;font-size:12px;line-height:1.8;margin:12px 0;padding-left:13px;border-left:2px solid transparent}.toc a.current{color:var(--accent);border-color:#80b8a7}.toc .view-source{margin-top:34px;font-size:11px;color:#a0a7ad}
@media(max-width:1120px){.layout{grid-template-columns:minmax(0,740px);padding:55px 44px;max-width:830px}.toc{display:none}}
@media(max-width:780px){.sidebar{display:none}.main{margin-left:0}.topbar{height:58px;padding:0 24px}.layout{padding:40px 24px 65px}.reading-controls small{display:none}h1{font-size:27px;margin-bottom:30px}article{line-height:1.9}article h2{font-size:21px;margin-top:40px}article table{font-size:14px}article td,article th{padding:10px 8px}.eyebrow{font-size:11px}article .diagram{border-radius:6px}article pre{font-size:12px;padding:12px}article blockquote{padding-left:14px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style>
</head>
<body>
<aside class="sidebar" aria-label="知识目录">
<a class="brand" href="https://github.com/zben777/MatInfer">MatInfer<span style="color:#76a28e">.</span></a>
<p class="tagline">理解推理，记录问题。</p>
<div class="nav-label">${esc(meta.navLabel || '')}</div><div class="nav-title">${esc(meta.navTitle || '')}</div>
${sideLinks}
<div class="side-footer">MatInfer · 阅读页</div>
</aside>
<main class="main"><header class="topbar"><div class="crumb">${(meta.crumb || []).map(esc).join('<span>/</span>')}</div><div class="reading-controls"><small>阅读字号</small><button id="smaller" aria-label="减小字号">−</button><button id="larger" aria-label="增大字号">＋</button></div></header>
<div class="layout"><div><div class="eyebrow">${esc(meta.eyebrow || '图解专题')}</div><h1 id="article-title">${esc(title)}</h1><article id="content">
${body.join('\n')}
<footer class="article-footer">MatInfer · 用图建立直觉，用推导解释细节。</footer>
</article></div>
<nav class="toc" aria-label="本页目录"><div class="toc-label">本页内容</div>${tocHtml}<a class="view-source" href="${REPO}/${mdRelToRepo}">在 GitHub 阅读原文 ↗</a></nav>
</div></main>
<dialog id="diagram-dialog" aria-label="查看专题图" style="padding:16px;border:1px solid #ddd;border-radius:12px;width:min(1200px,95vw);max-height:95vh"><button id="close-diagram" autofocus style="display:block;margin:0 0 12px auto">关闭 ×</button><img id="large-diagram" alt="" style="width:100%;min-width:900px;height:auto"></dialog>
<script>
(()=>{let size=18;const change=d=>{size=Math.max(16,Math.min(22,size+d));document.documentElement.style.setProperty('--font-size',size+'px');document.getElementById('smaller').disabled=size===16;document.getElementById('larger').disabled=size===22};document.getElementById('smaller').addEventListener('click',()=>change(-1));document.getElementById('larger').addEventListener('click',()=>change(1));const links=[...document.querySelectorAll('.toc a[href^="#"]')];const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){links.forEach(l=>l.classList.toggle('current',l.hash==='#'+e.target.id))}},{rootMargin:'-10% 0px -65% 0px'});document.querySelectorAll('article h2').forEach(h=>observer.observe(h));})();
document.querySelectorAll('.diagram-link').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();const img=a.querySelector('img');const big=document.getElementById('large-diagram');big.src=img.src;big.alt=img.alt;document.getElementById('diagram-dialog').showModal()}));document.getElementById('close-diagram').onclick=()=>document.getElementById('diagram-dialog').close();
</script>
</body>
</html>
`;

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(outPath, html, 'utf8');

  return { title, sections: sections.length, figures: blocks.filter((b) => b.type === 'figure').length, bytes: Buffer.byteLength(html) };
}

/* ---------------------------------- CLI ---------------------------------- */

const [mdArg, outArg] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
if (!mdArg) {
  console.error('用法: node tools/build_reading_page.mjs <input.md> [output.html]');
  console.error('输出路径与导航信息默认来自 tools/topics.json，未登记时按路径推导。');
  process.exit(1);
}
const mdPath = path.resolve(mdArg);

const registry = loadRegistry();
const mdRel = path.relative(REPO_ROOT, mdPath).split(path.sep).join('/');
const entry = (registry.topics || []).find((t) => t.md === mdRel) || {};
const outPath = path.resolve(outArg || entry.out || mdPath.replace(/\.md$/, '.html'));

const meta = { ...deriveMeta(mdRel, registry.layerLabels), ...entry };

const stat = build(mdPath, outPath, meta);
console.log(`  标题   ${stat.title}`);
console.log(`  章节   ${stat.sections}`);
console.log(`  插图   ${stat.figures}`);
console.log(`  体积   ${(stat.bytes / 1024).toFixed(0)} KiB`);
console.log(`  输出   ${path.relative(process.cwd(), outPath)}`);
