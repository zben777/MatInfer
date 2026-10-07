#!/usr/bin/env node
/**
 * 一条命令重建 MatInfer 的全部图源与阅读页，并检查链接。
 *
 *   node tools/build_all.mjs                 # 全流程
 *   node tools/build_all.mjs --check-only    # 只做链接检查，不重建
 *   node tools/build_all.mjs --skip-figures  # 跳过图源重生成
 *
 * 依赖（装在隔离的 node workspace，不污染本机）：
 *   NODE_PATH=/Users/benzhang/.workbuddy/binaries/node/workspace/node_modules
 *
 * 设计要点：
 *   - 专题清单与导航信息来自 tools/topics.json，不在脚本里写死。
 *   - managed=true 的专题会被重建；managed=false 的（含手工样式/侧栏）只做链接检查，
 *     避免把手工内容覆盖掉。
 *   - 链接检查按 HTML **实际所在层级**解析相对路径——docs/ 下只需 ../assets/，
 *     docs/<层>/<专题>/ 下才需要 ../../../assets/。这个坑真的发生过。
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TOOLS = path.join(REPO_ROOT, 'tools');
const NODE = process.execPath;
const PYTHON = process.env.PYTHON || 'python3';

const argv = process.argv.slice(2);
const OPT = {
  checkOnly: argv.includes('--check-only'),
  skipFigures: argv.includes('--skip-figures') || argv.includes('--check-only'),
};

const rel = (p) => path.relative(REPO_ROOT, p).split(path.sep).join('/');
const sha = (p) =>
  fs.existsSync(p) ? crypto.createHash('sha1').update(fs.readFileSync(p)).digest('hex') : null;
const isExternal = (u) => /^(https?:|data:|mailto:|#|\/\/)/.test(u);

function walk(dir, filter, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === 'node_modules' || e.name.startsWith('.')) continue;
      walk(p, filter, out);
    } else if (filter(p)) out.push(p);
  }
  return out;
}

function run(cmd, args, label) {
  try {
    return { ok: true, out: execFileSync(cmd, args, { cwd: REPO_ROOT, encoding: 'utf8' }) };
  } catch (e) {
    return { ok: false, out: `${e.stdout || ''}${e.stderr || ''}${e.message}` };
  }
}

/* ------------------------------- 链接检查 ------------------------------- */

/** 检查一个 HTML 里所有相对引用是否能落到真实文件。 */
function checkHtml(htmlPath) {
  const dir = path.dirname(htmlPath);
  const html = fs.readFileSync(htmlPath, 'utf8');
  const issues = [];
  const test = (u, what) => {
    if (isExternal(u) || !u) return;
    const clean = decodeURIComponent(u.split('#')[0]);
    if (!clean) return;
    if (!fs.existsSync(path.resolve(dir, clean))) issues.push(`${what}断链 → ${u}`);
  };

  for (const m of html.matchAll(/class="diagram-link" href="([^"]+)"/g)) test(m[1], '放大图');
  for (const m of html.matchAll(/<img\b[^>]*\ssrc="([^"]+)"/g)) test(m[1], '图片');
  for (const m of html.matchAll(/<a\b[^>]*\shref="([^"]+)"/g)) {
    if (m[0].includes('diagram-link')) continue; // 放大图已单独检查，避免重复报同一条
    test(m[1], '相对链接');
  }
  return issues;
}

/** 检查 md 里的插图引用。 */
function checkMd(mdPath) {
  const dir = path.dirname(mdPath);
  const src = fs.readFileSync(mdPath, 'utf8');
  const issues = [];
  for (const m of src.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
    const u = m[1];
    if (isExternal(u)) continue;
    if (!fs.existsSync(path.resolve(dir, u))) issues.push(`插图断链 → ${u}`);
  }
  return issues;
}

/* --------------------------------- 主流程 -------------------------------- */

const registry = JSON.parse(fs.readFileSync(path.join(TOOLS, 'topics.json'), 'utf8'));
const topics = registry.topics || [];
let problems = 0;

console.log('\nMatInfer 构建总览');
console.log('='.repeat(62));

/* 1. 图源 */
if (!OPT.skipFigures) {
  console.log('\n[1/3] 图源');
  const gen = fs
    .readdirSync(TOOLS)
    .filter((f) => /^generate_.*_figures\.py$/.test(f))
    .sort();
  if (!gen.length) console.log('  （没有找到 tools/generate_*_figures.py）');
  for (const g of gen) {
    const before = new Map(
      walk(path.join(REPO_ROOT, 'assets', 'images'), (p) => p.endsWith('.svg')).map((p) => [p, sha(p)])
    );
    const r = run(PYTHON, [path.join(TOOLS, g)], g);
    const after = walk(path.join(REPO_ROOT, 'assets', 'images'), (p) => p.endsWith('.svg'));
    const changed = after.filter((p) => before.get(p) !== sha(p));
    const added = after.filter((p) => !before.has(p));
    if (!r.ok) {
      problems++;
      console.log(`  ✗ ${g} 执行失败`);
      console.log(String(r.out).split('\n').slice(-6).map((l) => '      ' + l).join('\n'));
    } else {
      const note =
        changed.length || added.length
          ? `变化 ${changed.length} 张，新增 ${added.length} 张`
          : `assets/images 共 ${after.length} 张 SVG，无变化`;
      console.log(`  ✓ ${g.padEnd(34)} ${note}`);
      for (const p of [...added, ...changed]) console.log(`      · ${rel(p)}`);
    }
  }
} else {
  console.log('\n[1/3] 图源（已跳过）');
}

/* 2. 阅读页 */
console.log('\n[2/3] 阅读页');
const builtPages = [];
for (const t of topics) {
  const mdPath = path.join(REPO_ROOT, t.md);
  if (!fs.existsSync(mdPath)) {
    problems++;
    console.log(`  ✗ ${t.md} —— md 不存在`);
    continue;
  }
  const outPath = path.resolve(REPO_ROOT, t.out || t.md.replace(/\.md$/, '.html'));
  console.log(`  ${t.md}`);
  if (t.managed === false) {
    console.log(`      跳过重建（managed=false，含手工内容）→ 仅检查 ${rel(outPath)}`);
    if (fs.existsSync(outPath)) builtPages.push(outPath);
    continue;
  }
  if (OPT.checkOnly) {
    console.log('      跳过重建（--check-only）');
    if (fs.existsSync(outPath)) builtPages.push(outPath);
    continue;
  }
  const before = sha(outPath);
  const r = run(NODE, [path.join(TOOLS, 'build_reading_page.mjs'), t.md], t.md);
  if (!r.ok) {
    problems++;
    console.log('      ✗ 构建失败');
    console.log(String(r.out).split('\n').slice(0, 10).map((l) => '        ' + l).join('\n'));
    continue;
  }
  const after = sha(outPath);
  const stat = String(r.out)
    .trim()
    .split('\n')
    .map((l) => l.trim())
    .join(' · ');
  console.log(`      ✓ ${stat}`);
  console.log(`      ${before === after ? '未变化' : before ? '已更新' : '新建'} → ${rel(outPath)}`);
  builtPages.push(outPath);
}

/* 3. 链接检查 */
console.log('\n[3/3] 链接检查');

// 3a. 所有登记专题的 HTML（含 managed=false 的）
const seen = new Set();
for (const t of topics) {
  const outPath = path.resolve(REPO_ROOT, t.out || t.md.replace(/\.md$/, '.html'));
  if (seen.has(outPath)) continue;
  seen.add(outPath);
  if (!fs.existsSync(outPath)) continue;
  const issues = checkHtml(outPath);
  problems += issues.length;
  console.log(`  ${issues.length ? '✗' : '✓'} ${rel(outPath)}${issues.length ? '' : '   OK'}`);
  for (const i of issues) console.log(`      · ${i}`);
}

// 3b. 仓库里其它 HTML（未登记的）
const allHtml = walk(path.join(REPO_ROOT, 'docs'), (p) => p.endsWith('.html'));
let extra = 0;
for (const h of allHtml) {
  if (seen.has(h)) continue;
  const issues = checkHtml(h);
  extra++;
  problems += issues.length;
  console.log(`  ${issues.length ? '✗' : '✓'} ${rel(h)}${issues.length ? '' : '   OK'}（未登记）`);
  for (const i of issues) console.log(`      · ${i}`);
}
if (!extra) console.log('  ✓ 没有未登记的 HTML');

// 3c. md 的插图引用 + 未登记专题提醒
const mdFiles = walk(path.join(REPO_ROOT, 'docs'), (p) => p.endsWith('.md'));
let mdIssues = 0;
const withFigures = [];
for (const md of mdFiles) {
  const issues = checkMd(md);
  mdIssues += issues.length;
  if (/assets\/images\//.test(fs.readFileSync(md, 'utf8'))) withFigures.push(rel(md));
  for (const i of issues) console.log(`      · ${rel(md)}: ${i}`);
}
problems += mdIssues;
console.log(`  ${mdIssues ? '✗' : '✓'} md 插图引用（${mdFiles.length} 个文件）${mdIssues ? '' : '   OK'}`);

const registeredMd = new Set(topics.map((t) => t.md));
const unregistered = withFigures.filter((m) => !registeredMd.has(m));
if (unregistered.length) {
  console.log('\n  ! 引用了 assets/images 但未登记到 topics.json（不会生成阅读页）：');
  for (const m of unregistered) console.log(`      · ${m}`);
}

/* 汇总 */
console.log('\n' + '='.repeat(62));
console.log(
  problems
    ? `发现 ${problems} 个问题，请逐条处理。`
    : `全部通过：${topics.length} 个登记专题，${builtPages.length} 个阅读页，链接无断链。`
);
process.exit(problems ? 1 : 0);
