// Minimal text-run extractor for simple (Word-generated) PDFs: inflates content streams,
// collects Tj/TJ runs with positions, groups them into lines per page.
// Usage: node pdftext.js in.pdf out.json
const fs = require('fs');
const zlib = require('zlib');

const [, , inPath, outPath] = process.argv;
const buf = fs.readFileSync(inPath);
const lat = buf.toString('latin1');

function inflate(raw) {
  try { return zlib.inflateSync(raw, { finishFlush: zlib.constants.Z_SYNC_FLUSH }); } catch (e) { return null; }
}

// ---- collect streams ----
const streams = [];
const re = /stream\r?\n/g;
let m;
while ((m = re.exec(lat)) !== null) {
  const sIdx = m.index;
  if (lat.slice(sIdx - 3, sIdx) === 'end') continue;
  const dict = lat.slice(lat.lastIndexOf(' obj', sIdx), sIdx);
  const dataStart = sIdx + m[0].length;
  const lm = dict.match(/\/Length (\d+)(\s+\d+\s+R)?/);
  const dataEnd = lm && !lm[2] ? dataStart + +lm[1] : lat.indexOf('endstream', dataStart);
  const raw = buf.subarray(dataStart, dataEnd);
  const data = /\/FlateDecode/.test(dict) ? inflate(raw) : raw;
  if (data) streams.push(data.toString('latin1'));
  re.lastIndex = dataEnd;
}

// ---- content stream parser ----
function parseContent(s) {
  const runs = [];
  const stack = [];
  let tm = [1, 0, 0, 1, 0, 0], tlm = tm.slice(), font = '', size = 0, leading = 0;
  const n = s.length;
  let i = 0;
  const delim = /[\s\/\[\]<>()%{}]/;
  while (i < n) {
    const c = s[i];
    if (c === '(') {
      let depth = 1, j = i + 1, out = '';
      while (j < n) {
        const ch = s[j];
        if (ch === '\\') {
          const nx = s[j + 1];
          if (/[0-7]/.test(nx)) {
            let oct = nx, k = j + 2;
            while (k < j + 4 && /[0-7]/.test(s[k])) { oct += s[k]; k++; }
            out += String.fromCharCode(parseInt(oct, 8)); j = k; continue;
          }
          if (nx === '\r' || nx === '\n') { j += 2; if (nx === '\r' && s[j] === '\n') j++; continue; }
          const map = { n: '\n', r: '\r', t: '\t', b: '\b', f: '\f' };
          out += map[nx] !== undefined ? map[nx] : nx; j += 2; continue;
        }
        if (ch === '(') depth++;
        else if (ch === ')') { depth--; if (depth === 0) break; }
        out += ch; j++;
      }
      stack.push({ t: 'str', v: out }); i = j + 1; continue;
    }
    if (c === '<' && s[i + 1] === '<') {
      let depth = 0, j = i;
      while (j < n) {
        if (s.startsWith('<<', j)) { depth++; j += 2; }
        else if (s.startsWith('>>', j)) { depth--; j += 2; if (!depth) break; }
        else j++;
      }
      stack.push({ t: 'dict' }); i = j; continue;
    }
    if (c === '<') { const j = s.indexOf('>', i); stack.push({ t: 'hex', v: s.slice(i + 1, j) }); i = j + 1; continue; }
    if (c === '[') { stack.push({ t: 'arrStart' }); i++; continue; }
    if (c === ']') {
      const arr = [];
      while (stack.length && stack[stack.length - 1].t !== 'arrStart') arr.unshift(stack.pop());
      stack.pop(); stack.push({ t: 'arr', v: arr }); i++; continue;
    }
    if (c === '/') { let j = i + 1; while (j < n && !delim.test(s[j])) j++; stack.push({ t: 'name', v: s.slice(i + 1, j) }); i = j; continue; }
    if (c === '%') { const j = s.indexOf('\n', i); i = j === -1 ? n : j + 1; continue; }
    if (/\s/.test(c)) { i++; continue; }
    let j = i; while (j < n && !delim.test(s[j])) j++;
    if (j === i) { i++; continue; }
    const tok = s.slice(i, j); i = j;
    if (/^[+-]?(\d+\.?\d*|\.\d+)$/.test(tok)) { stack.push({ t: 'num', v: parseFloat(tok) }); continue; }
    const ops = stack.splice(0);
    const nv = k => (ops[k] && ops[k].t === 'num') ? ops[k].v : 0;
    switch (tok) {
      case 'BT': tm = [1, 0, 0, 1, 0, 0]; tlm = tm.slice(); break;
      case 'Tf': font = ops[0] && ops[0].v; size = nv(1); break;
      case 'Tm': tm = [nv(0), nv(1), nv(2), nv(3), nv(4), nv(5)]; tlm = tm.slice(); break;
      case 'Td': case 'TD': {
        const tx = nv(0), ty = nv(1);
        tlm = [tlm[0], tlm[1], tlm[2], tlm[3], tlm[4] + tx * tlm[0] + ty * tlm[2], tlm[5] + tx * tlm[1] + ty * tlm[3]];
        tm = tlm.slice(); if (tok === 'TD') leading = -ty; break;
      }
      case 'TL': leading = nv(0); break;
      case 'T*': tlm = [tlm[0], tlm[1], tlm[2], tlm[3], tlm[4] - leading * tlm[2], tlm[5] - leading * tlm[3]]; tm = tlm.slice(); break;
      case 'Tj': case "'": case '"': {
        const st = ops[ops.length - 1];
        if (st && st.t === 'str') runs.push({ x: tm[4], y: tm[5], size: size * Math.abs(tm[3] || 1), font, text: st.v });
        break;
      }
      case 'TJ': {
        const arr = (ops[0] && ops[0].v) || [];
        let t = '';
        for (const el of arr) { if (el.t === 'str') t += el.v; else if (el.t === 'num' && el.v < -250) t += ' '; }
        runs.push({ x: tm[4], y: tm[5], size: size * Math.abs(tm[3] || 1), font, text: t });
        break;
      }
    }
  }
  return runs;
}

const pages = [];
for (const s of streams) {
  if (!/\bBT\b/.test(s) || !/T[jJ]/.test(s)) continue;
  const runs = parseContent(s);
  if (!runs.length) continue;
  const all = runs.map(r => r.text).join(' ');
  const pm = all.match(/Page\s+(\d+)/);
  // group into lines by y (tolerance 2pt)
  runs.sort((a, b) => b.y - a.y || a.x - b.x);
  const lines = [];
  for (const r of runs) {
    const last = lines[lines.length - 1];
    if (last && Math.abs(last.y - r.y) <= 2) last.runs.push(r);
    else lines.push({ y: r.y, runs: [r] });
  }
  for (const l of lines) l.runs.sort((a, b) => a.x - b.x);
  pages.push({ page: pm ? +pm[1] : null, lines: lines.map(l => ({ y: +l.y.toFixed(1), runs: l.runs.map(r => ({ x: +r.x.toFixed(1), size: +r.size.toFixed(1), font: r.font, text: r.text })) })) });
}
pages.sort((a, b) => (a.page ?? 1e9) - (b.page ?? 1e9));
fs.writeFileSync(outPath, JSON.stringify(pages));
console.log('streams', streams.length, 'text pages', pages.length, 'page numbers', pages.map(p => p.page).join(','));
