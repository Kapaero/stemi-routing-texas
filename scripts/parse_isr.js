// Parse THCIC "Reporting Status of Texas Hospitals" (output of pdftext.js) -> rows: year, THCIC_ID, name, city, note
// Usage: node parse_isr.js year in.json >> out.tsv
const fs = require('fs');
const [, , year, inPath] = process.argv;
const pages = JSON.parse(fs.readFileSync(inPath));

// join text runs, inserting a space when the gap is wider than the previous run's estimated width
function joinRuns(rs) {
  let out = '';
  rs.forEach((r, i) => {
    if (i > 0) {
      const p = rs[i - 1];
      const est = p.text.length * p.size * 0.45;
      if (!/\s$/.test(out) && !/^\s/.test(r.text) && r.x - p.x > est + 1) out += ' ';
    }
    out += r.text;
  });
  return out.replace(/\s+/g, ' ').trim();
}

const allRuns = pages.flatMap(p => p.lines.flatMap(l => l.runs));
const mode = arr => { const c = {}; let best = null; for (const v of arr) { c[v] = (c[v] || 0) + 1; if (best === null || c[v] > c[best]) best = v; } return best; };
const idRuns = allRuns.filter(r => /^\d{6}$/.test(r.text.trim()));
const idX = +mode(idRuns.map(r => Math.round(r.x)));
const idFont = mode(idRuns.map(r => r.font));
const idSize = +mode(idRuns.map(r => r.size));
const hdr = allRuns.find(r => r.text.trim() === 'Reports');
const cut = hdr ? hdr.x - 2 : 285;
const legend = /Inpatient Reporting in Texas|Total facilities|Total reporting/i;

const rows = [];
let city = null, cur = null, done = false;
for (const pg of pages) {
  if (done) break;
  for (const l of pg.lines) {
    const rs = l.runs.filter(r => r.x < cut && r.text.trim());
    if (!rs.length) continue;
    const f = rs[0];
    const text = joinRuns(rs);
    if (/^Page\b/.test(text)) continue;
    if (legend.test(text)) { done = true; break; }
    const atIdCol = Math.abs(Math.round(f.x) - idX) <= 2;
    if (atIdCol && /^\d{6}$/.test(f.text.trim())) {
      cur = { id: f.text.trim(), name: joinRuns(rs.slice(1)), city, note: '' };
      rows.push(cur);
    } else if (atIdCol && (f.font !== idFont || f.size > idSize + 0.5)) {
      if (!cur && f.size > idSize + 2) continue; // title lines on page 1
      city = text; cur = null;
    } else if (cur && f.x > idX + 10) {
      if (f.size < idSize - 0.5 || /report|quarter|close/i.test(text)) cur.note += (cur.note ? '; ' : '') + text;
      else cur.name += ' ' + text;
    }
  }
}
const noteRe = /\s*((First|Last|Final)\s*report.*|Close[ds]?\b.*|Close\d.*|Close$|Merged.*|Formerly.*)$/i;
for (const r of rows) {
  r.name = r.name.replace(/(\S)(Close[ds]?)(\b|\d|$)/, '$1 $2$3');
  const mm = r.name.match(noteRe);
  if (mm) { r.note = mm[1] + (r.note ? '; ' + r.note : ''); r.name = r.name.slice(0, mm.index); }
  r.note = r.note.split('; ').filter(s => !/^(st|nd|rd|th|lv|OC|x)$/.test(s.trim())).join('; ');
  r.name = r.name.replace(/\s*-\s*/g, '-').replace(/\s+/g, ' ').trim();
}
for (const r of rows) console.log([year, r.id, r.name, r.city, r.note].join('\t'));
console.error(`${year}: rows=${rows.length} cities=${new Set(rows.map(r => r.city)).size} noCity=${rows.filter(r => !r.city).length}`);
