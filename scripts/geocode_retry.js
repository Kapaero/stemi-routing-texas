// Retry Census geocoding for unmatched hospital addresses using address variants
// (house number +/- small offsets, highway spelling variants, known alternate street addresses).
// Usage: node geocode_retry.js hosp_geocoded.csv retry.json
const fs = require('fs');
const [, , inPath, outPath] = process.argv;

function parseCSV(text) {
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n') { row.push(f.replace(/\r$/, '')); rows.push(row); row = []; f = ''; }
    else f += c;
  }
  if (f || row.length) { row.push(f); rows.push(row); }
  const h = rows.shift();
  return rows.filter(r => r.length === h.length).map(r => Object.fromEntries(h.map((k, i) => [k, r[i]])));
}

const ALT = {
  '229000': ['710 FM 1960 Rd W'], '026000': ['901 US Highway 83 N'], '832900': ['1 Medical Plz', 'One Medical Plaza'],
  '823400': ['1 Saint Marks Pl'], '356000': ['387 W Interstate 10'], '441000': ['200 S Interstate 35'],
  '847000': ['5016 S US Highway 75'], '820800': ['3000 N Interstate 35'], '285000': ['2400 N Interstate 35E', '2400 N Interstate 35 E'],
  '788002': ['5501 S Expressway 77', '5501 S US Highway 77'], '971000': ['850 W Central Texas Expy'],
  '975208': ['17201 Interstate 45 S'], '724700': ['18220 State Highway 249'], '534001': ['23900 Interstate 10', '23900 Katy Fwy'],
  '823000': ['16655 US Highway 59', '16655 Southwest Fwy'], '975201': ['27800 US Highway 290'], '975138': ['16100 State Highway 288'],
  '091001': ['6250 US Highway 83', '6250 Highway 83'], '141000': ['3201 W State Highway 22'], '466000': ['1717 US Highway 59 Loop N', '1717 Highway 59 Byp'],
  '000112': ['11937 US Highway 271', '11937 Highway 271'], '339002': ['11212 State Hwy 151', '11212 Highway 151'],
  '973120': ['16977 Interstate 35 N', '16977 IH 35 N'], '101100': ['8230 N Loop 1604 W', '8230 Loop 1604 N'],
  '206100': ['700 Scott and White Dr', '700 Scott & White Dr'],
};
const STOP = new Set(['N', 'S', 'E', 'W', 'ST', 'AVE', 'RD', 'DR', 'BLVD', 'PKWY', 'FWY', 'HWY', 'LN', 'WAY', 'CT', 'PL', 'LOOP',
  'STREET', 'AVENUE', 'ROAD', 'DRIVE', 'BOULEVARD', 'PARKWAY', 'FREEWAY', 'HIGHWAY', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'US', 'STATE']);
const streetTokens = a => a.toUpperCase().replace(/[^A-Z0-9 ]/g, ' ').split(/\s+/).slice(1).filter(t => t && !STOP.has(t) && !/^\d+$/.test(t));

function variants(r) {
  const base = [r.addr, ...(ALT[r.THCIC_ID] || [])];
  const out = [];
  for (const a of base) {
    const clean = a.replace(/\bIH\b/i, 'Interstate').replace(/\bI-?\s?(\d+)/i, 'Interstate $1').replace(/\bHWY\b/i, 'Highway')
      .replace(/(\d+)\/\d+/, '$1').replace(/\s+/g, ' ').trim();
    for (const v of new Set([a, clean])) {
      out.push(v);
      const m = v.match(/^(\d+)\s+(.*)$/);
      if (m) for (const d of [1, -1, 2, -2, 4, -4, 10, -10, 20, -20, 50, -50, 100, -100]) {
        const n = +m[1] + d; if (n > 0) out.push(`${n} ${m[2]}`);
      }
    }
  }
  return [...new Set(out)];
}

async function geocode(addr, city, zip) {
  const q = `${addr}, ${city}, TX ${zip}`;
  const url = 'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?benchmark=Public_AR_Current&format=json&address=' + encodeURIComponent(q);
  for (let k = 0; k < 3; k++) {
    try { const r = await fetch(url); if (r.ok) { const j = await r.json(); return j.result.addressMatches || []; } } catch (e) { }
    await new Promise(s => setTimeout(s, 1000));
  }
  return [];
}

(async () => {
  const rows = parseCSV(fs.readFileSync(inPath, 'utf8')).filter(r => r.geo_status === 'No_Match');
  const res = {};
  for (const r of rows) {
    const toks = streetTokens(r.addr).concat(...(ALT[r.THCIC_ID] || []).map(streetTokens));
    let hit = null;
    for (const v of variants(r)) {
      const m = await geocode(v, r.a_city, r.a_zip);
      const ok = m.find(x => toks.length === 0 || toks.some(t => x.matchedAddress.toUpperCase().includes(t)));
      if (ok) { hit = { variant: v, matched: ok.matchedAddress, lat: ok.coordinates.y, lon: ok.coordinates.x }; break; }
    }
    res[r.THCIC_ID] = hit;
    process.stderr.write(`${r.THCIC_ID} ${r.name}: ${hit ? hit.matched : 'none'}\n`);
  }
  fs.writeFileSync(outPath, JSON.stringify(res, null, 1));
  console.log('retried', rows.length, 'matched', Object.values(res).filter(Boolean).length);
})();
