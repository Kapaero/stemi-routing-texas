// Name-based lookup in OpenStreetMap Nominatim for hospitals the Census geocoder could not place.
// Public hospital names only; 1 request/second per the Nominatim usage policy.
// Usage: node geocode_osm.js hosp_geocoded.csv geocode_retry.json zcta2010_tx.csv osm.json
const fs = require('fs');
const [, , geoPath, retryPath, zctaPath, outPath] = process.argv;
const sleep = ms => new Promise(r => setTimeout(r, ms));

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
const hav = (a, b, c, d) => { const R = 6371, r = Math.PI / 180; const x = Math.sin((c - a) * r / 2) ** 2 + Math.cos(a * r) * Math.cos(c * r) * Math.sin((d - b) * r / 2) ** 2; return 2 * R * Math.asin(Math.sqrt(x)); };

(async () => {
  const retry = JSON.parse(fs.readFileSync(retryPath, 'utf8'));
  const z = Object.fromEntries(fs.readFileSync(zctaPath, 'utf8').trim().split('\n').slice(1).map(l => { const [k, la, lo] = l.split(','); return [k, [+la, +lo]]; }));
  const rows = parseCSV(fs.readFileSync(geoPath, 'utf8')).filter(r => r.geo_status === 'No_Match' && !retry[r.THCIC_ID]);
  const out = {};
  for (const r of rows) {
    const zc = z[r.a_zip];
    const queries = [`${r.name}, ${r.a_city}, Texas`, `${r.cname}, ${r.a_city}, Texas`, `${r.addr}, ${r.a_city}, Texas ${r.a_zip}`];
    let hit = null;
    for (const q of [...new Set(queries)]) {
      const url = 'https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&countrycodes=us&q=' + encodeURIComponent(q);
      let res = [];
      try { const resp = await fetch(url, { headers: { 'User-Agent': 'hospital-geocoding-research-script/1.0' } }); if (resp.ok) res = await resp.json(); } catch (e) { }
      await sleep(1100);
      const ok = res.map(x => ({ ...x, km: zc ? hav(zc[0], zc[1], +x.lat, +x.lon) : 0 }))
        .filter(x => x.km < 25)
        .sort((a, b) => ((b.category === 'amenity' || b.category === 'healthcare') - (a.category === 'amenity' || a.category === 'healthcare')) || a.km - b.km);
      if (ok.length) { hit = { query: q, display: ok[0].display_name, cat: `${ok[0].category}/${ok[0].type}`, lat: +ok[0].lat, lon: +ok[0].lon, km_from_zip: +ok[0].km.toFixed(1) }; break; }
    }
    out[r.THCIC_ID] = hit;
    process.stderr.write(`${r.THCIC_ID} ${r.name}: ${hit ? hit.cat + ' | ' + hit.display.slice(0, 90) : 'none'}\n`);
  }
  fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
  console.log('osm tried', rows.length, 'found', Object.values(out).filter(Boolean).length);
})();
