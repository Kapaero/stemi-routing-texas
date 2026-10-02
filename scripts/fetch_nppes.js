// Fetch NPI-2 (organization) records with hospital taxonomies for given Texas cities from the public NPPES API.
// Usage: node fetch_nppes.js cities.txt out.json
const fs = require('fs');
const [, , citiesPath, outPath] = process.argv;
const cities = fs.readFileSync(citiesPath, 'utf8').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
const TAX = ['General Acute Care Hospital'];
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function get(url) {
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (res.ok) return await res.json();
    } catch (e) { /* retry */ }
    await sleep(1500 * (attempt + 1));
  }
  return null;
}

(async () => {
  const out = {};
  let calls = 0;
  for (const city of cities) {
    for (const tax of TAX) {
      for (let skip = 0; skip <= 1000; skip += 200) {
        const url = `https://npiregistry.cms.hhs.gov/api/?version=2.1&enumeration_type=NPI-2&state=TX&city=${encodeURIComponent(city)}` +
          `&taxonomy_description=${encodeURIComponent(tax)}&limit=200&skip=${skip}`;
        const j = await get(url); calls++;
        await sleep(250);
        if (!j || !j.results || !j.results.length) break;
        for (const r of j.results) {
          const loc = (r.addresses || []).find(a => a.address_purpose === 'LOCATION') || {};
          out[r.number] = {
            npi: r.number, query_city: city, org: r.basic.organization_name || '',
            other: (r.other_names || []).map(o => o.organization_name).filter(Boolean),
            subpart: r.basic.organizational_subpart || '', status: r.basic.status || '',
            enum_date: r.basic.enumeration_date || '', updated: r.basic.last_updated || '',
            addr1: loc.address_1 || '', addr2: loc.address_2 || '', city: loc.city || '', zip: (loc.postal_code || '').slice(0, 5),
            tax: (r.taxonomies || []).map(t => (t.primary ? '*' : '') + t.desc).join('; '),
          };
        }
        if (j.results.length < 200) break;
      }
    }
    process.stderr.write(`${city}: total ${Object.keys(out).length}\n`);
  }
  fs.writeFileSync(outPath, JSON.stringify(Object.values(out)));
  console.log('calls', calls, 'records', Object.keys(out).length);
})();
