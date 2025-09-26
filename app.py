from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from flask import Flask, jsonify, Response, send_from_directory

from datetime import datetime
import os



app = Flask(__name__)

R = Path("reports")
P = Path("preds")

def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        return []

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def list_images() -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    if R.exists():
        for f in sorted(R.iterdir()):
            if f.suffix.lower() in exts:
                files.append(f.name)
    return files

def humanize_compare(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rename = {"model": "Model", "CRPS": "CRPS (↓ better)", "PIT_mean": "PIT Mean"}
    return [{rename.get(k, k): v for k, v in row.items()} for row in rows]

def humanize_coverage(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rename = {"model": "Model", "PIT_KS_p": "PIT K–S p-value", "VaR05_cov": "5% VaR coverage", "VaR95_cov": "95% VaR coverage"}
    return [{rename.get(k, k): v for k, v in row.items()} for row in rows]

@app.route("/api")
def api() -> Response:
    compare = humanize_compare(read_csv(R / "compare.csv"))
    coverage = humanize_coverage(read_csv(R / "coverage.csv"))
    params = read_json(R / "model_params.json")
    report_txt = (R / "report.txt").read_text() if (R / "report.txt").exists() else ""
    images = list_images()
    data = {
        "compare": compare,
        "coverage": coverage,
        "params": params,
        "report_text": report_txt,
        "images": images,
    }
    return jsonify(data)

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta http-equiv="refresh" content="{auto_refresh_sec}">
<meta charset="utf-8" />
<title>CL_Model — Local Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { --fg:#0f172a; --bg:#ffffff; --muted:#6b7280; --card:#f8fafc; --accent:#0ea5e9; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e5e7eb; --bg:#0b1020; --muted:#94a3b8; --card:#0f172a; --accent:#22d3ee; }
  }
  * { box-sizing: border-box; }
  body { margin: 2rem auto; max-width: 1140px; font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Helvetica, Arial; color: var(--fg); background: var(--bg); }
  header { display:flex; gap:1rem; align-items:baseline; margin-bottom:1.25rem; }
  h1 { margin:0; font-size:1.6rem; }
  .muted { color: var(--muted); font-size: .95rem; }
  .grid { display:grid; gap:1rem; grid-template-columns: repeat(12, 1fr); }
  .card { background: var(--card); border: 1px solid rgba(148,163,184,.25); border-radius: 14px; padding: 1rem; }
  .span-6 { grid-column: span 6; } .span-12 { grid-column: span 12; }
  table { width:100%; border-collapse: collapse; }
  th, td { padding: .55rem .6rem; border-bottom: 1px solid rgba(148,163,184,.25); text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  .chips { display:flex; flex-wrap: wrap; gap:.4rem; }
  .chip { padding: .35rem .55rem; border-radius: 999px; background: rgba(14,165,233,.15); color: var(--accent); font-size: .85rem; }
  .imgs { display:grid; gap: .8rem; grid-template-columns: repeat(2, 1fr); }
  .imgs figure { margin:0; }
  .imgs img { width: 100%; height: auto; border-radius: 10px; border: 1px solid rgba(148,163,184,.25); }
  pre { white-space: pre-wrap; font-size:.95rem; }
</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
  <header>
    <h1>CL_Model — Local Dashboard</h1>
    <div class="muted">serving from <code>reports/</code> and <code>preds/</code></div>
  </header>

  <div class="grid">
    <section class="card span-6">
      <h3>Model Parameters</h3>
      <div id="params"></div>
    </section>

    <section class="card span-6">
      <h3>Models Present</h3>
      <div class="chips" id="models"></div>
    </section>

    <section class="card span-6">
      <h3>CRPS (lower is better)</h3>
      <div id="crps_bar" style="height:320px;"></div>
      <div id="crps_tbl"></div>
    </section>

    <section class="card span-6">
      <h3>Coverage & PIT</h3>
      <div id="cov_bar" style="height:320px;"></div>
      <div id="cov_tbl"></div>
    </section>

    <section class="card span-12">
      <h3>PIT / Calibration Plots</h3>
      <div class="imgs" id="imgs"></div>
    </section>

    <section class="card span-12">
      <h3>Text Summary</h3>
      <pre id="txt"></pre>
    </section>
  </div>

<script>
async function load() {
  const res = await fetch('/api');
  const data = await res.json();

  const params = document.getElementById('params');
  params.innerHTML = Object.keys(data.params||{}).length
    ? '<pre>'+JSON.stringify(data.params, null, 2)+'</pre>'
    : '<div class="muted">No parameters file found</div>';

  const modelsDiv = document.getElementById('models');
  const predNames = [
    {"name":"Quantum Walk (raw)", "file":"model_qwalk.parquet"},
    {"name":"Quantum Walk (calibrated)", "file":"model_qwalk_cal.parquet"},
    {"name":"SV + Jumps", "file":"model_svjump.parquet"}
  ];
  predNames.forEach(p => {
    const el = document.createElement('span');
    el.className = 'chip';
    el.textContent = p.name;
    modelsDiv.appendChild(el);
  });

  function table(containerId, rows) {
    if (!rows || !rows.length) {
      document.getElementById(containerId).innerHTML = '<div class="muted">No data found</div>';
      return;
    }
    const cols = Object.keys(rows[0]);
    let html = '<table><thead><tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    html += rows.map(r => '<tr>'+cols.map(c=>`<td>${r[c]}</td>`).join('')+'</tr>').join('');
    html += '</tbody></table>';
    document.getElementById(containerId).innerHTML = html;
  }

  function bar(containerId, rows, xKey, yKey, title) {
    if (!rows || !rows.length) {
      document.getElementById(containerId).innerHTML = '';
      return;
    }
    const x = rows.map(r => r[xKey]);
    const y = rows.map(r => Number(r[yKey]));
    const trace = {type:'bar', x, y, marker:{line:{width:0}}};
    const layout = {
      title, margin: {t:30,l:40,r:10,b:60},
      xaxis: {tickangle: -20},
      yaxis: {rangemode: 'tozero'},
      paper_bgcolor:'transparent', plot_bgcolor:'transparent'
    };
    Plotly.newPlot(containerId, [trace], layout, {displayModeBar:false, responsive:true});
  }

  table('crps_tbl', data.compare);
  bar('crps_bar', data.compare, 'Model', 'CRPS (↓ better)', 'CRPS by Model');

  table('cov_tbl', data.coverage);
  bar('cov_bar', data.coverage, 'Model', '95% VaR coverage', '95% VaR Coverage');

  const imgs = (data.images||[]).filter(n => n.toLowerCase().includes('pit'));
  const imgWrap = document.getElementById('imgs');
  if (!imgs.length) {
    imgWrap.innerHTML = '<div class="muted">No PIT images found</div>';
  } else {
    imgWrap.innerHTML = imgs.map(n => `
      <figure>
        <img src="/reports/${n}" alt="${n}" />
        <figcaption>${n.replace('pit_','').replace('.png','').replaceAll('_',' ')}</figcaption>
      </figure>
    `).join('');
  }

  document.getElementById('txt').textContent = data.report_text || 'No report.txt found.';
}

load();
</script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return INDEX_HTML

@app.route("/reports/<path:fname>")
def serve_report_file(fname: str):
    if not (R / fname).exists():
        return ("Not found", 404)
    return send_from_directory(str(R), fname)

if __name__ == "__main__":
    R.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
