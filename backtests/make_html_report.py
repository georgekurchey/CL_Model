from pathlib import Path
import pandas as pd
import base64, json, datetime

R = Path("reports")
R.mkdir(parents=True, exist_ok=True)
now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def try_read_csv(p):
    p = R / p
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def try_read_text(p):
    p = R / p
    return p.read_text() if p.exists() else None

def try_read_json(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None

def img_tag(path):
    p = R / path
    if not p.exists():
        return ""
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{path}" />'

compare = try_read_csv("compare.csv")
coverage = try_read_csv("coverage.csv")
report_txt = try_read_text("report.txt")
params = try_read_json("model_params.json")

def table_html(df, caption):
    if df is None or df.empty:
        return f"<h3>{caption}</h3><p><em>No data.</em></p>"
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].map(lambda x: f"{x:.6f}")
    return f"<h3>{caption}</h3>" + df2.to_html(index=False, classes='tbl', border=0, escape=False)

def pre_html(text, caption):
    if not text:
        return f"<h3>{caption}</h3><p><em>No data.</em></p>"
    return f"<h3>{caption}</h3><pre>{text}</pre>"

def kv_html(obj, caption):
    if not obj:
        return f"<h3>{caption}</h3><p><em>No data.</em></p>"
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [("qwalk.mix", obj.get("qwalk",{}).get("mix")),
                     ("qwalk.df", obj.get("qwalk",{}).get("df")),
                     ("qwalk.paths", obj.get("qwalk",{}).get("paths")),
                     ("qwalk.cal_scale", obj.get("qwalk",{}).get("cal_scale")),
                     ("svjump.ridge", obj.get("svjump",{}).get("ridge"))]
    )
    return f"<h3>{caption}</h3><table class='tbl'><tbody>{rows}</tbody></table>"

imgs = [
    ("PIT — Quantum Walk (calibrated)", "pit_model_qwalk_cal.png"),
    ("PIT — Quantum Walk (raw)", "pit_model_qwalk.png"),
    ("PIT — SV+Jump", "pit_model_svjump.png"),
    ("PIT — Quantile Regression", "pit_quantile_reg.png"),
    ("PIT — Baseline Quantiles", "pit_baseline_quantiles.png"),
    ("PIT — ARX+GARCH", "pit_arx_garch.png"),
]
img_html = "".join(
    f"<figure><figcaption>{title}</figcaption>{img_tag(fname)}</figure>"
    for title,fname in imgs
)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>CL_Model Results</title>
<style>
  :root {{ --fg:#101418; --bg:#ffffff; --muted:#6b7280; --accent:#0ea5e9; }}
  body {{ margin:2rem auto; max-width:1100px; font-family: ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial; color:var(--fg); background:var(--bg); line-height:1.45; }}
  h1,h2,h3 {{ margin: 0.6em 0 0.4em; }}
  header {{ display:flex; align-items:baseline; gap:1rem; margin-bottom:1rem; }}
  header .stamp {{ color:var(--muted); font-size:0.9rem; }}
  .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:1.2rem; }}
  .full {{ grid-column: 1 / -1; }}
  .card {{ border:1px solid #e5e7eb; border-radius:12px; padding:1rem; }}
  .tbl {{ border-collapse: collapse; width:100%; }}
  .tbl th, .tbl td {{ border-bottom:1px solid #e5e7eb; padding:0.5rem 0.6rem; text-align:left; }}
  pre {{ background:#0b1020; color:#e5e7eb; border-radius:10px; padding:1rem; overflow:auto; }}
  figure {{ margin:0; padding:0.8rem; border:1px solid #e5e7eb; border-radius:12px; display:flex; flex-direction:column; gap:0.6rem; }}
  figcaption {{ color:var(--muted); font-size:0.9rem; }}
  img {{ width:100%; height:auto; border-radius:8px; }}
  footer {{ margin-top:2rem; color:var(--muted); font-size:0.9rem; }}
</style>
</head>
<body>
  <header>
    <h1>CL_Model — Results Dashboard</h1>
    <div class="stamp">Generated {now}</div>
  </header>

  <section class="grid">
    <div class="card">{kv_html(params, "Model Parameters")}</div>
    <div class="card">{table_html(compare, "CRPS Comparison")}</div>
    <div class="card">{table_html(coverage, "Coverage & PIT Summary")}</div>
    <div class="card">{pre_html(report_txt, "Text Summary (report.txt)")}</div>
    <div class="card full">
      <h3>Calibration Plots (PIT)</h3>
      <div class="grid">{img_html}</div>
    </div>
  </section>

  <footer>
    <p>Files sourced from <code>reports/</code> and <code>preds/</code>. Missing items are skipped automatically.</p>
  </footer>
</body>
</html>
"""
out = R / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}")
