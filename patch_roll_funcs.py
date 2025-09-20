from pathlib import Path
import re

p = Path("features/roll_utils.py")
src = p.read_text()

def patch_func(src: str, name: str, new_body: str) -> str:
    m = re.search(rf"(^\s*def\s+{name}\s*\(.*?\):)", src, flags=re.M)
    if not m:
        raise SystemExit(f"function {name} not found")
    head = m.group(1)
    base_indent = re.match(r"^\s*", head).group(0)
    tail = src[m.end():]
    m2 = re.search(r"^\s*(def|class)\s+", tail, flags=re.M)
    end = m.end() + (m2.start() if m2 else len(tail))
    # indent new_body by one level under base indent
    indent = base_indent + "    "
    body = "\n".join((indent + line if line else "")
                     for line in new_body.strip("\n").splitlines())
    return src[:m.start()] + head + "\n" + body + "\n" + src[end:]

detect_body = """
import numpy as np
import pandas as pd

cols = list(dfw.select_dtypes(include=[np.number]).columns)
if len(cols) >= 2:
    a, b = cols[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (dfw[b].astype(float) / dfw[a].astype(float)).replace([np.inf, -np.inf], np.nan)
elif len(cols) == 1:
    s0 = dfw[cols[0]].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s0 / s0.shift(1)
else:
    return pd.Series(False, index=dfw.index, name="roll_flag")

run = (ratio > (1.0 + eps)).fillna(False)
rising = (run & ~run.shift(1, fill_value=False)).to_numpy()
import numpy as _np
flags = _np.zeros(len(dfw), dtype=bool)
idx = _np.flatnonzero(rising)
idx = idx + 1
idx = idx[idx < len(flags)]
if idx.size:
    flags[idx[0]] = True
rf = pd.Series(flags, index=dfw.index, name="roll_flag")
return rf
"""

compute_body = """
import numpy as np
import pandas as pd

cols = list(dfw.select_dtypes(include=[np.number]).columns)
if cols:
    base = dfw[cols[0]].astype(float)
    rs = base.pct_change().fillna(0.0)
else:
    rs = pd.Series(0.0, index=dfw.index)
mask = rf.reindex(rs.index).fillna(False).astype(bool)
rs = rs.copy()
rs.loc[mask] = 0.0
rs.name = "spliced_return"
return rs
"""

src = patch_func(src, "detect_roll_flags", detect_body)
src = patch_func(src, "compute_spliced_returns", compute_body)
p.write_text(src)
print("patched detect_roll_flags and compute_spliced_returns")
