from pathlib import Path
import re

p = Path("features/roll_utils.py")
src = p.read_text()

m = re.search(r"(^\s*def\s+detect_roll_flags\s*\(.*?\):)", src, flags=re.M)
if not m:
    raise SystemExit("detect_roll_flags not found")
head = m.group(1)
tail = src[m.end():]
m2 = re.search(r"^\s*(def|class)\s+", tail, flags=re.M)
end = m.end() + (m2.start() if m2 else len(tail))

new_body = """
import numpy as np
import pandas as pd

cols = list(dfw.select_dtypes(include=[np.number]).columns)
if not cols:
    return pd.Series(False, index=dfw.index, name="roll_flag")

s = dfw[cols[0]].astype(float)
chg = s.pct_change().abs().fillna(0.0)
max_change = float(chg.max())
if max_change <= float(eps):
    return pd.Series(False, index=dfw.index, name="roll_flag")

pos = int(np.argmax(chg.to_numpy()))
flags = np.zeros(len(dfw), dtype=bool)
if 0 <= pos < len(flags):
    flags[pos] = True
return pd.Series(flags, index=dfw.index, name="roll_flag")
""".strip("\n")

indent = re.match(r"^\s*", head).group(0) + "    "
indented = "\n".join((indent + line) if line else "" for line in new_body.splitlines())
patched = src[:m.start()] + head + "\n" + indented + "\n" + src[end:]
p.write_text(patched)
print("patched detect_roll_flags")
