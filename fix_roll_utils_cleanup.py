from pathlib import Path
import re

p = Path("features/roll_utils.py")
if not p.exists():
    raise SystemExit("missing: features/roll_utils.py")

s = p.read_text()

# 1) Strip any previous broken wrappers and literal \n artifacts
patterns_to_remove = [
    r"_prev_detect_roll_flags\s*=.*",
    r"_final_detect_roll_flags\s*=.*",
    r"detect_roll_flags\s*=\s*_final_detect_roll_flags",
    r"detect_roll_flags\s*=\s*_wrapped_detect_roll_flags",
    r"_orig_detect_roll_flags\s*=.*",
    r"_prev_compute_spliced_returns\s*=.*",
    r"_final_compute_spliced_returns\s*=.*",
    r"_prev2_compute_spliced_returns\s*=.*",
    r"_final2_compute_spliced_returns\s*=.*",
    r"compute_spliced_returns\s*=\s*_final_compute_spliced_returns",
    r"compute_spliced_returns\s*=\s*_final2_compute_spliced_returns",
]
for pat in patterns_to_remove:
    s = re.sub(pat, "", s)

# Remove any stray literal backslash-n sequences left in code
s = s.replace("\\n", "\n")

# Compress multiple blank lines
s = re.sub(r"\n{3,}", "\n\n", s).rstrip() + "\n"

# 2) Append clean, recursion-safe wrappers
wrapper = """
# --- begin auto-patch (idempotent) ---
try:
    _orig_detect_roll_flags  # type: ignore[name-defined]
except NameError:
    try:
        _orig_detect_roll_flags = detect_roll_flags  # type: ignore
    except NameError:
        _orig_detect_roll_flags = None

if _orig_detect_roll_flags is not None:
    def detect_roll_flags(*args, **kwargs):  # type: ignore[override]
        rf = _orig_detect_roll_flags(*args, **kwargs)
        rf = rf.astype(bool)
        rf = (rf & ~rf.shift(1, fill_value=False)).shift(1, fill_value=False)
        rf.name = "roll_flag"
        return rf

try:
    _orig_compute_spliced_returns  # type: ignore[name-defined]
except NameError:
    try:
        _orig_compute_spliced_returns = compute_spliced_returns  # type: ignore
    except NameError:
        _orig_compute_spliced_returns = None

if _orig_compute_spliced_returns is not None:
    def compute_spliced_returns(dfw, rf, *args, **kwargs):  # type: ignore[override]
        rs = _orig_compute_spliced_returns(dfw, rf, *args, **kwargs)
        try:
            import numpy as _np
            pos = _np.where(rf.astype(bool).to_numpy())[0]
            rs.iloc[pos] = 0.0
        except Exception:
            pass
        return rs
# --- end auto-patch ---
""".lstrip("\n")

backup = p.with_suffix(p.suffix + ".cleanbak")
backup.write_text(p.read_text())
p.write_text(s + wrapper)
print(f"cleaned and patched OK; backup -> {backup}")
