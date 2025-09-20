from pathlib import Path
p = Path("features/roll_utils.py")
if not p.exists():
    raise SystemExit("missing: features/roll_utils.py")
s = p.read_text()
if "detect_roll_flags = _wrapped_detect_roll_flags" in s:
    print("already patched"); raise SystemExit(0)
if "def detect_roll_flags" not in s:
    raise SystemExit("could not find detect_roll_flags in text")
backup = p.with_suffix(p.suffix + ".bak")
backup.write_text(s)
block = """
try:
    _orig_detect_roll_flags = detect_roll_flags
    def _wrapped_detect_roll_flags(*args, **kwargs):
        rf = _orig_detect_roll_flags(*args, **kwargs)
        rf = rf.astype(bool)
        rf = rf & ~rf.shift(1, fill_value=False)
        rf.name = "roll_flag"
        return rf
    detect_roll_flags = _wrapped_detect_roll_flags
except Exception:
    pass
"""
p.write_text(s.rstrip() + "\n" + block)
print("patched OK; backup ->", backup)
