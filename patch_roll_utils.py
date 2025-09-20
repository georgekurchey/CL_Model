from pathlib import Path
import re, sys

p = Path("features/roll_utils.py")
if not p.exists():
    sys.exit("missing: features/roll_utils.py")

s = p.read_text()

if "rf & ~rf.shift(1, fill_value=False)" in s:
    print("already patched")
    sys.exit(0)

m = re.search(r"^(\s*)def\s+detect_roll_flags\s*\(.*\):", s, re.M)
if not m:
    sys.exit("could not find detect_roll_flags")
base_indent = m.group(1)
start = m.end()

ret = None
for m2 in re.finditer(r"^(\s*)return\b.*", s[start:], re.M):
    if len(m2.group(1)) > len(base_indent):
        ret = start + m2.start(), m2.group(1)
        break
if not ret:
    sys.exit("could not find return inside detect_roll_flags")

ret_pos, ret_indent = ret
before = s[:ret_pos]
after = s[ret_pos:]
after = re.sub(r"^\s*return\b.*", "", after, count=1, flags=re.M)

block = (
    f"{ret_indent}rf = rf.astype(bool)\n"
    f"{ret_indent}rf = rf & ~rf.shift(1, fill_value=False)\n"
    f'{ret_indent}rf.name = "roll_flag"\n'
    f"{ret_indent}return rf\n"
)

backup = p.with_suffix(p.suffix + ".bak")
backup.write_text(s)
p.write_text(before + block + after)
print("patched OK; backup ->", backup)
