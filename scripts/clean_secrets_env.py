import re, sys, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/georgekurchey/CL_Model")
SECRETS = ROOT / "config" / "secrets.env"

# Vars we care about
VARS = ["NASDAQ_DATA_LINK_API_KEY", "FRED_API_KEY", "EIA_API_KEY"]

def clean_val(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    # If prompt text sneaked in, keep only plausible key tokens (last one)
    candidates = re.findall(r"[A-Za-z0-9_-]{16,}", s)
    if candidates:
        return candidates[-1]
    # Fallback: last whitespace-delimited token
    parts = s.split()
    return parts[-1] if parts else s

def parse_exports(text: str) -> dict:
    vals = {}
    for line in text.splitlines():
        m = re.match(r'^\s*export\s+([A-Z0-9_]+)\s*=\s*(.*)$', line)
        if not m: 
            continue
        var, rhs = m.group(1), m.group(2).strip()
        if var in VARS:
            # drop inline comments
            rhs = rhs.split("#", 1)[0].strip()
            if rhs.startswith(("'", '"')) and rhs.endswith(("'", '"')) and len(rhs) >= 2:
                rhs = rhs[1:-1]
            vals[var] = clean_val(rhs)
    return vals

def main():
    if not SECRETS.exists():
        print(f"Secrets file not found: {SECRETS}")
        sys.exit(1)

    raw = SECRETS.read_text(encoding="utf-8")
    vals = parse_exports(raw)

    # Back up
    bk = SECRETS.with_suffix(".env.bak_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(SECRETS, bk)

    # Rewrite with just clean exports (preserving only our three keys)
    lines = [f"# Cleaned {datetime.utcnow().isoformat()}Z"]
    for v in VARS:
        val = vals.get(v, "")
        if not val:
            print(f"WARNING: {v} was not found; leaving blank")
        lines.append(f'export {v}="{val}"')
    SECRETS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote cleaned secrets to: {SECRETS}")
    print(f"Backup saved to        : {bk}")

    # Show masked preview
    def mask(s): 
        return "****" if len(s) <= 4 else "****" + s[-4:]
    for v in VARS:
        val = vals.get(v, "")
        print(f"{v:28s} -> {mask(val)} (len={len(val)})")

if __name__ == "__main__":
    main()
