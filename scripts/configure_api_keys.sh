#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/georgekurchey/CL_Model"
CONF_DIR="$ROOT/config"
SECRETS="$CONF_DIR/secrets.env"

mkdir -p "$CONF_DIR"

# Backup existing file (if any)
backup=""
if [ -f "$SECRETS" ]; then
  backup="$SECRETS.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$SECRETS" "$backup"
fi

# Trim surrounding double quotes if present
trim_quotes() {
  local v="$1"
  case "$v" in
    \"*\") v="${v%\"}"; v="${v#\"}";;
  esac
  printf '%s' "$v"
}

# Extract existing value for VAR from secrets.env
get_existing() {
  local var="$1"
  [ -f "$SECRETS" ] || { printf ''; return; }
  # match: export VAR="value"  (allow spaces)
  local line
  line="$(grep -E '^[[:space:]]*export[[:space:]]+'"$var"'=' "$SECRETS" | tail -n1 || true)"
  [ -z "$line" ] && { printf ''; return; }
  line="${line#*=}"
  trim_quotes "$line"
}

# Prompt for a var with visible input; Enter keeps existing
prompt_visible() {
  local var="$1" label="$2" existing="$3" input=""
  if [ -n "$existing" ]; then
    local show_tail="${existing: -4}"
    printf "%s (press Enter to keep ****%s): " "$label" "$show_tail"
    IFS= read -r input || true
    [ -z "$input" ] && input="$existing"
  else
    while :; do
      printf "%s: " "$label"
      IFS= read -r input || true
      [ -n "$input" ] && break
      echo "Value cannot be empty."
    done
  fi
  printf '%s' "$input"
}

# Escape any embedded double quotes for shell
esc() { printf '%s' "${1//\"/\\\"}"; }

echo "Configure API keys (visible input; you can paste long values)."

EX_NDL="$(get_existing NASDAQ_DATA_LINK_API_KEY)"
EX_FRED="$(get_existing FRED_API_KEY)"
EX_EIA="$(get_existing EIA_API_KEY)"

NDL="$(prompt_visible NASDAQ_DATA_LINK_API_KEY 'NASDAQ_DATA_LINK_API_KEY' "$EX_NDL")"
echo
FRED="$(prompt_visible FRED_API_KEY 'FRED_API_KEY' "$EX_FRED")"
echo
EIA="$(prompt_visible EIA_API_KEY 'EIA_API_KEY' "$EX_EIA")"
echo

{
  echo "# Generated $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "export NASDAQ_DATA_LINK_API_KEY=\"$(esc "$NDL")\""
  echo "export FRED_API_KEY=\"$(esc "$FRED")\""
  echo "export EIA_API_KEY=\"$(esc "$EIA")\""
} > "$SECRETS"

chmod 600 "$SECRETS"
echo "Saved: $SECRETS"
[ -n "$backup" ] && echo "Backup: $backup"
echo "To load into this shell now:  source \"$SECRETS\""
