#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/georgekurchey/CL_Model"
SECRETS="$ROOT/config/secrets.env"

usage() {
  echo "Usage: bash $0 [--raw]"
  echo "  --raw   show full keys (unmasked)"
  exit 1
}

SHOW_RAW=false
[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && usage
[[ "${1:-}" == "--raw" ]] && SHOW_RAW=true

if [ ! -f "$SECRETS" ]; then
  echo "Secrets file not found: $SECRETS"
  echo "Run the key configurator first:"
  echo "  bash /Users/georgekurchey/CL_Model/scripts/configure_api_keys.sh"
  exit 1
fi

trim_quotes() {
  local v="$1"
  v="${v%\"}"; v="${v#\"}"
  printf '%s' "$v"
}

get_val() {
  local var="$1"
  local line
  line="$(grep -E "^[[:space:]]*export[[:space:]]+$var=" "$SECRETS" | tail -n1 || true)"
  [[ -z "$line" ]] && { printf ''; return; }
  line="${line#export }"
  line="${line#"$var"=}"
  trim_quotes "$line"
}

mask() {
  local s="$1"; local n="${#s}"
  if $SHOW_RAW; then printf '%s' "$s"; return; fi
  if (( n <= 4 )); then printf '****'; else printf '****%s' "${s: -4}"; fi
}

print_key() {
  local var="$1" label="$2"
  local v; v="$(get_val "$var")"
  if [[ -z "$v" ]]; then
    printf "%-28s : %s\n" "$label" "(not set)"
  else
    printf "%-28s : %s  (len=%d)\n" "$label" "$(mask "$v")" "${#v}"
  fi
}

echo "Reading: $SECRETS"
print_key "NASDAQ_DATA_LINK_API_KEY" "NASDAQ_DATA_LINK_API_KEY"
print_key "FRED_API_KEY"             "FRED_API_KEY"
print_key "EIA_API_KEY"              "EIA_API_KEY"
