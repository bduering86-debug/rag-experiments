#!/usr/bin/env bash
set -euo pipefail

# Export Mermaid diagrams from Markdown files in docs/diagrams to SVG and PNG.
# Usage:
#   scripts/export_diagrams.sh
#   scripts/export_diagrams.sh --svg-only
#   scripts/export_diagrams.sh --png-only
#   scripts/export_diagrams.sh --width 2400 --height 1600

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIAGRAM_DIR="$ROOT_DIR/docs/diagrams"
TMP_DIR="$DIAGRAM_DIR/.tmp_mermaid"

EXPORT_SVG=true
EXPORT_PNG=true
WIDTH=2400
HEIGHT=1600
BACKGROUND="transparent"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --svg-only)
      EXPORT_SVG=true
      EXPORT_PNG=false
      shift
      ;;
    --png-only)
      EXPORT_SVG=false
      EXPORT_PNG=true
      shift
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --background)
      BACKGROUND="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,22p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_mmdc() {
  if command -v mmdc >/dev/null 2>&1; then
    mmdc "$@"
  elif command -v npx >/dev/null 2>&1; then
    npx -y @mermaid-js/mermaid-cli "$@"
  elif command -v docker >/dev/null 2>&1; then
    docker run --rm \
      -u "$(id -u):$(id -g)" \
      -v "$ROOT_DIR:$ROOT_DIR" \
      -w "$ROOT_DIR" \
      minlag/mermaid-cli "$@"
  else
    echo "No Mermaid CLI found. Install one of: mmdc, npx (Node.js), or docker." >&2
    exit 127
  fi
}

extract_mermaid_block() {
  local input_file="$1"
  local output_file="$2"

  awk '
    BEGIN { in_block=0; found=0 }
    /^```mermaid[[:space:]]*$/ { in_block=1; found=1; next }
    /^```[[:space:]]*$/ { if (in_block==1) { in_block=0; exit } }
    { if (in_block==1) print }
    END { if (found==0) exit 3 }
  ' "$input_file" > "$output_file"
}

if [[ ! -d "$DIAGRAM_DIR" ]]; then
  echo "Diagram directory not found: $DIAGRAM_DIR" >&2
  exit 1
fi

mkdir -p "$TMP_DIR"

processed=0
skipped=0

shopt -s nullglob
for md_file in "$DIAGRAM_DIR"/*.md; do
  base_name="$(basename "$md_file" .md)"
  mmd_file="$TMP_DIR/${base_name}.mmd"

  if ! extract_mermaid_block "$md_file" "$mmd_file"; then
    echo "[skip] No Mermaid block in: $md_file"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$EXPORT_SVG" == true ]]; then
    svg_out="$DIAGRAM_DIR/${base_name}.svg"
    run_mmdc -i "$mmd_file" -o "$svg_out"
    echo "[ok] SVG  $svg_out"
  fi

  if [[ "$EXPORT_PNG" == true ]]; then
    png_out="$DIAGRAM_DIR/${base_name}.png"
    run_mmdc -i "$mmd_file" -o "$png_out" -w "$WIDTH" -H "$HEIGHT" -b "$BACKGROUND"
    echo "[ok] PNG  $png_out"
  fi

  processed=$((processed + 1))
done

rm -rf "$TMP_DIR"

echo "Done. Processed: $processed, skipped: $skipped"
