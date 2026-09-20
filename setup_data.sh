#!/usr/bin/env bash
# Places the three datasets under data/ and checks their file counts.
#
#   SDNET2018  manual download (instructions printed if absent)
#   MVTec AD   manual download (instructions printed if absent)
#   VISION     downloaded from Hugging Face if absent (requires huggingface_hub
#              and, if the dataset is gated for your account, prior access
#              approval and `huggingface-cli login`)
#
# Usage (from the code directory):  bash setup_data.sh
# Exits non-zero if any dataset is missing or incomplete.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=data
SDNET_DIR="$DATA_DIR/sdnet2018"
MVTEC_DIR="$DATA_DIR/mvtec_anomaly_detection"
VISION_DIR="$DATA_DIR/vision_dataset"
mkdir -p "$DATA_DIR"

status=0

count() {   # count <dir> <name pattern>
  if [ -d "$1" ]; then find "$1" -type f -iname "$2" | wc -l | tr -d ' '; else echo 0; fi
}

expect() {  # expect <label> <found> <expected>
  if [ "$2" -eq "$3" ]; then
    printf '  %-34s %6d files (ok)\n' "$1" "$2"
  else
    printf '  %-34s %6d files, expected %d\n' "$1" "$2" "$3"
    status=1
  fi
}

# ── SDNET2018 ─────────────────────────────────────────────────────────────────
echo "SDNET2018"
if [ -d "$SDNET_DIR/W" ] && [ -d "$SDNET_DIR/D" ] && [ -d "$SDNET_DIR/P" ]; then
  expect "W/CW (cracked walls)" "$(count "$SDNET_DIR/W/CW" '*.jpg')" 3851
  expect "W/UW (uncracked walls)" "$(count "$SDNET_DIR/W/UW" '*.jpg')" 14287
  expect "D/CD (cracked decks)" "$(count "$SDNET_DIR/D/CD" '*.jpg')" 2025
  expect "D/UD (uncracked decks)" "$(count "$SDNET_DIR/D/UD" '*.jpg')" 11595
  expect "P/CP (cracked pavements)" "$(count "$SDNET_DIR/P/CP" '*.jpg')" 2608
  expect "P/UP (uncracked pavements)" "$(count "$SDNET_DIR/P/UP" '*.jpg')" 21726
else
  status=1
  cat <<EOF
  Not found. Manual download (about 0.6 GB, CC BY 4.0):
    1. https://digitalcommons.usu.edu/all_datasets/48/  ->  Download
    2. Extract so that these directories exist:
         $SDNET_DIR/D/CD  $SDNET_DIR/D/UD
         $SDNET_DIR/P/CP  $SDNET_DIR/P/UP
         $SDNET_DIR/W/CW  $SDNET_DIR/W/UW
EOF
fi

# ── MVTec AD ──────────────────────────────────────────────────────────────────
echo "MVTec AD"
if [ -d "$MVTEC_DIR/screw" ]; then
  for spec in metal_nut:220:115:93 screw:320:160:119 tile:230:117:84 \
              wood:247:79:60 grid:264:78:57; do
    IFS=: read -r cat n_train n_test n_gt <<<"$spec"
    expect "$cat/train" "$(count "$MVTEC_DIR/$cat/train" '*.png')" "$n_train"
    expect "$cat/test" "$(count "$MVTEC_DIR/$cat/test" '*.png')" "$n_test"
    expect "$cat/ground_truth" "$(count "$MVTEC_DIR/$cat/ground_truth" '*.png')" "$n_gt"
  done
else
  status=1
  cat <<EOF
  Not found. Manual download (about 4.9 GB, CC BY-NC-SA 4.0):
    1. https://www.mvtec.com/company/research/datasets/mvtec-ad
       -> accept the licence and download mvtec_anomaly_detection.tar.xz
    2. mkdir -p $MVTEC_DIR && tar -xf mvtec_anomaly_detection.tar.xz -C $MVTEC_DIR
       so that $MVTEC_DIR/<category>/{train,test,ground_truth} exist.
    Only metal_nut, screw, tile, wood and grid are used.
EOF
fi

# ── VISION ────────────────────────────────────────────────────────────────────
echo "VISION"
VISION_CATEGORIES=(Casting Ring Screw Cylinder)
missing=()
for cat in "${VISION_CATEGORIES[@]}"; do
  if [ "$(count "$VISION_DIR/$cat" '*.jpg')" -eq 0 ]; then
    missing+=("$cat")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "  downloading ${missing[*]} from huggingface.co/datasets/VISION-Workshop/VISION-Datasets"
  if ! python - "$VISION_DIR" "${missing[@]}" <<'PYEOF'
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
for cat in sys.argv[2:]:
    archive = hf_hub_download(repo_id="VISION-Workshop/VISION-Datasets",
                              repo_type="dataset", filename=f"{cat}.tar.gz",
                              local_dir=str(dest))
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    Path(archive).unlink()
    print(f"  {cat}: extracted")
PYEOF
  then
    status=1
    cat <<EOF
  Download failed. If access is denied, request access at
    https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets
  run \`huggingface-cli login\` and re-run this script.
EOF
  fi
fi
for spec in Casting:108:102:3510 Ring:90:84:928 Screw:114:126:3396 Cylinder:276:290:2482; do
  IFS=: read -r cat n_train n_val n_inf <<<"$spec"
  expect "$cat/train" "$(count "$VISION_DIR/$cat/train" '*.jpg')" "$n_train"
  expect "$cat/val" "$(count "$VISION_DIR/$cat/val" '*.jpg')" "$n_val"
  expect "$cat/inference" "$(count "$VISION_DIR/$cat/inference" '*.jpg')" "$n_inf"
done

echo
if [ "$status" -ne 0 ]; then
  echo "Some datasets are missing or incomplete; see above."
  exit 1
fi
echo "All datasets in place. Next: ./reproduce.sh"
