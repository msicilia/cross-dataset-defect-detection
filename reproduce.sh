#!/usr/bin/env bash
# Reproduces the results, figures and tables of the paper.
#
# Usage:  ./reproduce.sh [STAGE ...]
#   data         split manifests and VISION masks
#   preflight    smoke test of datasets and detectors on CPU
#   benchmark    cross-dataset benchmark (seven detectors) and backbone ablation
#   experiments  mixed-source, MVTec split, reference-set size, contamination,
#                proportions, confounders, localisation, cost
#   check        integrity of all result sets
#   analysis     integrity check, statistics and figures
#   tables       LaTeX tables
#   all          every stage above, in order (default)
#
# Uses the `python` of the active environment, or the interpreter given in
# PYTHON (see README.md), and expects the datasets under data/ (see
# setup_data.sh). DEVICE selects the device for the experiment scripts: auto
# (default: cuda, then mps, then cpu), cuda, mps or cpu. Each step writes
# results/logs/<step>.log, appending to earlier logs. Complete result cells
# are not recomputed, so an interrupted run resumes at the first incomplete one.
set -euo pipefail

# On macOS, keep the machine from sleeping for the duration of the run.
if [[ "$(uname)" == "Darwin" && -z "${REPRODUCE_AWAKE:-}" ]] && command -v caffeinate >/dev/null; then
  export REPRODUCE_AWAKE=1
  exec caffeinate -ims "$0" "$@"
fi

cd "$(dirname "$0")"

export DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"
LOG_DIR=results/logs

# Logs are appended, so those of an interrupted run are kept on resume.
step() {
  local name=$1
  shift
  echo "[$(date '+%F %T')] $name: $*"
  echo "=== $(date '+%F %T') $*" >> "$LOG_DIR/$name.log"
  "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
}

stage_data() {
  step preprocess "$PYTHON" preprocess.py
  step vision_masks "$PYTHON" vision_masks.py
}

stage_preflight() {
  step smoke_test "$PYTHON" smoke_test.py
}

stage_benchmark() {
  step run_experiments "$PYTHON" run_experiments.py
  step run_experiments_backbones "$PYTHON" run_experiments.py \
    --methods dino_patchcore_small dino_patchcore_large
  step ablate_spade_layers "$PYTHON" ablate_spade_layers.py
}

stage_experiments() {
  step run_mixed_source "$PYTHON" run_mixed_source.py
  step run_mvtec_split "$PYTHON" run_mvtec_split.py
  step run_fewshot "$PYTHON" run_fewshot.py
  step run_contamination "$PYTHON" run_contamination.py
  step run_proportions "$PYTHON" run_proportions.py
  step run_confounders "$PYTHON" run_confounders.py
  step run_localisation "$PYTHON" run_localisation.py
  step measure_cost "$PYTHON" measure_cost.py
}

stage_check() {
  step check_consistency "$PYTHON" check_consistency.py
}

stage_analysis() {
  stage_check
  step analyze_significance "$PYTHON" analyze_significance.py
  step analyze_inversion "$PYTHON" analyze_inversion.py
  step analyze_per_category "$PYTHON" analyze_per_category.py
  step analyze_localisation "$PYTHON" analyze_localisation.py
  step analyze_divergence "$PYTHON" analyze_divergence.py
  step analyze_mvtec_split "$PYTHON" analyze_mvtec_split.py
  step analyze_mixed_fewshot "$PYTHON" analyze_mixed_fewshot.py
  step analyze_contamination "$PYTHON" analyze_contamination.py
  step analyze_confounders "$PYTHON" analyze_confounders.py
  step make_figures "$PYTHON" make_figures.py
  step make_sample_montage "$PYTHON" make_sample_montage.py
}

stage_tables() {
  step make_tables "$PYTHON" make_tables.py
}

run_stage() {
  case "$1" in
    data | preflight | benchmark | experiments | check | analysis | tables)
      "stage_$1" ;;
    all)
      stage_data
      stage_preflight
      stage_benchmark
      stage_experiments
      stage_analysis
      stage_tables ;;
    *)
      echo "unknown stage: $1" >&2
      sed -n '4,13p' "$0" >&2
      exit 2 ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- all
fi
# Validate every stage name before running anything (run_stage exits on an unknown one).
for stage in "$@"; do
  case "$stage" in
    data | preflight | benchmark | experiments | check | analysis | tables | all) ;;
    *) run_stage "$stage" ;;
  esac
done

if ! command -v "$PYTHON" >/dev/null || ! "$PYTHON" -c "import torch, transformers" 2>/dev/null; then
  echo "no usable Python ($PYTHON): activate the environment from README.md or set PYTHON" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
echo "python: $(command -v "$PYTHON") ($("$PYTHON" --version 2>&1)), DEVICE=$DEVICE"

for stage in "$@"; do
  run_stage "$stage"
done
echo "[$(date '+%F %T')] done: $*"
