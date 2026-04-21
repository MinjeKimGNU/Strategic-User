#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME:-mjKimTS}" python)
SCRIPT="$SCRIPT_DIR/su_extractor.py"
DEVICE="${DEVICE:-cuda:0}"
MODEL_NAME="${MODEL_NAME:-SASRec}"
DATASETS=(Amazon_Beauty Amazon_Sports_and_Outdoors Amazon_Toys_and_Games yelp)
CLUSTERING_METHODS=(kmeans len_based pop_based)
INJECTION_MODES=(no_injection target random pop)
TOTAL=0

build_checkpoint_list() {
  local dataset="$1"
  local -n out_ref=$2
  shopt -s nullglob
  out_ref=("$SCRIPT_DIR"/saved/"${MODEL_NAME}"*"$dataset"*.pth)
  shopt -u nullglob
}

run_dataset() {
  local dataset="$1"
  local model_files=()
  build_checkpoint_list "$dataset" model_files

   if [ ${#model_files[@]} -eq 0 ]; then
    echo "[SKIP] no ${MODEL_NAME} checkpoints found for dataset=${dataset} under $SCRIPT_DIR/saved"
    return
  fi

  for clustering in "${CLUSTERING_METHODS[@]}"; do
    for model_file in "${model_files[@]}"; do
      local model_basename
      model_basename=$(basename "$model_file")
      local run_id="${model_basename%.pth}"
      for injection in "${INJECTION_MODES[@]}"; do
        TOTAL=$((TOTAL + 1))
        echo "[${TOTAL}] ${dataset} | ${clustering} | ${injection} | ${run_id}"
        "${PYTHON_CMD[@]}" "$SCRIPT" \
          --model_file "$model_file" \
          --run_id "$run_id" \
          --clustering_method "$clustering" \
          --injection_mode "$injection" \
          --device "$DEVICE"
      done
    done
  done
}

cd "$SCRIPT_DIR"

for dataset in "${DATASETS[@]}"; do
  run_dataset "$dataset"
done

echo
echo "All ${TOTAL} integrated runs processed."
