#!/usr/bin/env bash
# Run the ExDark low-contrast detection comparison on one Linux host.
# Experiments run sequentially so both GPUs are dedicated to each method.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-2}"
PORT_BASE="${PORT_BASE:-29500}"
WORK_ROOT="${WORK_ROOT:-${ROOT_DIR}/work_dirs/low_contrast}"
AMP="${AMP:-1}"
RESUME="${RESUME:-0}"
RUN_TEST="${RUN_TEST:-1}"

if [[ "${GPUS}" -ne 2 ]]; then
  echo "ERROR: This script is configured for two GPUs. Set GPUS=2." >&2
  exit 2
fi

for required_path in \
  "data/exdark/images" \
  "data/exdark/annotations/exdark_train.json" \
  "data/exdark/annotations/exdark_val.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: Missing required path: ${ROOT_DIR}/${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${WORK_ROOT}" "${WORK_ROOT}/logs" "${WORK_ROOT}/metrics"

if ! python -c 'import mmengine, mmdet, torch' >/dev/null 2>&1; then
  echo "ERROR: Python environment cannot import torch, mmengine and mmdet." >&2
  echo "Activate the MMDetection environment before running this script." >&2
  exit 2
fi

if [[ ! -f "${ROOT_DIR}/tools/dist_train.sh" ]] || [[ ! -f "${ROOT_DIR}/tools/dist_test.sh" ]]; then
  echo "ERROR: Expected MMDetection distributed launch scripts were not found." >&2
  exit 2
fi

AMP_ARGS=()
if [[ "${AMP}" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--resume auto)
fi

run_one() {
  local name="$1"
  local config="$2"
  local port="$3"
  local work_dir="${WORK_ROOT}/${name}"
  local log_file="${WORK_ROOT}/logs/${name}.log"

  echo "============================================================"
  echo "[$(date '+%F %T')] Training ${name}"
  echo "Config: ${config}"
  echo "Work dir: ${work_dir}"
  echo "============================================================"

  PORT="${port}" bash tools/dist_train.sh "${config}" "${GPUS}" \
    --work-dir "${work_dir}" "${AMP_ARGS[@]}" "${RESUME_ARGS[@]}" \
    2>&1 | tee "${log_file}"

  if [[ "${RUN_TEST}" != "1" ]]; then
    return
  fi

  local checkpoint
  checkpoint="$(find "${work_dir}" -maxdepth 1 -type f -name 'best_coco_bbox_mAP*.pth' -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${checkpoint}" ]]; then
    echo "ERROR: No best_coco_bbox_mAP checkpoint found in ${work_dir}" >&2
    exit 3
  fi

  echo "[$(date '+%F %T')] Evaluating ${name}: ${checkpoint}"
  PORT="$((port + 100))" bash tools/dist_test.sh "${config}" "${checkpoint}" "${GPUS}" \
    --work-dir "${work_dir}/eval" 2>&1 | tee "${WORK_ROOT}/logs/${name}_eval.log"

}

run_one "a_raw_dino" \
  "configs/low_contrast_detection/dino_r50_exdark_baseline.py" \
  "${PORT_BASE}"
run_one "b_clahe_dino" \
  "configs/low_contrast_detection/dino_r50_exdark_clahe.py" \
  "$((PORT_BASE + 10))"
run_one "c_dark_dino" \
  "configs/low_contrast_detection/dark_dino_r50_exdark.py" \
  "$((PORT_BASE + 20))"

echo "============================================================"
echo "All experiments completed."
echo "Logs: ${WORK_ROOT}/logs"
echo "Checkpoints and evaluation output: ${WORK_ROOT}/<experiment>/"
echo "============================================================"
