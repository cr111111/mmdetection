#!/usr/bin/env bash
# ============================================================
# Dark-DINO v2: Low-Light Object Detection — Full Pipeline
# From git clone to experiment results, one script.
# Usage:
#   chmod +x tools/run_low_contrast_experiments.sh
#   bash tools/run_low_contrast_experiments.sh [STAGE]
# Stages: all | env | data | exp | test | clean (default: all)
# Env vars (override as needed):
#   REPO_URL    – git remote (default: origin repo)
#   BRANCH      – branch name (default: dark-dino-v2)
#   GPUS        – number of GPUs (default: 2)
#   WORK_ROOT   – work directory (default: ./work_dirs/low_contrast)
#   AMP         – mixed precision (default: 1)
#   RESUME      – resume from checkpoint (default: 0)
#   RUN_TEST    – run eval after training (default: 1)
# ============================================================

set -Eeuo pipefail

# ── Stage selection ────────────────────────────────────────
STAGE="${1:-all}"
case "${STAGE}" in
  env|data|exp|test|clean|all) ;;
  *) echo "Usage: $0 [env|data|exp|test|clean|all]" >&2; exit 1 ;;
esac

# ── Paths & defaults ───────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REPO_URL="${REPO_URL:-https://github.com/cr111111/mmdetection.git}"
BRANCH="${BRANCH:-dark-dino-v2}"
GPUS="${GPUS:-2}"
PORT_BASE="${PORT_BASE:-29500}"
WORK_ROOT="${WORK_ROOT:-${ROOT_DIR}/work_dirs/low_contrast}"
AMP="${AMP:-1}"
RESUME="${RESUME:-0}"
RUN_TEST="${RUN_TEST:-1}"

EXDATA_ROOT="${EXDATA_ROOT:-${ROOT_DIR}/data/exdark}"
EXDATA_ZIP="${EXDATA_ZIP:-ExDark.zip}"
EXDATA_URL="${EXDATA_URL:-https://github.com/cs-chan/ExDark/raw/master/ExDark.zip}"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "Dark-DINO v2 Full Pipeline"
echo "Stage : ${STAGE}"
echo "Root  : ${ROOT_DIR}"
echo "Branch: ${BRANCH}"
echo "GPUs  : ${GPUS}"
echo "Date  : $(date '+%F %T')"
echo "============================================================"


# ══════════════════════════════════════════════════════════════
# STAGE: clean  —  remove work dirs / venv / data
# ══════════════════════════════════════════════════════════════
stage_clean() {
  echo "[clean] Removing generated directories..."
  rm -rf "${WORK_ROOT}" "${VENV_DIR}" "${ROOT_DIR}/build" "${ROOT_DIR}/dist" "*.egg-info"
  find "${ROOT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  echo "[clean] Done."
}


# ══════════════════════════════════════════════════════════════
# STAGE: env  —  git clone + create venv + install deps
# ══════════════════════════════════════════════════════════════
stage_env() {
  # --- Git clone if needed ---
  if [[ ! -d "${ROOT_DIR}/.git" ]]; then
    echo "[env] Cloning ${REPO_URL} (branch=${BRANCH}) ..."
    # We are already in ROOT_DIR which may be empty or non-git;
    # clone into a temp dir and move contents here
    tmp_dir=$(mktemp -d)
    git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${tmp_dir}/repo"
    # Move all hidden files (including .git) and regular files
    shopt -s dotglob nullglob
    mv "${tmp_dir}/repo/"* "${ROOT_DIR}/" 2>/dev/null || true
    mv "${tmp_dir}/repo"/.* "${ROOT_DIR}/" 2>/dev/null || true
    shopt -u dotglob nullglob
    rm -rf "${tmp_dir}"
    cd "${ROOT_DIR}"
    echo "[env] Clone complete."
  else
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo "[env] Git repo exists (branch=${current_branch}). Pulling latest..."
    git fetch origin "${BRANCH}"
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}" || true
  fi

  # --- Create virtual environment ---
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[env] Creating virtual environment at ${VENV_DIR} ..."
    ${PYTHON} -m venv "${VENV_DIR}"
  fi
  echo "[env] Activating venv..."
  source "${VENV_DIR}/bin/activate"

  # --- Upgrade pip ---
  pip install --upgrade pip setuptools wheel 2>&1 | tail -3

  # --- Install PyTorch (CUDA 12.4) ---
  if ! python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "[env] Installing PyTorch with CUDA support ..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
  fi

  # --- Install mmcv (compiled with CUDA) ---
  if ! python -c 'import mmcv.ops' 2>/dev/null; then
    echo "[env] Installing mmcv with CUDA extensions ..."
    pip install "mmcv>=2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.5/index.html 2>&1 | tail -5 || {
      echo "[env] Pre-built mmcv not available, building from source..."
      pip install "mmcv>=2.1.0" --no-build-isolation 2>&1 | tail -5
    }
  fi

  # --- Install mmdet + dependencies ---
  echo "[env] Installing MMDetection ..."
  pip install -e ".[optional]" 2>&1 | tail -5

  # --- Verify ---
  echo "[env] Verifying installation ..."
  python -c "
import torch, mmengine, mmdet, mmcv
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU[{i}]: {torch.cuda.get_device_name(i)}')
print(f'mmcv: {mmcv.__version__}')
print(f'mmdet: {mmdet.__version__}')
"

  echo "[env] Environment ready."
}


# ══════════════════════════════════════════════════════════════
# STAGE: data  —  download ExDark + convert to COCO format
# ══════════════════════════════════════════════════════════════
stage_data() {
  # Activate venv for Python scripts
  [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

  # --- Download ExDark dataset ---
  if [[ ! -d "${EXDATA_ROOT}/images" ]]; then
    echo "[data] Downloading ExDark dataset ..."
    mkdir -p "${EXDATA_ROOT}"

    # Try to find existing zip or download
    if [[ ! -f "${EXDATA_ROOT}/${EXDATA_ZIP}" ]]; then
      echo "[data] Downloading from ${EXDATA_URL} (~1.2 GB) ..."
      wget -q --show-progress -O "${EXDATA_ROOT}/${EXDATA_ZIP}" "${EXDATA_URL}" || \
      curl -L -o "${EXDATA_ROOT}/${EXDATA_ZIP}" "${EXDATA_URL}"
    fi

    # Extract
    echo "[data] Extracting ExDark ..."
    mkdir -p "${EXDATA_ROOT}/_temp_downloads"
    unzip -qo "${EXDATA_ROOT}/${EXDATA_ZIP}" -d "${EXDATA_ROOT}/_temp_downloads/"
    echo "[data] Extraction complete."
  fi

  # --- Convert to COCO format ---
  if [[ ! -f "${EXDATA_ROOT}/annotations/exdark_train.json" ]]; then
    echo "[data] Converting ExDark annotations to COCO format ..."
    python tools/download_exdark.py --data-root "${EXDATA_ROOT}"
  fi

  # --- Validate ---
  echo "[data] Validating dataset ..."
  python -c "
import json, os
for split in ['train', 'val']:
    path = f'${EXDATA_ROOT}/annotations/exdark_{split}.json'
    with open(path) as f:
        d = json.load(f)
    print(f'  {split}: {len(d[\"images\"])} images, {len(d[\"annotations\"])} anns')
print('[data] Dataset ready.')
"
}


# ══════════════════════════════════════════════════════════════
# STAGE: exp  —  train all three experiments sequentially
# ══════════════════════════════════════════════════════════════
stage_exp() {
  [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

  if [[ "${GPUS}" -ne 2 ]]; then
    echo "WARNING: Script tuned for 2 GPUs. Running with ${GPUS}."
  fi

  for required_path in \
    "data/exdark/images" \
    "data/exdark/annotations/exdark_train.json" \
    "data/exdark/annotations/exdark_val.json"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "ERROR: Missing required path: ${ROOT_DIR}/${required_path}" >&2
      echo "Run '$0 data' first." >&2
      exit 2
    fi
  done

  if ! python -c 'import mmengine, mmdet, torch' >/dev/null 2>&1; then
    echo "ERROR: Python environment missing dependencies." >&2
    echo "Run '$0 env' first." >&2
    exit 2
  fi

  mkdir -p "${WORK_ROOT}" "${WORK_ROOT}/logs" "${WORK_ROOT}/metrics"

  AMP_ARGS=()
  if [[ "${AMP}" == "1" ]]; then AMP_ARGS+=(--amp); fi

  RESUME_ARGS=()
  if [[ "${RESUME}" == "1" ]]; then RESUME_ARGS+=(--resume auto); fi

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

    if [[ "${RUN_TEST}" != "1" ]]; then return; fi

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
  echo "Checkpoints and evaluation output: ${WORK_ROOT}<experiment>/"
  echo "============================================================"
}

# ══════════════════════════════════════════════════════════════
# STAGE: test  —  only evaluate existing checkpoints
# ══════════════════════════════════════════════════════════════
stage_test() {
  [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

  for name in a_raw_dino b_clahe_dino c_dark_dino; do
    local work_dir="${WORK_ROOT}/${name}"
    local checkpoint
    checkpoint="$(find "${work_dir}" -maxdepth 1 -type f -name 'best_coco_bbox_mAP*.pth' -printf '%T@ %p\n' \
      | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "${checkpoint}" ]]; then
      local cfg_name
      case "${name}" in
        a_raw_dino)  cfg_name="configs/low_contrast_detection/dino_r50_exdark_baseline.py" ;;
        b_clahe_dino) cfg_name="configs/low_contrast_detection/dino_r50_exdark_clahe.py" ;;
        c_dark_dino)  cfg_name="configs/low_contrast_detection/dark_dino_r50_exdark.py" ;;
      esac
      echo "[test] Evaluating ${name}: ${checkpoint}"
      PORT="$((PORT_BASE + 200))" bash tools/dist_test.sh "${cfg_name}" "${checkpoint}" "${GPUS}" \
        --work-dir "${work_dir}/eval_re" 2>&1 | tee "${WORK_ROOT}/logs/${name}_eval_re.log"
    else
      echo "[test] No checkpoint found for ${name}, skipping."
    fi
  done
}


# ══════════════════════════════════════════════════════════════
# MAIN  —  dispatch to stage(s)
# ══════════════════════════════════════════════════════════════
case "${STAGE}" in
  clean) stage_clean ;;
  env)   stage_env ;;
  data)  stage_data ;;
  exp)   stage_exp ;;
  test)  stage_test ;;
  all)
    stage_env
    stage_data
    stage_exp
    ;;
esac

echo ""
echo "[DONE] Stage '${STAGE}' completed at $(date '+%F %T')"
