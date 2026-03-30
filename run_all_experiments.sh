#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/orchestration_logs/${TIMESTAMP}}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_all_experiments.sh
  ./run_all_experiments.sh all
  ./run_all_experiments.sh e1 e2 e3
  ./run_all_experiments.sh --dry-run e4 e5

Environment overrides:
  PYTHON_BIN   Python executable to use. Default: python
  LOG_ROOT     Log directory. Default: nft_experiments/orchestration_logs/<timestamp>
  HF_HUB_DOWNLOAD_TIMEOUT  Hugging Face download timeout in seconds. Default: 60
  HF_HUB_ETAG_TIMEOUT      Hugging Face metadata timeout in seconds. Default: 60
  E1_CONFIG    Override config path for E1
  E2_CONFIG    Override config path for E2
  E3_CONFIG    Override config path for E3
  E4_CONFIG    Override config path for E4
  E5_CONFIG    Override config path for E5
EOF
}

stage_runner() {
  case "$1" in
    e1) printf '%s' "${ROOT_DIR}/e1_controlled/src/run_e1_controlled.py" ;;
    e2) printf '%s' "${ROOT_DIR}/e2_predictor/src/run_e2_predictor.py" ;;
    e3) printf '%s' "${ROOT_DIR}/e3_attribution/src/run_e3_attribution.py" ;;
    e4) printf '%s' "${ROOT_DIR}/e4_forecasting/src/run_e4_forecasting.py" ;;
    e5) printf '%s' "${ROOT_DIR}/e5_bgss/src/run_e5_bgss.py" ;;
    *) return 1 ;;
  esac
}

stage_config() {
  case "$1" in
    e1) printf '%s' "${E1_CONFIG:-${ROOT_DIR}/e1_controlled/configs/default.json}" ;;
    e2) printf '%s' "${E2_CONFIG:-${ROOT_DIR}/e2_predictor/configs/default.json}" ;;
    e3) printf '%s' "${E3_CONFIG:-${ROOT_DIR}/e3_attribution/configs/default.json}" ;;
    e4) printf '%s' "${E4_CONFIG:-${ROOT_DIR}/e4_forecasting/configs/default.json}" ;;
    e5) printf '%s' "${E5_CONFIG:-${ROOT_DIR}/e5_bgss/configs/default.json}" ;;
    *) return 1 ;;
  esac
}

append_stage() {
  local stage="$1"
  for existing in "${STAGES[@]:-}"; do
    if [[ "${existing}" == "${stage}" ]]; then
      return 0
    fi
  done
  STAGES+=("${stage}")
}

run_stage() {
  local stage="$1"
  local runner
  local config
  local log_file
  runner="$(stage_runner "${stage}")"
  config="$(stage_config "${stage}")"
  log_file="${LOG_ROOT}/${stage}.log"

  if [[ ! -f "${runner}" ]]; then
    echo "[ERROR] Missing runner for ${stage}: ${runner}" >&2
    exit 1
  fi
  if [[ ! -f "${config}" ]]; then
    echo "[ERROR] Missing config for ${stage}: ${config}" >&2
    exit 1
  fi

  echo "================================================================"
  echo "[INFO] Stage ${stage}"
  echo "[INFO] Runner: ${runner}"
  echo "[INFO] Config: ${config}"
  echo "[INFO] Log:    ${log_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  mkdir -p "${LOG_ROOT}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${runner}" "${config}"
  ) 2>&1 | tee "${log_file}"
}

STAGES=()

if [[ $# -eq 0 ]]; then
  STAGES=(e1 e2 e3 e4 e5)
else
  for arg in "$@"; do
    case "${arg}" in
      --dry-run)
        DRY_RUN=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      all)
        append_stage e1
        append_stage e2
        append_stage e3
        append_stage e4
        append_stage e5
        ;;
      e1|e2|e3|e4|e5)
        append_stage "${arg}"
        ;;
      *)
        echo "[ERROR] Unknown argument: ${arg}" >&2
        usage
        exit 1
        ;;
    esac
  done
fi

if ! "${PYTHON_BIN}" --version >/dev/null 2>&1; then
  echo "[ERROR] Python executable is not available: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${LOG_ROOT}"
fi

echo "[INFO] Root directory: ${ROOT_DIR}"
echo "[INFO] Python:         ${PYTHON_BIN}"
echo "[INFO] Stages:         ${STAGES[*]}"
echo "[INFO] Log root:       ${LOG_ROOT}"
echo "[INFO] HF timeouts:    download=${HF_HUB_DOWNLOAD_TIMEOUT}s etag=${HF_HUB_ETAG_TIMEOUT}s"

export HF_HUB_DOWNLOAD_TIMEOUT
export HF_HUB_ETAG_TIMEOUT

for stage in "${STAGES[@]}"; do
  run_stage "${stage}"
done

echo "================================================================"
echo "[INFO] Completed stages: ${STAGES[*]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "[INFO] Logs saved under: ${LOG_ROOT}"
fi
