#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Centroids visualization runner
# ==========================================================
# Usage examples:
#   bash run_visualize_centroids_llama-3.1-8b.sh
#   TARGET_LAYER=1 bash run_visualize_centroids_llama-3.1-8b.sh
#   TARGET_LAYER=1 TARGET_KIND=k GROUP_ID=3 bash run_visualize_centroids_llama-3.1-8b.sh
#
# Config via env vars:
#   DATA_ROOT       default: .../output/llama-3.1-8b-2c4b
#   CENTROIDS_DIR   default: ${DATA_ROOT}/centroids
#   VIS_ROOT        default: ${PROJECT_ROOT}/visualization/centroids/llama-3.1-8b_4c8b
#   TARGET_LAYER    default: all       (or a layer index, e.g. 7)
#   TARGET_LAYERS   default: 0,4,8,12,16,20,24,28,31 (used when TARGET_LAYER=all)
#   TARGET_KIND     default: both      (k | v | both)
#   GROUP_ID        default: 0
#   BINS            default: 120
#   PCA_MAX_POINTS  default: 10000
# ==========================================================

PROJECT_ROOT="/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/output/llama-3.1-8b-2c8b}"
CENTROIDS_DIR="${CENTROIDS_DIR:-${DATA_ROOT}/centroids}"
VIS_ROOT="${VIS_ROOT:-${PROJECT_ROOT}/visualization/centroids/llama-3.1-8b_2c8b}"

TARGET_LAYER="${TARGET_LAYER:-all}"
TARGET_LAYERS="${TARGET_LAYERS:-0,4,8,12,16,20,24,28,31}"
TARGET_KIND="${TARGET_KIND:-both}"
GROUP_ID="${GROUP_ID:-0}"
BINS="${BINS:-120}"
PCA_MAX_POINTS="${PCA_MAX_POINTS:-10000}"

SCRIPT_PATH="${PROJECT_ROOT}/centroids_visualize.py"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[Error] Script not found: ${SCRIPT_PATH}"
  exit 1
fi

if [[ ! -d "${CENTROIDS_DIR}" ]]; then
  echo "[Error] Centroids dir not found: ${CENTROIDS_DIR}"
  echo "Hint: run generate_centroids first."
  exit 1
fi

case "${TARGET_KIND}" in
  k|v|both) ;;
  *)
    echo "[Error] TARGET_KIND must be one of: k | v | both"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Centroids Visualization Runner"
echo "Project Root : ${PROJECT_ROOT}"
echo "Centroids Dir: ${CENTROIDS_DIR}"
echo "Vis Output   : ${VIS_ROOT}"
echo "Target Layer : ${TARGET_LAYER}"
echo "Target Layers: ${TARGET_LAYERS}"
echo "Target Kind  : ${TARGET_KIND}"
echo "Group ID     : ${GROUP_ID}"
echo "Bins         : ${BINS}"
echo "PCA points   : ${PCA_MAX_POINTS}"
echo "=========================================================="

shopt -s nullglob
files=("${CENTROIDS_DIR}"/*_centroids_fisher_layer*.npy)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[Error] No centroid .npy files found in: ${CENTROIDS_DIR}"
  exit 1
fi

run_one() {
  local file_path="$1"
  local file_name layer kind out_dir
  file_name="$(basename "${file_path}")"

  if [[ "${file_name}" =~ ^([kv])_centroids_fisher_layer([0-9]+)\.npy$ ]]; then
    kind="${BASH_REMATCH[1]}"
    layer="${BASH_REMATCH[2]}"
  else
    echo "[Skip] Unrecognized filename format: ${file_name}"
    return
  fi

  if [[ "${TARGET_LAYER}" != "all" ]]; then
    if [[ "${layer}" != "${TARGET_LAYER}" ]]; then
      return
    fi
  else
    if [[ ",${TARGET_LAYERS}," != *",${layer},"* ]]; then
      return
    fi
  fi

  if [[ "${TARGET_KIND}" != "both" && "${kind}" != "${TARGET_KIND}" ]]; then
    return
  fi

  out_dir="${VIS_ROOT}/layer${layer}/${kind}"
  mkdir -p "${out_dir}"

  echo ""
  echo ">>> Visualizing ${file_name}"
  python "${SCRIPT_PATH}" \
    --centroid-path "${file_path}" \
    --group-id "${GROUP_ID}" \
    --bins "${BINS}" \
    --pca-max-points "${PCA_MAX_POINTS}" \
    --plot-mode pca3d \
    --output-dir "${out_dir}" \
    --no-show
}

ran_count=0
for f in "${files[@]}"; do
  before="${ran_count}"
  run_one "${f}" || true
  if [[ "${before}" -eq "${ran_count}" ]]; then
    :
  fi
  if [[ -f "${f}" ]]; then
    file_name="$(basename "${f}")"
    if [[ "${file_name}" =~ ^([kv])_centroids_fisher_layer([0-9]+)\.npy$ ]]; then
      kind="${BASH_REMATCH[1]}"
      layer="${BASH_REMATCH[2]}"
      layer_ok=0
      if [[ "${TARGET_LAYER}" != "all" ]]; then
        if [[ "${layer}" == "${TARGET_LAYER}" ]]; then
          layer_ok=1
        fi
      else
        if [[ ",${TARGET_LAYERS}," == *",${layer},"* ]]; then
          layer_ok=1
        fi
      fi

      if [[ "${layer_ok}" -eq 1 && ( "${TARGET_KIND}" == "both" || "${kind}" == "${TARGET_KIND}" ) ]]; then
        ran_count=$((ran_count + 1))
      fi
    fi
  fi
done

if [[ "${ran_count}" -eq 0 ]]; then
  echo ""
  echo "[Warning] No files matched TARGET_LAYER=${TARGET_LAYER}, TARGET_KIND=${TARGET_KIND}"
  exit 1
fi

echo ""
echo "=========================================================="
echo "Done. Visualized ${ran_count} centroid file(s)."
echo "Output directory: ${VIS_ROOT}"
echo "=========================================================="
