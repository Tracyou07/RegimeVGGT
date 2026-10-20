#!/usr/bin/env bash
# RegimeVGGT — Tanks & Temples 6-scene pose evaluation.
#
# Reproduces Table 4 (camera pose on T&T) from the paper.
#
# Required env vars:
#   TNT_DIR         — root of T&T training set, contains scene subfolders
#                     (Barn, Caterpillar, Courthouse, Ignatius,
#                      Meetingroom, Truck), each holding RGB frames.
#   TNT_COLMAP_DIR  — directory of T&T ground-truth COLMAP poses,
#                     organised the same way (one subdir per scene).
#   CKPT            — path to VGGT-1B checkpoint (see README.md).
#
# Usage:
#   TNT_DIR=/data/tnt/training \
#   TNT_COLMAP_DIR=/data/tnt/poses \
#   CKPT=ckpt/model_tracker_fixed_e20.pt \
#       bash scripts/eval_tnt.sh

set -euo pipefail

: "${TNT_DIR:?Set TNT_DIR to your T&T training-set root.}"
: "${TNT_COLMAP_DIR:?Set TNT_COLMAP_DIR to your T&T COLMAP pose root.}"
: "${CKPT:?Set CKPT to your VGGT-1B checkpoint path.}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/results/regimevggt/tnt"
mkdir -p "${OUT_DIR}"

python "${REPO_ROOT}/eval/eval_pose_tnt.py" \
    --tnt_dir       "${TNT_DIR}" \
    --colmap_dir    "${TNT_COLMAP_DIR}" \
    --ckpt_path     "${CKPT}" \
    --save_dir      "${OUT_DIR}" \
    --kf_every      1

echo "Results written to ${OUT_DIR}"
