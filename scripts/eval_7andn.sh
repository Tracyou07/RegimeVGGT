#!/usr/bin/env bash
# RegimeVGGT — 7Scenes + NRGBD dense reconstruction evaluation.
#
# Requires the original data and metric packages described in README.md.
#
# Required env vars:
#   SEVEN_ROOT  — root of the 7Scenes dataset.
#   NRGBD_ROOT  — root of the NRGBD dataset.
#   CKPT        — path to model_tracker_fixed_e20.pt (see README.md).
#
# Optional env vars:
#   KF          — keyframe stride (default 10; paper also reports 3).
#
# Usage:
#   SEVEN_ROOT=/data/7scenes \
#   NRGBD_ROOT=/data/nrgbd \
#   CKPT=ckpt/model_tracker_fixed_e20.pt \
#       bash scripts/eval_7andn.sh

set -euo pipefail

KF="${KF:-10}"

: "${SEVEN_ROOT:?Set SEVEN_ROOT to your 7Scenes dataset root.}"
: "${NRGBD_ROOT:?Set NRGBD_ROOT to your NRGBD dataset root.}"
: "${CKPT:?Set CKPT to your VGGT-1B checkpoint path.}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/results/regimevggt"
mkdir -p "${OUT_DIR}"

python "${REPO_ROOT}/eval/eval_7andN.py" \
    --datasets       7scenes NRGBD \
    --7scenes_root   "${SEVEN_ROOT}" \
    --nrgbd_root     "${NRGBD_ROOT}" \
    --ckpt_path      "${CKPT}" \
    --kf             "${KF}" \
    --output_dir     "${OUT_DIR}"

echo "Results written to ${OUT_DIR}"
