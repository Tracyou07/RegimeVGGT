# RegimeVGGT

Code for **RegimeVGGT: Layer-Wise Spatially Preserving Redundancy Removal for Visual Geometry Grounded Transformer**.

[Paper](https://arxiv.org/abs/2606.18439) | [Method](https://arxiv.org/html/2606.18439v1#S3)

RegimeVGGT accelerates VGGT without training by combining saliency-guided token merging and selectively protected K/V downsampling. This repository exposes the final method described in Sections 3.2 and 3.3 of the linked paper.

<p align="center">
  <img src="image/overview.png" width="900" alt="RegimeVGGT pipeline overview">
</p>

## Final method

The same fixed policy is used by every evaluation entry point. Layer numbers below are one-based.

| Global layers | Merge ratio | K/V sampling factor |
| --- | --- | --- |
| 1-10 | 0.99 | 1.5 |
| 11-18 | 0.50 | 1.3 |
| 19-24 | 0.99 | 1.7 |

- Frozen DINOv2 CLS attention selects the top 10% salient patches per frame. They are protected during merging and K/V sampling.
- Camera/register tokens and the first frame are retained; the spatial K/V grid is shifted by frame index.
- Merge indices are cached within the three bands. Layers 11-14 recompute their indices at each layer.
- Size-aware attention preserves merged-token multiplicity through an extra Q/K dimension and head-dimension padding.
- Every global layer uses the final merge-and-sample path; DPT taps retain full token resolution after unmerging.

The public method interface is:

```python
import torch
from methods.regimevggt import run_aggregator_regimevggt

# model: pretrained VGGT backbone in eval mode
# images: RGB in [0, 1], shaped [1, frames, 3, height, width]
with torch.no_grad():
    features, patch_start_idx = run_aggregator_regimevggt(model, images)
```

The returned features feed the existing VGGT camera, depth and point heads. The VGGT backbone and its head dependencies are retained for checkpoint compatibility.

## Installation

```bash
git clone https://github.com/Tracyou07/RegimeVGGT.git
cd RegimeVGGT
python -m pip install -r requirements.txt
```

The paper reports measurements on one NVIDIA H800 80 GB at input resolution 518 x 392. Those measurements are paper results; the cleaned implementation still needs a full pretrained GPU benchmark run.

## Checkpoint

The method uses Meta's `model_tracker_fixed_e20.pt` checkpoint without fine-tuning.

```bash
mkdir -p ckpt
wget -P ckpt \
  https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
export CKPT="$PWD/ckpt/model_tracker_fixed_e20.pt"
```

The checkpoint remains subject to the [upstream VGGT terms](https://huggingface.co/facebook/VGGT_tracker_fixed).

## Evaluation

All three launchers run the same final method. Dataset locations, checkpoint paths, frame counts and output locations remain runtime inputs.

```bash
# Tanks & Temples pose
TNT_DIR=/path/to/tnt/training \
TNT_COLMAP_DIR=/path/to/tnt/poses \
CKPT="$CKPT" bash scripts/eval_tnt.sh

# 7Scenes and NRGBD reconstruction; KF can be 3 or 10
SEVEN_ROOT=/path/to/7scenes \
NRGBD_ROOT=/path/to/nrgbd \
KF=10 CKPT="$CKPT" bash scripts/eval_7andn.sh

# ScanNet-50 reconstruction and pose
SCANNET_DATA=/path/to/scannet/processed \
SCANNET_GT_PLY=/path/to/scannet/scans \
SCANNET_SPLIT=/path/to/scannet_50.txt \
FRAMES=1000 CKPT="$CKPT" bash scripts/eval_scannet.sh
```

Use the original benchmark scene list for `SCANNET_SPLIT`, with one scene ID per line. A custom scene subset is a different evaluation protocol.

The imported release did not include the `data` and `utils` protocol packages required by the 7Scenes/NRGBD evaluator. These dataset adapters and metric helpers must be restored before that evaluation can run. This cleanup does not supply a replacement metric or scene split.

Outputs are written under `results/`. Dataset preparation, pretrained inference and benchmark numbers have not been validated by the CPU checks below.

## Local checks

```bash
python -m unittest discover -s tests -v
```

The checks use small CPU tensors and inspect the real argument parsers and shell launchers. They require no pretrained checkpoint or dataset.

## Repository layout

```text
methods/        Final RegimeVGGT aggregator and token-merging primitive
vggt/           VGGT backbone, heads and evaluation utilities
eval/           Dataset evaluation entry points
scripts/        Evaluation launchers
tests/          CPU regression and entry-point checks
image/          Paper figures
```

## Citation

```bibtex
@misc{you2026regimevggt,
  title={RegimeVGGT: Layer-Wise Spatially Preserving Redundancy Removal for Visual Geometry Grounded Transformer},
  author={Jinhao You and Shuo Lyu and Zhuohang Lyu and Tanxuan Li and Zibo Zhao and Jiaxiang Hu and Kai Tang and Yichen Guo},
  year={2026},
  eprint={2606.18439},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.18439}
}
```

## Acknowledgements and license

This implementation builds on VGGT and the FastVGGT token-merging primitive. Original copyright notices are retained.

Repository code is licensed under Apache-2.0; see `LICENSE`. The original VGGT code under `vggt/` and its weights remain subject to the upstream VGGT license.
