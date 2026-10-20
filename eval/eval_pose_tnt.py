"""Evaluate the final RegimeVGGT method on Tanks & Temples camera poses."""

import os
import sys
import json
import struct
import argparse
import time
from pathlib import Path

import numpy as np
import torch

# Add parent paths so imports work
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR / ".."))
sys.path.insert(0, str(EVAL_DIR / ".." / ".."))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.rotation import mat_to_quat
from vggt.utils.geometry import closed_form_inverse_se3


# ── Pose error metrics ─────────────────────────────────────────


def build_pair_index(N):
    i1, i2 = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    return i1, i2


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def translation_angle(tvec_gt, tvec_pred, eps=1e-15):
    t = tvec_pred / (torch.norm(tvec_pred, dim=1, keepdim=True) + eps)
    t_gt = tvec_gt / (torch.norm(tvec_gt, dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = 1e6
    rel_tangle_deg = err_t * 180.0 / np.pi
    return torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    pair_i1, pair_i2 = build_pair_index(num_frames)
    rel_gt = gt_se3[pair_i1].bmm(closed_form_inverse_se3(gt_se3[pair_i2]))
    rel_pred = pred_se3[pair_i1].bmm(closed_form_inverse_se3(pred_se3[pair_i2]))
    r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_err = translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
    return r_err, t_err


def calculate_auc_np(r_error, t_error, max_threshold=30):
    error_matrix = np.stack([r_error, t_error], axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(float) / len(max_errors)
    return np.mean(np.cumsum(normalized_histogram))


# ── COLMAP binary reader (from LiteVGGT) ──────────────────────


def read_colmap_images_bin(bin_path):
    """Read COLMAP images.bin and return dict: filename -> c2w 4x4 matrix."""
    poses = {}
    with open(bin_path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.frombuffer(f.read(8 * 4), dtype=np.float64)  # qw,qx,qy,qz
            tvec = np.frombuffer(f.read(8 * 3), dtype=np.float64)  # tx,ty,tz
            _cam_id = struct.unpack("<I", f.read(4))[0]

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8").split("/")[-1]

            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, 1)

            # Convert w2c quaternion to c2w matrix
            qw, qx, qy, qz = qvec
            R_wc = np.array([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy + 2*qz*qw, 2*qx*qz - 2*qy*qw],
                [2*qx*qy - 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz + 2*qx*qw],
                [2*qx*qz + 2*qy*qw, 2*qy*qz - 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
            ])
            t_wc = -R_wc @ tvec
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_wc.astype(np.float32)
            c2w[:3, 3] = t_wc.astype(np.float32)
            poses[name] = c2w
    return poses


# ── T&T SfM .log reader ────────────────────────────────────────


def read_sfm_log(log_path):
    """Read T&T COLMAP SfM .log file.

    Format: every 5 lines = one camera
        Line 1: image_id image_id 0
        Lines 2-5: 4x4 w2c matrix (row-major)

    Returns:
        list of 4x4 w2c matrices, sorted by image_id
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    poses = []
    for i in range(0, len(lines), 5):
        if i + 4 >= len(lines):
            break
        header = lines[i].strip().split()
        img_id = int(header[0])
        mat = []
        for j in range(1, 5):
            row = [float(x) for x in lines[i + j].strip().split()]
            mat.append(row)
        w2c = np.array(mat, dtype=np.float32)
        poses.append((img_id, w2c))

    poses.sort(key=lambda x: x[0])
    return [p[1] for p in poses]


# ── T&T data loading ───────────────────────────────────────────


TNT_TRAINING_SCENES = [
    "Barn",
    "Caterpillar",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck",
]


def load_tnt_scene(tnt_dir, colmap_dir, scene_name, kf_every=1):
    """Load T&T scene images and GT poses.

    Supports two pose formats:
    1. COLMAP binary: colmap_dir/<scene>/0/images.bin (c2w)
    2. SfM .log: colmap_dir/<scene>/<scene>_COLMAP_SfM.log (w2c)

    Images are at: tnt_dir/<scene>/*.jpg

    Args:
        tnt_dir: Root dir containing <Scene>/*.jpg
        colmap_dir: Dir containing pose data (binary or .log)
        scene_name: e.g. "Barn"
        kf_every: Keyframe stride

    Returns:
        image_paths: list of image file paths
        gt_w2c: [N, 4, 4] world-to-camera matrices
    """
    scene_dir = Path(tnt_dir) / scene_name

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene image dir not found: {scene_dir}")

    # Try COLMAP binary first, then .log
    bin_path = Path(colmap_dir) / scene_name / "0" / "images.bin"
    log_path = Path(colmap_dir) / scene_name / f"{scene_name}_COLMAP_SfM.log"

    # Also try with sparse/ subdirectory (3DGS format)
    sparse_bin = Path(colmap_dir) / scene_name / "sparse" / "0" / "images.bin"

    if bin_path.exists():
        poses_dict = read_colmap_images_bin(str(bin_path))
        # c2w dict keyed by filename → convert to w2c list
        image_paths = []
        w2c_list = []
        for img_name in sorted(poses_dict.keys()):
            c2w = poses_dict[img_name]
            w2c = np.linalg.inv(c2w)
            # Find matching image file
            img_path = _find_image(scene_dir, img_name)
            if img_path:
                image_paths.append(str(img_path))
                w2c_list.append(w2c.astype(np.float32))
    elif sparse_bin.exists():
        poses_dict = read_colmap_images_bin(str(sparse_bin))
        image_paths = []
        w2c_list = []
        for img_name in sorted(poses_dict.keys()):
            c2w = poses_dict[img_name]
            w2c = np.linalg.inv(c2w)
            img_path = _find_image(scene_dir, img_name)
            if img_path:
                image_paths.append(str(img_path))
                w2c_list.append(w2c.astype(np.float32))
    elif log_path.exists():
        # .log format: T&T / Open3D Log convention is c2w (camera-to-world)
        # We need to invert to w2c for pose error computation
        c2w_poses = read_sfm_log(str(log_path))
        all_images = sorted([
            f for f in os.listdir(scene_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        n = min(len(all_images), len(c2w_poses))
        image_paths = [str(scene_dir / all_images[i]) for i in range(n)]
        w2c_list = [np.linalg.inv(c2w_poses[i]).astype(np.float32) for i in range(n)]
    else:
        raise FileNotFoundError(
            f"No pose data found for {scene_name}. "
            f"Checked: {bin_path}, {sparse_bin}, {log_path}"
        )

    # Apply keyframe stride
    if kf_every > 1:
        image_paths = image_paths[::kf_every]
        w2c_list = w2c_list[::kf_every]

    gt_w2c = np.stack(w2c_list, axis=0)  # [N, 4, 4]
    return image_paths, gt_w2c


def _find_image(scene_dir, img_name):
    """Find image file with flexible name matching."""
    # Direct match
    p = scene_dir / img_name
    if p.exists():
        return p
    # Try rgb/ subdirectory
    p = scene_dir / "rgb" / img_name
    if p.exists():
        return p
    # Try padding variants
    name, ext = os.path.splitext(img_name)
    digits = "".join(c for c in name if c.isdigit())
    if len(digits) == 5:
        p = scene_dir / (name.replace(digits, digits.zfill(6)) + ext)
        if p.exists():
            return p
    if len(digits) == 6:
        p = scene_dir / (digits[1:] + ext)
        if p.exists():
            return p
    return None


# ── Inference ──────────────────────────────────────────────────


@torch.no_grad()
def run_inference(model, images_tensor, dtype):
    """Run inference and return predicted w2c extrinsics [N, 3, 4]."""
    torch.cuda.synchronize()
    start = time.time()

    with torch.cuda.amp.autocast(dtype=dtype):
        imgs = images_tensor.cuda().to(dtype)
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)

        from methods.regimevggt import run_aggregator_regimevggt
        output_list, psi = run_aggregator_regimevggt(model, imgs)
        pose_enc = model.camera_head(output_list)[-1]

    with torch.amp.autocast("cuda", dtype=torch.float64):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_tensor.shape[-2:]
        )
        pred_extrinsic = extrinsic[0]  # [N, 3, 4]

    torch.cuda.synchronize()
    elapsed = time.time() - start
    return pred_extrinsic, elapsed


# ── Main ───────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser("RegimeVGGT Tanks & Temples pose evaluation")
    parser.add_argument("--tnt_dir", type=Path, required=True,
                        help="T&T images root (contains <Scene>/*.jpg).")
    parser.add_argument("--colmap_dir", type=Path, required=True,
                        help="Pose root with <Scene>/0/images.bin or <Scene>/<Scene>_COLMAP_SfM.log.")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="Scene names such as training__Barn; defaults to all six training scenes.")
    parser.add_argument("--kf_every", type=int, default=1, help="Keyframe stride.")
    parser.add_argument("--save_dir", type=Path, default=EVAL_DIR.parent / "results" / "regimevggt" / "tnt")
    parser.add_argument("--ckpt_path", type=str,
                        default=str(EVAL_DIR.parent / "ckpt" / "model_tracker_fixed_e20.pt"))
    return parser.parse_args()


def build_model(ckpt_path, dtype):
    model = VGGT()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval().to(dtype)
    return model


def main():
    args = parse_args()
    dtype = torch.bfloat16
    torch.manual_seed(0)
    np.random.seed(0)

    scenes = args.scenes if args.scenes else TNT_TRAINING_SCENES

    print(f"Tanks & Temples Pose Evaluation")
    print(f"  Scenes: {scenes}")
    print(f"  KF stride: {args.kf_every}")

    method_dir = args.save_dir
    method_dir.mkdir(parents=True, exist_ok=True)
    log_file = method_dir / "logs.txt"
    log_file.write_text("")

    model = build_model(args.ckpt_path, dtype)
    per_scene_results = {}

    for scene_idx, scene_name in enumerate(scenes):
        print(f"\n  [{scene_idx+1}/{len(scenes)}] {scene_name}", end=" ", flush=True)

        try:
            image_paths, gt_w2c = load_tnt_scene(
                args.tnt_dir, args.colmap_dir, scene_name, args.kf_every
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        N = len(image_paths)
        print(f"({N} frames)", end=" ", flush=True)

        # Load images
        images = load_and_preprocess_images(image_paths).to("cuda")

        patch_w = images.shape[-1] // 14
        patch_h = images.shape[-2] // 14
        model.update_patch_dimensions(patch_w, patch_h)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        pred_extr, elapsed = run_inference(model, images, dtype)

        with torch.amp.autocast("cuda", dtype=torch.float64):
            gt_se3 = torch.from_numpy(gt_w2c).to("cuda").double()
            add_row = torch.tensor([0, 0, 0, 1], device="cuda").expand(N, 1, 4).double()
            pred_se3 = torch.cat((pred_extr.double(), add_row), dim=1)

            r_err, t_err = se3_to_relative_pose_error(pred_se3, gt_se3, N)
            r_err_np = r_err.cpu().numpy()
            t_err_np = t_err.cpu().numpy()

        auc30 = calculate_auc_np(r_err_np, t_err_np, 30)
        auc15 = calculate_auc_np(r_err_np, t_err_np, 15)
        auc5 = calculate_auc_np(r_err_np, t_err_np, 5)

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"AUC@30={auc30:.4f} @15={auc15:.4f} @5={auc5:.4f}  "
              f"time={elapsed:.1f}s  VRAM={peak_vram:.1f}GB")

        per_scene_results[scene_name] = {
            "auc30": float(auc30), "auc15": float(auc15), "auc5": float(auc5),
            "time": float(elapsed), "vram_gb": float(peak_vram),
        }

        # Incremental log
        log_line = (
            f"Scene: {scene_name}, N: {N}, "
            f"AUC30: {auc30:.4f}, AUC15: {auc15:.4f}, AUC5: {auc5:.4f}, "
            f"Time_s: {elapsed:.2f}, VRAM_GB: {peak_vram:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    del model
    torch.cuda.empty_cache()

    # Summary
    if per_scene_results:
        print(f"\n{'─'*70}")
        print(f"  RegimeVGGT Summary — Tanks & Temples ({len(per_scene_results)} scenes)")
        print(f"{'─'*70}")
        for s in sorted(per_scene_results):
            r = per_scene_results[s]
            print(f"  {s:<15} AUC@30={r['auc30']:.4f}  @15={r['auc15']:.4f}  @5={r['auc5']:.4f}  time={r['time']:.1f}s")

        mean_auc = float(np.mean([r["auc30"] for r in per_scene_results.values()]))
        mean_auc15 = float(np.mean([r["auc15"] for r in per_scene_results.values()]))
        mean_auc5 = float(np.mean([r["auc5"] for r in per_scene_results.values()]))
        total_time = float(sum(r["time"] for r in per_scene_results.values()))
        print(f"  {'Mean/Total':<15} AUC@30={mean_auc:.4f}  @15={mean_auc15:.4f}  @5={mean_auc5:.4f}  time={total_time:.1f}s")

        summary = {
            "method": "regimevggt",
            "dataset": "Tanks_and_Temples",
            "num_scenes": len(per_scene_results),
            "mean_auc30": mean_auc,
            "mean_auc15": mean_auc15,
            "mean_auc5": mean_auc5,
            "total_time": total_time,
            "per_scene": per_scene_results,
        }
        summary_path = method_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        with open(log_file, "a") as f:
            f.write(f"\nMean AUC@30={mean_auc:.4f}  @15={mean_auc15:.4f}  @5={mean_auc5:.4f}  total_time={total_time:.1f}s\n")

        print(f"\n  Results saved to {method_dir}/")


if __name__ == "__main__":
    main()
