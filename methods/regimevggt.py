"""The final RegimeVGGT inference method from the paper.

All 24 global blocks use saliency-protected Q/K/V merging, phase-shifted
K/V downsampling, size-aware attention, and unmerging before the residual
and MLP. The zero-based bands are [0, 10), [10, 18), and [18, 24), with
merge rates (0.99, 0.50, 0.99) and nominal K/V rates (1.5, 1.3, 1.7).
DINO CLS top-10% patches, frame 0, and camera/register tokens are preserved.
Merge indices are cached per band, with layers 10-13 refreshed individually.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from vggt.models.aggregator import slice_expand_and_flatten
from methods.fastvggt_merge import clear_merge_cache, token_merge_bipartite2d


_SALIENCY_FRACTION = 0.1
_MERGE_RATIOS = (0.99, 0.50, 0.99)
_KV_SIGMAS = (1.5, 1.3, 1.7)


def _sigma_to_tile(sigma):
    """Discrete grids for the paper's nominal K/V downsampling rates."""
    if sigma == 1.3:
        return 4, 3
    if sigma == 1.5:
        return 3, 2
    if sigma == 1.7:
        return 5, 3
    raise ValueError(f"Unsupported paper K/V rate: {sigma}")


def _build_keeper_mask_on_merged(
    merged_global_idx: torch.Tensor,
    Hp: int,
    Wp: int,
    tokens_per_frame: int,
    num_special: int,
    sigma_sub: float,
    protected_global_idx: torch.Tensor,
) -> torch.Tensor:
    """Keep phase-shifted merged keys plus specials, frame 0, and saliency."""
    g = merged_global_idx.long()
    frame = g // tokens_per_frame
    local = g % tokens_per_frame
    is_special = local < num_special
    patch_idx = local - num_special
    r = patch_idx // Wp
    c = patch_idx % Wp

    tile, kept = _sigma_to_tile(sigma_sub)
    in_grid = (r < (Hp // tile) * tile) & (c < (Wp // tile) * tile)
    phase_r = (frame // tile) % tile
    phase_c = frame % tile
    keep = (((r - phase_r) % tile) < kept)
    keep = keep & (((c - phase_c) % tile) < kept) & in_grid
    return keep | is_special | (frame == 0) | protected_global_idx[g]


def _build_psi_protect_mask(
    importance_scores: torch.Tensor,
    S: int,
    Hp: int,
    Wp: int,
    tokens_per_frame: int,
    num_special: int,
) -> torch.Tensor:
    """Build the shared per-scene top-10% saliency mask for K/V protection."""
    T_patches = Hp * Wp
    num_protect = max(1, int(T_patches * _SALIENCY_FRACTION))
    dev = importance_scores.device
    protected_global = torch.zeros(
        S * tokens_per_frame, dtype=torch.bool, device=dev,
    )
    protected_global[num_special:num_special + T_patches] = True
    if S > 1:
        _, topk = importance_scores[1:S].topk(num_protect, dim=1)
        frame_offsets = (
            torch.arange(1, S, device=dev) * tokens_per_frame + num_special
        ).view(-1, 1)
        protected_global[(frame_offsets + topk).reshape(-1)] = True
    return protected_global


def _regimevggt_global_forward(
    block,
    x_all,
    pos,
    merge_ratio: float,
    importance_scores,
    merge_cache_key,
    Hp: int,
    Wp: int,
    tokens_per_frame: int,
    num_special: int,
    sigma_sub: float,
    psi_protect_mask: torch.Tensor,
):
    """Merge Q/K/V, sample protected K/V, attend with token-size bias, unmerge."""
    attn_mod = block.attn
    B, N_all, C = x_all.shape
    H = attn_mod.num_heads
    D = attn_mod.head_dim

    # ── 1. LayerNorm ──
    x_normed = block.norm1(x_all)

    # ── 2. QKV projection on full tokens ──
    qkv = attn_mod.qkv(x_normed).reshape(B, N_all, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each [B, H, N, D]
    q = attn_mod.q_norm(q)
    k = attn_mod.k_norm(k)
    if attn_mod.rope is not None and pos is not None:
        q = attn_mod.rope(q, pos)
        k = attn_mod.rope(k, pos)
    del qkv, x_normed

    # ── 3. Bipartite merge on x_all (metric), applied to Q, K, V jointly ──
    r_count = int(N_all * merge_ratio)
    generator = torch.Generator(device=x_all.device)
    generator.manual_seed(33)
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        x_all, Wp, Hp, 2, 2, r_count, False, generator,
        enable_protection=True,
        importance_scores=importance_scores,
        alpha=_SALIENCY_FRACTION,
        cache_key=merge_cache_key,
    )
    merged_global_idx = merge_fn.merged_global_idx  # [N_m] int64

    # Flatten q, k, v from [B, H, N, D] to [B, N, H*D] for merge_fn.
    q_flat = q.permute(0, 2, 1, 3).reshape(B, N_all, H * D)
    k_flat = k.permute(0, 2, 1, 3).reshape(B, N_all, H * D)
    v_flat = v.permute(0, 2, 1, 3).reshape(B, N_all, H * D)
    del q, k, v

    q_m, k_m, v_m = merge_fn(
        q_flat, mode="mean", extra_tensors=k_flat, extra_tensors_2=v_flat
    )
    del q_flat, k_flat, v_flat
    N_m = q_m.shape[1]

    # Reshape back to [B, H, N_m, D].
    q_m = q_m.reshape(B, N_m, H, D).permute(0, 2, 1, 3)
    k_m = k_m.reshape(B, N_m, H, D).permute(0, 2, 1, 3)
    v_m = v_m.reshape(B, N_m, H, D).permute(0, 2, 1, 3)

    keep_mask = _build_keeper_mask_on_merged(
        merged_global_idx, Hp, Wp, tokens_per_frame, num_special,
        sigma_sub, psi_protect_mask,
    )
    keeper_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    kept_size = merge_fn.size[keeper_idx]
    k_sub = k_m[:, :, keeper_idx, :]
    v_sub = v_m[:, :, keeper_idx, :]

    # ToMe size-aware bias on the (merge → subsample) K/V via
    # dim-append flash trick: encode log(size)·sqrt(d) into an extra
    # K-dim, append matching cols to Q (=1) and V (=0), pad to next
    # flash-supported head_dim, preserving flash SDPA without an attn_mask.
    d_head = q_m.shape[-1]
    flash_supported = (8, 16, 32, 64, 80, 96, 112, 128, 160, 192, 224, 256)
    target_d = next(
        (fd for fd in flash_supported if fd > d_head), d_head + 1
    )
    pad_extra = target_d - d_head - 1
    sqrt_d = math.sqrt(d_head)

    log_size_scaled = (
        (torch.log(kept_size.clamp_min(1.0)) * sqrt_d)
        .to(q_m.dtype)
        .view(1, 1, -1, 1)
        .expand(k_sub.shape[0], k_sub.shape[1], k_sub.shape[2], 1)
    )
    ones_q = torch.ones(
        *q_m.shape[:-1], 1, device=q_m.device, dtype=q_m.dtype
    )
    zero_v = torch.zeros(
        *v_sub.shape[:-1], 1, device=v_sub.device, dtype=v_sub.dtype
    )
    if pad_extra > 0:
        q_pad = torch.zeros(
            *q_m.shape[:-1], pad_extra, device=q_m.device, dtype=q_m.dtype
        )
        k_pad = torch.zeros(
            *k_sub.shape[:-1], pad_extra, device=k_sub.device, dtype=k_sub.dtype
        )
        v_pad = torch.zeros(
            *v_sub.shape[:-1], pad_extra, device=v_sub.device, dtype=v_sub.dtype
        )
        q_aug = torch.cat([q_m, ones_q, q_pad], dim=-1)
        k_aug = torch.cat([k_sub, log_size_scaled, k_pad], dim=-1)
        v_aug = torch.cat([v_sub, zero_v, v_pad], dim=-1)
    else:
        q_aug = torch.cat([q_m, ones_q], dim=-1)
        k_aug = torch.cat([k_sub, log_size_scaled], dim=-1)
        v_aug = torch.cat([v_sub, zero_v], dim=-1)
    attn_aug = F.scaled_dot_product_attention(
        q_aug, k_aug, v_aug, scale=1.0 / sqrt_d
    )
    attn_out = attn_aug[..., :d_head]
    del q_aug, k_aug, v_aug, attn_aug, log_size_scaled, ones_q, zero_v
    del q_m, k_m, v_m, k_sub, v_sub

    # ── 6. Output projection (on merged-token layout) ──
    attn_out = attn_out.transpose(1, 2).reshape(B, N_m, C)
    attn_out = attn_mod.proj(attn_out)
    attn_out = attn_mod.proj_drop(attn_out)

    # ── 7. Unmerge and residual ──
    attn_out = unmerge_fn(attn_out)    # [B, N_all, C]
    x = x_all + block.ls1(attn_out)
    del attn_out
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x


def run_aggregator_regimevggt(model, images):
    """Run the fixed paper method; return decoder taps and patch-start index.

    ``images`` has shape [1, S, 3, H, W]: one scene with S frames per call.
    """
    agg = model.aggregator
    B, S, C_in, H, W = images.shape
    if B != 1:
        raise ValueError("RegimeVGGT expects one scene per call (batch size 1)")
    Hp = H // agg.patch_size
    Wp = W // agg.patch_size
    captured_psi = []

    def _psi_hook(module, inp, out):
        x_in = inp[0]
        B_h, N_h, C_h = x_in.shape
        qkv = module.qkv(x_in).reshape(
            B_h, N_h, 3, module.num_heads, C_h // module.num_heads
        )
        q, k, _ = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q_cls = q[:, :, 0:1]
        attn = torch.matmul(q_cls, k.transpose(-2, -1)) * (module.head_dim ** -0.5)
        cls_attn = F.softmax(attn, dim=-1)[:, :, 0, 1:].mean(dim=1)
        captured_psi.append(cls_attn.detach())

    clear_merge_cache()
    hook_handle = agg.patch_embed.blocks[-1].attn.register_forward_hook(_psi_hook)
    imgs = (images - agg._resnet_mean) / agg._resnet_std
    imgs_flat = imgs.view(B * S, C_in, H, W)
    try:
        patch_tokens = agg.patch_embed(imgs_flat)
    finally:
        hook_handle.remove()
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    if not captured_psi:
        raise RuntimeError("DINO CLS attention was not captured for saliency protection")
    psi_raw = torch.cat(captured_psi, dim=0)
    T_patches = Hp * Wp
    if psi_raw.shape[1] < T_patches:
        raise ValueError("DINO CLS attention has fewer entries than the patch grid")
    psi = psi_raw[:, -T_patches:].contiguous()
    del captured_psi, psi_raw

    camera_token = slice_expand_and_flatten(agg.camera_token, B, S)
    register_token = slice_expand_and_flatten(agg.register_token, B, S)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
    del camera_token, register_token, patch_tokens, imgs_flat, imgs

    pos = None
    if agg.rope is not None:
        pos = agg.position_getter(B * S, Hp, Wp, device=images.device) + 1
        pos_special = torch.zeros(
            B * S, agg.patch_start_idx, 2, device=images.device, dtype=pos.dtype
        )
        pos = torch.cat([pos_special, pos], dim=1)

    _, P, C = tokens.shape
    patch_start = agg.patch_start_idx
    psi_protect_mask = _build_psi_protect_mask(psi, S, Hp, Wp, P, patch_start)
    del images
    torch.cuda.empty_cache()

    block4DPT_idx = [4, 11, 17, 23]
    output_list = []
    for i in range(agg.depth):
        tokens = tokens.view(B * S, P, C)
        frame_pos = pos.view(B * S, P, 2) if pos is not None else None
        tokens = agg.frame_blocks[i](tokens, pos=frame_pos)
        frame_inter = tokens.view(B, S, P, C) if i in block4DPT_idx else None

        tokens_global = tokens.view(B, S * P, C)
        global_pos = pos.view(B, S * P, 2) if pos is not None else None
        band = 0 if i < 10 else 1 if i < 18 else 2
        layer_ratio = _MERGE_RATIOS[band]
        merge_cache_key = None if 10 <= i < 14 else (
            "regimevggt_band", band, layer_ratio,
        )
        tokens_global = _regimevggt_global_forward(
            agg.global_blocks[i], tokens_global, global_pos,
            merge_ratio=layer_ratio,
            importance_scores=psi,
            merge_cache_key=merge_cache_key,
            Hp=Hp, Wp=Wp, tokens_per_frame=P, num_special=patch_start,
            sigma_sub=_KV_SIGMAS[band], psi_protect_mask=psi_protect_mask,
        )
        tokens = tokens_global.view(B * S, P, C)

        if i in block4DPT_idx:
            global_inter = tokens.view(B, S, P, C)
            concat_inter = torch.cat([frame_inter, global_inter], dim=-1)
            if concat_inter.dtype != torch.bfloat16:
                concat_inter = concat_inter.to(torch.bfloat16)
            output_list.append(concat_inter)
            del frame_inter, global_inter, concat_inter

        if os.environ.get("GEOVGGT_NO_PER_LAYER_CLEAN", "0") != "1":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if agg.rope is not None and hasattr(agg.rope, "frequency_cache"):
                agg.rope.frequency_cache.clear()

    del tokens, pos
    torch.cuda.empty_cache()
    return output_list, agg.patch_start_idx
