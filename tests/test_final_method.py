"""CPU checks of the released method using small, untrained tensor fixtures."""

import os
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods import fastvggt_merge, regimevggt
from vggt.layers.block import Block


class TinyDinoAttention(nn.Module):
    """A real single-head attention layer with known CLS saliency ordering."""

    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.head_dim = 8
        self.qkv = nn.Linear(8, 24, bias=False)
        with torch.no_grad():
            self.qkv.weight.zero_()
            self.qkv.weight[0, 0] = 1
            self.qkv.weight[8, 0] = 1

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return F.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None]
        ).squeeze(1)


class TinyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn = TinyDinoAttention()
        self.blocks = nn.ModuleList([block])

    def forward(self, images):
        patches = images[:, 0].flatten(1).unsqueeze(-1).expand(-1, -1, 8)
        cls = torch.ones(patches.shape[0], 1, 8)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.blocks[0].attn(tokens)
        return {"x_norm_patchtokens": tokens[:, 1:]}


def tiny_model():
    """Use the actual transformer Block; no checkpoint or GPU is involved."""
    torch.manual_seed(7)
    agg = SimpleNamespace(
        depth=24,
        patch_size=1,
        patch_start_idx=5,
        patch_embed=TinyEmbedding(),
        camera_token=torch.randn(1, 2, 1, 8),
        register_token=torch.randn(1, 2, 4, 8),
        _resnet_mean=torch.zeros(1, 1, 3, 1, 1),
        _resnet_std=torch.ones(1, 1, 3, 1, 1),
        rope=None,
        frame_blocks=[Block(8, 1, mlp_ratio=1).eval() for _ in range(24)],
        global_blocks=[Block(8, 1, mlp_ratio=1).eval() for _ in range(24)],
    )
    return SimpleNamespace(aggregator=agg)


class FinalMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = []
        cls.kept = []
        cls.merged = []
        real_similarity = fastvggt_merge.fast_similarity_chunks
        real_keeper = regimevggt._build_keeper_mask_on_merged

        def trace_similarity(*args, **kwargs):
            result = real_similarity(*args, **kwargs)
            cls.matches.append(result)
            return result

        def trace_keeper(*args, **kwargs):
            mask = real_keeper(*args, **kwargs)
            cls.merged.append(args[0])
            cls.kept.append(args[0][mask])
            return mask

        # Patch values increase with position, so each frame's last ten
        # patches are the known top-10% CLS-attention tokens.
        images = torch.linspace(0.01, 1, 100).view(1, 1, 1, 10, 10)
        images = images.expand(1, 3, 3, 10, 10).contiguous()
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            # This existing runtime switch avoids CUDA synchronization on CPU.
            with patch.dict(os.environ, {"GEOVGGT_NO_PER_LAYER_CLEAN": "1"}), \
                 patch.object(fastvggt_merge, "fast_similarity_chunks", trace_similarity), \
                 patch.object(regimevggt, "_build_keeper_mask_on_merged", trace_keeper), \
                 torch.no_grad():
                cls.outputs, cls.patch_start = regimevggt.run_aggregator_regimevggt(
                    tiny_model(), images
                )
        finally:
            torch.set_num_threads(thread_count)

    def test_sensitive_layers_recompute_matches_instead_of_reusing_them(self):
        # One shallow match, four individual L11-L14 refreshes, one remaining
        # middle-band match, and one deep-band match.
        self.assertEqual(len(self.matches), 7)

    def test_salient_keys_survive_sampling_in_every_global_layer(self):
        self.assertEqual(len(self.kept), 24)
        for layer, kept in enumerate(self.kept):
            with self.subTest(layer=layer):
                for frame in (1, 2):
                    important = torch.arange(frame * 105 + 95, (frame + 1) * 105)
                    self.assertTrue(torch.isin(important, kept).all().item())

    def test_frame_zero_and_special_tokens_survive_every_global_layer(self):
        required = torch.cat([torch.arange(105), torch.arange(105, 110),
                              torch.arange(210, 215)])
        for layer, kept in enumerate(self.kept):
            with self.subTest(layer=layer):
                self.assertTrue(torch.isin(required, kept).all().item())

    def test_layer_bands_select_the_paper_phase_shifted_keys(self):
        # Hand-enumerated rows/columns for frames 1 and 2 on a 10x10 grid.
        # The middle grid truncates to 8x8, the shallow grid to 9x9.
        grids = (
            ({0, 1, 3, 4, 6, 7}, ({1, 2, 4, 5, 7, 8}, {0, 2, 3, 5, 6, 8})),
            ({0, 1, 2, 4, 5, 6}, ({1, 2, 3, 5, 6, 7}, {0, 2, 3, 4, 6, 7})),
            ({0, 1, 2, 5, 6, 7}, ({1, 2, 3, 6, 7, 8}, {2, 3, 4, 7, 8, 9})),
        )
        for layer, (merged, kept) in enumerate(zip(self.merged, self.kept)):
            rows, cols_by_frame = grids[0 if layer < 10 else 1 if layer < 18 else 2]
            allowed = set(range(105))
            for frame in (1, 2):
                allowed.update(range(frame * 105, frame * 105 + 5))
                allowed.update(range(frame * 105 + 95, (frame + 1) * 105))
                allowed.update(frame * 105 + 5 + r * 10 + c
                               for r in rows for c in cols_by_frame[frame - 1])
            with self.subTest(layer=layer):
                self.assertEqual(set(kept.tolist()), set(merged.tolist()) & allowed)

    def test_unmerged_outputs_remain_available_at_all_four_decoder_taps(self):
        self.assertEqual(self.patch_start, 5)
        self.assertEqual(len(self.outputs), 4)
        for output in self.outputs:
            self.assertEqual(tuple(output.shape), (1, 3, 105, 16))
            self.assertTrue(torch.isfinite(output).all().item())

    def test_multiple_scenes_are_rejected_with_a_clear_input_error(self):
        images = torch.zeros(2, 3, 3, 10, 10)
        with self.assertRaisesRegex(ValueError, "one scene"), \
             patch.dict(os.environ, {"GEOVGGT_NO_PER_LAYER_CLEAN": "1"}), \
             torch.no_grad():
            regimevggt.run_aggregator_regimevggt(tiny_model(), images)

    def test_padded_attention_matches_explicit_log_size_softmax(self):
        # Dropping the size term, using the padded head dimension as scale,
        # or forgetting to unmerge must change this numerical result.
        torch.manual_seed(11)
        block = Block(8, 1, mlp_ratio=1).eval()
        x = torch.randn(1, 63, 8)
        scores = torch.linspace(0, 1, 16).expand(3, -1)
        protect = regimevggt._build_psi_protect_mask(scores, 3, 4, 4, 21, 5)
        generator = torch.Generator().manual_seed(33)
        merge, unmerge = fastvggt_merge.token_merge_bipartite2d(
            x, 4, 4, 2, 2, 31, False, generator,
            enable_protection=True, importance_scores=scores, alpha=0.1,
        )
        with torch.no_grad():
            q, k, v = block.attn.qkv(block.norm1(x)).chunk(3, dim=-1)
            q, k, v = merge(q, mode="mean", extra_tensors=k, extra_tensors_2=v)
            keep = regimevggt._build_keeper_mask_on_merged(
                merge.merged_global_idx, 4, 4, 21, 5, 1.5, protect
            )
            sizes = merge.size[keep]
            self.assertGreater(sizes.max().item(), 1)
            logits = q @ k[:, keep].transpose(-2, -1) / math.sqrt(8)
            attended = logits.add(sizes.log()).softmax(-1) @ v[:, keep]
            expected = x + block.ls1(unmerge(block.attn.proj(attended)))
            expected = expected + block.ls2(block.mlp(block.norm2(expected)))
            actual = regimevggt._regimevggt_global_forward(
                block, x, None, 0.5, scores, None, 4, 4, 21, 5, 1.5, protect
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class ProtectedMergeAccountingTests(unittest.TestCase):
    def setUp(self):
        # Seed 33 puts eight of these top-10% patches in the destination pool.
        torch.manual_seed(7)
        self.q = torch.randn(1, 315, 8)
        self.k = torch.randn(1, 315, 6)
        self.v = torch.randn(1, 315, 4)
        self.merge, self.unmerge = fastvggt_merge.token_merge_bipartite2d(
            self.q, 10, 10, 2, 2, int(315 * 0.5), False,
            torch.Generator().manual_seed(33), enable_protection=True,
            importance_scores=torch.arange(100).expand(3, -1), alpha=0.1,
        )

    def test_salient_destinations_have_only_one_merged_parent_entry(self):
        parents = self.merge.merged_global_idx
        self.assertEqual(parents.unique().numel(), parents.numel())
        protected = torch.cat([torch.arange(200, 210), torch.arange(305, 315)])
        self.assertTrue(torch.isin(protected, parents).all().item())

    def test_token_sizes_conserve_original_mass_for_q_k_and_v(self):
        self.assertEqual(self.merge.size.sum().item(), 315)
        merged = self.merge(self.q, extra_tensors=self.k, extra_tensors_2=self.v)
        for original, reduced in zip((self.q, self.k, self.v), merged):
            represented_sum = (reduced * self.merge.size[None, :, None]).sum(dim=1)
            torch.testing.assert_close(represented_sum, original.sum(dim=1),
                                       rtol=1e-5, atol=1e-5)

    def test_unmerge_and_remerge_preserve_all_three_reduced_tensors(self):
        merged = self.merge(self.q, extra_tensors=self.k, extra_tensors_2=self.v)
        q, k, v = (self.unmerge(tensor) for tensor in merged)
        roundtrip = self.merge(q, extra_tensors=k, extra_tensors_2=v)
        for actual, expected in zip(roundtrip, merged):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
