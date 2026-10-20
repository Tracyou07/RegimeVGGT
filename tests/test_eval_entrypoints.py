"""Exercise public CLI and shell argument forwarding without model dependencies."""

import argparse
import ast
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if os.name == "nt" and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH = "C:/Program Files/Git/bin/bash.exe"


def load_parser(filename):
    """Execute only the real argparse declarations, leaving GPU imports unloaded."""
    tree = ast.parse((ROOT / "eval" / filename).read_text(encoding="utf-8"))
    container = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)
         and node.name in {"get_args_parser", "parse_args"}),
        None,
    )
    if container is None:
        container = next(node for node in tree.body if isinstance(node, ast.If)
                         and any(isinstance(item, ast.Assign) and any(
                             isinstance(target, ast.Name) and target.id == "parser"
                             for target in item.targets) for item in node.body))
    statements = [
        node for node in container.body
        if (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"parser", "scene_selection"}
            for target in node.targets
        )) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id in {"parser", "scene_selection"})
    ]
    namespace = dict(argparse=argparse, Path=Path, os=os, osp=os.path,
                     ROOT_DIR=str(ROOT), REPO_ROOT=ROOT, EVAL_DIR=ROOT / "eval")
    exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), namespace)
    return namespace["parser"]


class EvaluationEntrypoints(unittest.TestCase):
    def parse_call(self, call):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                return load_parser(Path(call[0]).name).parse_args(call[1:])
            except SystemExit:
                self.fail(f"Wrapper forwarded unsupported CLI arguments: {call[1:]}")

    def run_wrapper(self, name, **extra):
        if not BASH:
            self.skipTest("bash is required for shell wrapper checks")
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory)
            (sandbox / "scripts").mkdir()
            script = sandbox / "scripts" / name
            shutil.copyfile(ROOT / "scripts" / name, script)
            capture = sandbox / "arguments.txt"
            env = dict(os.environ, SEVEN_ROOT="/data/seven scenes", NRGBD_ROOT="/data/nrgbd",
                       TNT_DIR="/data/tnt", TNT_COLMAP_DIR="/data/poses",
                       SCANNET_DATA="/data/scannet", SCANNET_GT_PLY="/data/gt",
                       SCANNET_SPLIT="/data/protocol/scannet50.txt",
                       CKPT="/weights/checkpoint with space.pt", CAPTURE=capture.as_posix())
            env.pop("FRAMES", None)
            env.update(extra)
            command = ('python() { printf "%s\\n" __CALL__ "$@" __END__ >> "$CAPTURE"; }; '
                       'export -f python; bash "$1"')
            result = subprocess.run([BASH, "-c", command, "test", script.as_posix()],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = []
            for block in capture.read_text().split("__CALL__\n")[1:]:
                calls.append(block.splitlines()[:-1])
            return calls

    def test_wrappers_forward_checkpoint_to_supported_cli(self):
        for script in ("eval_7andn.sh", "eval_tnt.sh", "eval_scannet.sh"):
            with self.subTest(script=script):
                for call in self.run_wrapper(script):
                    self.assertIn("--ckpt_path", call)
                    parsed = self.parse_call(call)
                    self.assertEqual(parsed.ckpt_path, "/weights/checkpoint with space.pt")

    def test_scannet_default_sweeps_all_paper_frame_counts(self):
        calls = self.run_wrapper("eval_scannet.sh")
        parsers = [self.parse_call(call) for call in calls]
        self.assertEqual([args.input_frame for args in parsers], [1000, 500, 300, 100])

    def test_scannet_frame_override_selects_one_run(self):
        calls = self.run_wrapper("eval_scannet.sh", FRAMES="300")
        self.assertEqual(len(calls), 1)
        parsed = self.parse_call(calls[0])
        self.assertEqual(parsed.input_frame, 300)

    def test_scannet_wrapper_forwards_exact_benchmark_split(self):
        parsed = self.parse_call(self.run_wrapper("eval_scannet.sh", FRAMES="100")[0])
        self.assertEqual(parsed.scene_list, Path("/data/protocol/scannet50.txt"))

    def test_scannet_requires_explicit_split_or_custom_subset(self):
        required = ["--data_dir", "/data/scannet", "--gt_ply_dir", "/data/gt"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                load_parser("eval_scannet50.py").parse_args(required)
        subset = load_parser("eval_scannet50.py").parse_args(required + ["--num_scenes", "2"])
        self.assertEqual(subset.num_scenes, 2)

    def test_default_checkpoints_match_documented_repository_location(self):
        for filename in ("eval_7andN.py", "eval_pose_tnt.py", "eval_scannet50.py"):
            with self.subTest(filename=filename):
                default = load_parser(filename).get_default("ckpt_path")
                self.assertEqual(Path(default).resolve(),
                                 ROOT / "ckpt" / "model_tracker_fixed_e20.pt")

    def test_legacy_method_and_ablation_arguments_are_rejected(self):
        fixtures = (
            ("eval_7andN.py", []),
            ("eval_pose_tnt.py", ["--tnt_dir", "/data/tnt", "--colmap_dir", "/data/poses"]),
            ("eval_scannet50.py", ["--data_dir", "/data/scannet", "--gt_ply_dir", "/data/gt",
                                   "--num_scenes", "2"]),
        )
        for filename, required in fixtures:
            for option in (["--method", "baseline"], ["--merge_ratio", "0.8"]):
                with self.subTest(filename=filename, option=option):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            load_parser(filename).parse_args(required + option)


if __name__ == "__main__":
    unittest.main()
