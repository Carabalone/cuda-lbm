import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_compute_capability_formats(self):
        self.assertEqual(benchmark.parse_compute_capability("9.0"), (9, 0))
        self.assertEqual(benchmark.parse_compute_capability("sm_75"), (7, 5))

    def test_visible_devices(self):
        self.assertIsNone(benchmark.parse_visible_devices(None))
        self.assertEqual(benchmark.parse_visible_devices("2, GPU-abc"), ["2", "GPU-abc"])
        self.assertEqual(benchmark.parse_visible_devices("-1"), [])

    def test_select_gpu_uses_visible_ordinals(self):
        gpus = [{"index": "0", "uuid": "GPU-a"}, {"index": "2", "uuid": "GPU-b"}]
        gpu, token = benchmark.select_gpu(gpus, 1, "0,2")
        self.assertEqual(gpu["uuid"], "GPU-b")
        self.assertEqual(token, "2")
        with self.assertRaisesRegex(RuntimeError, "select one"):
            benchmark.select_gpu(gpus, None, None)

    def test_warning_and_median(self):
        self.assertTrue(benchmark.contains_warning("x [WARNING] unstable"))
        self.assertFalse(benchmark.contains_warning("all good"))
        self.assertEqual(benchmark.median([9.0, 2.0, 4.0]), 4.0)

    def test_legacy_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.txt"
            path.write_text("# GPU: Test GPU\n# CUDA arch: sm_90\n\n"
                            "label,collision,layout,run1,run2,run3,median_mlups\n"
                            "BGK_SoA,BGK,SOA,1,3,2,2\n")
            metadata, rows = benchmark.parse_legacy_result(path)
        self.assertEqual(metadata["gpu"], "Test GPU")
        self.assertEqual(metadata["cuda_arch"], "sm_90")
        self.assertEqual(rows[0]["median_mlups"], "2")

    def test_manifest_is_complete_and_isolated(self):
        self.assertEqual(len(benchmark.CASES), 10)
        commands = [benchmark.make_command(case, "sm_90", 128,
                    Path(".benchmark/.build") / case.label) for case in benchmark.CASES]
        self.assertTrue(all("-DLBM_BENCHMARK_GRID=128" in command[-2] for command in commands))
        self.assertEqual(len({command[-3] for command in commands}), 10)

    def test_unsupported_blackwell_has_actionable_error(self):
        with mock.patch.object(benchmark, "command_output", return_value="sm_75 sm_90"):
            with self.assertRaisesRegex(RuntimeError, r"CUDA 12.8\+"):
                benchmark.validate_architecture("sm_100", {"toolkit_version": "12.0"})

    def test_dry_run_writes_full_manifest_without_building(self):
        gpu = {"index": "0", "name": "NVIDIA H100", "uuid": "GPU-test",
               "memory.total": "80000", "compute_cap": "9.0", "driver_version": "555",
               "temperature.gpu": "30", "clocks.current.sm": "1000",
               "clocks.current.memory": "1200"}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(benchmark, "query_gpus", return_value=[gpu]), \
             mock.patch.object(benchmark, "nvcc_info", return_value={"toolkit_version": "12.0"}), \
             mock.patch.object(benchmark, "validate_architecture"), \
             mock.patch.object(benchmark, "git_metadata", return_value={"commit": "x", "dirty": False}), \
             mock.patch("builtins.print"):
            result = benchmark.main(["--dry-run", "--output", tmp])
            run_dirs = [path for path in Path(tmp).iterdir() if path.is_dir()]
            metadata = json.loads((run_dirs[0] / "metadata.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(len(metadata["manifest"]), 10)

    def test_plotting_with_synthetic_h100(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = []
            for case in benchmark.CASES:
                for sample, value in enumerate((1000.0, 1100.0, 1050.0), 1):
                    rows.append({"gpu_key": "h100", "case": case.label,
                                 "collision": case.collision, "layout": case.layout,
                                 "sample": sample, "mlups": value, "median_mlups": 1050.0,
                                 "average_timestep_ms": 1.0, "steps_per_second": 1000.0,
                                 "status": "ok", "warmup_steps": 100, "measured_steps": 250,
                                 "date": "synthetic", "protocol": "test", "log": ""})
            benchmark.write_csv(run_dir / "results.csv", rows)
            benchmark.render_plots(run_dir, "h100")
            names = {path.name for path in (run_dir / "plots").iterdir()}
        self.assertIn("matrix_h100.png", names)
        self.assertIn("summary_soa_by_gpu.svg", names)


if __name__ == "__main__":
    unittest.main()
