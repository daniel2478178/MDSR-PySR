import multiprocessing as mp
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.discovery import pysr_with_params


def inspect_spawned_config(config):
    return {
        "id_folder": pysr_with_params.resolve_id_folder("P21", config),
        "iterations": config.iterations,
        "pysr_processes": config.pysr_processes,
        "run_id": config.run_id,
    }


class NullQueue:
    def put(self, _message):
        pass


class FakePySRRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class RunConfigTests(unittest.TestCase):
    def build_config(self, data_root):
        args = pysr_with_params.build_parser().parse_args([
            "physicsMDSR_Range_20_59.xlsx",
            str(data_root),
            "--iterations", "400",
            "--outer-processes", "4",
            "--pysr-processes", "3",
            "--run-id", "warm_0_7",
        ])
        return pysr_with_params.build_config(args)

    def test_cli_data_root_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "physicsMDSR_Range_CSV_Noise_01"
            config = self.build_config(data_root)

            self.assertEqual(config.data_root, data_root.resolve())
            self.assertEqual(
                pysr_with_params.resolve_id_folder("P21", config),
                data_root.resolve() / "P21",
            )

    def test_spawned_worker_keeps_cli_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "physicsMDSR_Range_CSV_Noise_01"
            config = self.build_config(data_root)

            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                result = executor.submit(inspect_spawned_config, config).result()

            self.assertEqual(result["id_folder"], data_root.resolve() / "P21")
            self.assertEqual(result["iterations"], 400)
            self.assertEqual(result["pysr_processes"], 3)
            self.assertEqual(result["run_id"], "warm_0_7")

    def test_actual_spawned_task_uses_configured_data_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "physicsMDSR_Range_CSV_Noise_01"
            id_folder = data_root / "P_CONFIG_TEST"
            id_folder.mkdir(parents=True)
            (id_folder / "pysr_warm_0_7_summary.csv").write_text(
                "completed\n",
                encoding="utf-8",
            )
            config = self.build_config(data_root)

            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                result = executor.submit(
                    pysr_with_params.process_directory_task,
                    "P_CONFIG_TEST",
                    {},
                    config,
                    NullQueue(),
                ).result()

            self.assertEqual(result["status"], "SKIPPED")

    def test_model_uses_cli_runtime_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "physicsMDSR_Range_CSV_Noise_01"
            id_folder = data_root / "P21"
            id_folder.mkdir(parents=True)
            config = self.build_config(data_root)

            fake_pysr = SimpleNamespace(PySRRegressor=FakePySRRegressor)
            with patch.dict(sys.modules, {"pysr": fake_pysr}):
                model = pysr_with_params.build_model(id_folder, config)

            self.assertEqual(model.kwargs["niterations"], 400)
            self.assertEqual(model.kwargs["procs"], 3)
            self.assertEqual(model.kwargs["run_id"], "warm_0_7")
            self.assertEqual(
                Path(model.kwargs["output_directory"]),
                id_folder / "pysr_runs",
            )


if __name__ == "__main__":
    unittest.main()
