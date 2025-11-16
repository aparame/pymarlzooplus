import os
import sys
import subprocess
from pathlib import Path
import unittest

from pymarlzooplus.utils.plot_utils import (
    plot_single_experiment_results,
    plot_multiple_experiment_results,
    plot_average_per_algo_for_all_tasks_of_a_benchmark,
)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEARCH_CMD = [
    sys.executable,
    str(PROJECT_ROOT / "pymarlzooplus/search.py"),
    "run", "--config", "./tests/search.config.yaml", "--yes","locally","--cpus","1"
]

SINGLE_PATH = os.path.expanduser(f"{PROJECT_ROOT}/pymarlzooplus/results/sacred/maa2c/pistonball_v6/3/")
SINGLE_ALGO_NAME = "maa2c"
SINGLE_ENV_NAME = "pistonball_v6"

MULTI_ENV_NAMES = ["pistonball_v6", "cooperative_pong_v5"]
MULTI_ALGO_NAMES = ["QPLEX", "MAA2C"]

BENCHMARK_PICKLES = [
    os.path.expanduser(
        f"{PROJECT_ROOT}/pymarlzooplus/results/multiple-exps-plots/pistonball_v6/all_results_env=pistonball_v6.pkl"
    ),
    os.path.expanduser(
        f"{PROJECT_ROOT}/pymarlzooplus/results/multiple-exps-plots/cooperative_pong_v5/all_results_env=cooperative_pong_v5.pkl"
    )
]
BENCHMARK_TITLE = "PettingZoo"
BENCHMARK_SAVE_PATH = os.path.expanduser(f"{PROJECT_ROOT}/pymarlzooplus/results/multiple-exps-plots/")
BENCHMARK_PLOT_LEGEND = True


class TestPlotUtilsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not SEARCH_CMD:
            raise RuntimeError(
                "SEARCH_CMD is empty. Fill it with the correct command to run search.py."
            )

        print("Running search.py to generate real data:")
        print(" ".join(SEARCH_CMD))

        result = subprocess.run(
            SEARCH_CMD,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
        )

        if result.returncode == 0:
            print(f"STDOUT from search.py:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR from search.py:\n{result.stderr}")

        if result.returncode != 0:
            msg = (
                f"search.py failed with return code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
            raise RuntimeError(msg)

    def test_single_algo_plot(self):
        self.assertTrue(
            os.path.isdir(SINGLE_PATH),
            f"Single-experiment path does not exist: {SINGLE_PATH}. "
            "Did search.py write runs there?",
        )

        plot_single_experiment_results(
            path_to_results=SINGLE_PATH,
            algo_name=SINGLE_ALGO_NAME,
            env_name=SINGLE_ENV_NAME,
        )

        # Check that a 'plots' dir was created and has content
        plots_dir = os.path.join(SINGLE_PATH, "plots")
        self.assertTrue(
            os.path.isdir(plots_dir),
            f"'plots' directory was not created at {plots_dir}",
        )
        self.assertTrue(
            os.listdir(plots_dir),
            f"No plot files were created in {plots_dir}",
        )

    def test_multiple_algos_plot(self):
        for MULTI_ENV_NAME in MULTI_ENV_NAMES:
            multi_paths = [
                os.path.expanduser(f"{PROJECT_ROOT}/pymarlzooplus/results/sacred/qplex/{MULTI_ENV_NAME}"),
                os.path.expanduser(f"{PROJECT_ROOT}/pymarlzooplus/results/sacred/maa2c/{MULTI_ENV_NAME}"),
            ]


            multi_save_path = os.path.expanduser(f"{PROJECT_ROOT}/pymarlzooplus/results/multiple-exps-plots/{MULTI_ENV_NAME}/")

            existing_paths = [p for p in multi_paths if os.path.isdir(p)]
            existing_names = [
                name
                for p, name in zip(multi_paths, MULTI_ALGO_NAMES)
                if os.path.isdir(p)
            ]

            self.assertTrue(
                existing_paths,
                "None of the MULTI_PATHS exist. "
                "Make sure search.py generated data to those Sacred directories.",
            )

            os.makedirs(multi_save_path, exist_ok=True)

            plot_multiple_experiment_results(
                paths_to_results=existing_paths,
                algo_names=existing_names,
                env_name=MULTI_ENV_NAME,
                path_to_save=multi_save_path,
                plot_train=False,
                plot_legend_bool=True,
            )

            files = os.listdir(multi_save_path)
            self.assertTrue(
                files,
                f"No files created in {multi_save_path} by plot_multiple_experiment_results.",
            )

            self.all_results_pickle = os.path.join(
                multi_save_path,
                f"all_results_env={MULTI_ENV_NAME}.pkl",
            )
            self.assertTrue(
                os.path.isfile(self.all_results_pickle),
                f"{self.all_results_pickle} not found – "
                "plot_multiple_experiment_results should create it.",
            )

    def test_benchmark_average_plot(self):
        os.makedirs(BENCHMARK_SAVE_PATH, exist_ok=True)

        existing_pickles = [p for p in BENCHMARK_PICKLES if os.path.isfile(p)]
        self.assertTrue(
            existing_pickles,
            "None of the BENCHMARK_PICKLES exist. "
            "Make sure you ran multiple-exps plotting first.",
        )

        plot_average_per_algo_for_all_tasks_of_a_benchmark(
            paths_to_pickle_results=existing_pickles,
            plot_title=BENCHMARK_TITLE,
            path_to_save=BENCHMARK_SAVE_PATH,
            plot_legend_bool=BENCHMARK_PLOT_LEGEND,
        )

        files = os.listdir(BENCHMARK_SAVE_PATH)
        self.assertTrue(
            files,
            f"No files created in benchmark save path {BENCHMARK_SAVE_PATH}.",
        )


if __name__ == "__main__":
    # Create a TestSuite to enforce specific order
    suite = unittest.TestSuite()

    suite.addTest(TestPlotUtilsIntegration('test_single_algo_plot'))
    suite.addTest(TestPlotUtilsIntegration('test_multiple_algos_plot'))
    suite.addTest(TestPlotUtilsIntegration('test_benchmark_average_plot'))

    # Run the suite
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    sys.exit(not result.wasSuccessful())
