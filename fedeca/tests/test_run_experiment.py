"""Test of the experiment pipeline."""

import subprocess
from pathlib import Path

import pandas as pd

from fedeca.utils.experiment_utils import load_dataframe_from_pickles


def test_example_experiment(tmp_path):
    """Test hydra run with the example experiment."""
    subprocess.run(
        [
            "python",
            "-m",
            "experiments.run_experiment",
            "-m",
            "experiment=example",
            f"hydra.sweep.dir={tmp_path}",
        ],
        check=False,
    )

    df_res = load_dataframe_from_pickles(tmp_path / "results_Example_experiment.pkl")
    df_res = df_res.drop(columns=["fit_time"]).reset_index(drop=True)
    # Not ideal, but save to and load from csv file to avoid dtype issues
    path_csv = tmp_path / "results.csv"
    df_res.to_csv(path_csv, index=False)
    df_res = pd.read_csv(path_csv)

    path_csv_true = (
        Path(__file__).parent / "artifacts/results_run_experiment_example.csv"
    )
    df_true = pd.read_csv(path_csv_true)

    pd.testing.assert_frame_equal(df_res, df_true)
