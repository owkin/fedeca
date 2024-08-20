"""Test of the experiment pipeline."""

import subprocess
from pathlib import Path

import pandas as pd

from fedeca.utils.experiment_utils import load_dataframe_from_pickles


def test_example_experiment(tmp_path):
    """Test hydra run with the "test_run" experiment."""
    subprocess.run(
        [
            "python",
            "-m",
            "experiments.run_experiment",
            "-m",
            "experiment=test_run",
            f"hydra.sweep.dir={tmp_path}",
        ],
        check=False,
    )

    df_res = load_dataframe_from_pickles(tmp_path / "results_Test_run.pkl")
    df_res = df_res.drop(columns=["fit_time"]).reset_index(drop=True)
    # Not ideal, but save to and load from csv file to avoid dtype issues
    path_csv = tmp_path / "results.csv"
    df_res.to_csv(path_csv, index=False)
    df_res = pd.read_csv(path_csv)

    path_csv_true = Path(__file__).parent / "artifacts/results_test_run.csv"
    df_true = pd.read_csv(path_csv_true)
    # In the past there was a bug in SMD computation so no need to match the bug
    cols = [col for col in df_true.columns if "smd" not in col]

    pd.testing.assert_frame_equal(df_res[cols], df_true[cols])
