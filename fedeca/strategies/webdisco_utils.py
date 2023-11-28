"""Webdisco utils."""
import tempfile
from math import sqrt
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from substra import Client
from substrafl.algorithms import Algo
from substrafl.model_loading import _load_from_files

from fedeca.utils.substrafl_utils import download_train_task_models_by_round


def get_final_cox_model_function(
    client: Client,
    compute_plan_key: Union[str, Algo],
    num_rounds: int,
    standardize_data: bool,
    duration_col: str,
    event_col: str,
    simu_mode: bool = False,
    robust: bool = False,
):
    """Retreive first converged Cox model and corresponding hessian.

    Parameters
    ----------
    Client : Client
        The susbtrafl Client that registered the CP.
    compute_plan_key : Union[str, Algo]
        The key of the CP.
    num_rounds : int
        The number of rounds of the CP.
    standardize_data : float, optional
        Whether or not the data was standadized, by default 0.05
    duration_col : str
        The name of the duration column.
    event_col : str
        The name of the event column.
    simu_mode : bool
        Whether or not we are using simu mode. Note this could be inferred from
        the Client.
    robust : bool, optional
        Retreive global statistics for robust variance estimation.

    Returns
    -------
    tuple
        Returns hessian, log-likelihood, Cox model's weights, global moments
    """
    found_params = False
    found_hessian = False

    for i in range(0, num_rounds):
        actual_round = get_last_algo_from_round_count(i, standardize_data, simu_mode)
        if not simu_mode:
            # We have to use a custom function instead of download_algo_state
            # to bypass substrafl limitation on number of local tasks per
            # round
            with tempfile.TemporaryDirectory() as temp_dir:
                download_train_task_models_by_round(
                    client=client,
                    dest_folder=temp_dir,
                    compute_plan_key=compute_plan_key,
                    round_idx=actual_round,
                )
                algo = _load_from_files(input_folder=temp_dir)
        else:
            algo = compute_plan_key.intermediate_states[actual_round]

        if algo.server_state["success"]:
            if not found_params:
                found_params = True
                algo_params = algo
            # Unfortunately finding params is only half of the job as we
            # need the hessian computed on those params
            else:
                found_hessian = True
                algo_hessian = algo
                break

        if i == max(num_rounds - 2, 0) and not found_params:
            print(
                """Cox model did not converge ! Taking params from the before final
                round"""
            )
            found_params = True
            algo_params = algo

        if i == (num_rounds - 1):
            if algo.server_state["success"]:
                print("You are one round short to get the true final hessian !")
            found_hessian = True
            algo_hessian = algo

    assert (
        found_params and found_hessian
    ), """Do more rounds it needs to converge and then do one more round
        to get the final hessian"""

    model = algo_params.model
    hessian = algo_hessian.server_state["hessian"]
    ll = algo_hessian.server_state["past_ll"]

    if standardize_data:

        global_moments = algo.global_moments
        computed_vars = global_moments["vars"]
        # We need to match pandas standardization
        bias_correction = global_moments["bias_correction"]

        computed_stds = computed_vars.transform(
            lambda x: sqrt(x * bias_correction + 1e-16)
        )
    else:
        computed_stds = pd.Series(np.ones((model.fc1.weight.shape)).squeeze())
        global_moments = {}

    # We unstandardize the weights
    final_params = model.fc1.weight.data.numpy().squeeze() / computed_stds.to_numpy()

    # Robust estimation
    global_robust_statistics = {}
    if robust:
        global_robust_statistics = algo_hessian.server_state["global_robust_statistics"]
        global_robust_statistics["global_moments"] = global_moments

    return hessian, ll, final_params, computed_stds, global_robust_statistics


def compute_summary_function(final_params, variance_matrix, alpha=0.05):
    """Compute summary function.

    Parameters
    ----------
    final_params : np.ndarray
        The estimated vallues of Cox model coefficients.
    variance_matrix : np.ndarray
        Computed variance matrix whether using robust estimation or not.
    alpha : float, optional
        The quantile level to test, by default 0.05

    Returns
    -------
    pd.DataFrame
        Summary of IPTW analysis as in lifelines.
    """
    se = np.sqrt(variance_matrix.diagonal())
    ci = 100 * (1 - alpha)
    z = stats.norm.ppf(1 - alpha / 2)
    Z = final_params / se
    U = Z**2
    pvalues = stats.chi2.sf(U, 1)
    summary = pd.DataFrame()
    summary["coef"] = final_params
    summary["se(coef)"] = se
    summary[f"coef lower {round(ci)}%"] = final_params - z * se
    summary[f"coef upper {round(ci)}%"] = final_params + z * se
    summary["z"] = Z
    summary["p"] = pvalues

    return summary


def get_last_algo_from_round_count(num_rounds, standardize_data=True, simu_mode=False):
    """Get true number of rounds.

    Parameters
    ----------
    num_rounds : list[int]
        _description_
    standardize_data : bool, optional
        _description_, by default True
    simu_mode : bool
        Whether or not we are in simu mode.

    Returns
    -------
    _type_
        _description_
    """
    # One count for each aggregation starting at 1 (init round): +1 for
    # standardization +1 for global_survival_statistics
    if not simu_mode:
        actual_number_of_rounds = 2 * (num_rounds + 1) + 2
    # Minus 1 stems from the fact that simu mode is peculiar
    # and that we start adding to it only in the build_compute_plan
    # aka 1 before

    if simu_mode:
        actual_number_of_rounds = (num_rounds + 1) + 2
    if not standardize_data:
        actual_number_of_rounds -= 1
    return actual_number_of_rounds
