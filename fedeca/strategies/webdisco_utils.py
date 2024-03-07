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

from fedeca.utils.substrafl_utils import (
    download_train_task_models_by_round,
    get_simu_state_from_round,
)


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
    """Retrieve first converged Cox model and corresponding hessian.

    In case of bootstrapping retrieves the first converged Cox models for each
    seed.

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
    # We retrieve the end task of the first round to see if it is bootstrapped or
    # not
    actual_round = get_last_algo_from_round_count(0, standardize_data, simu_mode)
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
        # compute_plan_key is a tuple with (scores, train_states, aggregate_states)
        train_state = get_simu_state_from_round(
            simu_memory=compute_plan_key[1],
            client_id=client.organization_info().organization_id,
            round_idx=actual_round,
        )

        algo = train_state.algo

    if hasattr(algo, "individual_algos"):
        num_seeds = len(algo.individual_algos)

        def test_convergence(algo_arg):
            return [
                (indiv_algo, indiv_algo.server_state["success"])
                for indiv_algo in algo_arg.individual_algos
            ]

    else:
        num_seeds = 1

        def test_convergence(algo_arg):
            return [(algo_arg, algo_arg.server_state["success"])]

    found_params_dict = {}
    found_hessian_dict = {}
    algo_params = {}
    algo_hessian = {}
    for i in range(num_seeds):
        found_params_dict[i] = False
        found_hessian_dict[i] = False
        algo_params[i] = None
        algo_hessian[i] = None

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
            # compute_plan_key is a tuple with (scores, train_states, aggregate_states)
            train_state = get_simu_state_from_round(
                simu_memory=compute_plan_key[1],
                client_id=client.organization_info().organization_id,
                round_idx=actual_round,
            )

            algo = train_state.algo

        convergence_of_algos = test_convergence(algo)

        for j in range(num_seeds):
            if convergence_of_algos[j][1]:
                if not found_params_dict[j]:
                    found_params_dict[j] = True
                    algo_params[j] = convergence_of_algos[j][0]
                # Unfortunately finding params is only half of the job as we
                # need the hessian computed on those params
                else:
                    found_hessian_dict[j] = True
                    algo_hessian[j] = convergence_of_algos[j][0]

        # we only break if all hessians have been found across seeds
        if all(found_hessian_dict.values()):
            break

        for j in range(num_seeds):
            if i == max(num_rounds - 2, 0) and not found_params_dict[j]:
                print(
                    f"""Cox model did not converge for seed {j} ! Taking params
                    from the before final round"""
                )
                found_params_dict[j] = True
                algo_params[j] = convergence_of_algos[j][0]

            if i == (num_rounds - 1):
                if convergence_of_algos[j][1]:
                    print(
                        f"For seed {j} You are one round short to get the"
                        " true final hessian !"
                    )
                found_hessian_dict[j] = True
                algo_hessian[j] = convergence_of_algos[j][0]

    assert all([list(found_params_dict.values())]) and all(
        [list(found_hessian_dict.values())]
    ), """Do more rounds all seeds have not converged and then do one more round
        after convergence to to get the final hessian"""

    # For each seed we extract the final quantities
    models = [algo_p.model for algo_p in algo_params.values()]
    hessians = [algo_h.server_state["hessian"] for algo_h in algo_hessian.values()]
    lls = [algo_h.server_state["past_ll"] for algo_h in algo_hessian.values()]

    if standardize_data:

        global_moments_list = [algo_p.global_moments for algo_p in algo_params.values()]
        computed_vars_list = [g["vars"] for g in global_moments_list]
        # We need to match pandas standardization across seeds
        bias_correction_list = [g["bias_correction"] for g in global_moments_list]
        computed_stds_list = [
            computed_v.transform(lambda x: sqrt(x * bias_c + 1e-16))
            for computed_v, bias_c in zip(computed_vars_list, bias_correction_list)
        ]

    else:
        computed_stds_list = [
            pd.Series(np.ones((models[0].fc1.weight.shape)).squeeze())
            for _ in range(num_seeds)
        ]
        global_moments_list = [{} for _ in range(num_seeds)]

    # We unstandardize the weights across seeds
    final_params_list = [
        model.fc1.weight.data.numpy().squeeze() / computed_stds.to_numpy()
        for model, computed_stds in zip(models, computed_stds_list)
    ]

    # Robust estimation
    global_robust_statistics = {}
    if robust:
        assert num_seeds == 1, "cannot use robust in combination with bootstrapping"
        global_robust_statistics = algo_hessian[0].server_state[
            "global_robust_statistics"
        ]
        global_robust_statistics["global_moments"] = global_moments_list[0]

    return (
        hessians,
        lls,
        final_params_list,
        computed_stds_list,
        global_robust_statistics,
    )


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
    actual_number_of_rounds = 2 * (num_rounds + 1) + 2

    if not standardize_data:
        actual_number_of_rounds -= 1
    return actual_number_of_rounds
