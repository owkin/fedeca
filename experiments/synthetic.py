"""Module for synthetic experiment."""
from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError
from omegaconf.dictconfig import DictConfig

from fedeca.fedeca_core import FedECA
from fedeca.utils.experiment_utils import effective_sample_size, std_mean_differences
from fedeca.utils.survival_utils import CoxData
from fedeca.utils.typing import _SeedType


def single_experiment(
    data_gen: CoxData,
    n_samples: int,
    model_configs: Mapping[str, dict | DictConfig],
    duration_col: str = "time",
    event_col: str = "event",
    treated_col: str = "treatment",
    ps_col: str = "propensity_scores",
    seed: _SeedType = None,
    fit_fedeca: Optional[dict[str, Any]] = None,
    return_propensities: bool = True,
    return_weights: bool = True,
):
    """Perform a single experiment comparing survival models.

    Parameters
    ----------
    data_gen : CoxData
        Data generator instance.
    n_samples : int
        Number of samples to generate.
    model_configs : Mapping[str, dict | DictConfig]
        Dictionary of configs used to initialize survival models.
    duration_col : str, optional
        Column name for event duration, by default "time".
    event_col : str, optional
        Column name for event indicator, by default "event".
    treated_col : str, optional
        Column name for treatment indicator, by default "treatment".
    ps_col : str, optional
        Column name for propensity scores, by default "propensity_scores".
    seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}, optional
        The seed for reproducibility. Defaults to None.
    fit_fedeca: dict[str, Any], optional, by default None
        Dictionary of kwargs for the fit function of
        :class:`fedeca.fedeca_core.FedECA`.
    return_propensities: bool
        return propensity scores in the results dataframe, by default True
    return_weights: bool
        return samples weights in the results dataframe, by default True

    Returns
    -------
    pd.DataFrame
        Results of the experiment of `n` rows where `n` is the number of models.
    """
    if seed is None:
        seed = data_gen.rng
    if fit_fedeca is None:
        fit_fedeca = {}
    rng = np.random.default_rng(seed)

    models = dict(
        (name, hydra.utils.instantiate(model)) for name, model in model_configs.items()
    )
    for model in models.values():
        model.set_random_state(rng)

    # prepare dataframe
    data = data_gen.generate_dataframe(
        n_samples,
        prefix="X_",
        duration_col=duration_col,
        event_col=event_col,
        treated_col=treated_col,
        ps_col=ps_col,
        seed=rng,
    )

    res = []

    non_cov = [treated_col, event_col, duration_col, ps_col]
    covariates = [x for x in data.columns if x not in non_cov]

    mask_treated = data[treated_col].eq(1)
    smd_true_ps = std_mean_differences(
        data[ps_col][mask_treated],
        data[ps_col][~mask_treated],
    )
    df_smd_raw = (
        data[covariates]
        .apply(lambda s: std_mean_differences(s[mask_treated], s[~mask_treated]))
        .to_frame()
        .transpose()
        .add_prefix("smd_raw_")
    )
    ate_true = data_gen.average_treatment_effect_
    percent_ties = data_gen.percent_ties
    models_fit_times: dict[str, Optional[float]] = {
        model_name: None for model_name, _ in models.items()
    }

    for name, model in models.items():
        model.treated_col = treated_col
        model.event_col = event_col
        model.duration_col = duration_col
        targets = None
        if name.lower() == "oracleiptw":
            targets = data[ps_col]
        if isinstance(model, FedECA):
            data_fedeca = copy.deepcopy(data).drop(columns=[ps_col])
            # Note that for now FedECA cannot use the targets argument
            # if you want to use it we need to do another PR
            backend_type = fit_fedeca.get("backend_type", "subprocess")
            if backend_type == "remote":
                fit_fedeca["urls"] = fit_fedeca["urls"][: fit_fedeca["n_clients"]]
                fit_fedeca["tokens"] = [
                    open(
                        f"/home/owkin/tokens/api_key{i + 1}",
                        "r",
                    ).read()
                    for i in range(0, fit_fedeca["n_clients"])
                ]

            model.fit(data_fedeca, targets, **fit_fedeca)
            # For some reasons sometimes parameters are passed directly to the
            # instance model wo being included in fit_fedeca hence the slightly
            # convoluted syntax
            dp_target_epsilon = model.__dict__.pop("dp_target_epsilon", np.nan)
            dp_target_delta = model.__dict__.pop("dp_target_delta", np.nan)

            models_fit_times[name] = model.total_fit_time
        else:
            backend_type = "N/A"
            dp_target_epsilon = np.nan
            dp_target_delta = np.nan
            try:
                t1 = time.time()
                model.fit(data, targets)
                t2 = time.time()
                models_fit_times[name] = t2 - t1
            except ConvergenceError:
                # More likely to happen with small sample size and large covariate shift
                pass

        if model.results_ is not None:
            df_smd_weighted = None
            smd_estim_ps = None
            ess = None
            if model.propensity_scores_ is not None:
                smd_estim_ps = std_mean_differences(
                    model.propensity_scores_[mask_treated],
                    model.propensity_scores_[~mask_treated],
                )

            if model.weights_ is not None:
                ess = effective_sample_size(model.weights_[mask_treated])
                df_smd_weighted = (
                    data[covariates]
                    .multiply(model.weights_, axis=0)
                    .apply(
                        lambda s: std_mean_differences(
                            s[mask_treated], s[~mask_treated]
                        )
                    )
                    .to_frame()
                    .transpose()
                    .add_prefix("smd_weighted_")
                )

            log_likelihood = model.log_likelihood_

            df_res_single = model.results_.assign(
                method=name,
                variance_method=getattr(model, "variance_method", None),
                ess=ess,
                smd_estim_ps=smd_estim_ps,
                smd_true_ps=smd_true_ps,
                ate_true=ate_true,
                log_likelihood=log_likelihood,
                fit_time=models_fit_times[name],
                backend_type=backend_type,
                dp_target_epsilon=dp_target_epsilon,
                dp_target_delta=dp_target_delta,
                percent_ties=percent_ties,
            ).reset_index(drop=True)

            if return_propensities:
                df_res_single["propensity_scores"] = [model.propensity_scores_]
            if return_weights:
                df_res_single["weights"] = [model.weights_]
            if df_smd_weighted is not None:
                df_res_single = df_res_single.join(df_smd_weighted)

            res.append(df_res_single)

    if "n_clients" in fit_fedeca:
        n_clients = fit_fedeca["n_clients"]
    else:
        n_clients = None

    df_res = (
        pd.concat(res)
        .join(df_smd_raw)
        .reset_index(drop=True)
        .assign(
            n_samples=n_samples,
            n_events=int(data["event"].sum()),
            ndim=data_gen.ndim,
            features_type=data_gen.features_type,
            overlap=data_gen.overlap,
            cov_corr=data_gen.cov_corr,
            prop_treated=data_gen.prop_treated,
            scale_t=data_gen.scale_t,
            shape_t=data_gen.shape_t,
            censoring_factor=data_gen.censoring_factor,
            percent_ties=data_gen.percent_ties,
            random_censoring=data_gen.random_censoring,
            standardize_features=data_gen.standardize_features,
            n_clients=n_clients,
        )
    )

    return df_res
