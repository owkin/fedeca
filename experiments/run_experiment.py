"""Main module for running hydra config based experiments."""
import pickle
import re
from collections.abc import Mapping

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from experiments.synthetic import single_experiment
from fedeca.utils.survival_utils import BaseSurvivalEstimator, CoxData


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def run_experiment(cfg: DictConfig):
    """Run experiment with hydra configs."""
    with open_dict(cfg):
        models_common = cfg.pop("models_common")
        for model in cfg.models.values():
            model.update(models_common)
    config_hydra = HydraConfig.get()
    job_num = config_hydra.job.num
    if (job_range := cfg.get("job_range", None)) is not None and (
        job_num < job_range[0] or job_num >= job_range[1]
    ):
        return
    if (job_list := cfg.get("job_list", None)) is not None and job_num not in job_list:
        return
    # Set seed for each job in a deterministic manner using the initial_seed
    seed_seq = np.random.SeedSequence(cfg.initial_seed).spawn(job_num + 1)[-1]
    cfg.data.seed = int(seed_seq.generate_state(1)[0])

    output = re.sub(r"\s", "_", cfg.name)
    if (batch_id := cfg.get("batch_id", None)) is not None:
        output = output + f"_batch_{batch_id}"
    output = f"{config_hydra.sweep.dir}/results_{output}.pkl"

    data_gen: CoxData = hydra.utils.instantiate(cfg.data)
    models: Mapping[str, BaseSurvivalEstimator] = dict(
        (name, hydra.utils.instantiate(model)) for name, model in cfg.models.items()
    )
    for model in models.values():
        model.set_random_state(data_gen.rng)
    if "fit_fedeca" in cfg.keys():
        fedeca_config = hydra.utils.instantiate(cfg.fit_fedeca)
    else:
        fedeca_config = None

    results = [
        single_experiment(
            data_gen,
            n_samples=cfg.parameters.n_samples,
            models=models,
            treated_col=models_common.treated_col,
            event_col=models_common.event_col,
            duration_col=models_common.duration_col,
            ps_col=models_common.ps_col,
            fit_fedeca=fedeca_config,
            return_propensities=cfg.parameters.return_propensities,
            return_weights=cfg.parameters.return_weights,
        ).assign(
            rep_id=rep_id,
            cate=cfg.data.cate,
            propensity=cfg.data.propensity,
        )
        for rep_id in range(cfg.parameters.n_reps)
    ]
    with open(output, "ab") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    run_experiment()  # pylint:disable=no-value-for-parameter
