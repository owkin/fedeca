"""Federated IPTW script."""
import torch

from fedeca import FedECA
from fedeca.utils.data_utils import generate_survival_data

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    N_CLIENTS = 2
    NDIM = 10
    URLS = []
    TOKENS = []
    # Choose BACKEND_TYPE between subprocess, remote and docker

    BACKEND_TYPE = "subprocess"
    if BACKEND_TYPE == "remote":
        # If you use BACKEND_TYPE="remote", download your API key with SSO login then
        #  copy-paste it in a file called api_key inside the tokens folder otherwise
        # comment the following two lines
        URLS = [f"https://api.org-{i + 1}.demo.cg.owkin.tech" for i in range(N_CLIENTS)]
        TOKENS = [open(f"tokens/api_key{i + 1}", "r").read() for i in range(N_CLIENTS)]

    df, cox_model_coeffs = generate_survival_data(
        na_proportion=0.0,
        ncategorical=0,
        ndim=NDIM,
        seed=seed,
        n_samples=1000,
        use_cate=False,
        censoring_factor=0.3,
    )

    # We can choose not to give any clients or data of any kind to FedECA
    # they will be given to it by the fit method
    IPTW = FedECA(
        ndim=NDIM,
        treated_col="treated",
        duration_col="T",
        event_col="E",
        num_rounds_list=[2, 4],
        dp_target_epsilon=10.0,
        dp_max_grad_norm=1.1,
        dp_target_delta=0.1,
        dp_propensity_model_training_params={"batch_size": 50, "num_updates": 200},
        dp_propensity_model_optimizer_kwargs={"lr": 0.001},
    )
    IPTW.fit(
        df,
        None,
        N_CLIENTS,
        split_method="split_control_over_centers",
        split_method_kwargs={"treatment_info": "treated"},
        backend_type=BACKEND_TYPE,
        urls=URLS,
        tokens=TOKENS,
        robust=True,
    )
