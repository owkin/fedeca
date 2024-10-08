"""Launch test fl run with synthetic data."""
import copy
import itertools
import json

import numpy as np
from substrafl.nodes import AggregationNode, TrainDataNode

from fedeca import FedECA
from fedeca.competitors import PooledIPTW
from fedeca.utils.data_utils import (
    split_control_over_centers,
    split_dataframe_across_clients,
)
from fedeca.utils.substra_utils import Client
from fedeca.utils.survival_utils import CoxData

# for the script to work you need to SSO to the frontend
# DS_FRONTEND_URL
# then you need to generate an API token (expiration=never) and copy it
# and uncomment and run the following line after copying the token
# dict1 ={
#    "token": "copypastetokengeneratedfromthefrontend
# DS_FRONTEND_URL",
# }
# out_file = open("tokens.json", "w")
# json.dump(dict1, out_file, indent = 6)
# out_file.close()
# we also need ffcd token (FFCD_FRONTEND_URL)
# to be able to download models.

DS_BACKEND_URL = "DS_BACKEND_URL"
FFCD_BACKEND_URL = "FFCD_BACKEND_URL"
with open("tokens.json") as f:
    d = json.load(f)

token = d["token"]

owkin_ds = Client(
    url=DS_BACKEND_URL,
    token=token,
    backend_type="remote",
)
print(owkin_ds.list_dataset())

# We need FFCD or IDIBIGi tokens to be able to download models
with open("ffcd_tokens.json") as f:
    ffcd_d = json.load(f)

ffcd_token = ffcd_d["token"]
ffcd_client = Client(
    url=FFCD_BACKEND_URL,
    token=ffcd_token,
    backend_type="remote",
)
# check connexion
ffcd_client.list_dataset()

IDIBIGI_dataset_key = "abaca912-1cde-4f0e-90f3-a653519f0b90"
IDIBIGI_datasamples_keys = ["c0cdd6bb-2328-4cfb-9221-ba269d495893"]
IDIBIGI_org_id = "IDIBIGi"


FFCD_dataset_key = "734a1396-7c35-43f5-ae52-847db4fd7aaf"
FFCD_datasamples_keys = ["448f2e0e-3d64-4b48-a643-0ab48d0fbf97"]
FFCD_org_id = "FFCDMSP"


PANCAN_dataset_key = "de43d4c5-f1bb-4be5-b01f-be0eab9f6592"
PANCAN_datasamples_keys = ["b77277dc-75fc-415c-a3f9-2122fe91fbf4"]
PANCAN_org_id = "PanCan"


aggregation_node = AggregationNode("OwkinMSP")

# Create the Train Data Nodes (or training tasks) and save them in a list
idibigi_node = TrainDataNode(
    organization_id=IDIBIGI_org_id,
    data_manager_key=IDIBIGI_dataset_key,
    data_sample_keys=IDIBIGI_datasamples_keys,
)

ffcd_node = TrainDataNode(
    organization_id=FFCD_org_id,
    data_manager_key=FFCD_dataset_key,
    data_sample_keys=FFCD_datasamples_keys,
)

pancan_node = TrainDataNode(
    organization_id=PANCAN_org_id,
    data_manager_key=PANCAN_dataset_key,
    data_sample_keys=PANCAN_datasamples_keys,
)


# Let's generate 1000 data samples with 10 covariates
data = CoxData(seed=42, n_samples=1000, ndim=10)
df = data.generate_dataframe()

# We remove the true propensity score
df = df.drop(columns=["propensity_scores"], axis=1)

# We split it across 2 clients
simu_clients, simu_train_data_nodes, _, dfs, _ = split_dataframe_across_clients(
    df,
    n_clients=3,
    split_method="split_control_over_centers",
    split_method_kwargs={"treatment_info": "treatment"},
    data_path="./data",
    backend_type="simu",
)


indices_list = split_control_over_centers(
    df, n_clients=3, treatment_info="treatment", seed=42
)

global_clients_indices = list(itertools.chain(*indices_list))
N_BTST = 200
SEED = 42
kwargs = {
    "ndim": 10,
    "treated_col": "treatment",
    "event_col": "event",
    "duration_col": "time",
    "bootstrap_function": "global",
    "bootstrap_seeds": SEED,
    "clients_sizes": [500, 250, 250],
    "client_identifier": "center",
    "num_rounds_list": [20, 20],
    "training_strategy": "iptw",
    "indices_in_global_dataset": global_clients_indices,
    "n_bootstrap": N_BTST,
}  # "aiptw"}

res = {}
for variance_method in ["naïve", "robust", "bootstrap"]:
    results = []
    pooled_iptw = PooledIPTW(
        treated_col="treatment",
        event_col="event",
        duration_col="time",
        variance_method=variance_method,
        seed=SEED,
        n_bootstrap=N_BTST,
    )
    pooled_iptw.fit(data=df, targets=None)
    results.append(copy.deepcopy(pooled_iptw.results_))
    for backend_type in ["simu", "remote"]:
        current_kwargs = copy.deepcopy(kwargs)
        current_kwargs["variance_method"] = variance_method
        if backend_type == "remote":
            current_kwargs["ds_client"] = owkin_ds
            current_kwargs["train_data_nodes"] = [idibigi_node, ffcd_node, pancan_node]
            current_kwargs["partner_client"] = ffcd_client
            current_kwargs["clients_names"] = [
                "center0",
                "center1",
                "center2",
            ]  # [IDIBIGI_org_id, FFCD_org_id, PANCAN_org_id]
            current_kwargs["aggregation_node"] = aggregation_node
            # current_kwargs["dependencies"] = [Path(os.path.abspath(sys.argv[0])).parent.parent / "ffcd_loop" / "base_opener.py"]  # noqa: E501

        else:
            # We take the first client, no one cares
            # current_kwargs["ds_client"] = simu_clients[list(simu_clients.keys())[0]]
            # current_kwargs["train_data_nodes"] = simu_train_data_nodes
            current_kwargs["clients_names"] = ["center0", "center1", "center2"]

        fedeca_cp = FedECA(**current_kwargs)
        if backend_type != "remote":
            fedeca_cp.fit(
                data=df,
                targets=None,
                n_clients=3,
                split_method="split_control_over_centers",
                split_method_kwargs={"treatment_info": "treatment", "seed": 42},
                backend_type=backend_type,
                data_path="./data",
            )
        else:
            fedeca_cp.run()
        results.append(copy.deepcopy(fedeca_cp.results_))

    try:
        if not np.allclose(
            np.asarray(results[0][results[1].columns]),
            np.asarray(results[1]),
            rtol=1e-2,
        ):
            raise ValueError("Results are not the same between pooled and subprocess")
        if not np.allclose(
            np.asarray(results[1][results[2].columns]),
            np.asarray(results[2]),
            rtol=1e-2,
        ):
            raise ValueError("Results are not the same between subprocess and remote")
    except ValueError:
        breakpoint()
