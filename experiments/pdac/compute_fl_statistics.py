"""Compute Federated statistics on the data."""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import git
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from substra.sdk.models import ComputePlanStatus
from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment
from substrafl.model_loading import download_aggregate_shared_state
from substrafl.nodes import AggregationNode
from torch import nn

import fedeca
from fedeca.fedeca_core import LogisticRegressionTorch
from fedeca.strategies.fed_kaplan import FedKaplan
from fedeca.strategies.fed_smd import FedSMD
from fedeca.utils.plot_fed_kaplan import fed_km_plot
from fedeca.utils.substra_utils import Client
from fedeca.viz.plot import owkin_palette

with open("tokens.json") as f:
    d = json.load(f)

token = d["token"]
DS_BACKEND_URL = "DS_BACKEND_URL"

owkin_ds = Client(
    url=DS_BACKEND_URL,
    token=token,
    backend_type="remote",
)  # noqa: E501


f = open("results_2024-10-04_12-29-48.pkl", "rb")
res = pickle.load(f)
f.close()


train_data_nodes = res["kwargs"]["train_data_nodes"]
aggregation_node = AggregationNode(res["kwargs"]["aggregation_node"])
propensity_model_weights = res["naïve"]["propensity_model"]
propensity_model = LogisticRegressionTorch(ndim=propensity_model_weights["weight"].size)

propensity_model.fc1.weight.data = nn.parameter.Parameter(
    torch.from_numpy(propensity_model_weights["weight"])
)
propensity_model.fc1.bias.data = nn.parameter.Parameter(
    torch.from_numpy(propensity_model_weights["bias"])
)
# Testing inference
X = torch.ones(([64, 4])).to(torch.float64)

propensity_scores = propensity_model(X)

strategy = FedSMD(
    treated_col=res["kwargs"]["treated_col"],
    duration_col=res["kwargs"]["duration_col"],
    event_col=res["kwargs"]["event_col"],
    propensity_model=propensity_model,
    client_identifier=res["kwargs"]["client_identifier"],
)
fedeca_path = fedeca.__path__[0]
repo_folder = Path(
    git.Repo(fedeca_path, search_parent_directories=True).working_dir
).resolve()
wheel_folder = repo_folder / "temp"
os.makedirs(wheel_folder, exist_ok=True)
for stale_wheel in wheel_folder.glob("fedeca*.whl"):
    stale_wheel.unlink()
process = subprocess.Popen(
    f"python -m build --wheel --outdir {wheel_folder} {repo_folder}",
    shell=True,
    stdout=subprocess.PIPE,
)
process.wait()
assert process.returncode == 0, "Failed to build the wheel"
fedeca_wheel_path = next(wheel_folder.glob("fedeca*.whl"))
# Building base_opener dependency as a wheel because of bug of local_code
# position in the opener docker
base_opener_path = Path(os.path.abspath(sys.argv[0])).parent / "base_opener.py"
assert base_opener_path.exists(), f"Could not find {base_opener_path}"
temp_folder = "./temp"
package_folder = str(Path(temp_folder) / "base_opener")
wheel_folder = str(Path(temp_folder) / "wheels")
os.makedirs(package_folder, exist_ok=True)
os.makedirs(wheel_folder, exist_ok=True)

copy_process = subprocess.Popen(
    f"cp {base_opener_path} ./temp/base_opener/base_opener.py",
    shell=True,
    stdout=subprocess.PIPE,
)
copy_process.wait()
assert copy_process.returncode == 0, "Failed to copy the base_opener"
setup_code = """
from setuptools import setup, find_packages


setup(
    name="base_opener",
    version="1.0",
    packages=find_packages(),
)
    """
with open("./temp/setup.py", "w") as f:
    f.write(setup_code)

init_code = """
from .base_opener import ZPDACOpener, FedECACenters
"""
with open("./temp/base_opener/__init__.py", "w") as f:
    f.write(init_code)

for stale_wheel in Path(wheel_folder).glob("base_opener*.whl"):
    stale_wheel.unlink()

build_process = subprocess.Popen(
    f"python -m build --wheel --outdir {wheel_folder} ./temp",
    shell=True,
    stdout=subprocess.PIPE,
)
build_process.wait()
assert build_process.returncode == 0, "Failed to build the wheel"
wheel_path = next(Path(wheel_folder).glob("base_opener*.whl"))

compute_plan = execute_experiment(
    client=owkin_ds,
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    num_rounds=1,
    experiment_folder=str(Path(".") / "experiment_summaries"),
    dependencies=Dependency(
        local_installable_dependencies=[Path(fedeca_wheel_path), Path(wheel_path)]
    ),
    name="Fed SMD",
)
t1 = time.time()
t2 = time.time()
timeout = 3600
status = owkin_ds.get_compute_plan(compute_plan.key).status
while (
    (status != ComputePlanStatus.done)
    and (t2 - t1) < timeout
    and status != ComputePlanStatus.failed
):  # noqa: E501
    time.sleep(60)
    t2 = time.time()
    status = owkin_ds.get_compute_plan(compute_plan.key).status

if status == ComputePlanStatus.failed:
    raise ValueError("The CP has failed ! Check the frontend.")

fl_results = download_aggregate_shared_state(
    client=owkin_ds,
    compute_plan_key=compute_plan.key,  # '969be4e0-b0f1-41f1-b032-b91cc659b3a9'
    round_idx=0,
)

with open(f"fed_smd_{compute_plan.key}.pkl", "wb") as f:
    pickle.dump(fl_results, f)


fl_results["weighted_smd"] = pd.DataFrame(
    fl_results["weighted_smd"], columns=["smd"]
)  # noqa: E501
fl_results["unweighted_smd"] = pd.DataFrame(
    fl_results["unweighted_smd"], columns=["smd"]
)  # noqa: E501

fl_results["weighted_smd"]["covariate"] = fl_results[
    "weighted_smd"
].index.tolist()  # noqa: E501
fl_results["unweighted_smd"]["covariate"] = fl_results[
    "unweighted_smd"
].index.tolist()  # noqa: E501

fl_results["weighted_smd"]["weighted"] = True
fl_results["unweighted_smd"]["weighted"] = False

fl_results["weighted_smd"].reset_index(drop=True, inplace=True)
fl_results["unweighted_smd"].reset_index(drop=True, inplace=True)

df = pd.concat(
    [fl_results["weighted_smd"], fl_results["unweighted_smd"]],
    axis=0,
    ignore_index=True,
)  # noqa: E501

g = sns.scatterplot(
    data=df, x="smd", y="covariate", hue="weighted", palette=owkin_palette.values()
)
g.set_xlabel("Standardized mean difference")
g.set_ylabel("Covariate")

g.axvline(0.1, color="black", linestyle="--", alpha=0.2)
g.axvline(-0.1, color="black", linestyle="--", alpha=0.2)

plt.savefig("smd_real_data.pdf", bbox_inches="tight", dpi=300)
plt.clf()

strategy = FedKaplan(
    treated_col=res["kwargs"]["treated_col"],
    duration_col=res["kwargs"]["duration_col"],
    event_col=res["kwargs"]["event_col"],
    propensity_model=propensity_model,
    client_identifier=res["kwargs"]["client_identifier"],
)
compute_plan = execute_experiment(
    client=owkin_ds,
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    num_rounds=1,
    experiment_folder=str(Path(".") / "experiment_summaries"),
    dependencies=Dependency(
        local_installable_dependencies=[Path(fedeca_wheel_path), Path(wheel_path)]
    ),
    name="Fed Kaplan",
)
t1 = time.time()
t2 = time.time()
timeout = 3600
status = owkin_ds.get_compute_plan(compute_plan.key).status
while (
    (status != ComputePlanStatus.done)
    and (t2 - t1) < timeout
    and status != ComputePlanStatus.failed
):  # noqa: E501
    time.sleep(60)
    t2 = time.time()
    status = owkin_ds.get_compute_plan(compute_plan.key).status

if status == ComputePlanStatus.failed:
    raise ValueError("The CP has failed ! Check the frontend.")

fl_results = download_aggregate_shared_state(
    client=owkin_ds, compute_plan_key=compute_plan.key, round_idx=0
)
with open(f"fed_kaplan_{compute_plan.key}.pkl", "wb") as f:
    pickle.dump(fl_results, f)

fl_grid_treated, fl_s_treated, fl_var_s_treated, fl_cumsum_treated = fl_results[
    "treated"
]
(
    fl_grid_untreated,
    fl_s_untreated,
    fl_var_s_untreated,
    fl_cumsum_untreated,
) = fl_results["untreated"]
fed_treated_plot = fed_km_plot(
    fl_grid_treated,
    fl_s_treated,
    fl_var_s_treated,
    fl_cumsum_treated,
    label="FOLFIRINOX",  # noqa: E501
)

fed_untreated_plot = fed_km_plot(
    fl_grid_untreated,
    fl_s_untreated,
    fl_var_s_untreated,
    fl_cumsum_untreated,
    label="GEM+NAB",  # noqa: E501
)
plt.ylim(0.0, 1.0)
plt.ylabel("Probability of survival")
plt.xlabel("Survival time (months)")
plt.legend()
plt.savefig("fed_km.pdf", bbox_inches="tight", dpi=300)
