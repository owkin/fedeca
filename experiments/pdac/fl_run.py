"""Run the federated ECA on the pancreatic cancer dataset."""
import copy
import json
import os
import pickle
import argparse
# assumes datasets' names contain the number of samples ..._NXXXX_... in their names
# and the original colum names ..._colXX;YY;ZZ;..._NXXXX where XX, YY, ZZ are
# the covariates
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import git
from substrafl.nodes import AggregationNode, TrainDataNode
from itertools import combinations

import fedeca
from fedeca import FedECA
from fedeca.utils.substra_utils import Client

NO_PANCAN = False
VERSION = 5
IDIBIGI_org_id = "IDIBIGi"
FFCD_org_id = "FFCDMSP"
PANCAN_org_id = "PanCan"

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
IDIBIGI_BACKEND_URL = "IDIBIGI_BACKEND_URL"

with open("tokens.json") as f:
    d = json.load(f)

token = d["token"]

owkin_ds = Client(
    url=DS_BACKEND_URL,
    token=token,
    backend_type="remote",
)  # noqa: E501
print(owkin_ds.list_dataset())

# We need FFCD or IDIBIGi or PANCAN tokens to be able to download models see above
with open("ffcd_tokens.json") as f:
    ffcd_d = json.load(f)

with open("idibigi_tokens.json") as f:
    idibigi_d = json.load(f)

ffcd_token = ffcd_d["token"]
ffcd_client = Client(
    url=FFCD_BACKEND_URL,
    token=ffcd_token,
    backend_type="remote",
)  # noqa: E501

idibigi_token = idibigi_d["token"]
idibigi_client = Client(
    url=IDIBIGI_BACKEND_URL,
    token=idibigi_token,
    backend_type="remote",
)  # noqa: E501
# check connexion
#datasets = owkin_ds.list_dataset()
#breakpoint()
datasets = owkin_ds.list_dataset()


if __name__ == "__main__":
    # All datasets have to be registered prior to launching this, this only
    # selects a dataset from registered datasets with certain properties based
    # on its name
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', "-SYNTH", action='store_true', default=False, help="Whether or not to use synthetic data")
    parser.add_argument('--fake-treatment', "-FT", action='store_true', default=False, help="Whether or not to use a fake treatment dataset.")
    parser.add_argument('--is-folfirinox', "-FOLFI", action='store_true', default=False)
    parser.add_argument("--treatment", "-T", type=str, default="idibigi")
    args = parser.parse_args()

    # Note that by default it will run all pairs of centers and the 3 centers for all variance methods

    SYNTHETIC = args.synthetic
    FAKE_TREATMENT = args.fake_treatment
    IS_FOLFIRINOX = args.is_folfirinox
    TREATMENT = args.treatment


    # Here we have a whole logic that obtains hashes from the system based on the dataset name

    idibigi_datasets = [ds for ds in datasets if "idibigi" in ds.name and ds.name.endswith(f"_v{VERSION}")]
    ffcd_datasets = [ds for ds in datasets if "ffcd" in ds.name and ds.name.endswith(f"_v{VERSION}")]
    pancan_datasets = [ds for ds in datasets if "pancan" in ds.name and ds.name.endswith(f"_v{VERSION}")]



    if SYNTHETIC:
        idibigi_datasets = [ds for ds in idibigi_datasets if ds.name.startswith("SYNTH_")]
        ffcd_datasets = [ds for ds in ffcd_datasets if ds.name.startswith("SYNTH_")]
        pancan_datasets = [ds for ds in pancan_datasets if ds.name.startswith("SYNTH_")]

    else:
        if FAKE_TREATMENT:
            # filtering out non fake treatment datasets
            idibigi_datasets = [ds for ds in idibigi_datasets if re.fullmatch(f"T[IFP]F[TF]idibigi_.*", ds.name)]
            ffcd_datasets = [ds for ds in ffcd_datasets if re.fullmatch(f"T[IFP]F[TF]ffcd_.*", ds.name)]
            pancan_datasets = [ds for ds in pancan_datasets if re.fullmatch(f"T[IFP]F[TF]pancan_.*", ds.name)]
            # filtering out the correct treatment
            idibigi_datasets = [ds for ds in idibigi_datasets if ds.name.startswith(f"T{TREATMENT[0].upper()}")]
            ffcd_datasets = [ds for ds in ffcd_datasets if ds.name.startswith(f"T{TREATMENT[0].upper()}")]
            pancan_datasets = [ds for ds in pancan_datasets if ds.name.startswith(f"T{TREATMENT[0].upper()}")]
            # filtering out the correct group
            idibigi_datasets = [ds for ds in idibigi_datasets if ds.name[2:4] == f"F{str(IS_FOLFIRINOX)[0]}"]
            ffcd_datasets = [ds for ds in ffcd_datasets if ds.name[2:4] == f"F{str(IS_FOLFIRINOX)[0]}"]
            pancan_datasets = [ds for ds in pancan_datasets if ds.name[2:4] == f"F{str(IS_FOLFIRINOX)[0]}"]
        else:
            # filtering out fake treatment datasets
            idibigi_datasets = [ds for ds in idibigi_datasets if re.fullmatch(f"idibigi_.*", ds.name)]
            ffcd_datasets = [ds for ds in ffcd_datasets if re.fullmatch(f"ffcd_.*", ds.name)]
            pancan_datasets = [ds for ds in pancan_datasets if re.fullmatch(f"pancan_.*", ds.name)]



    assert len(idibigi_datasets) == 1
    assert len(ffcd_datasets) == 1
    assert len(pancan_datasets) == 1

    IDIBIGI_dataset_key = idibigi_datasets[0].key
    IDIBIGI_datasamples_keys = idibigi_datasets[0].data_sample_keys
    FFCD_dataset_key = ffcd_datasets[0].key
    FFCD_datasamples_keys = ffcd_datasets[0].data_sample_keys
    PANCAN_dataset_key = pancan_datasets[0].key
    PANCAN_datasamples_keys = pancan_datasets[0].data_sample_keys

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

    NCLIENTS = 3 if not NO_PANCAN else 2

    clients_sizes = [
        int(re.findall(r"(?<=_N)[0-9]+(?=_)", owkin_ds.get_dataset(dataset_key).name)[0])
        for dataset_key in [IDIBIGI_dataset_key, FFCD_dataset_key, PANCAN_dataset_key][
            :NCLIENTS
        ]
    ]  # noqa: E501
    cols = [
        re.findall(r"(?<=[0-9]_).+(?=_N[0-9]+)", owkin_ds.get_dataset(dataset_key).name)[0]
        for dataset_key in [IDIBIGI_dataset_key, FFCD_dataset_key, PANCAN_dataset_key][
            :NCLIENTS
        ]
    ]  # noqa: E501
    # v0 had inexact ordering of column names in dataset name
    assert all(
        [
            ";".join(sorted(cols[i].split(";"))) == ";".join(sorted(cols[0].split(";")))
            for i in range(1, len(cols))
        ]
    )
    col_names = cols[0].split(";")
    ndim = len(col_names) - 3
    print(f"Found {ndim} covariates in the original colnames {col_names}")

    kwargs = {
        "ndim": ndim,
        "treated_col": "treatment",
        "event_col": "event",
        "duration_col": "Overall survival",  # we can change it back to time if needed
        "bootstrap_function": "global",
        "bootstrap_seeds": 42,
        "client_identifier": "center",
        "num_rounds_list": [20, 20],
        "training_strategy": "iptw",
        "n_bootstrap": 200,
        "ds_client": owkin_ds,
        "aggregation_node": aggregation_node,
        "l2_coeff_nr": 1e-16,
    }


    fedeca_path = fedeca.__path__[0]
    commit_fedeca = git.Repo(fedeca_path, search_parent_directories=True).head.object.hexsha
    run_script = open(__file__).read()
    picklizable_kwargs = copy.deepcopy(kwargs)
    # aggregation_node is not picklizable
    picklizable_kwargs["aggregation_node"] = picklizable_kwargs[
        "aggregation_node"
    ].organization_id


    # We iterate on the full loop as well as all possible combinations of 2 clients
    clients = ["idibigi", "ffcd", "pancan"]
    clients_pairs = list(combinations(clients, 2))
    # Doing it on all pairs and the 3 centers
    clients_combination = clients_pairs + [tuple(clients)]
    idibigi_info = {"node": idibigi_node, "org_id": IDIBIGI_org_id, "size": clients_sizes[0]}
    ffcd_info = {"node": ffcd_node, "org_id": FFCD_org_id, "size": clients_sizes[1]}
    pancan_info = {"node": pancan_node, "org_id": PANCAN_org_id, "size": clients_sizes[2]}
    clients_info = {"idibigi": idibigi_info, "ffcd": ffcd_info, "pancan": pancan_info}
    global_res = {current_clients: None for current_clients in clients_combination}
    template_res = {
        "kwargs": picklizable_kwargs,
        "commit_fedeca": commit_fedeca,
        "run_script": run_script,
        }
    for current_clients in clients_combination:
        current_kwargs_clients = copy.deepcopy(kwargs)
        # This arguments were not stored before
        current_kwargs_clients["train_data_nodes"] = [clients_info[c_client]["node"] for c_client in current_clients]
        current_kwargs_clients["clients_names"] = [clients_info[c_client]["org_id"] for c_client in current_clients]
        current_kwargs_clients["clients_sizes"] = [clients_info[c_client]["size"] for c_client in current_clients]
        current_kwargs_clients["partner_client"] = ffcd_client if "ffcd" in current_clients else idibigi_client
        print(current_kwargs_clients)
        res = copy.deepcopy(template_res)
        # for debugging purposes and also for further use when computing weighted KM
        res["kwargs"]["train_data_nodes"] = current_kwargs_clients["train_data_nodes"]
        res["kwargs"]["clients_names"] = current_kwargs_clients["clients_names"]
        res["kwargs"]["clients_sizes"] = current_kwargs_clients["clients_sizes"]
        res["kwargs"]["partner_client"] = current_kwargs_clients["partner_client"]

        for idx, variance_method in enumerate(["bootstrap", "robust", "naïve"]):

            current_kwargs = copy.deepcopy(current_kwargs_clients)
            current_kwargs["variance_method"] = variance_method

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
            setup_code = """from setuptools import setup, find_packages

setup(
    name="base_opener",
    version="1.0",
    packages=find_packages(),
)
            """
            with open("./temp/setup.py", "w") as f:
                f.write(setup_code)

            init_code = """from .base_opener import ZPDACOpener, FedECACenters"""
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

            current_kwargs["dependencies"] = [Path(wheel_path)]

            # That's it !

            fedeca_cp = FedECA(**current_kwargs)
            fedeca_cp.run()

            # Kaplan-Meier
            print("==========================================================")
            print(variance_method)
            print("==========================================================")
            print(fedeca_cp.results_)
            res[variance_method] = {}
            res[variance_method]["results"] = copy.deepcopy(fedeca_cp.results_)
            res[variance_method]["ll"] = fedeca_cp.log_likelihood_
            res[variance_method]["compute_plan_keys"] = fedeca_cp.compute_plan_keys
            res[variance_method]["computed_stds_list"] = fedeca_cp.computed_stds_list
            if hasattr(fedeca_cp, "propensity_models"):
                res[variance_method]["propensity_models"] = [
                    {
                        "weight": model.fc1.weight.detach().cpu().data.numpy(),
                        "bias": model.fc1.bias.detach().cpu().data.numpy(),
                    }
                    for model in fedeca_cp.propensity_models
                ]  # noqa: E501
            else:
                model = fedeca_cp.propensity_model
                res[variance_method]["propensity_model"] = {
                    "weight": model.fc1.weight.detach().cpu().data.numpy(),
                    "bias": model.fc1.bias.detach().cpu().data.numpy(),
                }  # noqa: E501
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_file_path = f"{current_clients}_results_{timestamp}"
        if SYNTHETIC:
            current_file_path += "_synth"
        if FAKE_TREATMENT:
            current_file_path += f"_is_folfirinox{IS_FOLFIRINOX}_treatment{TREATMENT}"
        # We dump each result when we have it
        with open(current_file_path + ".pkl", "wb") as f:
            pickle.dump(res, f)

        global_res[current_clients] = res


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    global_file_path = f"global_results_{timestamp}"
    if SYNTHETIC:
        global_file_path += "_synth"
    if FAKE_TREATMENT:
        global_file_path += f"_is_folfirinox{IS_FOLFIRINOX}_treatment{TREATMENT}"
    with open(global_file_path + ".pkl", "wb") as f:
        pickle.dump(global_res, f)
