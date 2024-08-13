"""Utility functions of data generation."""
import copy
import math
import os
import random
import zlib
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from substra.sdk.schemas import DataSampleSpec, DatasetSpec, Permissions
from substrafl.nodes import TestDataNode, TrainDataNode

import fedeca
from fedeca.utils.substra_utils import Client
from fedeca.utils.survival_utils import generate_survival_data


def generate_cox_data_and_substra_clients(
    n_clients: int = 2,
    ndim: int = 10,
    split_method_kwargs: Union[dict, None] = None,
    backend_type: str = "subprocess",
    data_path: Union[str, None] = None,
    urls: Union[list, None] = None,
    tokens: Union[list, None] = None,
    seed: int = 42,
    n_per_client: int = 200,
    add_treated: bool = False,
    ncategorical: int = 0,
):
    """Generate Cox data on disk for several clients.

    Generate Cox data and register them with different
    fake clients.

    Parameters
    ----------
    n_clients : int, (optional)
        Number of clients. Defaults to 2.
    ndim : int, (optional)
        Number of covariates. Defaults to 10.
    split_method_kwargs = Union[dict, None]
        The argument to the split_method uniform.
    backend_type : str, (optional)
        Type of backend. Defaults to "subprocess".
    data_path : str, (optional)
       Path to save the data. Defaults to None.
    seed : int, (optional)
        Random seed. Defaults to 42.
    n_per_client : int, (optional)
        Number of samples per client. Defaults to 200.
    add_treated : bool, (optional)
        Whether or not to keep treated column.
    ncategorical: int, (optional)
        Number of features to make categorical a posteriori (moving away from Cox
        assumptions).
    """
    assert backend_type in ["remote", "docker", "subprocess"]
    assert n_clients >= 2
    if split_method_kwargs is not None:
        split_method_kwargs = copy.deepcopy(split_method_kwargs)
        if "seed" in split_method_kwargs:
            raise ValueError("You provided splitting seed twice")
    else:
        split_method_kwargs = {}
    split_method_kwargs["seed"] = seed
    df, cox_model_coeffs = generate_survival_data(
        na_proportion=0.0,
        ncategorical=ncategorical,
        ndim=ndim,
        seed=seed,
        n_samples=n_per_client * n_clients,
        use_cate=False,
        censoring_factor=0.3,
    )

    if not (add_treated):
        df.drop("treated", axis=1, inplace=True)

    return (
        *split_dataframe_across_clients(
            df=df,
            n_clients=n_clients,
            split_method="uniform",
            split_method_kwargs=split_method_kwargs,
            backend_type=backend_type,
            data_path=data_path,
            urls=urls,
            tokens=tokens,
        ),
        cox_model_coeffs,
    )


def split_dataframe_across_clients(
    df,
    n_clients,
    split_method: Union[Callable, str] = "uniform",
    split_method_kwargs: Union[dict, None] = None,
    backend_type="subprocess",
    data_path: Union[str, None] = None,
    urls=[],
    tokens=[],
):
    """Split patients over the centers.

    Parameters
    ----------
    df : pandas.DataFrame,
        Dataframe containing features of the patients.
    n_clients : int,
        Number of clients.
    split_method : Union[Callable, str]
        How to split the dataset across all clients, if callable should have the
        signature: df, n_clients, kwargs -> list[list[int]]
        if str should be an existing key, which will invoke the corresponding
        callable. Possible values are `uniform` which splits the patients
        uniformly across centers or `split_control_over_centers` where one
        center has all the treated patients and the control is split over the
        remaining ones.
    split_method_kwargs: Union[dict, None]
        Optional kwargs for the split_method method.
    backend_type : str, (optional)
        Backend type. Default is "subprocess".
    data_path : Union[str, None],
        Path on where to save the data on disk.
    urls : List,
        List of urls.
    tokens : List,
        List of tokens.
    """
    # Deterministic hashing of non human-readable objects: df and
    # split_method_kwargs
    # Adding float32 conversion to avoid hash differences due to float64 rounding
    # differences on different machines
    to_hash = hash_pandas_object(
        df.select_dtypes(include="number").astype("float32"), index=True
    ).values.tolist() + [split_method_kwargs]
    to_hash = str.encode("".join([str(e) for e in to_hash]))
    hash_df = zlib.adler32(to_hash)
    clients = []
    if backend_type == "remote":
        assert (
            len(urls) == n_clients
        ), f"You should provide a list of {n_clients} URLs for the different clients"
        assert (
            len(tokens) == n_clients
        ), "You should provide a token for each client in remote mode"
        for i in range(n_clients):
            clients.append(Client(url=urls[i], token=tokens[i], backend_type="remote"))
    else:
        for i in range(n_clients):
            clients.append(Client(backend_type=backend_type))

    clients = {c.organization_info().organization_id: c for c in clients}
    # Store organization IDs
    ORGS_ID = list(clients.keys())

    ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
    DATA_PROVIDER_ORGS_ID = ORGS_ID

    if data_path is None:
        (Path.cwd() / "tmp").mkdir(exist_ok=True)
        data_path = Path.cwd() / "tmp" / "data_eca"
    else:
        data_path = Path(data_path)
        # All paths need to be absolute paths
        data_path = data_path.resolve()

    (data_path).mkdir(exist_ok=True)
    if isinstance(split_method, str):
        assert split_method in [
            "uniform",
            "split_control_over_centers",
        ], f"split_method name {split_method} not recognized"
        if split_method == "uniform":
            split_method = uniform_split
        else:
            split_method = split_control_over_centers

    if split_method_kwargs is None:
        split_method_kwargs = {}
    # Now split_method is a Callable
    clients_indices_list = split_method(df, n_clients, **split_method_kwargs)
    all_indices = set(chain.from_iterable(clients_indices_list))
    # Check that split methods is valid (no drop_last) could be removed for
    # more flexibility
    assert len(all_indices) == len(df.index)
    assert set(all_indices) == set(range(len(df.index)))
    dfs = []
    for i in range(n_clients):
        os.makedirs(data_path / f"center{i}", exist_ok=True)
        cdf = copy.deepcopy(df.iloc[clients_indices_list[i]])
        cdf["center"] = f"center{i}"
        df_path = data_path / f"center{i}" / "data.csv"
        if df_path.exists():
            df_path.unlink()
        cdf.to_csv(df_path, index=False)
        dfs.append(cdf)

    assets_directory = Path(fedeca.__file__).parent / "scripts" / "substra_assets"

    # Sample registration: will fill two dicts with relevant handles to retrieve
    # data
    dataset_keys = {}
    datasample_keys = {}

    dataset_name = (
        f"ECA-{hash_df}-nclients{n_clients}-split-method{split_method.__name__}"
    )

    for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
        client = clients[org_id]
        found_eca_datasets = [
            dataset
            for dataset in client.list_dataset(filters={"owner": [org_id]})
            if dataset.name == dataset_name
        ]
        if len(found_eca_datasets) == 0:
            permissions_dataset = Permissions(
                public=False, authorized_ids=[ALGO_ORG_ID]
            )
            # DatasetSpec is the specification of a dataset. It makes sure every field
            # is well defined, and that our dataset is ready to be registered.
            # The real dataset object is created in the add_dataset method.
            dataset = DatasetSpec(
                name=dataset_name,
                data_opener=assets_directory / "csv_opener.py",
                description=assets_directory / "description.md",
                permissions=permissions_dataset,
                logs_permission=permissions_dataset,
            )
            dataset_keys[org_id] = client.add_dataset(dataset)
            assert dataset_keys[org_id], "Missing dataset key"

            # Add the training data on each organization.
            data_sample = DataSampleSpec(
                data_manager_keys=[dataset_keys[org_id]],
                path=data_path / f"center{i}",
            )
            datasample_keys[org_id] = client.add_data_sample(data_sample)
        # Maybe samples already exist in the platform
        else:
            dataset_keys[org_id] = found_eca_datasets[0].key
            datasample_keys[org_id] = found_eca_datasets[0].data_sample_keys[0]

    # Actual creation of objects of interest
    train_data_nodes = []
    test_data_nodes = []
    for org_id in DATA_PROVIDER_ORGS_ID:
        # Create the Train Data Node (or training task) and save it in a list
        train_data_node = TrainDataNode(
            organization_id=org_id,
            data_manager_key=dataset_keys[org_id],
            data_sample_keys=[datasample_keys[org_id]],
        )

        train_data_nodes.append(train_data_node)

        # Create the Train Data Node (or training task) and save it in a list
        test_data_node = TestDataNode(
            organization_id=org_id,
            data_manager_key=dataset_keys[org_id],
            data_sample_keys=[datasample_keys[org_id]],
        )

        test_data_nodes.append(test_data_node)

    return clients, train_data_nodes, test_data_nodes, dfs, df


def uniform_split(
    df: pd.DataFrame, n_clients: int, use_random: bool = True, seed: int = 42
):
    """Split patients uniformly over n_clients.

    Parameters
    ----------
    df : pandas.DataFrame,
        Dataframe containing features of the patients.
    n_clients : int,
        Number of clients.
    use_random : bool
        Whether or not to shuffle data before splitting. Defaults to True.
    seed : int, (optional)
        Seeding for shuffling
    """
    # We don't want to alter df
    df_func = copy.deepcopy(df)
    df_func_size = len(df_func.index)
    indices = list(range(df_func_size))
    if use_random:
        random.seed(seed)
        random.shuffle(indices)
    indices_list = []
    n_samples_per_client = math.ceil(df_func_size / n_clients)

    for i in range(n_clients):
        start = i * n_samples_per_client
        stop = min((i + 1) * n_samples_per_client, len(indices))
        indices_center = indices[start:stop]
        indices_list.append(indices_center)
    return indices_list


def split_control_over_centers(
    df,
    n_clients,
    treatment_info="treatment_allocation",
    use_random: bool = True,
    seed: int = 42,
):
    """Split patients in the control group over the centers.

    Parameters
    ----------
    df : pandas.DataFrame,
        Dataframe containing features of the patients.
    n_clients : int,
        Number of clients.
    treatment_info : str, (optional)
        Column name for the treatment allocation covariate.
        Defaults to "treatment_allocation".
    use_random : bool
        Whether or not to shuffle the control group indices before splitting.
    seed: int
        The seed of the shuffling.
    """
    # We don't want to alter df
    df_func = copy.deepcopy(df)
    # Making sure there is no funny business with indices
    df_func.reset_index(inplace=True)

    df_func_size = len(df_func.index)

    # Computing number of samples in each center with control data
    n_control_center = math.ceil((df_func[treatment_info] == 0).sum() / (n_clients - 1))

    indices_treated_df = np.flatnonzero(df[treatment_info] == 1).tolist()
    indices_list = [indices_treated_df]
    indices_control_df = [
        idx for idx in range(df_func_size) if idx not in indices_treated_df
    ]
    if use_random:
        random.seed(seed)
        random.shuffle(indices_control_df)

    for i in range(1, n_clients):
        start = (i - 1) * n_control_center
        stop = min(i * n_control_center, len(indices_control_df))
        indices_center = indices_control_df[start:stop]
        indices_list.append(indices_center)
    return indices_list
