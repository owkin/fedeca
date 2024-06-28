"""Bootstrap data globally."""
from functools import partial
from typing import Union

import numpy as np
import pandas as pd

from fedeca.utils.survival_utils import BootstrapMixin


def make_global_bootstrap_function(
    clients_sizes: list,
    n_bootstrap=200,
    bootstrap_seeds: Union[None, list] = None,
    client_identifier: Union[None, str] = None,
    clients_names: Union[None, list] = None,
):
    """Create a function that will return a bootstrap sample for a given seed.

    Parameters
    ----------
    clients_sizes : list
        List of integers representing the sizes of the clients
    n_bootstrap : int, optional
        Number of bootstrap samples to generate, by default 200
    bootstrap_seeds : Union[None, list], optional
        List of seeds to use for the bootstrap, by default None
    client_identifier : Union[None, str], optional
        Name of the column that identifies the client, by default None

    Returns
    -------
    Callable
        The bootstrap function to be used to select indices for the bootstrap
    """
    global_btst_indices = {}
    per_client_btst_indices = {}
    indices = np.arange(sum(clients_sizes))
    if clients_names is not None:
        assert len(clients_names) == len(
            clients_sizes
        ), "You should give as many clients names as there are clients"
    else:
        clients_names = [f"client{i}" for i in range(len(clients_sizes))]
    # A bit silly but we want to follow exactly the BootstrapMixin sampling
    global_df = pd.DataFrame({"indices": indices})
    bootstrap_sample = partial(BootstrapMixin.bootstrap_sample, self=None)
    # We assume data is laid out in the order of the given centers sizes
    # but frankly we do not really care
    clients_indices_list = []
    start_idx = 0
    for client_size in clients_sizes:
        end_idx = start_idx + client_size
        clients_indices_list.append(indices[start_idx:end_idx].tolist())
        start_idx = end_idx
    if not isinstance(bootstrap_seeds, list):
        # It is either a number or None
        # First call as in init of SurvivalEstimator
        rng = np.random.default_rng(bootstrap_seeds)
        # second call in bootstrap_std
        rng = np.random.default_rng(rng)
        # We need to make sure the seeds are different as we create a dict with
        # keys being the seed but we want to make sure that the seeds are not valid
        # seed as this is not seeded
        assert n_bootstrap is not None
        bootstrap_seeds_list = [-seed for seed in np.arange(n_bootstrap).tolist()]
    else:
        rng = None
        assert n_bootstrap is not None
        bootstrap_seeds_list = bootstrap_seeds

    client_identifier = client_identifier if client_identifier is not None else "client"
    for seed in bootstrap_seeds_list:
        per_client_btst_indices[seed] = {}
        global_indices_list_per_client = [[] for _ in range(len(clients_sizes))]
        # Avoid edge cases of bootstrap where some clients might not have any data
        # or not enough data to compute a variance
        while not all(
            [
                len(global_indices) >= 2
                for global_indices in global_indices_list_per_client
            ]
        ):

            if rng is None:
                temp_rng = seed
            else:
                temp_rng = rng
            # This changes temp_rng (hence rng) on purpose following bootstrap_std
            global_indices_list = bootstrap_sample(data=global_df, seed=temp_rng)[
                "indices"
            ].tolist()
            global_indices_list_per_client = []
            for client_indices in clients_indices_list:
                global_indices_list_per_client.append(
                    [
                        global_idx
                        for global_idx in global_indices_list
                        if global_idx in client_indices
                    ]
                )
        global_btst_indices[seed] = global_indices_list

        # Now we "just" need to translate global_indices in per-client indices
        for idx_c, client_indices in enumerate(clients_indices_list):
            per_client_btst_indices[seed][clients_names[idx_c]] = [
                clients_indices_list[idx_c].index(global_idx)
                for global_idx in global_indices_list_per_client[idx_c]
            ]

    def global_bootstrap(data, seed):
        assert (
            client_identifier in data.columns
        ), f"Data in each center should have a center identifier {client_identifier}"
        assert (
            data[client_identifier].nunique() == 1
        ), "Data from one center should come from only one center"
        center = data[client_identifier].unique()[0]
        assert (
            center in clients_names
        ), f"{center} is not a valid center name. Those are {clients_names}"
        # we drop the center column so that it doesn't affect FedECA
        data = data.drop(columns=[client_identifier])
        indices_center = per_client_btst_indices[seed][center]
        return data.iloc[indices_center]

    return global_bootstrap, bootstrap_seeds_list, global_btst_indices
