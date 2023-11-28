"""File containing constants for the repo."""
import socket
from os.path import join

import git

if socket.gethostname().startswith("abstra"):
    EXPE_PATH = "/home/owkin/project/results_experiments"
else:
    repo = git.Repo(".", search_parent_directories=True)
    EXPE_PATH = join(repo.working_dir, "experiments", "results")

EXPERIMENTS_PATHS = {
    "pooled_equivalent": EXPE_PATH + "/pooled_equivalent/",
    "nb_clients": EXPE_PATH + "/nb_clients/",
    "power": EXPE_PATH + "/power/",
    "dp_results": EXPE_PATH + "/pooled_equivalent_dp/",
    "real_world": EXPE_PATH + "/real-world/",
    "robust_pooled_equivalence": EXPE_PATH + "/robust_pooled_equivalence/",
}
