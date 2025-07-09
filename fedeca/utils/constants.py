"""File containing constants for the repo."""
import socket
from importlib.resources import as_file, files
from pathlib import Path

if socket.gethostname().startswith("abstra"):
    EXPE_PATH = Path("/home/owkin/project/results_experiments")
else:
    with as_file(files("experiments")) as module_path:
        EXPE_PATH = module_path / "results"

EXPERIMENTS_PATHS = {
    "pooled_equivalent": EXPE_PATH / "pooled_equivalent",
    "nb_clients": EXPE_PATH / "nb_clients",
    "power": EXPE_PATH / "power_analysis",
    "dp_results": EXPE_PATH / "pooled_equivalent_dp",
    "real_world": EXPE_PATH / "real-world",
    "robust_pooled_equivalence": EXPE_PATH / "robust_pooled_equivalence",
    "robust_pooled_equivalence_ties": EXPE_PATH / "pooled_equivalent_ties",
    "smd_results": EXPE_PATH / "smd",
    "pdac": EXPE_PATH / "pdac",
}
