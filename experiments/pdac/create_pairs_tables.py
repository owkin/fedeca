from pathlib import Path
import pickle
from fedeca.utils.substra_utils import Client
import json
import os

# Find all pickle files in folder FOLDER
FOLDER = Path(".")
results_exchangeability = list(FOLDER.glob("*.pkl"))

with open("tokens.json") as f:
    d = json.load(f)

token = d["token"]

URL = "url"
owkin_ds = Client(
    url=URL,
    token=token,
    backend_type="remote",
)

f = open("table_pairs.tex", "w")
f.write("\\begin{table}\n")
f.write("\\begin{tabular}{ lllll } \n") # pair (treatment) HR z p
f.write("\\toprule\n")
f.write("Method & \\#centers & HR (95\% CI) & z & p \\\\ \n")
f.write("\\midrule\n")

names_mapping = {"ffcd": "FFCD", "idibigi": "IDIBGI", "pancan": "PanCAN"}
for pickle_file in results_exchangeability:
    # open pickle pickle_file
    with open(pickle_file, "rb") as pick:
        current_res = pickle.load(pick)

    current_train_datasets = current_res["kwargs"]["train_data_nodes"]

    current_train_datasets_names = [owkin_ds.get_dataset(dataset.data_manager_key).name for dataset in current_train_datasets]
    sizes = [int(name.split("_")[-2][1:]) for name in current_train_datasets_names]
    id_pop_treatment = current_train_datasets_names[0][:4]
    centers = [name.split("_")[0] for name in current_train_datasets_names]
    assert all([c in list(names_mapping.keys()) for c in centers])
    centers = [names_mapping[c] for c in centers]
    for variance_method in ["bootstrap", "robust", "naïve"]:
        current_results = current_res[variance_method]["results"]
        f.write(
            f"FedECA ({variance_method}) & "
            f"\\scriptsize{{{centers[0]}, {centers[1]}}} & "
            f"\\num{{{current_results['exp(coef)'].item()}}}"
            f"(\\num{{{current_results['exp(coef) lower 95%'].item()}}}, \\num{{{current_results['exp(coef) upper 95%'].item()}}}) & "
            f"\\num{{{current_results['z'].item()}}} & \\num[round-mode=figures,round-precision=3]{{{current_results['p'].item()}}} \\\\ \n"
        )
f.write("\\bottomrule \n") 
f.write("\end{tabular}\n")
f.write("\caption{FL between pairs of centers.}\n")
f.write("\label{tab:fl_pairs}\n")
f.write("\end{table}\n")
f.close()
os.system("cat table_pairs.tex")