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

f = open("table_exchangeability.tex", "w")
f.write("\\begin{table}\n")
f.write("\\begin{tabular}{ lllll } \n") # pair (treatment) HR z p
f.write("\\toprule\n")
f.write("Pair and treament & n & HR (95\% CI) & z & p \\\\ \n")
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

    assert all([name[:4] == id_pop_treatment for name in current_train_datasets_names])
    if id_pop_treatment[1].lower() == "i":
        treatment = "idibigi"
    elif id_pop_treatment[1].lower() == "f":
        treatment = "ffcd"
    elif id_pop_treatment[1].lower() == "p":
        treatment = "pancan"
    population = "FOLFIRINOX" if id_pop_treatment[3] == "T" else "GEM + NAB"

    assert any([treatment in name for name in current_train_datasets_names]), "The treatment is wrong"
    centers_pairs = "-".join([names_mapping[name[4:].split("_")[0]] for name in current_train_datasets_names])

    current_results = current_res["bootstrap"]["results"]
    f.write(
        f"{centers_pairs} (tested T='{names_mapping[treatment]}', real T='{population}') & "
        f"{sizes[0]} / {sizes[1]} ({sum(sizes)}) & "
        f"\\num{{{current_results['exp(coef)'].item()}}}"
        f"(\\num{{{current_results['exp(coef) lower 95%'].item()}}}, \\num{{{current_results['exp(coef) upper 95%'].item()}}}) & "
        f"\\num{{{current_results['z'].item()}}} & \\num[round-mode=figures,round-precision=3]{{{current_results['p'].item()}}} \\\\ \n"
    )
f.write("\\bottomrule \n") 
f.write("\end{tabular}\n")
f.write("\caption{Exchangeability test between pairs of centers.}\n")
f.write("\label{tab:exchang}\n")
f.write("\end{table}\n")
f.close()
os.system("cat table_exchangeability.tex")


