# from pathlib import Path
import pickle
# from fedeca.utils.substra_utils import Client
# import json
import os

# with open("tokens.json") as f:
#     d = json.load(f)

# token = d["token"]

# URL = "url"

# owkin_ds = Client(
#     url=URL,
#     token=token,
#     backend_type="remote",
# )

f = open("table.tex", "w")
f.write("\\begin{table}\n")
f.write("\\begin{tabular}{ lllll } \n") # pair (treatment) HR z p
f.write("\\toprule\n")
f.write("Method & \\#centers & HR (95\% CI) & z & p \\\\ \n")
f.write("\\midrule\n")

names_mapping = {"ffcd": "FFCD", "idibigi": "IDIBGI", "pancan": "PanCAN"}


with open("/Users/jterrail/Desktop/fedeca_dataset/fedeca_dataset/fl/results_2024-10-04_12-29-48.pkl", "rb") as pick:
    res = pickle.load(pick)

# train_datasets = res["kwargs"]["train_data_nodes"]
# train_datasets_names = [owkin_ds.get_dataset(dataset.data_manager_key).name for dataset in train_datasets]
# sizes = [int(name.split("_")[-2][1:]) for name in train_datasets_names]
for variance_method in ["bootstrap", "robust", "naïve"]:

    current_results = res[variance_method]["results"]
    f.write(
        f"FedECA ({variance_method}) & "
        "{\\scriptsize FFCD, IDIBGI, PanCAN } & "
        f"\\num{{{current_results['exp(coef)'].item()}}}"
        f"(\\num{{{current_results['exp(coef) lower 95%'].item()}}}, \\num{{{current_results['exp(coef) upper 95%'].item()}}}) & "
        f"\\num{{{current_results['z'].item()}}} & \\num[round-mode=figures,round-precision=3]{{{current_results['p'].item()}}} \\\\ \n"
    )
f.write("\\bottomrule \n") 
f.write("\end{tabular}\n")
f.write("\caption{Effect on overall survival of FOLFIRINOX over gemcitabine + nab placlitaxel as estimated by FedECA versus single-center estimates.}\n")
f.write("\label{tab:real_world_stats}\n")
f.write("\end{table}\n")
f.close()
os.system("cat table.tex")


