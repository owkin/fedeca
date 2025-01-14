import matplotlib.pyplot as plt
import pandas as pd
FFCD_URL, IDIBIGI_URL, PANCAN_URL, FL_DATA_PATH = (
    "FFCD_URL",
    "IDIBIGI_URL",
    "PANCAN_URL",
    "FL_DATA_PATH_vXX",
)
from ffcd.ffcd_opener import FFCDOpener
from idibigi.idibigi_opener import IDIBIGIOpener
from pancan.pancan_opener import PanCanOpener
from pandas.util import hash_pandas_object
import seaborn as sns
import pickle
from fedeca.fedeca_core import LogisticRegressionTorch
from torch import nn
import torch
import numpy as np

from itertools import cycle

POTENTIAL_HOSPITALS = ["idibigi", "ffcd", "pancan"]
PATH_TO_DATA = FL_DATA_PATH
hospitals_mapping = {
    "idibigi": {
        "url": IDIBIGI_URL,
        "opener": IDIBIGIOpener,
        "owner": "IDIBIGi",
        "synth_center_id": 0,
    },
    "ffcd": {
        "url": FFCD_URL,
        "opener": FFCDOpener,
        "owner": "FFCDMSP",
        "synth_center_id": 1,
    },
    "pancan": {
        "url": PANCAN_URL,
        "opener": PanCanOpener,
        "owner": "PanCan",
        "synth_center_id": 2,
    },
}

hospital = ""
while hospital not in POTENTIAL_HOSPITALS:
    hospital = input("Enter the name of the hospital you are in: ")
    if hospital not in POTENTIAL_HOSPITALS:
        print("hospital {hospital} not recognized")


OpenerClass = hospitals_mapping[hospital]["opener"]

op = OpenerClass()
# We give it a list of path corresponding to all data samples' folders
# in this specific case we put all the samples into one sample
# (recommended way to use substra now)

X = op.get_data([str(PATH_TO_DATA)])
assert X["center"].nunique() == 1
assert (
    X["center"].iloc[0] == hospitals_mapping[hospital]["owner"])
# No standardization in the propensity model

f = open("results_2024-10-04_12-29-48.pkl", "rb")
res = pickle.load(f)
f.close()

propensity_model_weights = res["naïve"]["propensity_model"]
propensity_model = LogisticRegressionTorch(ndim=propensity_model_weights["weight"].size)

propensity_model.fc1.weight.data = nn.parameter.Parameter(
    torch.from_numpy(propensity_model_weights["weight"])
)
propensity_model.fc1.bias.data = nn.parameter.Parameter(
    torch.from_numpy(propensity_model_weights["bias"])
)
CONT_COLUMNS = [
    "Age at cancer diagnosis",
    "Performance Status (ECOG) at cancer diagnosis",
]
CAT_COLUMNS = ["Biological gender_F", "Liver metastasis_True"]
# ["treatment", "event", "Overall survival"]

# CENTER_COLUMN = ["center"]

Xprop = X[CONT_COLUMNS + CAT_COLUMNS].to_numpy().astype("float64")
treated = X["treatment"].to_numpy().flatten()
Xprop = torch.from_numpy(Xprop)

with torch.no_grad():
    propensity_scores = propensity_model(Xprop)


propensity_scores = propensity_scores.detach().numpy().flatten()

tol = 1e-16

# We robustify the division
weights = treated * 1.0 / np.maximum(propensity_scores, tol) + (
            1 - treated
        ) * 1.0 / (np.maximum(1.0 - propensity_scores, tol))


X["weights"] = weights.flatten()
print("Maximum weights: " + str(weights.flatten().max()))
print("Minimum weights: " + str(weights.flatten().min()))
X["ps_score"] = propensity_scores.flatten()

fig, axs = plt.subplots(figsize=(10, 5))
N = 25 + 1
bins=np.linspace(1.0, 10., N)
sns.histplot(data=X, x="weights", hue="treatment", bins=bins)
for b0, b1, color in zip(bins[:-1], bins[1:], cycle(['crimson', 'lightblue'])):
    axs.axvspan(b0, b1, color=color, alpha=0.1, zorder=0)
plt.savefig(f"histograms_weights_{hospital}.pdf", bbox_inches="tight", dpi=300)
plt.clf()


fig, axs = plt.subplots(figsize=(10, 5))
N = 25 + 1
bins=np.linspace(0, 1, N)
sns.histplot(data=X, x="ps_score", hue="treatment", bins=bins)
for b0, b1, color in zip(bins[:-1], bins[1:], cycle(['crimson', 'lightblue'])):
    axs.axvspan(b0, b1, color=color, alpha=0.1, zorder=0)

plt.savefig(f"histograms_ps_{hospital}.pdf", bbox_inches="tight", dpi=300)
