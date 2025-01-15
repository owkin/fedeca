"""Register data in the network."""
# This scripts cannot be readily executed anywhere as it assumes the existence
# of a deployed network
import os
import zlib
from pathlib import Path

import pandas as pd
from pandas.util import hash_pandas_object
from substra.sdk.schemas import DataSampleSpec, DatasetSpec, Permissions

import fedeca
# This depends on the network you are using
# this assumes further that data has been preprocessed and is available
# in the same path on each partner's nodes FL_DATA_PATH
FFCD_URL, IDIBIGI_URL, PANCAN_URL, FL_DATA_PATH = (
    "FFCD_URL",
    "IDIBIGI_URL",
    "PANCAN_URL",
    "FL_DATA_PATH_vXX",
)
from fedeca.utils.substra_utils import Client
from ffcd.ffcd_opener import FFCDOpener
from idibigi.idibigi_opener import IDIBIGIOpener
from pancan.pancan_opener import PanCanOpener


POTENTIAL_HOSPITALS = ["idibigi", "ffcd", "pancan"]
SYNTHETIC = False
# All variables below are only active in the case FAKE_TREATMENT is True
FAKE_TREATMENT = True
IS_FOLFIRINOX = False
IS_TREATMENT_CENTER = True
TREATMENT = "IDIBIGi"
 


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
assert TREATMENT is in [hospitals_mapping[k]["owner"] for k in list(hospitals_mapping.keys())]


hospital = None

while hospital not in POTENTIAL_HOSPITALS:
    hospital = input("Enter the name of the hospital you are in: ")
    if hospital not in POTENTIAL_HOSPITALS:
        print("hospital {hospital} not recognized")

move_forward = False
while not move_forward:
    move_forward = input(
        "Would you like to proceed doing data registration on {hospital} (Y(es)|N(o))"
    ).lower() in [
        "y",
        "yes",
    ]  # noqa: E501

data_path = FL_DATA_PATH
# this assumes the version is the last part of the path
VERSION = FL_DATA_PATH[-1]

if SYNTHETIC:
    from fedeca.utils.survival_utils import CoxData  # noqa: E402

    # Let's generate 1000 data samples with 10 covariates
    data = CoxData(seed=42, n_samples=1000, ndim=10)
    df = data.generate_dataframe()

    # We remove the true propensity score
    df = df.drop(columns=["propensity_scores"], axis=1)

    from fedeca.utils.data_utils import split_dataframe_across_clients  # noqa: E402

    _, _, _, _, _ = split_dataframe_across_clients(
        df,
        n_clients=3,
        split_method="split_control_over_centers",
        split_method_kwargs={"treatment_info": "treatment"},
        data_path=data_path + "_synthetic",
        backend_type="simu",
    )
    center_id = hospitals_mapping[hospital]["synth_center_id"]
    data_path += f"_synthetic/center{center_id}"

PATH_TO_DATA = Path(data_path)

# this assumes the token of the organization is stored in a file available
# at /home/owkin/project/token
URL = hospitals_mapping[hospital]["url"]
with open("/home/owkin/project/token", "r") as f:
    TOKEN = f.read()

if SYNTHETIC:
    from fedeca.scripts.substra_assets.csv_opener import CSVOpener  # noqa: E402

    OpenerClass = CSVOpener
else:
    OpenerClass = hospitals_mapping[hospital]["opener"]


client = Client(url=URL, token=TOKEN, backend_type="remote")

# Check connexion
client.list_dataset()


op = OpenerClass()
# We give it a list of path corresponding to all data samples' folders
# in this specific case we put all the samples into one sample
# (recommended way to use substra now)

X = op.get_data([str(PATH_TO_DATA)])
assert X["center"].nunique() == 1
assert (
    X["center"].iloc[0] == hospitals_mapping[hospital]["owner"]
    if not SYNTHETIC
    else f"center{center_id}"
)


print(X)


# Let's register this data, we need to give:
# - permissions
# - opener path
# - data samples' folders paths
# - potential description of the dataset
# - dataset's name

# But first let's check what already exists
print(client.organization_info())


datasets = client.list_dataset(
    filters={"owner": [hospitals_mapping[hospital]["owner"]]}
)


# In[18]:


len(datasets)


# In[19]:


[d.name for d in datasets]


# In[20]:


# In[21]:


permissions_dataset = Permissions(
    public=False, authorized_ids=[hospitals_mapping[hospital]["owner"], "OwkinMSP"]
)


# In[27]:


df = pd.read_csv(PATH_TO_DATA / "data.csv")

if FAKE_TREATMENT:
    TREATMENT_COL = "Treatment actually received"
    import numpy as np
    np.random.seed(42)
    if IS_FOLFIRINOX:
        df = df[df[TREATMENT_COL] == "FOLFIRINOX"]
    else:
        df = df[df[TREATMENT_COL] == "Gemcitabine + Nab-Paclitaxel"]

    assert len(df.index) > 0
    group = "FOL" if IS_FOLFIRINOX else "GEM"
    # Random treatment allocation within the chosen group
    if not IS_TREATMENT_CENTER:
        df["temp"] = (np.random.uniform(0, 1, len(df.index)) > 0.5)
    else:
        df["temp"] = (X["center"] == TREATMENT)
    df.loc[df["temp"], TREATMENT_COL] = "FOLFIRINOX"
    df.loc[~df["temp"], TREATMENT_COL] = "Gemcitabine + Nab-Paclitaxel"
    df.drop(columns=["temp"], inplace=True)
    path_fake_data = PATH_TO_DATA.parent / f"fake_allocation_folfirinox{IS_FOLFIRINOX}_treatment{TREATMENT}"
    os.makedirs(path_fake_data, exist_ok=True)
    PATH_TO_DATA = path_fake_data
    df.to_csv(path_fake_data / "data.csv", index=False)
    pd.testing.assert_frame_equal(pd.read_csv(PATH_TO_DATA / "data.csv").reset_index(drop=True), df.reset_index(drop=True))
    

# The float32 conversion is necessary to avoid hash differences because of weird
# float64 rounding issues on different machines
to_hash_numbers = hash_pandas_object(
    df.select_dtypes(include="number").astype("float32"), index=True
).values.tolist()
to_hash_numbers = str.encode("".join([str(e) for e in to_hash_numbers]))
to_hash_cat = hash_pandas_object(
    df.select_dtypes(exclude="number"), index=True
).values.tolist()
to_hash_cat = str.encode("".join([str(e) for e in to_hash_cat]))
hash_df = zlib.adler32(to_hash_numbers + to_hash_cat)

# Note that we used the original data for the hash BUT the data from the opener
# for the columns
colnames = ";".join([str(col).replace(" ", "")[:9] for col in sorted(df.columns)])
dataset_name = f"{hospital}_{hash_df}_{colnames}_N{len(df.index)}_v{VERSION}"
if SYNTHETIC:
    dataset_name = "SYNTH_" + dataset_name
if FAKE_TREATMENT:
    dataset_name = f"T{TREATMENT[0]}P{str(IS_FOLFIRINOX)[0]}" + dataset_name

assert (
    len(dataset_name) <= 100
), f"Length of dataset name {len(dataset_name)} Find a shorter name than {dataset_name} but that still contains all infos most important is hash"  # noqa: E501


found_eca_datasets = [
    dataset
    for dataset in client.list_dataset(
        filters={"owner": [hospitals_mapping[hospital]["owner"]]}
    )
    if dataset.name == dataset_name
]
if len(found_eca_datasets) > 0:
    print("WARNING !!!! Found a dataset from the same owner with the same name !")
    hash_dataset = found_eca_datasets[0].key
    assert (
        len(found_eca_datasets[0].data_sample_keys) == 0
    ), "Datasample already registered with this dataset, aborting"
else:
    if not SYNTHETIC:
        opener_path = os.path.abspath(f"{hospital}/{hospital}_opener.py")
    else:
        opener_path = str(
            Path(fedeca.__path__[0]) / "scripts/substra_assets/csv_opener.py"
        )
    description_path = str(
        Path(fedeca.__path__[0]) / "scripts/substra_assets/description.md"
    )
    assert os.path.exists(opener_path)
    assert os.path.exists(description_path), description_path

    # Note that there is no sample yet
    dataset = DatasetSpec(
        name=dataset_name,
        data_opener=opener_path,
        description=description_path,
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )

    hash_dataset = client.add_dataset(dataset)


# In[29]:


datasets = client.list_dataset(
    filters={"owner": [hospitals_mapping[hospital]["owner"]]}
)


# In[30]:


len(datasets)


# In[31]:


[d.name for d in datasets]

# Add the training data on each organization.
data_sample = DataSampleSpec(
    data_manager_keys=[hash_dataset],  # can also be found in the frontend
    path=str(PATH_TO_DATA),
)
client.add_data_sample(data_sample)
