datasets = owkin_ds.list_dataset()
import re


VERSION = 5
IDIBIGI_org_id = "IDIBIGi"
FFCD_org_id = "FFCDMSP"
PANCAN_org_id = "PanCan"

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
        idibigi_datasets = [ds for ds in idibigi_datasets if ds.name.startswith(f"T{TREATMENT}")]
        ffcd_datasets = [ds for ds in ffcd_datasets if ds.name.startswith(f"T{TREATMENT}")]
        pancan_datasets = [ds for ds in pancan_datasets if ds.name.startswith(f"T{TREATMENT}")]
        # filtering out the correct group
        idibigi_datasets = [ds for ds in idibigi_datasets if ds.name[2:4] == f"F{GROUP}"]
        ffcd_datasets = [ds for ds in ffcd_datasets if ds.name[2:4] == f"F{GROUP}"]
        pancan_datasets = [ds for ds in pancan_datasets if ds.name[2:4] == f"F{GROUP}"]

    else:
        # filtering out fake treatment datasets
        idibigi_datasets = [ds for ds in idibigi_datasets if re.fullmatch(f"idibigi_.*", ds.name)]
        ffcd_datasets = [ds for ds in ffcd_datasets if re.fullmatch(f"ffcd_.*", ds.name)]
        pancan_datasets = [ds for ds in pancan_datasets if re.fullmatch(f"pancan_.*", ds.name)]



assert len(idibigi_datasets) == 1
assert len(ffcd_datasets) == 1
assert len(pancan_datasets) == 1

IDIBIGI_dataset_key = idibigi_datasets[0].key
IDIBIGI_data_samples_keys = idibigi_datasets[0].data_samples_keys
FFCD_dataset_key = ffcd_datasets[0].key
FFCD_data_samples_keys = ffcd_datasets[0].data_samples_keys
PANCAN_dataset_key = pancan_datasets[0].key
PANCAN_data_samples_keys = pancan_datasets[0].data_samples_keys






