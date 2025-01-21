# Federated Learning FFCD<->IDIBIGI<->PanCan loop

This assumes the existence of a federated research network of 3 partners (labeled
`pancan`, `idibigi` and `ffcd`) deployed using substra, therefore it cannot be run
anywhere wo modifications.
It assumes each partner has a csv `data.csv` that is compatible with `base_opener.py`.
Meaning the following code runs wo error on each center.

```python
from pancan.pancan_opener import PanCanOpener
#from idibigi.idibigi_opener import IdibigiOpener
#from ffcd.ffcd_opener import FCCDOpener

op = PanCanOpener()
# op = IdibigiOpener()
# op = FFCDOpener()
# This does not throw any error
data = op.get_data([FL_DATA_PATH])
```

## Get data scientist's substra token to be able to launch FL tasks

## Get one of the partner's center's substra token to be able to download models

## Each partner registers synthetic data for test purposes

Each partner should execute the `data_registration.py` script by inputing the name
of the center from `["pancan", "idibigi", "ffcd"]` and confirming the registration.
This create a dataframe in a reproducible fashion and split it into 3 parts.

## Each partner registers its data

Set the `SYNTHETIC` variable to `False` in the `data_registration.py` script.
Preprocess the data in each center and store in a folder ending with a version number
such as `0` (see above). Store the path towards this folder in `FL_DATA_PATH` in `data_registration.py`.

Execute each script after making sure `FL_DATA_PATH` contains a `data.csv`.

:warning: Careful there should be only one file in the folder to avoid undefined
behavior as the whole content of the folder will be registered :warning:

Note: instructions below are somewhat stale as the scripts have been improved
now, providing all the right datasets have been registered, there is a logic
based on naming conventions to select the right ones using only user-defined
arguments.

# Execute FedECA on real data

Change dataset key and datasample keys in `fl_run.py` to the actual
values of data from the centers then run the script.

This should save a pickle with all necessary results with the 3 variance estimation
methods. Runtime is expected to be less than 3 hours.

# Plot KM curves and SMD

Open the script `compute_fl_statistics.py` after having run an `fl_run.py` (in order to
get the trained propensity model weights for SMD). Replace the results' file name by your
own (you just have to change the timestamp normally), then run the script.
