# YODA experiments

This scripts assumes the access to data from the [Yale University
Open Data Access (YODA)](https://yoda.yale.edu/) project.

Data is preprocessed and then missing values are imputed using MissForest (see `../pdac/base_opener.py`).

In the code there is a placeholder `path_to_imputed_data` that needs to be overwritten with the actual path of the data.

In this case FL clients are simulated such as control and treated are in two separate groups. Note that changing the number of clients or the data splits should not impact the results much.