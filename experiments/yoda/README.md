# YODA experiments

This scripts assumes the access to data from the [Yale University
Open Data Access (YODA)](https://yoda.yale.edu/) project.

Data is preprocessed and then missing values are imputed using MissForest (see
`./yoda_preprocessing.R`).

Imputed dataframe `df_imputed` should be exported to be used by `./yoda_experiment.py`,
where there is a placeholder `path_to_imputed_data` that needs to be overwritten with
the actual path of the exported data.

In this case FL clients are simulated such as control and treated are in two separate groups. Note that changing the number of clients or the data splits should not impact the results much.
