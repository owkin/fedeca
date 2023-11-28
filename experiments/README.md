### The `experiments` module
```shell
experiments/
├── config/
│   ├── data/
│   │   └── cox_data.yaml
│   ├── default_config.yaml
│   ├── experiment/
│   │   ├── example.yaml
│   │   └── ...
│   ├── model/
│   │   ├── common.yaml
│   │   ├── covariate_adjusted.yaml
│   │   ├── fl_iptw.yaml
│   │   ├── matching_adjusted.yaml
│   │   ├── naive_comparison.yaml
│   │   └── pooled_iptw.yaml
└── run_experiment.py
```

#### Configuration files
Experiments are defined by composing different configuration groups under `config`.
* `config/data` contains data generation models
* `config/model` contains estimation models
* `config/experiment` contains assembled experiment settings

#### Running experiments
`config/experiment/example.yaml` provides an example of experiment setting to help users
assemble more complex experiments. To run the experiment defined by `example.yaml`:
```shell
python -m experiments.run_experiment -m experiment=example
```
To run user-defined experiments, read and adapt (with copy) `example.yaml`, put the new
config file under `config/experiment`, then run the above command modifying the
`experiment` kwarg.

### User guide to generate figures of the FedECA article
---
#### Figure 2: Pooled IPTW vs. Fedeca
To reproduce the results on the relative error between FedECA and pooled IPTW, run the following command line:
```
python -m experiments.run_experiment -m experiment=robust_pooled_equivalent
```

To change some parameters of the experiment, the config file is in `experiments/config/experiment/robust_pooled_equivalent.yaml`.

Once done, the results are saved in the shared folder on Abstra, `/home/owkin/project/results_experiments/pooled_equivalent/results_Robust_Pooled_Equivalent.pkl`. 

In order to plot the figure, simply run the following python script
```
python /experiments/robust_pooled_equivalent/robust_plot_pooled_equivalent.py
```
The figure will be saved in a pdf format in your current directory.

To obtain the equivalent figure wo using robust just drop all robust_ prefixes.


---
#### Figure 3: Statistical power and Type I error benchmark.
To reproduce the results on statistical power and type I error, run the following commands:
```
python -m experiments.run_experiment -m experiment=power_and_type_one_error_cov_shift
python -m experiments.run_experiment -m experiment=power_and_type_one_error_n_samples
```

To change some parameters of the experiment, the config files are:
* `experiments/config/experiment/power_and_type_one_error_cov_shift.yaml`
* `experiments/config/experiment/power_and_type_one_error_n_samples.yaml`

Once done, the results are saved in the shared folder on Abstra:
* `/home/owkin/project/results_experiments/power_and_type_one_error_cov_shift/results_Power_and_type_one_error_analyses.pkl`.
* `/home/owkin/project/results_experiments/power_and_type_one_error_n_samples/results_Power_and_type_one_error_analyses.pkl`. 

In order to plot the figure, simply run the following command:
```
python /experiments/power/plot_power_type_one_error.py
```
The figure with 4 subfigures (power and type I error, varied covariate shift and number
of samples) will be saved in a pdf format in your current directory.

---
#### Figure S2: DP-FedECA

To reproduce the results on relative errors with changing DP params, run the following command line:
```
python -m experiments.run_experiment -m experiment=pooled_equivalence_dp
```

To change some parameters of the experiment, the config file is in `experiments/config/experiment/pooled_equivalence_dp.yaml`.
Once done, the results are saved in the shared folder on Abstra, `results_Pooled_equivalent_DP.pkl`.
Currently it's stored in the repository in `pooled_equivalent_dp folder`. 
In order to plot the figure, put the pickle on abstra and simply run the following python script
```
python /experiments/dp/plot_dp_hydra.py
```

---
#### Table 1: Real-world experiments

To reproduce the results on timings, first create and download API tokens from
the demo-env from all 3 organizations. Create a tokens folder in your current
directory and copy paste the tokens in api_key1 / ai_key2 and api_key3 files
corresondiing to org1 / org2 and org3.
Then run the following command line:
```
python -m experiments.run_experiment -m experiment=real_world_runtimes
```

To change some parameters of the experiment, the config file is in `experiments/config/experiment/real_world_runtimes.yaml`.
Once done, the results are saved in the shared folder on Abstra, `results_Real-world_experiments.pkl`.
In order to plot the figure, put the pickle on abstra and simply run the following python script
```
python ./experiments/real-world/plot_real_world_hydra.py
```

---
#### Figure S1: pooled equivalent
To reproduce the results on the relative error between FedECA and pooled IPTW with increasing the number of clients, run the following command line:
```
python -m experiments.run_experiment -m experiment=robust_pooled_equivalent_nb_clients
```

To change some parameters of the experiment, the config file is in `experiments/config/experiment/robust_pooled_equivalent_nb_clients.yaml`.

Once done, the results are saved in the shared folder on Abstra, `/home/owkin/project/results_experiments/robust_pooled_equivalent/results_Robust_Pooled_Equivalent_nb_clients.pkl`. 

In order to plot the figure, simply run the following python script
```
python /experiments/pooled_equivalent/robust_plot_pooled_equivalent_nb_clients.py
```
The figure will be saved in a pdf format in your current directory.

For the figure illustrating the effect of ties run:

```
python -m experiments.run_experiment -m experiment=pooled_equivalent_ties
```
Then plot with:

```
python ./experiments/ties/plot_ties_hydra.py
```

