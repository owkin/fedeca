# FedECA

[![pr-validation](https://github.com/owkin/fedeca/actions/workflows/pr_validation.yml/badge.svg?branch=main)](https://github.com/owkin/fedeca/actions/workflows/pr_validation.yml)
<img src="/badges/cov_badge.svg" >

:arrow_right: [The API doc is available here](https://owkin.github.io/fedeca/) :arrow_left:

This repository contains code associated with the [FedECA paper](https://doi.org/10.1038/s41467-025-62525-z).

Please [cite our paper](#citation) if you use our code !


## License

Before using the code be sure to check our [license](./license.md) first.


## The official Flower implementation

FedECA is also implemented in the [Flower framework](https://flower.ai/), available [here](https://github.com/owkin/flower-fedeca), as the offical implementation. Please consider using it for your FedECA analysis.


## Installation instructions

In the following, all estimated runtimes of installation and computations use as reference machine a Mac M3 Pro with 18G of RAM
and an internet connexion with a speed of 120Mbps (as measured by fast.com).  
The installation has been tested so far mostly on MacOS (Sonoma 14.1) but the CI is running on both Linux (Ubuntu latest) as well as Windows (2019) (for the installation only).  
The preferred way to run fedeca is on UNIX OS (Linux, MacOs) as substra itself is not thoroughly tested on Windows at the time of writing. We recommend Windows users to proceed with care and
report anything unexpected by opening an issue.  
Regarding hardware, fedeca is lightweight and does not require any non-standard setup and should therefore run on any modern laptop.  

To install the package, create an env with python `3.9` with conda

```bash
conda create -n fedeca python=3.9
conda activate fedeca
```
Creating the conda environment should take under 20 seconds on a modern computer with a reasonably good internet connexion (see top of the section for details).  
Within the environment, install the package by running:

```bash
git clone https://github.com/owkin/fedeca.git
cd fedeca
pip install -e ".[all_extra]"
```
The cloning itself should take less than 5 seconds and the pip install
should take around 1 min and 30 seconds.
Please note, however, that the repo contains artefacts of experiments' results, 
which allow for the reproduction of figures contained in the article without having to rerun
corresponding experiments. The size of the repo is therefore not negligeable (~88M), and
the actual time spent on cloning may vary depending on the internet bandwidth.

If you plan on developing, you should also install the pre-commit hooks

```bash
pre-commit install
```

This will run all the pre-commit hooks at each commit, ensuring a clean repo.

## Quickstart

In order to quickly get a sense on how FedECA works visit [here](./quickstart/quickstart.md).

Copy-pasting all cells of the tutorial into a regular python script gives a runtime of approximately 1 minute,
with less than 2 seconds spent on each run of the FedECA algorithm in-RAM (using the same Mac M3 Pro as a reference).

## Figures reproduction

To reproduce figures in the [FedECA paper](https://doi.org/10.1038/s41467-025-62525-z),
use the following commands:
<details>
  <summary>Equivalence between pooled IPTW and FedECA</summary>

  - Figure 2
    ```shell
    python -m experiments.pooled_equivalent.plot_pooled_equivalent
    ```
  - Supplementary Figure 2
    ```shell
    python -m experiments.pooled_equivalent.plot_pooled_equivalent_nb_clients
    ```
  - Supplementary Figure 3 (differential privacy)
    ```shell
    python -m experiments.dp.plot_dp_hydra
    ```
  - Supplementary Figure 4 (tied times)
    ```shell
    python -m experiments.ties.plot_ties_hydra
    ```
</details>
<details>
  <summary>Standardized mean difference and power analysis</summary>

  - Figure 3
    (remove the `--ensemble` flag to create subfigures separatedly)
    ```shell
    python -m experiments.smd_power.plot_smd_power --ensemble
    ```
</details>
<details>
  <summary>Real-world FedECA</summary>

  - Figure 4, Supplementary Figure 6
    (remove the `--ensemble` flag to create subfigures separatedly)
    ```shell
    python -m experiments.pdac.plot_smd_and_km --ensemble
    ```
  - Supplementary Figure 7 (exchangeability)
    ```shell
    python -m experiments.pdac.plot_exchangeability_kms
    ```
  - Supplementary Figure 8 (weights histogram)
    ```shell
    python -m experiments.pdac.plot_histogram
    ```
</details>

## <a name="citation"></a>Citing FedECA

```bibtex
@article{OgierduTerrail2025,
  author = {Jean Ogier du Terrail and Quentin Klopfenstein and Honghao Li and Imke Mayer and Nicolas Loiseau and Mohammad Hallal and Michael Debouver and Thibault Camalon and Thibault Fouqueray and Jorge Arellano Castro and Zahia Yanes and Laëtitia Dahan and Julien Taïeb and Pierre Laurent-Puig and Jean-Baptiste Bachet and Shulin Zhao and Remy Nicolle and Jérôme Cros and Daniel Gonzalez and Robert Carreras-Torres and Adelaida Garcia Velasco and Kawther Abdilleh and Sudheer Doss and Félix Balazard and Mathieu Andreux},
  title = {FedECA: federated external control arms for causal inference with time-to-event data in distributed settings},
  journal = {Nature Communications},
  year = {2025},
  volume = {16},
  number = {1},
  pages = {7496},
  doi = {10.1038/s41467-025-62525-z},
  url = {https://doi.org/10.1038/s41467-025-62525-z},
  abstract = {External control arms can inform early clinical development of experimental drugs and provide efficacy evidence for regulatory approval. However, accessing sufficient real-world or historical clinical trials data is challenging. Indeed, regulations protecting patients’ rights by strictly controlling data processing make pooling data from multiple sources in a central server often difficult. To address these limitations, we develop a method that leverages federated learning to enable inverse probability of treatment weighting for time-to-event outcomes on separate cohorts without needing to pool data. To showcase its potential, we apply it in different settings of increasing complexity, culminating with a real-world use-case in which our method is used to compare the treatment effect of two approved chemotherapy regimens using data from three separate cohorts of patients with metastatic pancreatic cancer. By sharing our code, we hope it will foster the creation of federated research networks and thus accelerate drug development.},
  issn = {2041-1723}
}

```
