# FedECA

[![pr-validation](https://github.com/owkin/fedeca/actions/workflows/pr_validation.yml/badge.svg?branch=main)](https://github.com/owkin/fedeca/actions/workflows/pr_validation.yml)
<img src="/badges/cov_badge.svg" >

:arrow_right:[The API doc is available here](https://owkin.github.io/fedeca/):arrow_left:

This repository contains code associated with the [FedECA arXiv paper](https://arxiv.org/abs/2311.16984).

Please [cite our paper](#citation) if you use our code !


## License

Before using the code be sure to check our [license](./license.md) first.


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

To reproduce figures in the [FedECA arXiv paper](https://arxiv.org/abs/2311.16984),
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

  - Figure 3(a), Figure 3(b)
    ```shell
    python -m experiments.smd.plot_smd
    ```
  - Figure 3(c)
    ```shell
    python -m experiments.power.plot_power_type_one_error
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
@ARTICLE{terrail2023fedeca,
       author = {{Ogier du Terrail}, Jean and {Klopfenstein}, Quentin and {Li}, Honghao and {Mayer}, Imke and {Loiseau}, Nicolas and {Hallal}, Mohammad and {Debouver}, Michael and {Camalon}, Thibault and {Fouqueray}, Thibault and {Arellano Castro}, Jorge and {Yanes}, Zahia and {Dahan}, Laetitia and {Ta{\"\i}eb}, Julien and {Laurent-Puig}, Pierre and {Bachet}, Jean-Baptiste and {Zhao}, Shulin and {Nicolle}, Remy and {Cros}, J{\'e}rome and {Gonzalez}, Daniel and {Carreras-Torres}, Robert and {Garcia Velasco}, Adelaida and {Abdilleh}, Kawther and {Doss}, Sudheer and {Balazard}, F{\'e}lix and {Andreux}, Mathieu},
        title = "{FedECA: Federated External Control Arms for Causal Inference with Time-To-Event Data in Distributed Settings}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Methodology, Computer Science - Distributed, Parallel, and Cluster Computing, Computer Science - Machine Learning},
         year = 2023,
        month = nov,
          eid = {arXiv:2311.16984},
        pages = {arXiv:2311.16984},
          doi = {10.48550/arXiv.2311.16984},
archivePrefix = {arXiv},
       eprint = {2311.16984},
 primaryClass = {stat.ME},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231116984O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
