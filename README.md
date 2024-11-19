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
Creating the conda environment should take under 20 seconds on a modern computer with a reasonably good internet connexion (see below for details).  
Within the environment, install the package by running:

```
git clone https://github.com/owkin/fedeca.git
cd fedeca
pip install -e ".[all_extra]"
```
The cloning itself should take less than 5 seconds and the pip install
should take around 1 min and 30 seconds.
Please note, however, that the repo contains artefacts of results of all
experiments, which allow for the reproduction of figures contained in the
article without having to rerun all the experiments. The size of the repo is
therefore considerable (~88M), and the actual time spent on cloning may
therefore vary depending on the internet bandwidth.

If you plan on developing, you should also install the pre-commit hooks

```bash
pre-commit install
```

This will run all the pre-commit hooks at each commit, ensuring a clean repo.

## Quickstart

Go [here](./quickstart/quickstart.md).

Copy-pasting all cells of the tutorial into a regular python script gives a runtime of approximately 1 minute,
with less than 2 seconds spent on each run of the FedECA algorithm in-RAM (using the same Mac M3 Pro as a reference).

## <a name="citation"></a>Citing FedECA

```
@ARTICLE{terrail2023fedeca,
       author = {{Ogier du Terrail}, Jean and {Klopfenstein}, Quentin and {Li}, Honghao and {Mayer}, Imke and {Loiseau}, Nicolas and {Hallal}, Mohammad and {Debouver}, Michael and {Camalon}, Thibault and {Fouqueray}, Thibault and {Arellano Castro}, Jorge and {Yanes}, Zahia and {Dahan}, Laetitia and {Ta{\"\i}eb}, Julien and {Laurent-Puig}, Pierre and {Bachet}, Jean-Baptiste and {Zhao}, Shulin and {Nicolle}, Remy and {Cros}, J{\'e}rome and {Gonzalez}, Daniel and {Carreras-Torres}, Robert and {Garcia Velasco}, Adelaida and {Abdilleh}, Kawther and {Doss}, Sudheer and {Balazard}, F{\'e}lix and {Andreux}, Mathieu},
        title = "{FedECA: A Federated External Control Arm Method for Causal Inference with Time-To-Event Data in Distributed Settings}",
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
