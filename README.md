# FedECA
<img src="/badges/cov_badge.svg" >

:arrow_right:[The API doc is available here](https://owkin.github.io/fedeca/):arrow_left:

This repository contains code associated with the [FedECA arXiv paper](https://arxiv.org/abs/2311.16984).

Please [cite our paper](#citation) if you use our code !


## License

Before using the code be sure to check our [license](./license.md) first.


## Installation instructions

To install the package, create an env with python `3.9` with conda

```bash
conda create -n fedeca python=3.9
conda activate fedeca
```

Within the environment, install the package by running:
```
git clone https://github.com/owkin/fedeca.git
pip install -e ".[all_extra]"
```

If you plan on developing, you should also install the pre-commit hooks

```bash
pre-commit install
```

This will run all the pre-commit hooks at each commit, ensuring a clean repo.

## Quickstart

Go [here](./quickstart/quickstart.md).

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
