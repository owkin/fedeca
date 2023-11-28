# FedECA

:arrow_right:[The API doc is available here](https://owkin.github.io/fedeca/):arrow_left:  

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
