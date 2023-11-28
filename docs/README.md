# Documentation

The documentation of FedECA is generated using Sphinx and hosted on github.
If you want to build the documentation from source, start by cloning the repository,
using instructions in the main readme, and install it with the `[all_extra]` option.

```bash
pip install -e ".[all_extra]"
```

You can now trigger the build using `make` command line in the `docs` folder.

```bash
cd docs
make clean html
```
