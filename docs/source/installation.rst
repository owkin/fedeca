
Installation
============

To install the package, create an env with python ``3.9`` with conda

.. code-block:: bash

   conda create -n fedeca python=3.9
   conda activate fedeca

Within the environment, install the package by running:

.. code-block::

   git clone https://github.com/owkin/fedeca.git
   cd fedeca
   pip install -e ".[all_extra]"

If you plan on contributing, you should also install the pre-commit hooks

.. code-block:: bash

   pre-commit install
