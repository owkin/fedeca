"""Setup script for fedeca."""
from setuptools import find_packages, setup

deps = ["docformatter"]
tests = ["pytest", "coverage"]
docs = [
    "jupyter",
    "sphinx==4.4.0",
    "sphinx_rtd_theme==1.0.0",
    "sphinx-autobuild==2020.9.1",
    "texttable==1.6.3",
    "myst-parser==0.16.1",
    # Sphinx 3.3.1 does not require a specific version of docutils
    # but docutils 0.17 changed the output html markup, breaking the RTD theme
    # original issue: https://github.com/sphinx-doc/sphinx/issues/9051
    "docutils==0.16",
    "sphinx_click==3.1.0",
    # "sphinx_gallery==0.11.1",
    "sphinx_autodoc_typehints",
    "sphinxcontrib-bibtex==2.5.0",
    "gitpython>=3.1.27",
]
all_extra = deps + tests + docs

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fedeca",
    version="0.0.2",
    python_requires=">=3.9.0,<3.11",
    license="MIT",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        # We cannot use the git+https syntax here because of docker build issues
        "substrafl @ https://github.com/Substra/substrafl/archive/refs/heads/feat/substrafl-simu-mode.zip",  # noqa: E501
        "argparse",
        "numpy",
        "pandas",
        "pre-commit",
        "scipy",
        "seaborn",
        "opacus",
        "lifelines",
        "git-python",
        "build",
        "torch==1.13.1",
        "scikit-learn==1.2.1",
        "pydantic<2.0",  # Need to be updated to > 2.0 to use latest Substra
        "indcomp==0.2.1",
        "hydra-core",
    ],
    extras_require={
        "all_extra": all_extra,
    },
    description="Federated External Control Arm with substra",
    long_description=long_description,
    author="""
        Jean Ogier du Terrail, Quentin Klopfenstein,
        Honghao Li, Nicolas Loiseau, Mathieu Andreux,
        Félix Balazard""",
    author_email="jean.du-terrail@owkin.com",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
