"""Setup script for fedeca."""
from setuptools import find_packages, setup

deps = ["docformatter"]
tests = ["pytest", "coverage"]
docs = [
    "jupyter",
    "sphinx<7",
    "sphinx-rtd-theme==0.4.2",
    "gitpython>=3.1.27",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]
dev = ["flake8", "flake8-docstrings", "pre-commit"]
all_extra = deps + tests + docs + dev

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
        # We cannot use the git+https / git+ssh syntax here because of docker
        # build issues related to git not being installed
        # Commented pinned versions are only there as references but should
        # not matter as long as the versions are compatible with the ones
        # uncommented. 
        "substrafl==0.46.0",
        "numpy==1.26.4",
        "pandas",#==2.2.3
        "pre-commit",#==4.0.1
        "scipy",#==1.13.1
        "seaborn",#==0.13.2
        "opacus",#==1.4.0
        "lifelines",#==0.29.0
        "git-python",#==1.0.3
        "build",#==1.2.2.post1
        "torch==1.13.1",
        "scikit-learn==1.2.1",
        "pydantic", #==2.23.4
        "indcomp==0.2.1",
        "hydra-core",#==1.3.2
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
