FedECA documentation
======================

This package allows to perform both simulations and deployments of federated
external control arms (FedECA) analyses.

Before using this code make sure to: 

#. read and accept the terms of the license license.md that can be found at the root of the repository.
#. read `substra's privacy strategy <https://docs.substra.org/en/stable/additional/privacy-strategy.html>`_
#. read our `companion article <https://arxiv.org/abs/2311.16984>`_
#. `activate secure rng in Opacus <https://opacus.ai/docs/faq#:~:text=What%20is%20the%20secure_rng,the%20security%20this%20brings.>`_ if you plan on using differential privacy.



Citing this work
----------------

::

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


License
-------

FedECA is released under a custom license that can be found under license.md at the root of the repository.

.. toctree::
   :maxdepth: 0
   :caption: Installation
   
   installation

.. toctree::
   :maxdepth: 0
   :caption: Getting Started Instructions
   
   quickstart

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: API

   api/fedeca
   api/competitors
   api/algorithms
   api/metrics
   api/scripts
   api/strategies
   api/utils
