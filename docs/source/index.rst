FedECA documentation
======================

This package allows to perform both simulations and deployments of federated
external control arms (FedECA) analyses.

Before using this code make sure to: 

#. read and accept the terms of the license license.md that can be found at the root of the repository.
#. read `substra's privacy strategy <https://docs.substra.org/en/stable/additional/privacy-strategy.html>`_
#. read the `companion article <https://arxiv.org/abs/2311.16984>`_
#. `activate secure rng in Opacus <https://opacus.ai/docs/faq#:~:text=What%20is%20the%20secure_rng,the%20security%20this%20brings.>`_ if you plan on using differential privacy.



Citing this work
----------------

::

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
