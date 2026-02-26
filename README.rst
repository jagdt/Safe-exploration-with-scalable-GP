=======================================================================================================
 Learning-based Model Predictive Control for Safe Exploration with scalable Gaussian process regression
=======================================================================================================

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
    :target: https://www.python.org/downloads/
    :alt: Python 3.7+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: LICENSE
    :alt: MIT License


Overview
--------

This implementation builds upon the SafeMPC framework proposed in the following paper:

.. [1] T. Koller, F. Berkenkamp, M. Turchetta, A. Krause,
  `Learning-based Model Predictive Control for Safe Exploration <https://arxiv.org/abs/1803.08287>`_
  in Proc. of the Conference on Decision and Control (CDC), 2018

and extends it with a scalable GP model.

**Supported Environment:**

- Inverted Pendulum

**Exploration Modes:**

1. **Static Exploration**: Find informative state-action pairs and reset the environment in every iteration step
2. **Dynamic Exploration**: Execute informative trajectories over multiple time steps

Installation
------------


**Basic Installation:**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/safe-exploration-with-scalable-gp.git
   cd safe-exploration-with-scalable-gp

   # Install with minimal dependencies
   pip install -e .

**Full Installation (Recommended):**

.. code-block:: bash

   # Install with all optional dependencies
   pip install -e ".[test,visualization,ssm_gpy,ssm_pytorch]"

**Installation Options:**

- ``test``: Testing tools (pytest, flake8, pydocstyle)
- ``visualization``: Plotting libraries (matplotlib, pygame)
- ``ssm_gpy``: GPy-based state space models
- ``ssm_pytorch``: PyTorch/GPyTorch-based models


Quick Start
-----------

**Run a Single Experiment:**

.. code-block:: bash

   cd experiments
   python run.py with scenario_file=journal_experiment_configs/dynamic_expl_pendulum_numpy.py

**Run Parameter Sweeps:**

.. code-block:: bash

   cd experiments
   python run_exploration_sweeps.py

**Visualize Results:**

.. code-block:: bash

   python evaluate_exploration_sweeps.py



License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

