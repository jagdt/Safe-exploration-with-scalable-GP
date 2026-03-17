=======================================================================================================
 Learning-based Model Predictive Control with scalable Gaussian process regression for safe exploration
=======================================================================================================

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
    :target: https://www.python.org/downloads/
    :alt: Python 3.7+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: LICENSE
    :alt: MIT License


Overview
--------

**Thesis Scope:**

This repository contains the code accompanying the author's Master's thesis, which develops a scalable Gaussian process (GP) framework that retains uniform uncertainty guarantees for use in learning-based Model Predictive Control (MPC). A paper describing this work is planned and will be linked here once available.

**Thesis Abstract:**

::

   Learning-based Model Predictive Control (MPC) using Gaussian processes (GPs) is an effective approach for safe control in the presence of model mismatch. High probability safety guarantees typically require uniform uncertainty bounds that hold over the entire state–input domain, but existing bounds are available only for full GP regression. Since exact GP inference scales poorly with the number of data points, these approaches become impractical in large-data regimes.

   This thesis closes this gap by developing a scalable GP framework that retains uniform uncertainty guarantees. We propose a finite-dimensional kernel approximation based on discretized trigonometric features, reducing GP regression to Bayesian linear regression in feature space and enabling efficient online updates. We derive a high-probability uniform uncertainty bound for the proposed finite-dimensional kernel approximation, and derive its closed-form solution for the squared-exponential kernel case. Finally, we integrate the proposed scalable GP into a safe learning-based MPC scheme and demonstrate that it achieves uncertainty bounds and exploration performance comparable to a standard GP while improving computational efficiency in large-data regimes.

This implementation builds upon the SafeMPC framework proposed in the following paper:

.. [1] T. Koller, F. Berkenkamp, M. Turchetta, A. Krause,
  `Learning-based Model Predictive Control for Safe Exploration <https://arxiv.org/abs/1803.08287>`_
  in Proc. of the Conference on Decision and Control (CDC), 2018

and extends it with a scalable GP model.

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

