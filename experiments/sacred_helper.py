"""Sets up Sacred, and contains the default parameters."""
import sys

import sacred
from sacred.observers import MongoObserver, FileStorageObserver

from experiments import sacred_auth_details

ex = sacred.Experiment()

config_updates, _ = sacred.arg_parser.get_config_updates(sys.argv)

# Disable saving to mongo using "with save_to_db=False"
if ("save_to_db" not in config_updates) or config_updates["save_to_db"]:
    # TODO: Enable MongoDB observer
    # mongo_observer = MongoObserver.create(url=sacred_auth_details.db_url, db_name='safe-exploration')
    # ex.observers.append(mongo_observer)
    pass
else:
    ex.observers.append(FileStorageObserver.create('safe_exploration_results'))


@ex.config
def base_config():
    save_to_db = False
    # File name of the base config file, or None to use the file associated with the environment below.
    scenario_file = None

    # Device to force Tensors to. None, or a valid pytorch device (e.g. cpu, cuda:0)
    device = None

    # When True, plots created by plot_episode_trajectory, plot_states will be saved to sacred. Otherwise they will be
    # displayed using plt.show().
    save_plots_to_sacred = True

    # -- Environment
    # The environment to use. One of 'pendulum', 'lander' or None (to use the scenario file above).
    environment = 'pendulum'
    render = True
    visualize = True
    plot_episode_trajectory = False
    # Number of dimensions that the pendulum moves in, n>=2. If n=2 then InvertedPendulum environment will be used,
    # otherwise NDPendulum. Currently only n=2 works properly.
    pendulum_dimensions = 2
    # When True, constraints will specify constant velocity requirements, instead of different at different angles.
    pendulum_simple_constraints = False
    # Number of random dimensions to add to the state before passing to the SSM.
    junk_state_dimen = 0
    # Number of random dimensions to add to the action before passing to the SSM.
    junk_action_dimen = 0
    # Whether to enable the environment objectives, otherwise the default objective of maximum variance is used.
    enable_objectives = False

    # -- Lunar Lander environment
    # The width of the environment, in metres.
    lander_env_width = 3
    # The distance from the origin to the surface of the moon, in metres
    lander_surface_y = 2

    # Type of state space model to use, one of exact_gp, mc_dropout, mc_dropout_gal.
    cem_ssm = 'exact_gp'

    # -- Episodic
    # The number of repeats of the experiment, over which we will average the metrics.
    n_scenarios = 6
    # Total number of steps to run each repeat for. Will start new episodes until this number is reached.
    total_steps = 400
    # Maximum number of steps in a single episode.
    n_steps = 50
    # Whether to initialise the ssm with no data, random data or safe data.
    # One of None, 'random_rollouts' or 'safe_samples'.
    init_mode = 'safe_samples'
    # How many initial samples to give to the ssm.
    n_safe_samples = 10
    # Standard deviation of the initial samples.
    init_sample_std = 0.01
    # Whether to plot the locations of the initial samples given to the ssm.
    plot_initial_samples = False
    # When True, at the end of each episode will plot all the states samples so far.
    plot_states = False
    # Whether to optimize the hyperparameters of a non-parametric model every episode, or just after the initial
    # episode.
    opt_hyp = True

    # -- CemSafeMPC
    mpc_time_horizon = 2
    cem_num_rollouts = 20
    cem_num_elites = 3
    cem_num_iterations = 8
    # Whether to plot the ellipsoids generated during the CEM optimisation process.
    plot_cem_optimisation = False
    plot_cem_terminal_states = False
    # If True then constrain all the states, otherwise just the terminal state.
    use_state_constraint = True
    cem_beta_safety = 3.0
    use_prior_model = True

    # -- MC dropout SSM
    mc_dropout_code_version = 2
    mc_dropout_training_iterations = 2001
    # List giving number of units in each hidden layer.
    mc_dropout_hidden_features = [64, 64]
    # Number of times we will sample the network during the forward pass, to compute the mean + var.
    mc_dropout_num_samples = 30
    # Whether to predict the aleatoric uncertainty as well as computing the epistemic uncertainty. (?)
    mc_dropout_predict_std = True
    # Whether to reinitialize the network weights before training.
    mc_dropout_reinitialize = True
    # Whether to use a fixed dropout probability or concrete dropout. One of 'fixed' or 'concrete'.
    mc_dropout_type = 'concrete'
    # The initial dropout rate, if mc_dropout_type = 'concrete'.
    mc_dropout_concrete_initial_probability = 0.5
    # The dropout rate, if mc_dropout_type = 'fixed'.
    mc_dropout_fixed_probability = 0.1
    # Whether to apply dropout to the input layer.
    mc_dropout_on_input = True
    mc_dropout_lengthscale = 1e-4

    # -- Exact GP SSM
    exact_gp_code_version = 2
    exact_gp_training_iterations = 1000
    # Kernel for the gp. One of 'rbf', 'nn'.
    exact_gp_kernel = 'rbf'
    nn_kernel_layers = [64, 128, 256, 512]


@ex.named_config
def server_config():
    """Contains appropriate flags for when running on a headless gpu server."""
    render = False
    visualize = False


def get_experiment():
    """Returns the Sacred experiment object."""
    return ex
