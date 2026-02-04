# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:37:59 2017

@author: tkoller
"""

import warnings
import numpy as np
import time

from .gp_reachability import verify_trajectory_safety, trajectory_inside_ellipsoid
from .safempc_exploration import StaticSafeMPCExploration, DynamicSafeMPCExploration
from .utils import generate_initial_samples, unavailable
from .utils_config import create_env, create_solver
from .visualization import (
    plot_model_error_comparison,
    setup_trajectory_plot,
    add_trajectory_colorbar_and_legend,
    save_trajectory_plot,
    RWTH_BLUE, RWTH_BLACK, RWTH_GREEN, RWTH_ORANGE,
    RWTH_CMAP,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    _has_matplotlib = True
except:
    _has_matplotlib = False


@unavailable(not _has_matplotlib, "matplotlib", conditionals=["visualize, save_vis"])
def run_exploration(conf, visualize=False):
    """ Runs exploration algorithm for static and dynamic exploration

    Implementation of the exploration experiments, where we learn about the underlying system as
    quickly as possible. As in the paper, we consider two settings:
        1. Static Exploration:
            Here, we try to find in each iteration the most informative state,action pair in the state space that
            is part of a feasible return trajectory to the safe set. Hence the gathered samples are not part of a trajectory,
            but we reset the system to a different state every time step
        2. Dynamic Exploration:
            We run the system over "n_iterations" time steps without resetting it and we want to execute the most
            "informative" trajectory on the system. Hence, we do not reset the system.
            In this setting we can decide to use an additional performance trajectory by setting "n_perf > 0" in the config
            (again, see the paper for details)

    Parameters
    ----------
    conf: Config
        The Config class for the exploration setting (see DefaultConfigExploration for details)

    """
    conf.create_savedirs(conf.file_path)
    # Get configs (see DefaultConfigExploration for Details)

    static_exploration = conf.static_exploration
    n_iterations = conf.n_iterations
    visualize = conf.visualize
    save_vis = conf.save_vis
    save_path = conf.save_path
    verify_safety = conf.verify_safety
    n_experiments = conf.n_experiments

    l_inf_gain = []
    l_inf_gain_reference = []  # Information gain with reference kernel
    l_sigm_sum = []
    l_sigm = []
    l_z_all = []
    l_x_next_obs_all = []
    l_x_next_pred = []
    l_x_next_prior = []
    l_timing = []
    l_projection_error = []

    for jj in range(n_experiments):
        env = create_env(conf, conf.env_name, conf.env_options)
        safempc, safe_policy = create_solver(conf, env)
        X, y = generate_initial_samples(env, conf, conf.relative_dynamics, safempc,
                                        safe_policy)
        
        # Time the initial training
        t_initial_train_start = time.time()
        safempc.update_model(X, y, opt_hyp=conf.train_gp, reinitialize_solver=False)
        t_initial_training = time.time() - t_initial_train_start
        x_train_init = safempc.x_train
        
        # Initialize reference GP with ground truth hyperparameters if provided
        reference_gp = None
        if conf.reference_gp:
            print("Initializing reference GP with ground truth hyperparameters...")
            from .ssm_numpy import NumpyGPModel
            
            reference_gp = NumpyGPModel(
                n_s_out=env.n_s,
                n_s_in=env.n_s,
                n_u=env.n_u,
                kern_types=conf.reference_kern_types,
                hyp=conf.reference_hyp,
                train=False
            )
            reference_gp.noise_var = conf.reference_noise_var
            
            reference_gp.train(X, y, opt_hyp=False)
            print(f"Reference GP initialized with fixed hyperparameters")

        if static_exploration:
            exploration_module = StaticSafeMPCExploration(safempc, env, conf.n_restarts_optimizer,
                                                          conf.init_m_initial_data, conf.init_std_initial_data, conf.verbosity)

            if verify_safety:
                warnings.warn("Safety_verification not possible in static mode")
                verify_safety = False
        else:
            exploration_module = DynamicSafeMPCExploration(safempc, env)

        # Initialize some logging variables
        inf_gain = np.empty((n_iterations, env.n_s))
        inf_gain_reference = np.empty((n_iterations, env.n_s)) if reference_gp is not None else None
        sigm_sum = np.empty(
            (n_iterations, 1))  # the sum of the confidence intervals per dimension
        sigm = np.empty((n_iterations, env.n_s))  # the individual confidence intervals
        z_all = np.empty((n_iterations, env.n_s + env.n_u))
        x_next_obs_all = np.empty((n_iterations, env.n_s))
        x_next_pred = np.empty((n_iterations, env.n_s))
        x_next_prior = np.empty((n_iterations, env.n_s))
        
        # Initialize timing tracking
        timing_per_iteration = {
            'initial_training': t_initial_training,
            'mpc_optimization': np.empty(n_iterations),
            'gp_training': np.empty(n_iterations),
            'total': np.empty(n_iterations),
        }

        # Initialize color code for plotting the states using RWTH colors
        if _has_matplotlib and n_iterations > 1:
            # Generate colors for each iteration using unified RWTH colormap
            iter_colors = [RWTH_CMAP((i+1) / max(n_iterations, 1)) for i in range(n_iterations)]
            c_sample = lambda it: iter_colors[it] if it < len(iter_colors) else RWTH_BLUE
        else:
            c_sample = lambda it: RWTH_BLUE

        if visualize or conf.save_vis:
            fig, ax = setup_trajectory_plot(env, exploration_module, conf, n_iterations)
            ell = None
            traj = None

        # Add colorbar and legend
        if (visualize or save_vis) and n_iterations > 1:
            add_trajectory_colorbar_and_legend(fig, ax, conf, exploration_module, 
                                              n_iterations, verify_safety)
            if visualize:
                plt.show(block=False)
                plt.pause(0.5)

        safety_all = None
        inside_ellipsoid = None
        if verify_safety:
            safety_all = np.zeros((n_iterations,), dtype=bool)
            inside_ellipsoid = np.zeros((n_iterations, safempc.n_safe))

        if static_exploration:
            x_i = None  # in static setting we optimize over x_i
        else:
            x_i = env.reset(env.p_origin)

        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}")
            t_iter_start = time.time()
            
            # find the most informative sample
            t_mpc_start = time.time()
            if verify_safety:
                x_i, u_i, feasible, k_fb_all, k_ff_all, p_ctrl, q_all = exploration_module.find_max_variance_verbose(x_i
                                                                                                                     )

                if feasible:
                    h_m_safe_norm, h_safe_norm, h_m_obs_norm, h_obs_norm = env.get_safety_constraints(
                        normalize=True)
                    safety_all[i], x_traj_safe = verify_trajectory_safety(env,
                                                                          x_i.squeeze(),
                                                                          k_fb_all, \
                                                                          k_ff_all,
                                                                          p_ctrl,
                                                                          h_m_safe_norm,
                                                                          h_safe_norm,
                                                                          h_m_obs_norm,
                                                                          h_obs_norm)
                    inside_ellipsoid[i, :] = trajectory_inside_ellipsoid(env,
                                                                         x_i.squeeze(),
                                                                         p_ctrl, q_all,
                                                                         k_fb_all,
                                                                         k_ff_all)
                
                    print(f"  Safety verification: {'SAFE' if inside_ellipsoid[i].all() else 'UNSAFE'}")

                    if visualize or save_vis:
                        if conf.visualize_ellipsoids:
                            if not ell is None:
                                for j in range(len(ell)):
                                    ell[j].remove()
                            ax, ell = env.plot_ellipsoid_trajectory(p_ctrl, q_all, vis_safety_bounds=False, ax=ax,
                                                                    unnormalize=True, color=RWTH_ORANGE)
                        
                        if conf.visualize_safe_trajectory:
                            if traj is not None:
                                for t in traj:
                                    t.remove()
                            
                            # Plot the planned trajectory under optimized control law
                            traj = []
                            if x_traj_safe is not None and len(x_traj_safe) > 0:
                                for j in range(len(x_traj_safe)):
                                    x_unnorm, _ = env.unnormalize(x_traj_safe[j].squeeze())
                                    line, = ax.plot(x_unnorm[0], x_unnorm[1], color=RWTH_GREEN, 
                                                  marker='o', markersize=2, linestyle='')
                                    traj.append(line)
                        
                        fig.canvas.draw()

                        if visualize:
                            plt.show(block=False)
                            plt.pause(0.5)

            else:
                x_i, u_i = exploration_module.find_max_variance(x_i)
            
            t_mpc_end = time.time()
            timing_per_iteration['mpc_optimization'][i] = t_mpc_end - t_mpc_start

            if visualize or save_vis:
                ax = env.plot_state(ax, x=x_i, color=c_sample(i), normalize=False, unnormalize=True)
                fig.canvas.draw()
                if visualize:
                    plt.show(block=False)
                    plt.pause(0.25)

            # Apply to system and observe next state
            # only reset the system to a different state in static mode
            if static_exploration:
                x_next, x_next_obs = env.simulate_onestep(x_i.squeeze(), u_i.squeeze())
            else:
                _, x_next, x_next_obs, _, _ = env.step(u_i.squeeze())
            
            x_next_obs_all[i, :] = x_next_obs

            # gather some information
            z_i = np.vstack((x_i, u_i)).T
            z_all[i] = z_i.squeeze()
            
            mu_next, s2_next = exploration_module.ssm_predict(z_i)
            
            pred_conf = np.sqrt(s2_next)
            sigm[i] = pred_conf.squeeze()
            x_next_prior[i, :] = safempc.eval_prior(x_i.T, u_i.T).squeeze()
            x_next_pred[i, :] = mu_next.squeeze() + safempc.eval_prior(x_i.T,
                                                                       u_i.T).squeeze()
            sigm_sum[i] = np.sum(pred_conf)

            # update model and information gain
            retrain = ((i+1) % conf.retrain_gp_interval == 0) if conf.retrain_gp_interval is not None else False
            t_train_start = time.time()
            exploration_module.update_model(z_i, x_next_obs.reshape((1, env.n_s)),
                                            train=retrain, replace_old=False)
            timing_per_iteration['gp_training'][i] = time.time() - t_train_start
            
            inf_gain[i, :] = exploration_module.get_information_gain()
            
            # Update reference GP and compute reference information gain
            if reference_gp is not None:
                reference_gp.update_model(z_i, x_next_obs.reshape((1, env.n_s)),
                                        opt_hyp=False, replace_old=False)
                inf_gain_reference[i, :] = reference_gp.information_gain()
            
            timing_per_iteration['total'][i] = time.time() - t_iter_start

            x_i = x_next

        if save_vis and save_path is not None:
            save_trajectory_plot(fig, save_path)
            plt.close(fig)

        l_inf_gain += [inf_gain]
        if reference_gp is not None:
            l_inf_gain_reference += [inf_gain_reference]
        l_sigm_sum += [sigm_sum]
        l_sigm += [sigm]
        l_z_all += [z_all]
        l_x_next_obs_all += [x_next_obs_all]
        l_x_next_pred += [x_next_pred]
        l_x_next_prior += [x_next_prior]
        l_timing += [timing_per_iteration]
        
        # Extract projection error if available
        if hasattr(exploration_module.safempc.ssm, 'projection_error_per_dim'):
            proj_error = exploration_module.safempc.ssm.projection_error_per_dim
            if proj_error is not None:
                l_projection_error.append(np.mean(proj_error))
            else:
                l_projection_error.append(np.nan)
        else:
            l_projection_error.append(np.nan)
        
        # Extract GP hyperparameters if available
        gp_hyperparameters = None
        if hasattr(exploration_module.safempc.ssm, 'hyp'):
            gp_hyperparameters = {
                'hyp': exploration_module.safempc.ssm.hyp,
                'noise_var': exploration_module.safempc.ssm.noise_var.tolist() if hasattr(exploration_module.safempc.ssm, 'noise_var') else None
            }

        if not save_path is None:
            # TODO extend saving method for CemSafeMPC
            if not hasattr(exploration_module.safempc, 'ssm'):
                raise AttributeError(
                    f"Cannot save results: The SafeMPC implementation "
                    f"'{type(exploration_module.safempc).__name__}' does not have an 'ssm' attribute. "
                    f"This save function is currently only compatible with SimpleSafeMPC. "
                    f"If using CemSafeMPC, you may need to implement a custom save method or disable saving."
                )
            save_results(save_path, l_sigm_sum, l_sigm, l_inf_gain, l_z_all, l_x_next_obs_all, l_x_next_pred,
                         x_next_prior, exploration_module.safempc.ssm, x_train_init, l_timing, safety_all=safety_all)
        if visualize or save_vis:
            plot_model_error_comparison(exploration_module.safempc, exploration_module.env, 
                                       save_dir=save_path, n_points=50, plot_bounds=conf.plot_bounds,
                                       n_initial_samples=x_train_init.shape[0])
    
    # Return aggregated results
    results = {
        'inf_gain': l_inf_gain,
        'sigm_sum': l_sigm_sum,
        'sigm': l_sigm,
        'z_all': l_z_all,
        'x_next_obs_all': l_x_next_obs_all,
        'x_next_pred': l_x_next_pred,
        'x_next_prior': l_x_next_prior,
        'timing': l_timing,
        'projection_error': l_projection_error,
        'gp_hyperparameters': gp_hyperparameters,
    }
    
    if l_inf_gain_reference:
        results['inf_gain_reference'] = l_inf_gain_reference

    if safety_all is not None:
        results['safety_all'] = safety_all
    if inside_ellipsoid is not None:
        results['inside_ellipsoid'] = inside_ellipsoid
    
    return results

def save_results(save_path, sigm_sum, sigm, inf_gain, z_all, x_next_obs_all,
                 x_next_pred, x_next_prior, gp, x_train_0, timing, safety_all=None):
    """ Create a dictionary from the results and save it """
    results_dict = dict()
    results_dict["sigm_sum"] = sigm_sum
    results_dict["sigm"] = sigm
    results_dict["inf_gain"] = inf_gain
    results_dict["z_all"] = z_all
    results_dict["x_next"] = x_next_obs_all
    results_dict["x_next_pred"] = x_next_pred
    results_dict["x_next_prior"] = x_next_prior
    results_dict["x_train_0"] = x_train_0
    results_dict["timing"] = timing
    
    # Save projection error if available
    if hasattr(gp, 'projection_error_per_dim') and gp.projection_error_per_dim is not None:
        results_dict["projection_error"] = np.mean(gp.projection_error_per_dim)

    if not safety_all is None:
        results_dict["safety_all"] = safety_all

    save_data_path = "{}/res_data".format(save_path)
    np.save(save_data_path, results_dict)

    gp_dict = gp.to_dict()
    save_data_gp_path = "{}/res_gp".format(save_path)
    np.save(save_data_gp_path, gp_dict)

    return results_dict
