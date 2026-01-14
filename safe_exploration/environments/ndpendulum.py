"""Contains the n dimensional inverted pendulum environment."""
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from polytope import polytope
from scipy.integrate import ode
from scipy.spatial.qhull import ConvexHull

from ..utils import assert_shape
from .environments import Environment


class NDPendulum(Environment):
    """N dimensional inverted pendulum environment.

    The pendulum is represented using hyperspherical coordinates, with a fixed value for r. Thus the state is (n-1)
    angles, along with their associated velocities:
                0. d_theta1
                   ...
            n-1-1. d_theta(n-1)
              n-1. theta1
                   ...
        (n-1)*2-1. theta(n-1)
    There are (n-1) actions, each exerting torque in the plane of one of the angles.
    """

    def __init__(self, name: str = "NDPendulum", n: int = 3, l: float = .5, m: float = .15, g: float = 9.82,
                 b: float = 0., dt: float = .05, init_m: Optional[float] = None, init_std: Optional[float] = None,
                 plant_noise: ndarray = np.array([0.01, 0.01, 0.01, 0.01]) ** 2, u_min: float = -1., u_max: float = 1,
                 target: ndarray = np.array([0.0, 0.0]), verbosity: int = 1, norm_x=None, norm_u=None):
        """
        :param name: name of the system
        :param n: number of dimensions, >=3
        :param l: length of the pendulum
        :param m: mass of the pendulum
        :param g: gravitation constant
        :param b: friction coefficient of the system
        :param init_m: [(n-1)*2 x 0] initial state mean
        :param init_std: standard deviation of the start state sample distribution. Note: This is not(!) the uncertainty
         of the state but merely allows for variation in the initial (deterministic) state.
        :param u_min: maximum negative torque applied to the system in any dimension
        :param u_max: maximum positive torque applied to the system in any dimension
        :param target: [(n-1)*2 x 0] target state
        """
        assert b == 0., 'Friction is not supported.'

        # We have n-1 angles, and for each a position and velocity.
        state_dimen = (n - 1) * 2
        # We can exert torque in the plane of each of the angles.
        action_dimen = (n - 1)
        num_angles = n - 1

        u_min = np.array([u_min] * num_angles)
        u_max = np.array([u_max] * num_angles)

        p_origin = np.array([0.0] * state_dimen)

        init_m = init_m if init_m is not None else np.array([0., ] * state_dimen)
        init_std = init_std if init_std is not None else np.array([0.01, ] * state_dimen)

        super().__init__(name, state_dimen, action_dimen, dt, init_m, init_std, plant_noise, u_min, u_max, target,
                         verbosity, p_origin)

        self.odesolver = ode(self._dynamics)
        self.l = l
        self.m = m
        self.g = g
        self.b = b
        self.target = target
        self.target_ilqr = init_m
        self.n = n
        self.num_angles = num_angles

        warnings.warn("Normalization turned off for now. Need to look into it")
        max_deg = 30
        if norm_x is None:
            norm_x = np.array([1.] * state_dimen)  # norm_x = np.array([np.sqrt(g/l), np.deg2rad(max_deg)])

        if norm_u is None:
            norm_u = np.array([1.] * action_dimen)  # norm_u = np.array([g*m*l*np.sin(np.deg2rad(max_deg))])

        self.norm = [norm_x, norm_u]
        self.inv_norm = [arr ** -1 for arr in self.norm]

        self._init_safety_constraints()

        raise NotImplementedError('NDPendulum doesn\'t work properly yet!')

    @property
    def l_mu(self) -> ndarray:
        return np.array(([0.05] * self.num_angles) + ([.02] * self.num_angles))

    @property
    def l_sigm(self) -> ndarray:
        return np.array(([0.05] * self.num_angles) + ([.02] * self.num_angles))

    def _reset(self):
        self.odesolver.set_initial_value(self.current_state, 0.0)

    def _check_current_state(self, state=None):
        if state is None:
            state = self.current_state

        # Check if the state lies inside the safe polytope i.e. A * x <= b.
        res = np.matmul(self.h_mat_safe, state) - self.h_safe.T
        satisfied = not (res > 0).any()
        # We don't use the status code.
        status_code = 0
        return not satisfied, status_code

    def _dynamics(self, t, state, action):
        """ Evaluate the system dynamics

        Parameters
        ----------
        t: float
            Input Parameter required for the odesolver for time-dependent
            odes. Has no influence in this system.
        state: n_sx1 array[float]
            The current state of the system
        action: n_ux1 array[float]
            The action to be applied at the current time step

        Returns
        -------
        dz: n_sx1 array[float]
            The ode evaluated at the given inputs.
        """
        assert_shape(state, (self.n_s,))
        assert_shape(action, (self.n_u,))
        velocity = state[:self.num_angles]
        position = state[self.num_angles:]

        gravity_proj = np.zeros_like(position)
        gravity_proj[0] = self.g / self.l * np.sin(position[0])

        inertia = self.m * self.l ** 2

        dvelocity = gravity_proj + action / inertia  # - b / inertia * state[0]
        dposition = velocity

        return np.concatenate((dvelocity.flat, dposition.flat))

    def _jac_dynamics(self):
        """ Evaluate the jacobians of the system dynamics

        Returns
        -------
        jac: (n_s) x (n_s+n_u) array[float]
            The jacobian of the dynamics w.r.t. the state and action

        """
        state = np.zeros((self.n_s,))
        position = state[self.num_angles:]
        theta1 = position[0]

        inertia = self.m * self.l ** 2

        jac_acl = np.array([[0., 0., self.g / self.l * np.cos(theta1), 0., 1/inertia, 0.],  #
                            [0., 0., 0., 0., 0., 1/inertia]])
        jac_vel = np.eye(self.num_angles, self.n_s + self.n_u)
        return np.vstack((jac_acl, jac_vel))

    def state_to_obs(self, state=None, add_noise=False):
        """ Transform the dynamics state to the state to be observed

        Parameters
        ----------
        state: n_sx0 1darray[float]
            The internal state of the system.
        add_noise: bool, optional
            If this is set to TRUE, a noisy observation is returned

        Returns
        -------
        state: 2x0 1darray[float]
            The state as is observed by the agent.
            In the case of the inverted pendulum, this is the same.

        """
        if state is None:
            state = self.current_state
        noise = 0
        if add_noise:
            noise += self.rng.standard_normal(self.n_s) * np.sqrt(self.plant_noise)

        state_noise = state + noise
        state_norm = state_noise * self.inv_norm[0]

        return state_norm

    def random_action(self) -> ndarray:
        c = 0.5
        return c * (self.rng.random(self.n_u) * (self.u_max - self.u_min) + self.u_min)

    def _init_safety_constraints(self):
        """ Get state and safety constraints

        We define the state constraints as:
            x_0 - 3*x_1 <= 1
            x_0 - 3*x_1 >= -1
            x_1 <= max_rad
            x_1 >= -max_rad
        """

        max_dx = 2.0
        max_theta1_deg = 20
        max_dtheta1 = 1.2
        max_dtheta1_at_vertical = 0.8

        max_theta1_rad = np.deg2rad(max_theta1_deg)

        # -max_dtheta <dtheta <= max_dtheta
        h_0_mat = np.asarray([[1., 0.], [-1., 0.]])
        h_0_vec = np.array([max_dtheta1, max_dtheta1])[:, None]

        #  (1/.4)*dtheta + (2/.26)*theta <= 1
        # 2*max_dtheta + c*max_rad <= 1
        # => c = (1+2*max_dtheta) / max_rad
        # for max_deg = 30, max_dtheta = 1.5 => c \approx 7.62
        corners_polygon = np.array([[max_dtheta1, max_dtheta1, -max_theta1_rad, -max_theta1_rad],  #
                                    [max_dtheta1, -max_dtheta1, -max_theta1_rad, max_theta1_rad],  #
                                    [max_dtheta1, max_dtheta1_at_vertical, -max_theta1_rad, 0.],  #
                                    [max_dtheta1, -max_dtheta1_at_vertical, -max_theta1_rad, 0.],  #
                                    [-max_dtheta1, max_dtheta1, max_theta1_rad, -max_theta1_rad],  #
                                    [-max_dtheta1, -max_dtheta1, max_theta1_rad, max_theta1_rad],  #
                                    [-max_dtheta1, max_dtheta1_at_vertical, max_theta1_rad, 0.],  #
                                    [-max_dtheta1, -max_dtheta1_at_vertical, max_theta1_rad, 0.],  #
                                    [max_dtheta1_at_vertical, max_dtheta1, 0., -max_theta1_rad],  #
                                    [max_dtheta1_at_vertical, -max_dtheta1, 0., max_theta1_rad],  #
                                    [max_dtheta1_at_vertical, max_dtheta1_at_vertical, 0., 0.],  #
                                    [max_dtheta1_at_vertical, -max_dtheta1_at_vertical, 0., 0.],  #
                                    [-max_dtheta1_at_vertical, max_dtheta1, 0., -max_theta1_rad],  #
                                    [-max_dtheta1_at_vertical, -max_dtheta1, 0., max_theta1_rad],  #
                                    [-max_dtheta1_at_vertical, max_dtheta1_at_vertical, 0., 0.],  #
                                    [-max_dtheta1_at_vertical, -max_dtheta1_at_vertical, 0., 0.]])

        ch = ConvexHull(corners_polygon)

        # returns the equation for the convex hull of the corner points s.t. eq = [H,h]
        # with Hx <= -h
        eq = ch.equations
        h_mat_safe = eq[:, :self.n_s]
        h_safe = -eq[:, self.n_s:]  # We want the form Ax <= b , hence A = H, b = -h

        p = polytope.qhull(corners_polygon)

        # normalize safety bounds
        self.h_mat_safe = h_mat_safe
        self.h_safe = h_safe
        self.h_mat_obs = None  # p.asarray([[0.,1.],[0.,-1.]])
        self.h_obs = None  # np.array([.6,.6]).reshape(2,1)

        # arrange the corner points such that it can be ploted via a line plot
        self.corners_polygon = corners_polygon
        self.ch_safety_bounds = ch

    def get_safety_constraints(self, normalize=True):
        """ Return the safe constraints

        Parameters
        ----------
        normalize: boolean, optional
            If TRUE: Returns normalized constraints
        """
        if normalize:
            m_x = np.diag(self.norm[0])
            h_mat_safe = np.dot(self.h_mat_safe, m_x)
        else:
            h_mat_safe = self.h_mat_safe

        return h_mat_safe, self.h_safe, self.h_mat_obs, self.h_obs

    def _render_env(self, screen, axis: [float], display_width: int, display_height: int):
        theta = self.current_state[2]
        phi = self.current_state[3]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.elev = 90
        ax1.azim = 90
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.plot([0, 1], [0, 0], [0, 0], color='grey')
        ax1.plot([0, 0], [0, 1], [0, 0], color='grey')
        ax1.plot([0, 0], [0, 0], [0, 1], color='grey')
        ax1.plot([0, x], [0, y], [0, z])
        ax1.scatter([x], [y], [z])

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.elev = 0
        ax2.azim = 90
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.plot([0, 1], [0, 0], [0, 0], color='grey')
        ax2.plot([0, 0], [0, 1], [0, 0], color='grey')
        ax2.plot([0, 0], [0, 0], [0, 1], color='grey')
        ax2.plot([0, x], [0, y], [0, z])
        ax2.scatter([x], [y], [z])

        plt.show()

    #     # Clear screen to black.
    #     screen.fill((0, 0, 0))
    #
    #     center_x = display_width / 2
    #     center_y = display_height / 2
    #
    #     length = min(display_width, display_height) / 3
    #
    #     theta = self.current_state[1]
    #     end_x = center_x - length * math.sin(theta)
    #     end_y = center_y - length * math.cos(theta)
    #
    #     pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 10)
    #     pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (end_x, end_y), width=3)

    def plot_ellipsoid_trajectory(self, p, q, vis_safety_bounds=True):
        raise NotImplementedError
