from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.integrate import ode
from scipy.spatial.qhull import ConvexHull
from torch import Tensor

from .environments import Environment

try:
    import pygame
    from pygame.rect import Rect

    _has_pygame = True
except:
    _has_pygame = False


class LunarLanderResult(Enum):
    """Describes the result of a step in the lunar lander environment."""
    FLYING = 0
    CRASHED = 1
    LANDED = 2


class LunarLander(Environment):
    """Environment similar to a simplified version of the lunar lander game.

    The coordinate system has y=0 10m above the surface of the moon. The surface of the moon is at y=10.
    state=[vx, vy, x, y]

    """

    def __init__(self, conf, dt=.05, verbosity=1):
        state_init_mean = np.array([0., 0., 0., .5])
        state_init_std = np.array([2.0, 0.5, 0., .5])
        super().__init__(name='LunearLander', n_s=4, n_u=2, dt=dt, init_m=state_init_mean, init_std=state_init_std,
                         plant_noise=np.array([0.001, 0.001, 0.001, 0.001]), u_min=np.array([-5., -5.]),
                         u_max=np.array([5., 5.]), target=None, verbosity=verbosity)
        self.odesolver = ode(self._dynamics)

        self._g = 1.62
        self._m = 1.
        self._n_d = 2
        self._width = conf.lander_env_width
        self._max_speed = 5.
        self._max_landing_speed = 0.5
        self._min_y = -1.
        self._lunar_surface_y = conf.lander_surface_y

        norm_x = np.array([self._max_speed, self._max_speed, self._width / 2, self._lunar_surface_y - self._min_y])
        norm_u = np.array(self.u_max - self.u_min)
        # norm_x = np.array([1., 1., 1., 1.])
        # norm_u = np.array([1., 1.])
        self.norm = [norm_x, norm_u]
        self.inv_norm = [arr ** -1 for arr in self.norm]

        self._init_safety_constraints()

    @property
    def l_mu(self) -> ndarray:
        # TODO: What should this actually be?
        return np.array([.1, .1, .1, .1])

    @property
    def l_sigm(self) -> ndarray:
        # TODO: What should this actually be?
        return np.array([.1, .1, .1, .1])

    def _reset(self):
        self.odesolver.set_initial_value(self.current_state)

    def _dynamics(self, t, state, action):
        velocity = state[:self._n_d]
        position = state[self._n_d:]

        # Make the gravity vary based on x position. The task of the ssm is to learn this.
        # Gravity is +ve because the vertical axis points downwards.
        gravity = self._g  # + 5 * (np.sin(position[0]) + 1)
        # print('gravity = ', gravity)

        dz = np.empty((4, 1))
        dz[0] = action[0] / self._m
        dz[1] = action[1] / self._m + gravity
        # The change in state is the velocity.
        dz[2] = velocity[0]
        dz[3] = velocity[1]
        return dz

    def _jac_dynamics(self):
        jac_acl = np.array([[0., 0., 0., 0., 1. / self._m, 0.],  #
                            [0., 0., 0., 0., 0., 1. / self._m]])
        jac_vel = np.eye(self._n_d, self.n_s + self.n_u)

        return np.vstack((jac_acl, jac_vel))

    def state_to_obs(self, state=None, add_noise=False):
        if state is None:
            state = self.current_state
        noise = 0
        if add_noise:
            noise += self.rng.standard_normal(self.n_s) * np.sqrt(self.plant_noise)

        state_noise = state + noise
        state_norm = state_noise * self.inv_norm[0]

        return state_norm

    def objective_cost_function(self, ps: Tensor) -> Optional[Tensor]:
        # Return the negative height above the surface of the moon, to encourage us to land.
        return - ps[:, -1]

    def get_safety_constraints(self, normalize=True):
        if normalize:
            m_x = np.diag(self.norm[0])
            h_mat_safe = np.dot(self.h_mat_safe, m_x)
        else:
            h_mat_safe = self.h_mat_safe

        return h_mat_safe, self.h_safe, self.h_mat_obs, self.h_obs

    def _check_current_state(self, state: Optional[ndarray] = None) -> Tuple[bool, int]:
        if state is None:
            state = self.current_state

        # Check if the state lies inside the safe polytope i.e. A * x <= b.
        res = np.matmul(self.h_mat_safe, state) - self.h_safe.T
        constraints_satisfied = not (res > 0).any()

        landed = state[-1] >= self._lunar_surface_y

        if not constraints_satisfied:
            result_code = LunarLanderResult.CRASHED.value
            print('Crashed!')
        elif landed:
            result_code = LunarLanderResult.LANDED.value
            print('Landed! at velocity', state[1])
            assert np.abs(state[-1]) <= 0.5, f'Landing velocity too high: {state[1]}'
        else:
            result_code = LunarLanderResult.FLYING.value

        done = not constraints_satisfied or landed
        return done, result_code

    def random_action(self) -> ndarray:
        return self.rng.uniform(self.u_min_norm, self.u_max_norm, self.n_u)

    def _render_env(self, screen, axis: [float], display_width: int, display_height: int):
        # Clear screen to black.
        screen.fill((0, 0, 0))

        screen_height_m = self._lunar_surface_y + 0.5
        surface_from_top = self._lunar_surface_y

        m_in_px = display_height / screen_height_m
        x_shift_px = display_width / 2

        white = (255, 255, 255)

        pygame.draw.rect(screen, white, Rect(0, surface_from_top * m_in_px, display_width,
                                             (screen_height_m - surface_from_top) * m_in_px))

        # Draw left and right of safe area.
        safe_left = (0 - self._width / 2) * m_in_px + x_shift_px
        safe_right = (0 + self._width / 2) * m_in_px + x_shift_px
        pygame.draw.line(screen, white, (safe_left, 0), (safe_left, display_height))
        pygame.draw.line(screen, white, (safe_right, 0), (safe_right, display_height))

        lander_x = self.current_state[2] * m_in_px + x_shift_px
        lander_y = self.current_state[3] * m_in_px
        lander_width_m = 0.2
        lander_width_px = lander_width_m * m_in_px
        pygame.draw.rect(screen, white,
                         Rect(lander_x - (lander_width_px / 2), lander_y - (lander_width_px / 2), lander_width_px,
                              lander_width_px))

    def plot_ellipsoid_trajectory(self, p, q, vis_safety_bounds=True):
        raise NotImplementedError

    def _init_safety_constraints(self):
        top = self._min_y
        bottom = self._lunar_surface_y + 0.3
        left = - self._width / 2
        right = self._width / 2

        max_speed_x = self._max_speed
        max_speed_top = self._max_speed

        # Work out what the speed at the bottom (which is below the surface of the moon) should be so we achieve the
        # correct speed at the surface.
        # max_speed_bottom = (self._lunar_surface_y - top) / (bottom - top) * self._max_landing_speed
        max_speed_bottom = self._max_landing_speed - ((max_speed_top - self._max_landing_speed) / (bottom - top)) * (
                bottom - self._lunar_surface_y)

        vertices = []
        for vel_x in (max_speed_x, -max_speed_x):
            for vel_y_sign in (-1, 1):
                for x in (left, right):
                    for y in (top, bottom):
                        if y == top:
                            vertices.append([vel_x, vel_y_sign * max_speed_top, x, y])
                        else:
                            vertices.append([vel_x, vel_y_sign * max_speed_bottom, x, y])

        ch = ConvexHull(np.array(vertices))

        # Returns the equation for the convex hull of the corner points s.t. eq = [H,h] with Hx <= -h
        eq = ch.equations
        h_mat_safe = eq[:, :self.n_s]
        h_safe = -eq[:, self.n_s:]  # We want the form Ax <= b , hence A = H, b = -h

        self.h_mat_safe = h_mat_safe
        self.h_safe = h_safe
        self.h_mat_obs = None  # p.asarray([[0.,1.],[0.,-1.]])
        self.h_obs = None  # np.array([.6,.6]).reshape(2,1)

        # arrange the corner points such that it can be ploted via a line plot
        self.corners_polygon = vertices
        self.ch_safety_bounds = ch
