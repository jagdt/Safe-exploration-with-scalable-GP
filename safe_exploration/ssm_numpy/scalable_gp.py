# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
from casadi import horzcat, vertcat, mtimes, solve, sum1, sqrt, fmax, cos, sin, jacobian, SX, Function, reshape, DM
from ..ssm_gp_base import GPModelBase
from .gp_bounds import ScalableGPBounds


class ScalableGPModel(GPModelBase):
    """Scalable GP implementation using spectral approximation
    
    Uses Fourier features to approximate stationary kernels, reducing
    computational complexity from O(N³) to O(NE²) where E is the number
    of features.
    
    Attributes:
        gp_trained (bool): Is set to TRUE once the train() method was called
        n_s (int): number of state dimensions of the dynamic system
        n_u (int): number of action/control dimensions of the dynamic system
        hyp (list[dict]): List of hyperparameter dictionaries
        n_frequencies (int): Number of trigonometric frequencies to use per dimension
        truncation_radius (float): Radius for ellipsoidal frequency truncation
        truncation_target (float): Target value for projection error in adaptive truncation
    """
    
    def __init__(self, n_s_out, n_s_in, n_u, X=None, y=None, kern_types=None,
                 hyp=None, train=False, n_frequencies=25, periods=10.0, domain_lengths=None, lengthscale_multiple=None,
                 use_global_opt_first=False, truncation_radius=12.0, truncation_target=1e-6, rkhs_norm=None, seed=None):
        """Initialize Scalable GP Model
        
        Parameters
        ----------
        n_s_out : int
            Number of output dimensions
        n_s_in : int
            Number of state input dimensions
        n_u : int
            Number of control input dimensions
        X : np.ndarray[float], optional
            Training inputs
        y : np.ndarray[float], optional
            Training targets
        kern_types : list[str], optional
            Kernel types (currently only supports 'rbf' equivalent via spectral approximation)
        hyp : list[dict], optional
            Hyperparameters for each kernel
        train : bool, optional
            Whether to train immediately
        n_frequencies : int, optional
            Maximum number of frequencies per dimension. The ellipsoid is sized such that
            each dimension gets maximum this many frequencies. (default: 25)
        period : float, sequence, or array, optional
            Period for periodization of Fourier features (default: 10.0).
            Accepted formats:
                - scalar: same period for all outputs and inputs
                - sequence of length input_dim: shared per-input periods across outputs
                - array of shape (n_s_out, input_dim): per-output, per-input periods
        truncation_radius : float, optional
            Radius r for ellipsoidal frequency truncation. Frequencies satisfy q^T A_tilde q <= r^2.
            If None, computed from n_frequencies to approximately match the count. (default: None)
        seed : int or None, optional
            Seed for random number generator.
        """

        self.n_s_out = n_s_out
        self.n_s_in = n_s_in
        self.n_u = n_u
        self.input_dim = n_s_in + n_u
        self.gp_trained = False
        
        self.rng = np.random.default_rng(seed)

        # Scalable GP specific attributes
        self.n_frequencies = n_frequencies
        self.truncation_radius = self._init_truncation_radius(truncation_radius)
        self.periods = self._init_periods(periods)
        self.domain_lengths = domain_lengths
        self.lengthscale_multiple = lengthscale_multiple
        self.hyp_optimized = False
        
        # Adaptive truncation control
        self._truncation_target = truncation_target
        self.rkhs_norms = self._init_truncation_radius(rkhs_norm)

        # Optimization settings
        self.use_global_opt_first = use_global_opt_first

        # Spectral parameters
        self.omegas = [None] * n_s_out
        self.lambdas = [None] * n_s_out
        
        self._init_frequencies()
        
        self._init_kernel_function(kern_types, hyp)
        self.hyp = self._create_hyp_dict(self.kern_types)
        self.noise_var = 1e-7 * np.ones((self.n_s_out,))

        self.beta_safety_per_dim = None
        self.projection_error_per_dim = None
        
        self.L_PhiT_Phi = [None] * n_s_out
        self.PhiT_y = [None] * n_s_out
        self.posterior_mean_coeffs = [None] * n_s_out
        
        if X is None or y is None:
            train = False
        
        if train:
            self.train(X, y)
        
        super(ScalableGPModel, self).__init__(n_s_out, n_u)
    
    def _init_periods(self, period):
        """Normalize the user period argument to an (n_s_out × input_dim) array."""
        if isinstance(period, (int, float)):
            return np.full((self.n_s_out, self.input_dim), float(period))

        period_array = np.array(period, dtype=float)

        if period_array.ndim == 1:
            if period_array.size != self.input_dim:
                raise ValueError(
                    f"period length ({period_array.size}) must match input_dim ({self.input_dim})"
                )
            return np.tile(period_array, (self.n_s_out, 1))

        if period_array.ndim == 2:
            if period_array.shape != (self.n_s_out, self.input_dim):
                raise ValueError(
                    "period array must have shape (n_s_out, input_dim); "
                    f"got {period_array.shape}"
                )
            return period_array.copy()

        raise ValueError(
            "period must be a scalar, vector of length input_dim, or "
            "(n_s_out × input_dim) array"
        )
    
    def _init_truncation_radius(self, truncation_radius):
        if truncation_radius is None:
            return [None] * self.n_s_out

        if np.isscalar(truncation_radius):
            return [truncation_radius] * self.n_s_out
        else:
            if len(truncation_radius) != self.n_s_out:
                raise ValueError("truncation_radius must be scalar or list of length n_s_out")
            return truncation_radius
    
    def _set_periods_based_on_domain_and_lengthscales(self, max_period_multiple=10.0, dampening_alpha=0.05):
        """Set periods based on domain lengths and lengthscale multiples.

        Returns
        -------
        periods : ndarray [n_s_out × input_dim]
            Updated periods for each output dimension
        """
        if self.domain_lengths is None or self.lengthscale_multiple is None:
            raise ValueError("domain_lengths and lengthscale_multiple must be provided at initialization")

        domain_lengths = np.asarray(self.domain_lengths, dtype=float)
        lengthscale_multiple = float(self.lengthscale_multiple)
        if domain_lengths.ndim != 1 or domain_lengths.size != self.input_dim:
            raise ValueError(
                f"domain_lengths must be a vector of length {self.input_dim}"
            )
        T_max = max_period_multiple * domain_lengths
        
        for dim_idx in range(self.n_s_out):
            print("Old periods for dim", dim_idx, ":", self.periods[dim_idx])
            if self.kern_types[dim_idx] == "rbf":
                decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
            elif self.kern_types[dim_idx] == "sum_lin_rbf":
                decay_rates = self.hyp[dim_idx]["rbf.exponential_decay_rates"]
            else:
                raise NotImplementedError(
                    "Adjusting periods based on lengthscales only implemented for 'rbf' and 'sum_lin_rbf' kernels."
                )
            lengthscales = np.sqrt(decay_rates * self.periods[dim_idx]**2 / (2 * np.pi**2))
            T_target = lengthscale_multiple * lengthscales + domain_lengths
            T_damped = (1 - dampening_alpha) * self.periods[dim_idx] + dampening_alpha * T_target
            T_bounded = np.minimum(T_damped, T_max)
            self.periods[dim_idx] = T_bounded
            print("New periods for dim", dim_idx, ":", self.periods[dim_idx])
        self._init_frequencies()
    
    def _init_frequencies(self):
        """Initialize trigonometric frequencies for spectral approximation using ellipsoidal half-lattice
        
        Generates frequency vectors ω ∈ ℝ^D from a half-lattice satisfying:
        - q^T A_tilde q <= r^2 (ellipsoidal truncation)
        - One representative from each pair {±q}
        """
        self.omegas = [None] * self.n_s_out
        for dim_idx in range(self.n_s_out):
            self.omegas[dim_idx] = self._create_ellipsoidal_frequencies(
                self.periods[dim_idx], 
                decay_rates=None,
                dim_idx=dim_idx,
                truncation_radius=self.truncation_radius[dim_idx]
            )

    def _create_ellipsoidal_frequencies(self, period_vector, decay_rates, dim_idx, truncation_radius=None):
        """Create ellipsoidal half-lattice of frequency vectors ω for spectral approximation
        
        Generates frequencies from integer lattice Z^d that satisfy:
        1. Ellipsoidal constraint: q^T A_tilde q <= r^2
        2. Half-lattice: only one representative from each pair {±q}
        
        Two modes:
        - decay_rates=None (initial): Isotropic allocation (same max_q per dimension)
        - decay_rates provided (after learning): Anisotropic ellipsoid based on lengthscales
        
        Parameters
        ----------
        period_vector : ndarray [input_dim]
            Periods for each input dimension
        decay_rates : ndarray [input_dim] or None
            Exponential decay rates a_i for each dimension.
            If None, uses isotropic allocation (equal frequencies per dimension).
        truncation_radius : float, optional
            Radius r for ellipsoidal truncation. If None, computed from n_frequencies.
        
        Returns
        -------
        omegas : ndarray [E × input_dim]
            Frequency vectors for Fourier features from half-lattice
        """
        if decay_rates is None:
            max_q = np.full(self.input_dim, self.n_frequencies, dtype=int)
            
            A_tilde_diag = np.ones(self.input_dim)
            
            truncation_radius = self.n_frequencies
        
        else:
            A_tilde_diag = decay_rates / (period_vector ** 2)
            
            if truncation_radius is None:
                truncation_radius = self.n_frequencies * np.sqrt(np.min(A_tilde_diag))
                print(f"Computed truncation radius: {truncation_radius}")
            
            max_q = np.ceil(truncation_radius / np.sqrt(A_tilde_diag)).astype(int)
            max_q = np.minimum(max_q, int(self.n_frequencies * 1.5))
        
        ranges = [np.arange(-max_q[i], max_q[i] + 1) for i in range(self.input_dim)]
        grids = np.meshgrid(*ranges, indexing='ij')
        
        q_candidates = np.stack([g.ravel() for g in grids], axis=1)
        
        quad_forms = np.sum(q_candidates * (A_tilde_diag * q_candidates), axis=1)
        mask_ellipsoid = quad_forms <= truncation_radius ** 2
        q_filtered = q_candidates[mask_ellipsoid]
        
        non_zero_mask = q_filtered != 0

        first_nonzero_idx = np.argmax(non_zero_mask, axis=1)
        
        all_zero_rows = ~np.any(non_zero_mask, axis=1)
        
        row_indices = np.arange(len(q_filtered))
        first_nonzero_vals = q_filtered[row_indices, first_nonzero_idx]
        
        flip_mask = (first_nonzero_vals < 0) & ~all_zero_rows
        q_normalized = q_filtered.copy()
        q_normalized[flip_mask] = -q_normalized[flip_mask]
        
        q_unique = np.unique(q_normalized, axis=0)
        
        omegas = q_unique / period_vector
        
        print(f"Created {omegas.shape[0]} frequencies for dimension {dim_idx}.")

        if len(omegas) == 0:
            omegas = np.zeros((1, self.input_dim))
        
        return omegas
    
    def _normalize_to_half_lattice(self, q):
        """Normalize integer vector to half-lattice representative
        
        For each pair {±q}, returns the representative with first non-zero
        component positive. Returns origin unchanged.
        
        Parameters
        ----------
        q : ndarray [input_dim]
            Integer vector
        
        Returns
        -------
        normalized_q : ndarray [input_dim]
            Half-lattice representative
        """
        for i in range(len(q)):
            if q[i] != 0:
                if q[i] < 0:
                    return -q
                else:
                    return q.copy()
        return q.copy()
    
    def _update_frequencies_for_dim(self, dim_idx):
        """Update frequencies for a specific dimension after hyperparameters change
        
        If adaptive truncation is enabled, recomputes the radius to achieve target error.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index
        """
        if self.kern_types[dim_idx] == "rbf":
            decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
        elif self.kern_types[dim_idx] == "sum_lin_rbf":
            decay_rates = self.hyp[dim_idx]["rbf.exponential_decay_rates"]
        else:
            return
        
        if self._truncation_target is not None:
            self.truncation_radius[dim_idx] = self.compute_truncation_radius_from_target_error(
                dim_idx, self._truncation_target
            )

        self.omegas[dim_idx] = self._create_ellipsoidal_frequencies(
            self.periods[dim_idx],
            decay_rates=decay_rates,
            dim_idx=dim_idx,
            truncation_radius=self.truncation_radius[dim_idx]
        )
  
    def _init_kernel_function(self, kern_types=None, hyp=None):
        """Initialize kernel functions based on name
        
        Parameters
        ----------
        kern_types : list[str], optional
            The names of the kernels / decay types for each dimension
        hyp : list[dict], optional
            Hyperparameters for each kernel
        """
        if kern_types is None:
            kern_types = ["rbf"] * self.n_s_out
        
        self.kern_types = kern_types
        
        for kern_type in kern_types:
            if kern_type not in ["rbf", "sum_lin_rbf", "polynomial_decay", "individual"]:
                raise ValueError(f"Unsupported kernel type for ScalableGP: {kern_type}")

    
    def _create_hyp_dict(self, kern_types):
        """Create hyperparameter dict for kernels
        
        Parameters
        ----------
        kern_types : list[str]
            Kernel identifiers
        
        Returns
        -------
        hyp : list[dict]
            List of hyperparameter dictionaries
        """
        hyp = [None] * self.n_s_out
        
        for i in range(self.n_s_out):
            hyp_i = dict()
            if kern_types[i] == "rbf":
                hyp_i["factor"] = 0.01
                hyp_i["exponential_decay_rates"] = 1.0 * np.ones(self.input_dim)
            elif kern_types[i] == "sum_lin_rbf":
                hyp_i["rbf.factor"] = 0.01
                hyp_i["rbf.exponential_decay_rates"] = 10.0 * np.ones(self.input_dim)
                hyp_i["linear.variances"] = 0.01 * np.ones(self.input_dim)
            elif kern_types[i] == "polynomial_decay":
                raise NotImplementedError("Polynomial decay not implemented yet")
            elif kern_types[i] == "individual":
                hyp_i["lambdas"] = np.ones(self.omegas[i].shape[0])
            else:
                raise ValueError("kernel type not supported")
            hyp[i] = hyp_i
        
        return hyp
    
    def _compute_lambdas(self, dim_idx, Q=None, r=None):
        """Compute spectral decay coefficients (lambdas)
        
        For RBF kernel, computes λ = C * exp(-0.5 * ω^T A ω) where:
        - C is a scaling factor
        - A is diagonal matrix with exponential_decay_rates
        - ω are the frequency vectors from self.omegas
        
        For sum_lin_rbf kernel, returns [rbf_lambdas, linear_variances] concatenated.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index
        Q : int, optional
            If provided, used to compute truncation radius for theoretical projection error.
        r : float, optional
            If provided, uses this truncation radius instead of computing from Q.
        
        Returns
        -------
        lambdas : ndarray
            Spectral decay coefficients (for sum_lin_rbf, includes linear variances at end)
        """
        if Q is not None or r is not None:
            if self.kern_types[dim_idx] in ["rbf", "sum_lin_rbf"]:
                if self.kern_types[dim_idx] == "rbf":
                    decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
                else:
                    decay_rates = self.hyp[dim_idx]["rbf.exponential_decay_rates"]
                
                if r is not None:
                    truncation_radius = r
                # else:
                #     A_tilde_diag = decay_rates / (self.periods[dim_idx] ** 2)
                #     truncation_radius = Q * np.sqrt(np.min(A_tilde_diag))
                #     max_truncation_radius = 2 * self.n_frequencies * np.sqrt(np.min(A_tilde_diag))
                #     truncation_radius = min(truncation_radius, max_truncation_radius)
                
                omegas = self._create_ellipsoidal_frequencies(
                    self.periods[dim_idx],
                    decay_rates=decay_rates,
                    dim_idx=dim_idx,
                    truncation_radius=truncation_radius
                )
            else:
                raise NotImplementedError("Q-based lambda computation only for rbf/sum_lin_rbf")
        else:
            omegas = self.omegas[dim_idx]
        
        E = omegas.shape[0]
        lambdas = np.zeros(E)
        
        if self.kern_types[dim_idx] == "rbf":
            factor = self.hyp[dim_idx]["factor"]
            exponential_decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
            quadratic_forms = -0.5 * np.sum(exponential_decay_rates * omegas ** 2, axis=1)
            lambdas = factor * np.exp(quadratic_forms)
        elif self.kern_types[dim_idx] == "sum_lin_rbf":
            factor = self.hyp[dim_idx]["rbf.factor"]
            exponential_decay_rates = self.hyp[dim_idx]["rbf.exponential_decay_rates"]
            quadratic_forms = -0.5 * np.sum(exponential_decay_rates * omegas ** 2, axis=1)
            rbf_lambdas = factor * np.exp(quadratic_forms)
            linear_variances = self.hyp[dim_idx]["linear.variances"]
            lambdas = np.concatenate([rbf_lambdas, linear_variances])
        elif self.kern_types[dim_idx] == "polynomial_decay":
            raise NotImplementedError("Polynomial decay not implemented yet")
        elif self.kern_types[dim_idx] == "individual":
            lambdas = self.hyp[dim_idx]["lambdas"]
        
        return lambdas
    
    def _phi_features(self, X, lambdas, dim_idx):
        """Compute Fourier features Φ(X) = [λ₀, λ₁·cos(2πω₁ᵀx), λ₂·sin(2πω₁ᵀx), ..., λ₂ₑ₋₁·sin(2πωₑᵀx)]
        
        Creates features using multivariate Fourier basis where each frequency
        ω is a vector in ℝ^D and features are cos(2πωᵀx) and sin(2πωᵀx).
        
        For sum_lin_rbf, lambdas contains [rbf_lambdas, linear_variances] concatenated.
        
        Parameters
        ----------
        X : ndarray [N × D]
            Input data with D dimensions
        lambdas : ndarray [E] or [E + D] for sum_lin_rbf
            Spectral coefficients for each feature
        
        Returns
        -------
        Phi : ndarray [N × (2E - 1)] or [N × (2E - 1 + D)] for sum_lin_rbf
            Fourier feature matrix
        """
        N = X.shape[0]
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]
        
        Phi = np.zeros((N, 2 * E - 1))

        Phi[:, 0] = lambdas[0]

        if E > 1:
            inner_products = 2 * np.pi * (omegas @ X.T)
            phases = inner_products[1:, :]
            cos_block = (lambdas[1:E, None] * np.cos(phases)).T
            sin_block = (lambdas[1:E, None] * np.sin(phases)).T
            Phi[:, 1::2] = cos_block
            Phi[:, 2::2] = sin_block
        
        if self.kern_types[dim_idx] == "sum_lin_rbf":
            linear_variances = lambdas[E:]
            linear_features = X * np.sqrt(linear_variances)
            Phi = np.hstack([Phi, linear_features])
        
        return Phi

    def train(self, X, y, opt_hyp=True):
        """Train a scalable GP for each state dimension
        
        Parameters
        ----------
        X : ndarray [N × (n_s_in + n_u)]
            Training inputs
        y : ndarray [N × n_s_out]
            Training targets
        opt_hyp : bool, optional
            If True, optimize hyperparameters (default: True)
        """
        n_data, _ = np.shape(X)
        
        self.x_train = X
        self.y_train = y
        
        if opt_hyp:
            if self.hyp_optimized and self.domain_lengths is not None and self.lengthscale_multiple is not None:
                # self._set_periods_based_on_domain_and_lengthscales()
                pass
            for i in range(self.n_s_out):
                self._optimize_hyperparameters(X, y[:, i], i)
            self.hyp_optimized = True
        
        for i in range(self.n_s_out):
            self.lambdas[i] = self._compute_lambdas(i)
            self._compute_posterior_params(X, y[:, i], i)
        self.gp_trained = True
    
    def _optimize_hyperparameters(self, X, y, dim_idx, max_iter=1000):
        """Optimize hyperparameters for a single output dimension
        
        Uses multi-start L-BFGS-B optimization to escape local minima.
        On first training pass, can optionally use differential evolution
        for global optimization.
        
        For sum_lin_rbf kernel, uses staged optimization:
        1. Optimize linear component first
        2. Optimize RBF component with linear fixed
        
        Parameters
        ----------
        X : ndarray [N × (n_s_in + n_u)]
            Training inputs
        y : ndarray [N]
            Training targets for one output dimension
        dim_idx : int
            Index of the output dimension being optimized
        max_iter : int, optional
            Maximum number of optimization iterations
        """
        try:
            print(f"[Dim {dim_idx}] Precomputing feature matrices for optimization...")
            C = self._compute_c_matrix(X, dim_idx)
            M_cache = C.T @ C
            v_cache = C.T @ y
            cache = (M_cache, v_cache)
            print(f"[Dim {dim_idx}] Cache created. Matrix size: {M_cache.shape}")
        except MemoryError:
            warnings.warn(f"[Dim {dim_idx}] Not enough memory for caching. Falling back to stanard optimization.")
            cache = None

        if self.kern_types[dim_idx] == "sum_lin_rbf":
            # Stage 1: Optimize linear component first
            print(f"[Dim {dim_idx}] Stage 1: Optimizing linear component...")
            initial_params = self._pack_hyperparameters(self.hyp[dim_idx], "sum_lin_rbf_linear_only", dim_idx)
            initial_params = np.log(initial_params)
            bounds = self._get_parameter_bounds("sum_lin_rbf_linear_only")
            
            result = minimize(
                self._neg_log_marginal_likelihood,
                initial_params,
                args=(X, y, dim_idx, "sum_lin_rbf_linear_only", cache),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success or result.status == 1:
                optimized_params = np.exp(result.x)
                partial_hyp = self._unpack_hyperparameters(optimized_params[:-1], "sum_lin_rbf_linear_only")
                self.hyp[dim_idx].update(partial_hyp)
                self.noise_var[dim_idx] = optimized_params[-1]
                print(f"[Dim {dim_idx}] Stage 1 succeeded, NLL: {result.fun:.4f}, Reason: {result.message}")
            else:
                warnings.warn(f"[Dim {dim_idx}] Linear optimization failed: {result.message}. Using initial values.")
            
            # Stage 2: Optimize RBF component with global opt and restarts
            print(f"[Dim {dim_idx}] Stage 2: Optimizing RBF component...")
            bounds = self._get_parameter_bounds("sum_lin_rbf_rbf_only")
            
            if self.use_global_opt_first and not self.hyp_optimized:
                best_params, best_nll = self._global_optimize(X, y, dim_idx, bounds, "sum_lin_rbf_rbf_only", max_iter, cache)
            else:
                best_params, best_nll = self._multi_start_optimize(X, y, dim_idx, bounds, "sum_lin_rbf_rbf_only", max_iter, cache)
            
            if best_params is not None:
                optimized_params = np.exp(best_params)
                partial_hyp = self._unpack_hyperparameters(optimized_params[:-1], "sum_lin_rbf_rbf_only")
                self.hyp[dim_idx].update(partial_hyp)
                self.noise_var[dim_idx] = optimized_params[-1]
                self._update_frequencies_for_dim(dim_idx)
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
                print(f"[Dim {dim_idx}] Final hyperparameters: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]:.2e}, NLL: {best_nll:.4f}")
            else:
                warnings.warn(f"[Dim {dim_idx}] RBF optimization failed. Using initial values.")
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
        else:
            bounds = self._get_parameter_bounds(self.kern_types[dim_idx])
            kern_type = self.kern_types[dim_idx]
            
            if self.use_global_opt_first and not self.hyp_optimized:
                print(f"[Dim {dim_idx}] Using differential evolution for global optimization...")
                best_params, best_nll = self._global_optimize(X, y, dim_idx, bounds, kern_type, max_iter, cache)
            else:
                best_params, best_nll = self._multi_start_optimize(X, y, dim_idx, bounds, kern_type, max_iter, cache)
            
            if best_params is not None:
                optimized_params = np.exp(best_params)
                self.hyp[dim_idx] = self._unpack_hyperparameters(optimized_params[:-1], kern_type)
                self.noise_var[dim_idx] = optimized_params[-1]
                self._update_frequencies_for_dim(dim_idx)
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
                print(f"[Dim {dim_idx}] Optimized hyperparameters: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]:.2e}, NLL: {best_nll:.4f}")
            else:
                warnings.warn(
                    f"Hyperparameter optimization failed for dimension {dim_idx}. "
                    f"Using initial hyperparameters."
                )
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
    
    def _compute_c_matrix(self, X, dim_idx):
        """Compute the unscaled feature matrix C
        
        Phi = C @ diag(scaling_vec), where scaling_vec comes from hyperparameters.
        """
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]
        N = X.shape[0]
        
        n_rbf_cols = 2 * E - 1
        n_cols = n_rbf_cols
        if self.kern_types[dim_idx] == "sum_lin_rbf":
            n_cols += self.input_dim
            
        C = np.zeros((N, n_cols))
        
        C[:, 0] = 1.0
        
        if E > 1:
            inner_products = 2 * np.pi * (omegas @ X.T)
            phases = inner_products[1:, :]
            C[:, 1:n_rbf_cols:2] = np.cos(phases).T
            C[:, 2:n_rbf_cols:2] = np.sin(phases).T
            
        if self.kern_types[dim_idx] == "sum_lin_rbf":
            C[:, n_rbf_cols:] = X
            
        return C

    def _make_scaling_vector(self, lambdas, dim_idx):
        """Construct the scaling vector L such that Phi = C @ diag(L)"""
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]
        
        n_rbf_cols = 2 * E - 1
        n_cols = n_rbf_cols
        if self.kern_types[dim_idx] == "sum_lin_rbf":
            n_cols += self.input_dim
            
        L_vec = np.empty(n_cols)
        
        L_vec[0] = lambdas[0]
        if E > 1:
            l_rest = lambdas[1:E]
            L_vec[1:n_rbf_cols:2] = l_rest
            L_vec[2:n_rbf_cols:2] = l_rest
            
        if self.kern_types[dim_idx] == "sum_lin_rbf":
            lin_vars = lambdas[E:]
            L_vec[n_rbf_cols:] = np.sqrt(lin_vars)
            
        return L_vec

    def _multi_start_optimize(self, X, y, dim_idx, bounds, kern_type, max_iter, cache=None):
        """Multi-start L-BFGS-B optimization from random starting points
        
        Runs optimization from multiple random starting points until one succeeds.
        
        Parameters
        ----------
        X : ndarray [N × D]
            Training inputs
        y : ndarray [N]
            Training targets
        dim_idx : int
            Output dimension index
        bounds : list of tuples
            Parameter bounds in log-space
        kern_type : str
            Kernel type
        max_iter : int
            Maximum iterations per optimization run
        cache : tuple, optional
            Precomputed (M, v) 
            
        Returns
        -------
        best_params : ndarray or None
            Best parameters found (log-space), or None if all failed
        best_nll : float
            Best negative log-likelihood found
        """
        best_params = None
        best_nll = np.inf
        
        initial_params = self._pack_hyperparameters(self.hyp[dim_idx], kern_type, dim_idx)
        initial_params_log = np.log(initial_params)
        
        results = []
        
        result = minimize(
            self._neg_log_marginal_likelihood,
            initial_params_log,
            args=(X, y, dim_idx, kern_type, cache),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        if result.success or result.status == 1:
            print(f"[Dim {dim_idx}] Multi-start 0: {result.message}, NLL: {result.fun:.4f}")
            results.append((result.x, result.fun))
        
        bounds_array = np.array(bounds)
        attempt = 0
        max_attempts = 100
        
        while len(results) == 0 and attempt < max_attempts:
            random_params = self.rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
            
            result = minimize(
                self._neg_log_marginal_likelihood,
                random_params,
                args=(X, y, dim_idx, kern_type, cache),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            attempt += 1
            if result.success or result.status == 1:
                print(f"[Dim {dim_idx}] Multi-start {attempt}: {result.message}, NLL: {result.fun:.4f}")
                results.append((result.x, result.fun))
        
        if results:
            best_idx = np.argmin([r[1] for r in results])
            best_params, best_nll = results[best_idx]
            print(f"[Dim {dim_idx}] Multi-start: {len(results)} successful, best NLL: {best_nll:.4f}")
        else:
            print(f"[Dim {dim_idx}] WARNING: All {attempt} optimization attempts failed!")
        
        return best_params, best_nll
    
    def _global_optimize(self, X, y, dim_idx, bounds, kern_type, max_iter, cache=None):
        """Global optimization using differential evolution
        
        Uses differential evolution to find a good global solution, then
        refines with L-BFGS-B. This is more expensive but more robust for
        the first training pass.
        
        Parameters
        ----------
        X : ndarray [N × D]
            Training inputs
        y : ndarray [N]
            Training targets
        dim_idx : int
            Output dimension index
        bounds : list of tuples
            Parameter bounds in log-space
        kern_type : str
            Kernel type
        max_iter : int
            Maximum iterations for refinement
        cache : tuple, optional
            Precomputed (M, v)
            
        Returns
        -------
        best_params : ndarray or None
            Best parameters found (log-space), or None if failed
        best_nll : float
            Best negative log-likelihood found
        """
        result_de = differential_evolution(
            self._neg_log_marginal_likelihood,
            bounds,
            args=(X, y, dim_idx, kern_type, cache),
            strategy='best1bin',
            maxiter=500,
            popsize=15,
            tol=1e-4,
            seed=self.rng,
            polish=False,
            workers=1,
            disp=False
        )
        
        print(f"[Dim {dim_idx}] Diff. evolution NLL: {result_de.fun:.4f}, Reason: {result_de.message}")
        
        result = minimize(
            self._neg_log_marginal_likelihood,
            result_de.x,
            args=(X, y, dim_idx, kern_type, cache),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success or result.status == 1:
            print(f"[Dim {dim_idx}] Refined NLL: {result.fun:.4f}, Reason: {result.message}")
            return result.x, result.fun
        else:
            return result_de.x, result_de.fun
    
    def _neg_log_marginal_likelihood(self, params, X, y, dim_idx, kern_type_override=None, cache=None):
        """Negative log marginal likelihood for scalable GP
        
        Uses Woodbury identity for efficient computation:
        log|K| = log|ΦᵀΦ + σ²I| + N*log(σ²)
        
        Optimizes in log-space for numerical stability and to ensure positivity.
        
        Parameters
        ----------
        params : ndarray
            Hyperparameters in log space
        X : ndarray [N × D]
            Training inputs
        y : ndarray [N]
            Training targets
        dim_idx : int
            Output dimension index
        kern_type_override : str, optional
            Override kernel type for staged optimization (e.g., "sum_lin_rbf_linear_only")
        cache : tuple, optional
            Precomputed (M, v) where M=C^T C and v=C^T y
        
        Returns
        -------
        nll : float
            Negative log marginal likelihood
        """
        params = np.exp(params)
        noise_var = params[-1]
        
        kern_type = kern_type_override if kern_type_override else self.kern_types[dim_idx]
        
        partial_hyp_dict = self._unpack_hyperparameters(params[:-1], kern_type)
        
        if kern_type in ["sum_lin_rbf_linear_only", "sum_lin_rbf_rbf_only"]:
            hyp_dict = self.hyp[dim_idx].copy()
            hyp_dict.update(partial_hyp_dict)
        else:
            hyp_dict = partial_hyp_dict
        
        old_hyp = self.hyp[dim_idx].copy()
        self.hyp[dim_idx] = hyp_dict
        lambdas = self._compute_lambdas(dim_idx)
        self.hyp[dim_idx] = old_hyp
        
        use_cache = cache is not None
        N = len(X)
        
        if use_cache:
            M, v = cache
            L_vec = self._make_scaling_vector(lambdas, dim_idx)
            E = M.shape[0]
            
            # A = Phi^T Phi + sigma^2 I
            # Phi^T Phi = diag(L) @ C^T @ C @ diag(L) = diag(L) @ M @ diag(L)            
            A = M * np.outer(L_vec, L_vec) + noise_var * np.eye(E)
            
            # Phi^T y = diag(L) @ C^T @ y = L_vec * v
            Phi_T_y = L_vec * v

        else:
            Phi_train = self._phi_features(X, lambdas, dim_idx)
            E = Phi_train.shape[1]
            
            # A = ΦᵀΦ + σ²I
            A = Phi_train.T @ Phi_train + noise_var * np.eye(E)
            Phi_T_y = Phi_train.T @ y

        try:
            L = np.linalg.cholesky(A)
            
            # Log determinant: log|ΦᵀΦ + σ²I| = 2 * ∑ log(diag(L))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            A_inv_Phi_T_y = np.linalg.solve(L, Phi_T_y)
            A_inv_Phi_T_y = np.linalg.solve(L.T, A_inv_Phi_T_y)
            
            y_T_y = np.sum(y**2)
            quadratic_term = (1/noise_var) * (y_T_y - (Phi_T_y.T @ A_inv_Phi_T_y))
            
            nll = 0.5 * (log_det + quadratic_term + N * np.log(2 * np.pi * noise_var))
            return float(nll)
        
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
            return 1e10
    
    def _compute_posterior_params(self, X, y, dim_idx):
        """Compute posterior parameters for a single dimension
        
        Parameters
        ----------
        X : ndarray [N × D]
            Training inputs
        y : ndarray [N]
            Training targets
        dim_idx : int
            Output dimension index
        """
        Phi = self._phi_features(X, self.lambdas[dim_idx], dim_idx)
        
        # A = ΦᵀΦ + σ²I
        n_features = Phi.shape[1]
        PhiT_Phi = Phi.T @ Phi + self.noise_var[dim_idx] * np.eye(n_features)
        
        self.L_PhiT_Phi[dim_idx] = np.linalg.cholesky(PhiT_Phi)
        
        self.PhiT_y[dim_idx] = Phi.T @ y
        
        # Solve (ΦᵀΦ + σ²I)⁻¹ Φᵀy
        temp = np.linalg.solve(self.L_PhiT_Phi[dim_idx], self.PhiT_y[dim_idx])
        self.posterior_mean_coeffs[dim_idx] = np.linalg.solve(self.L_PhiT_Phi[dim_idx].T, temp)
    
    def _pack_hyperparameters(self, hyp_dict, kern_type, dim_idx):
        """Pack hyperparameter dict into 1D array for optimization
        
        Parameters
        ----------
        hyp_dict : dict
            Hyperparameter dictionary
        kern_type : str
            Kernel type (including virtual types for staged optimization)
        dim_idx : int
            Output dimension index
            
        Returns
        -------
        hyp_array : ndarray
            1D array of hyperparameters (includes noise variance at end)
        """
        if kern_type == "rbf":
            hyp_array = np.concatenate([
                [hyp_dict["factor"]],
                hyp_dict["exponential_decay_rates"],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == "sum_lin_rbf":
            hyp_array = np.concatenate([
                [hyp_dict["rbf.factor"]],
                hyp_dict["rbf.exponential_decay_rates"],
                hyp_dict["linear.variances"],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == "sum_lin_rbf_linear_only":
            hyp_array = np.concatenate([
                hyp_dict["linear.variances"],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == "sum_lin_rbf_rbf_only":
            hyp_array = np.concatenate([
                [hyp_dict["rbf.factor"]],
                hyp_dict["rbf.exponential_decay_rates"],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == "individual":
            hyp_array = np.concatenate([
                hyp_dict["lambdas"],
                [self.noise_var[dim_idx]]
            ])
        else:
            raise ValueError(f"Unsupported kernel type: {kern_type}")
        
        return hyp_array
    
    def _unpack_hyperparameters(self, hyp_array, kern_type):
        """Unpack 1D array into hyperparameter dict
        
        Parameters
        ----------
        hyp_array : ndarray
            1D array of hyperparameters (without noise variance)
        kern_type : str
            Kernel type (including virtual types for staged optimization)
            
        Returns
        -------
        hyp_dict : dict
            Hyperparameter dictionary (may be partial for virtual types)
        """
        hyp_dict = {}
        
        if kern_type == "rbf":
            hyp_dict["factor"] = hyp_array[0]
            hyp_dict["exponential_decay_rates"] = hyp_array[1:]
        elif kern_type == "sum_lin_rbf":
            n = self.input_dim
            idx = 0
            hyp_dict["rbf.factor"] = hyp_array[idx]
            idx += 1
            hyp_dict["rbf.exponential_decay_rates"] = hyp_array[idx:idx+n]
            idx += n
            hyp_dict["linear.variances"] = hyp_array[idx:]
        elif kern_type == "sum_lin_rbf_linear_only":
            hyp_dict["linear.variances"] = hyp_array
        elif kern_type == "sum_lin_rbf_rbf_only":
            hyp_dict["rbf.factor"] = hyp_array[0]
            hyp_dict["rbf.exponential_decay_rates"] = hyp_array[1:]
        elif kern_type == "individual":
            hyp_dict["lambdas"] = hyp_array
        else:
            raise ValueError(f"Unsupported kernel type: {kern_type}")
        
        return hyp_dict
    
    def _get_parameter_bounds(self, kern_type):
        """Get parameter-specific bounds to prevent degenerate solutions
        
        Parameters
        ----------
        kern_type : str
            Kernel type (including virtual types for staged optimization)
        
        Returns
        -------
        bounds : list of tuples
            [(lower, upper), ...] in log-space for each parameter
        """
        bounds = []
        
        if kern_type == "rbf":
            # Factor: [1e-6, 1e-1]
            bounds.append((-13.8, -2.3))
            # Exponential decay rates: [1e-3, 1e2]
            bounds.extend([(-6.9, 4.6)] * self.input_dim)
            # Noise: [1e-9, 1e-5]
            bounds.append((-20.7, -11.5))
        
        elif kern_type == "sum_lin_rbf":
            # RBF factor: [1e-4, 1e4]
            bounds.append((-9.2, 9.2))
            # RBF exponential decay rates: [1e-4, 1e4]
            bounds.extend([(-9.2, 9.2)] * self.input_dim)
            # Linear variances: [1e-6, 1e-1]
            bounds.extend([(-13.8, -2.3)] * self.input_dim)
            # Noise: [1e-9, 1e-5]
            bounds.append((-20.7, -11.5))
        
        elif kern_type == "sum_lin_rbf_linear_only":
            # Linear variances: [1e-6, 1e-1]
            bounds.extend([(-13.8, -2.3)] * self.input_dim)
            # Noise: [1e-9, 1e-5]
            bounds.append((-20.7, -11.5))
        
        elif kern_type == "sum_lin_rbf_rbf_only":
            # RBF factor: [1e-2, 1e4]
            bounds.append((-4.6, 9.2))
            # RBF exponential decay rates: [1e2, 1e4]
            # bounds.extend([(4.6, 9.2),(-9.2, 9.2),(-9.2, 9.2)])
            # bounds.extend([(4.6, 9.2)] * self.input_dim)
            bounds.extend([(-2.3, 9.2)] * self.input_dim)
            # Noise: [1e-9, 1e-5]
            bounds.append((-20.7, -11.5))
        
        elif kern_type == "individual":
            # Individual lambdas: [1e-6, 1e2]
            n_omegas = self.omegas[0].shape[0] if len(self.omegas) > 0 else 0
            bounds.extend([(-13.8, 4.6)] * n_omegas)
            # Noise: [1e-9, 1e-5]
            bounds.append((-20.7, -11.5))
        
        else:
            raise ValueError(f"Unsupported kernel type: {kern_type}")
        
        return bounds
    
    def predict(self, x_new, quantiles=None, compute_gradients=False):
        """Compute predictive mean and variance at test points
        
        Parameters
        ----------
        x_new : ndarray [T × (n_s_in + n_u)]
            Test inputs
        quantiles : optional
            Not implemented
        compute_gradients : bool, optional
            Whether to compute gradients (default: False)
        
        Returns
        -------
        y_mu_pred : ndarray [T × n_s_out]
            Predictive mean
        y_sigm_pred : ndarray [T × n_s_out]
            Predictive standard deviation
        grad_mu : ndarray [T × n_s_out × input_dim], optional
            Gradients of mean (if compute_gradients=True)
        """
        T = np.shape(x_new)[0]
        y_mu_pred = np.empty((T, self.n_s_out))
        y_sigm_pred = np.empty((T, self.n_s_out))
        
        for i in range(self.n_s_out):
            Phi_test = self._phi_features(x_new, self.lambdas[i], i)
            
            y_mu_pred[:, i] = Phi_test @ self.posterior_mean_coeffs[i]
            
            V = np.linalg.solve(self.L_PhiT_Phi[i], Phi_test.T)
            V = np.linalg.solve(self.L_PhiT_Phi[i].T, V)
            
            variance = self.noise_var[i] * np.sum(Phi_test * V.T, axis=1)
            y_sigm_pred[:, i] = np.sqrt(np.maximum(variance, 1e-10))
        
        if quantiles is not None:
            raise NotImplementedError("Quantiles not implemented for ScalableGPModel")
        
        if compute_gradients:
            raise NotImplementedError("Predictive gradients not implemented")
            grad_mu = self.predictive_gradients(x_new)
            return y_mu_pred, y_sigm_pred, grad_mu
        
        return y_mu_pred, y_sigm_pred
    
    def predictive_gradients(self, x_new, grad_sigma=False):
        """Compute gradients of predictive mean/variance w.r.t. inputs
        
        Parameters
        ----------
        x_new: T x (n_s + n_u) array[float]
            The test inputs to compute the gradients at
        grad_sigma: bool, optional
            Additionaly returns the gradients of the predictive variance w.r.t. the inputs if
            this is set to TRUE
        """
        if grad_sigma:
            raise NotImplementedError("Gradient of sigma not implemented")

        T = np.shape(x_new)[0]

        grad_mu_pred = np.empty((T, self.n_s_out, self.input_dim))

        # for i in range(self.n_s_out):
        # ... compute gradients here ...

        raise NotImplementedError("Predictive gradients not implemented")

        return grad_mu_pred
    
    def update_model(self, x, y, opt_hyp=False, replace_old=True):
        """Update the model with new data
        
        Parameters
        ----------
        x : ndarray [n × (n_s_in + n_u)]
            New training inputs
        y : ndarray [n × n_s_out]
            New training targets
        opt_hyp : bool, optional
            Whether to re-optimize hyperparameters (default: False)
        replace_old : bool, optional
            If True, replace old data; if False, append (default: True)
        """
        if replace_old:
            x_new = x
            y_new = y
        else:
            x_new = np.vstack((self.x_train, x))
            y_new = np.vstack((self.y_train, y))
        
        self.train(x_new, y_new, opt_hyp=opt_hyp)
    
    def sample_from_gp(self, inp, size=10):
        """Sample from GP predictive distribution
        
        Parameters
        ----------
        inp : ndarray [n × (n_s_in + n_u)]
            Test inputs
        size : int, optional
            Number of samples per test point (default: 10)
        
        Returns
        -------
        S : ndarray [n × size × n_s_out]
            Samples from the posterior distribution
        """
        n = np.shape(inp)[0]
        S = np.empty((n, size, self.n_s_out))
        
        mu, sigma = self.predict(inp)
        
        for i in range(self.n_s_out):
            for j in range(size):
                S[:, j, i] = mu[:, i] + sigma[:, i] * self.rng.standard_normal(n)
        
        return S
    
    def information_gain(self, x=None):
        """Mutual information between samples and system
        
        Uses Cholesky decomposition for numerical stability to avoid overflow
        when computing log-determinant of large matrices.
        
        Parameters
        ----------
        x : ndarray, optional
            Test points (if None, use training points)
        
        Returns
        -------
        inf_gain_x_f : list
            Information gain for each output dimension
        """
        if x is None:
            x = self.x_train

        inf_gain_x_f = [None] * self.n_s_out
        for i in range(self.n_s_out):
            noise_var_i = self.noise_var[i]
            Phi = self._phi_features(x, self.lambdas[i], i)
            n_features = Phi.shape[1]
            PhiTPhi = Phi.T @ Phi
            
            # Compute log(det(I + PhiTPhi/noise_var)) using Cholesky for numerical stability
            try:
                A = np.eye(n_features) + (1 / noise_var_i) * PhiTPhi
                L = np.linalg.cholesky(A)
                inf_gain_x_f[i] = 2 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                warnings.warn(f"Cholesky decomposition failed for dimension {i} in information_gain. "
                             f"Matrix may be ill-conditioned. Returning NaN.")
                inf_gain_x_f[i] = np.nan

        return inf_gain_x_f

    def get_bounds(self, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0, projection_error=None):
        """Return a helper object for computing scalable GP confidence bounds."""

        return ScalableGPBounds(
            self,
            delta=delta,
            rkhs_norm=rkhs_norm,
            R_subgaussian=R_subgaussian,
            projection_error=projection_error,
        )
    
    def compute_bounds(self, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0, projection_error=None):
        """Compute β-values and projection-error offsets from current data."""

        if not self.gp_trained:
            raise ValueError("GP must be trained before integrating bounds")

        bounds = self.get_bounds(
            delta=delta,
            rkhs_norm=rkhs_norm,
            R_subgaussian=R_subgaussian,
            projection_error=projection_error,
        )

        self.beta_safety_per_dim = np.array([bounds.beta(dim_idx) for dim_idx in range(self.n_s_out)])
        self.projection_error_per_dim = np.array([bounds.projection_errors[dim_idx] for dim_idx in range(self.n_s_out)])
        print(f"Computed beta_safety_per_dim: {self.beta_safety_per_dim}")

    def compute_truncation_radius_from_target_error(self, dim_idx, target_error):
        """Compute truncation radius to achieve target projection error.
        
        Inverts the projection error formula to solve for r given a target error.
        Uses binary search since the tail integral is monotonically decreasing in r.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        target_error : float
            Desired projection error bound (e.g., 1e-6).
        
        Returns
        -------
        float
            Truncation radius r to achieve approximately the target error.
        """
        print(f"Computing truncation radius for dim {dim_idx} to achieve target error {target_error:.2e}...")   
        kern_type = self.kern_types[dim_idx]
        if kern_type == "rbf":
            C = self.hyp[dim_idx]["factor"]
            decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
        elif kern_type == "sum_lin_rbf":
            C = self.hyp[dim_idx]["rbf.factor"]
            decay_rates = self.hyp[dim_idx]["rbf.exponential_decay_rates"]
        else:
            raise NotImplementedError(f"Projection error not implemented for kernel type {kern_type}")
        
        periods = self.periods[dim_idx]
        A_tilde = decay_rates / (periods ** 2)
        
        B = self.rkhs_norms[dim_idx]
        d = self.input_dim
        
        rho = 0.5 * np.sqrt(np.sum(A_tilde))
        S_d_minus_1 = ScalableGPBounds._compute_sphere_surface_area(d)
        det_A_tilde = np.prod(A_tilde)
        sqrt_det_A_tilde = np.sqrt(det_A_tilde)
        
        # Compute target tail integral from target error
        # target_error = B * sqrt(2*C/sqrt(det) * S * I)
        # I_target = (target_error/B)^2 * sqrt(det) / (2*C*S)
        if B == 0:
            return rho + 1.0
        
        tail_integral_target = (target_error / B) ** 2 * sqrt_det_A_tilde / (2.0 * C * S_d_minus_1)
        
        r_min = rho + 0.5
        r_max = rho + 50.0
        
        integral_max = ScalableGPBounds._tail_integral(r_max - rho, rho, d)
        if integral_max > tail_integral_target:
            print(f"Warning: Target error {target_error} too small, using r_max={r_max}")
            return r_max
        
        integral_min = ScalableGPBounds._tail_integral(r_min - rho, rho, d)
        if integral_min < tail_integral_target:
            print(f"Warning: Target error {target_error} too large, using r_min={r_min}")
            return r_min
        
        tolerance = 1e-9
        print(tail_integral_target)
        print(tolerance * tail_integral_target)
        max_iterations = 50
        for iteration in range(max_iterations):
            r_mid = (r_min + r_max) / 2.0
            integral_mid = ScalableGPBounds._tail_integral(r_mid - rho, rho, d)
            
            if abs(integral_mid - tail_integral_target) < tolerance * tail_integral_target:
                print(f"Truncation radius search converged to r={r_mid:.4f} after {iteration+1} iterations")
                return r_mid
            
            if integral_mid > tail_integral_target:
                r_min = r_mid
            else:
                r_max = r_mid
        
        r_final = (r_min + r_max) / 2.0
        print(f"Truncation radius search converged to r={r_final:.4f} after {max_iterations} iterations")
        return r_final

    def predict_casadi_symbolic(self, x_new, compute_grads=False):
        """Return symbolic CasADi expressions for predictive mean/variance
        
        Uses Fourier feature representation for symbolic computation.
        
        Parameters
        ----------
        x_new : casadi.SX or casadi.MX
            Test input (1 × (n_s_in + n_u))
        compute_grads : bool, optional
            Whether to compute gradients of mean w.r.t. inputs (default: False)
            
        Returns
        -------
        mu_new : casadi expression
            Predictive mean (n_s_out × 1)
        sigma_new : casadi expression  
            Predictive standard deviation (n_s_out × 1)
        jac_mu : casadi expression, optional
            Jacobian of mean w.r.t. inputs (n_s_out × (n_s_in + n_u))
            Only returned if compute_grads=True
        """
        assert x_new.shape[0] == 1, \
            "We only support this for a single input vector right now"
        
        inp = SX.sym("input", x_new.shape)
        
        mu_all = []
        pred_sigma_all = []
        jac_mu_all = []
        
        for i in range(self.n_s_out):
            Phi = self._phi_features_casadi(inp, self.lambdas[i], i)
            
            mu_new = mtimes(Phi, self.posterior_mean_coeffs[i])
            
            V = solve(self.L_PhiT_Phi[i], Phi.T)
            V = solve(self.L_PhiT_Phi[i].T, V)
            variance = self.noise_var[i] * sum1(Phi @ V)
            sigma_new = fmax(variance, 1e-10)
            
            pred_func = Function("pred_func", [inp], [mu_new, sigma_new], ["inp"], ["mu_1", "sigma_1"])
            F_1 = pred_func(inp=x_new)
            mu_1 = F_1["mu_1"]
            mu_all = horzcat(mu_all, mu_1)
            pred_sigma = F_1["sigma_1"]
            pred_sigma_all = horzcat(pred_sigma_all, pred_sigma)
            
            if compute_grads:
                jac_func = pred_func.factory('dmudinp', ['inp'], ['jac:mu_1:inp'])
                F_1_jac = jac_func(inp=x_new)
                jac_mu = F_1_jac['jac_mu_1_inp']
                jac_mu_all = vertcat(jac_mu_all, jac_mu)

        if compute_grads:
            return mu_all.T, pred_sigma_all.T, jac_mu_all
        
        return mu_all.T, pred_sigma_all.T
    
    def _phi_features_casadi(self, X, lambdas, dim_idx):
        """Compute Fourier features symbolically using CasADi
        
        For sum_lin_rbf, lambdas contains [rbf_lambdas, linear_variances] concatenated.
        
        Parameters
        ----------
        X : casadi.SX or casadi.MX [1 × D]
            Input data (single point)
        lambdas : ndarray [E] or [E + D] for sum_lin_rbf
            Spectral coefficients for the current output dimension
        
        Returns
        -------
        Phi : casadi expression [1 × (2E-1)] or [1 × (2E-1+D)] for sum_lin_rbf
            Fourier feature vector
        """
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]

        inner_products = 2 * np.pi * mtimes(omegas, X.T)

        features = [lambdas[0]]
        
        if E > 1:
            phases = inner_products[1:]
            
            lambdas_rest = DM(lambdas[1:E])
            
            cos_vec = lambdas_rest * cos(phases)
            sin_vec = lambdas_rest * sin(phases)

            stacked = vertcat(cos_vec.T, sin_vec.T)
            interleaved = reshape(stacked, 1, 2 * (E - 1))
            
            features.append(interleaved)
        
        Phi_rbf = horzcat(*features)
        
        if self.kern_types[dim_idx] == "sum_lin_rbf":
            linear_variances = lambdas[E:]
            linear_features = X * sqrt(horzcat(*linear_variances))
            return horzcat(Phi_rbf, linear_features)
        
        return Phi_rbf
    
    @classmethod
    def from_dict(cls, gp_dict):
        """Initialize GP using data from a dict
        
        Parameters
        ----------
        gp_dict : dict
            Dictionary containing initialization data
        
        Returns
        -------
        model : ScalableGPModel
            Initialized model
        """
        data_available = False
        y = None
        x = None
        
        if "data_path" in gp_dict and gp_dict["data_path"] is not None:
            data_path = gp_dict["data_path"]
            data = np.load(data_path)
            x = data["S"]
            y = data["y"]
            data_available = True
        elif "x" in gp_dict and "y" in gp_dict:
            x = gp_dict["x"]
            y = gp_dict["y"]
            data_available = True
        else:
            warnings.warn(
                "GP needs either data_path or x,y to be trained. "
                "Instantiating without training."
            )
        
        if "prior_model" in gp_dict:
            prior_model = gp_dict["prior_model"]
            if data_available:
                y = y - prior_model(x)
        
        n_s_in = gp_dict["n_s_in"]
        n_s_out = gp_dict["n_s_out"]
        n_u = gp_dict["n_u"]
        
        kern_types = gp_dict.get("kern_types", None)
        
        train = gp_dict.get("train", False) and data_available
        
        hyp = gp_dict.get("hyp", None)
        
        n_frequencies = gp_dict.get("n_frequencies", 25)
        period = gp_dict.get("period", 10.0)
        
        return cls(n_s_out, n_s_in, n_u, x, y, kern_types, hyp, train,
                   n_frequencies, period)
    
    def to_dict(self):
        """Return a dict summarizing the object
        
        Returns
        -------
        gp_dict : dict
            Dictionary containing model state
        """
        gp_dict = dict()
        gp_dict["x"] = self.x_train
        gp_dict["y"] = self.y_train
        gp_dict["kern_types"] = self.kern_types
        gp_dict["hyp"] = self.hyp
        gp_dict["n_frequencies"] = self.n_frequencies
        gp_dict["period"] = self.periods.tolist()
        gp_dict["lambdas"] = self.lambdas
        gp_dict["noise_var"] = self.noise_var
        
        return gp_dict

    def estimate_true_rkhs_norm(self, X=None, y=None):
        """Estimate the RKHS norm of the true function in the current feature space RKHS.
        
        Parameters
        ----------
        X : ndarray, optional
            Pre-collected state-action samples [N x (n_s + n_u)]
        y : ndarray, optional
            True next states corresponding to X [N x n_s]
            
        Returns
        -------
        rkhs_norms : ndarray [n_s_out]
            Estimated RKHS norm for each output dimension.
        """
        if not self.gp_trained:
            raise ValueError("GP must be trained before estimating RKHS norm.")
        if X is None or y is None:
            raise ValueError("X and y must be provided for RKHS norm estimation.")

        norms = np.zeros(self.n_s_out)
        
        for i in range(self.n_s_out):
            Phi = self._get_features(X, i)
            
            A = Phi.T @ Phi
            b = Phi.T @ y[:, i]

            # Regularization
            lambda_reg = 1e-10
            A_lambda = A + lambda_reg * np.eye(A.shape[0])
            
            try:
                L = np.linalg.cholesky(A_lambda)
                w = np.linalg.solve(L.T, np.linalg.solve(L, b))
                
                norms[i] = np.sqrt(np.sum(w**2))
                
            except np.linalg.LinAlgError:
                warnings.warn(f"Failed to estimate RKHS norm for dim {i} (Cholesky failed)")
                norms[i] = np.nan
        
        return norms
