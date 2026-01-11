# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
from casadi import horzcat, vertcat, mtimes, solve, sum1, sqrt, fmax, cos, sin, jacobian, SX, Function
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
        n_frequencies (int): Number of trigonometric frequencies to use
    """
    
    def __init__(self, n_s_out, n_s_in, n_u, X=None, y=None, kern_types=None,
                 hyp=None, train=False, n_frequencies=25, periods=10.0, domain_lengths=None, lengthscale_multiple=None,
                 n_restarts=2, use_global_opt_first=True):
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
        n_frequencies : int or list[int], optional
            Number of Fourier frequencies per dimension. If int, same number used for all dimensions.
            If list, must have length equal to input_dim (n_s_in + n_u). (default: 25)
        period : float, sequence, or array, optional
            Period for periodization of Fourier features (default: 10.0).
            Accepted formats:
                - scalar: same period for all outputs and inputs
                - sequence of length input_dim: shared per-input periods across outputs
                - array of shape (n_s_out, input_dim): per-output, per-input periods
        """

        self.n_s_out = n_s_out
        self.n_s_in = n_s_in
        self.n_u = n_u
        self.input_dim = n_s_in + n_u
        self.gp_trained = False
        
        # Scalable GP specific attributes
        self.n_frequencies = n_frequencies
        self.n_frequencies_per_dim = None
        self.periods = self._init_periods(periods)
        self.domain_lengths = domain_lengths
        self.lengthscale_multiple = lengthscale_multiple
        self.hyp_optimized = False
        
        # Optimization settings
        self.n_restarts = n_restarts
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
        """Initialize trigonometric frequencies for spectral approximation
        
        Creates frequency vectors ω ∈ ℝ^D for computing features cos(2πω⊤x) and sin(2πω⊤x).
        If n_frequencies is an int E, generates E^D frequency vectors (full grid).
        If n_frequencies is a list [E_1, ..., E_D], generates E_1 × ... × E_D frequency vectors.
        """
        if isinstance(self.n_frequencies, int):
            n_frequencies_per_dim = [self.n_frequencies] * self.input_dim
        else:
            n_frequencies_per_dim = list(self.n_frequencies)
            if len(n_frequencies_per_dim) != self.input_dim:
                raise ValueError(
                    f"n_frequencies length ({len(n_frequencies_per_dim)}) must match "
                    f"input_dim ({self.input_dim})"
                )
        self.n_frequencies_per_dim = n_frequencies_per_dim

        self.omegas = [None] * self.n_s_out
        for dim_idx in range(self.n_s_out):
            self.omegas[dim_idx] = self._create_frequency_grid(n_frequencies_per_dim, self.periods[dim_idx])

    def _create_frequency_grid(self, n_frequencies_per_dim, period_vector):
        """Create grid of frequency vectors ω for spectral approximation
        
        Parameters
        ----------
        n_frequencies_per_dim : list[int]
            Number of frequencies per input dimension
        period_vector : ndarray [input_dim]
            Periods for each input dimension
        
        Returns
        -------
        omegas : ndarray [E × input_dim]
            Frequency vectors for Fourier features
        """
        freq_grids = []
        for i, E_d in enumerate(n_frequencies_per_dim):
            period_val = period_vector[i]
            if period_val <= 0:
                raise ValueError("period values must be positive")
            freq_grids.append(np.arange(0, E_d) / period_val)

        mesh = np.meshgrid(*freq_grids, indexing='ij')

        omega_list = []
        for idx in np.ndindex(*[len(g) for g in freq_grids]):
            omega = np.array([mesh[d][idx] for d in range(self.input_dim)])
            omega_list.append(omega)

        return np.array(omega_list)
  
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
    
    def _compute_lambdas(self, dim_idx, Q=None):
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
            If provided, computes lambdas using first Q frequencies for theoretical projection error.
        
        Returns
        -------
        lambdas : ndarray
            Spectral decay coefficients (for sum_lin_rbf, includes linear variances at end)
        """
        if Q is not None:
            n_frequencies_per_dim = [Q] * self.input_dim
            omegas = self._create_frequency_grid(n_frequencies_per_dim, self.periods[dim_idx])
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

        inner_products = 2 * np.pi * (omegas @ X.T)
        if E > 1:
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
        if self.kern_types[dim_idx] == "block_sum_lin_rbf":
            print(f"[Dim {dim_idx}] Stage 1: Optimizing linear component...")
            initial_params = self._pack_hyperparameters(self.hyp[dim_idx], "sum_lin_rbf_linear_only", dim_idx)
            initial_params = np.log(initial_params)
            bounds = self._get_parameter_bounds("sum_lin_rbf_linear_only")
            
            result = minimize(
                self._neg_log_marginal_likelihood,
                initial_params,
                args=(X, y, dim_idx, "sum_lin_rbf_linear_only"),
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success or result.status == 1:
                optimized_params = np.exp(result.x)
                partial_hyp = self._unpack_hyperparameters(optimized_params[:-1], "sum_lin_rbf_linear_only")
                self.hyp[dim_idx].update(partial_hyp)
                self.noise_var[dim_idx] = optimized_params[-1]
                print(f"[Dim {dim_idx}] Linear optimization succeeded: {partial_hyp}, noise={optimized_params[-1]}")
            else:
                warnings.warn(f"[Dim {dim_idx}] Linear optimization failed: {result.message}. Using initial values.")
            
            print(f"[Dim {dim_idx}] Stage 2: Optimizing RBF component...")
            initial_params = self._pack_hyperparameters(self.hyp[dim_idx], "sum_lin_rbf_rbf_only", dim_idx)
            initial_params = np.log(initial_params)
            bounds = self._get_parameter_bounds("sum_lin_rbf_rbf_only")
            
            result = minimize(
                self._neg_log_marginal_likelihood,
                initial_params,
                args=(X, y, dim_idx, "sum_lin_rbf_rbf_only"),
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success or result.status == 1:
                optimized_params = np.exp(result.x)
                partial_hyp = self._unpack_hyperparameters(optimized_params[:-1], "sum_lin_rbf_rbf_only")
                self.hyp[dim_idx].update(partial_hyp)
                self.noise_var[dim_idx] = optimized_params[-1]
                print(f"[Dim {dim_idx}] RBF optimization succeeded: {partial_hyp}, noise={optimized_params[-1]}")
            else:
                warnings.warn(f"[Dim {dim_idx}] RBF optimization failed: {result.message}. Using initial values.")
            
            self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
            print(f"[Dim {dim_idx}] Final hyperparameters: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]}")
        else:
            bounds = self._get_parameter_bounds(self.kern_types[dim_idx])
            kern_type = self.kern_types[dim_idx]
            
            if self.use_global_opt_first and not self.hyp_optimized:
                print(f"[Dim {dim_idx}] Using differential evolution for global optimization...")
                best_params, best_nll = self._global_optimize(X, y, dim_idx, bounds, kern_type, max_iter)
            else:
                best_params, best_nll = self._multi_start_optimize(X, y, dim_idx, bounds, kern_type, max_iter)
            
            if best_params is not None:
                optimized_params = np.exp(best_params)
                self.hyp[dim_idx] = self._unpack_hyperparameters(optimized_params[:-1], kern_type)
                self.noise_var[dim_idx] = optimized_params[-1]
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
                print(f"[Dim {dim_idx}] Optimized hyperparameters: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]:.2e}, NLL: {best_nll:.4f}")
            else:
                warnings.warn(
                    f"Hyperparameter optimization failed for dimension {dim_idx}. "
                    f"Using initial hyperparameters."
                )
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
    
    def _multi_start_optimize(self, X, y, dim_idx, bounds, kern_type, max_iter):
        """Multi-start L-BFGS-B optimization from random starting points
        
        Runs optimization from multiple random starting points within the
        parameter bounds, returning the solution with lowest NLL.
        
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
            args=(X, y, dim_idx, kern_type),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        if result.success or result.status == 1:
            results.append((result.x, result.fun))
        
        bounds_array = np.array(bounds)
        for i in range(self.n_restarts - 1):
            random_params = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])
            
            result = minimize(
                self._neg_log_marginal_likelihood,
                random_params,
                args=(X, y, dim_idx, kern_type),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            if result.success or result.status == 1:
                results.append((result.x, result.fun))
        
        if results:
            best_idx = np.argmin([r[1] for r in results])
            best_params, best_nll = results[best_idx]
            print(f"[Dim {dim_idx}] Multi-start: {len(results)}/{self.n_restarts} successful, best NLL: {best_nll:.4f}")
        
        return best_params, best_nll
    
    def _global_optimize(self, X, y, dim_idx, bounds, kern_type, max_iter):
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
            
        Returns
        -------
        best_params : ndarray or None
            Best parameters found (log-space), or None if failed
        best_nll : float
            Best negative log-likelihood found
        """
        # Run differential evolution
        result_de = differential_evolution(
            self._neg_log_marginal_likelihood,
            bounds,
            args=(X, y, dim_idx, kern_type),
            strategy='best1bin',
            maxiter=200,
            tol=1e-4,
            seed=42 + dim_idx,
            polish=False,
            workers=1,
            disp=False
        )
        
        print(f"[Dim {dim_idx}] Diff. evolution NLL: {result_de.fun:.4f}")
        
        # Refine with L-BFGS-B
        result = minimize(
            self._neg_log_marginal_likelihood,
            result_de.x,
            args=(X, y, dim_idx, kern_type),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success or result.status == 1:
            print(f"[Dim {dim_idx}] Refined NLL: {result.fun:.4f}")
            return result.x, result.fun
        else:
            # Fall back to DE result
            return result_de.x, result_de.fun
    
    def _neg_log_marginal_likelihood(self, params, X, y, dim_idx, kern_type_override=None):
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
        
        Phi_train = self._phi_features(X, lambdas, dim_idx)
        E = Phi_train.shape[1]
        N = len(X)
        
        # A = ΦᵀΦ + σ²I
        A = Phi_train.T @ Phi_train + noise_var * np.eye(E)
        try:
            L = np.linalg.cholesky(A)
            
            # Log determinant: log|ΦᵀΦ + σ²I| = 2 * ∑ log(diag(L))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            Phi_T_y = Phi_train.T @ y
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
            # Factor: [1e-6, 1e2]
            bounds.append((-13.8, 4.6))
            # Exponential decay rates: [1e-3, 1e3]
            bounds.extend([(-6.9, 6.9)] * self.input_dim)
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
        elif kern_type == "sum_lin_rbf":
            # RBF factor: [1e-6, 1e2]
            bounds.append((-13.8, 4.6))
            # RBF exponential decay rates: [1e-3, 1e3]
            bounds.extend([(-6.9, 6.9)] * self.input_dim)
            # Linear variances: [1e-6, 1e1]
            bounds.extend([(-13.8, 2.3)] * self.input_dim)
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
        elif kern_type == "sum_lin_rbf_linear_only":
            # Linear variances: [1e-6, 1e1]
            bounds.extend([(-13.8, 2.3)] * self.input_dim)
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
        elif kern_type == "sum_lin_rbf_rbf_only":
            # RBF factor: [1e-6, 1e2]
            bounds.append((-13.8, 4.6))
            # RBF exponential decay rates: [1e-3, 1e3]
            bounds.extend([(-6.9, 6.9)] * self.input_dim)
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
        elif kern_type == "individual":
            # Individual lambdas: [1e-6, 1e2]
            n_omegas = self.omegas[0].shape[0] if len(self.omegas) > 0 else 0
            bounds.extend([(-13.8, 4.6)] * n_omegas)
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
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
                S[:, j, i] = mu[:, i] + sigma[:, i] * np.random.randn(n)
        
        return S
    
    def information_gain(self, x=None):
        """Mutual information between samples and system
        
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
            Phi = self._phi_features(x, self.lambdas[i], i)#
            n_features = Phi.shape[1]
            PhiTPhi = Phi.T @ Phi
            inf_gain_x_f[i] = np.log(
                np.linalg.det(np.eye(n_features) + (1 / noise_var_i) * PhiTPhi))

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
            sigma_new = sqrt(fmax(variance, 1e-10))
            
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
            lambdas_rest = lambdas[1:E]
            
            cos_features = horzcat(*[lambdas_rest[i] * cos(phases[i]) for i in range(E-1)])
            sin_features = horzcat(*[lambdas_rest[i] * sin(phases[i]) for i in range(E-1)])
            
            interleaved = []
            for i in range(E-1):
                interleaved.append(cos_features[i])
                interleaved.append(sin_features[i])
            
            features.extend(interleaved)
        
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
