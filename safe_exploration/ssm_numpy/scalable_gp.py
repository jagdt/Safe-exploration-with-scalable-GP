# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
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
                 hyp=None, train=False, n_frequencies=25, periods=10.0, domain_lengths=None, lengthscale_multiple=None):
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

        # Spectral parameters
        self.n_features = None
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
            if self.kern_types[dim_idx] != "rbf":
                raise NotImplementedError(
                    "Adjusting periods based on lengthscales only implemented for 'rbf' kernel."
                )
            decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
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
        
        Total features: n_features = 2*(E_1 × ... × E_D) - 1
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

        self.omegas = []
        for dim_idx in range(self.n_s_out):
            period_vector = self.periods[dim_idx]
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

            self.omegas.append(np.array(omega_list))
        
        E_total = np.prod(n_frequencies_per_dim)
        self.n_features = 2 * E_total - 1
  
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
            if kern_type not in ["rbf", "polynomial_decay", "individual"]:
                raise ValueError(
                    "kernel type '{}' currently not supported for scalable GP".format(kern_type))

    
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
                hyp_i["factor"] = 1.0
                hyp_i["exponential_decay_rates"] = 1.0 * np.ones(self.input_dim)
            elif kern_types[i] == "polynomial_decay":
                raise NotImplementedError("Polynomial decay not implemented yet")
            elif kern_types[i] == "individual":
                hyp_i["lambdas"] = np.ones(self.omegas[i].shape[0])
            else:
                raise ValueError("kernel type not supported")
            hyp[i] = hyp_i
        
        return hyp
    
    def _compute_lambdas(self, dim_idx):
        """Compute spectral decay coefficients (lambdas)
        
        For RBF kernel, computes λ = C * exp(-0.5 * ω^T A ω) where:
        - C is a scaling factor
        - A is diagonal matrix with exponential_decay_rates
        - ω are the frequency vectors from self.omegas
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index
        
        Returns
        -------
        lambdas : ndarray
            Spectral decay coefficients
        """
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]
        lambdas = np.zeros(E)
        
        if self.kern_types[dim_idx] == "rbf":
            factor = self.hyp[dim_idx]["factor"]
            exponential_decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
            quadratic_forms = -0.5 * np.sum(exponential_decay_rates * omegas ** 2, axis=1)
            lambdas = factor * np.exp(quadratic_forms)
        elif self.kern_types[dim_idx] == "polynomial_decay":
            raise NotImplementedError("Polynomial decay not implemented yet")
        elif self.kern_types[dim_idx] == "individual":
            lambdas = self.hyp[dim_idx]["lambdas"]
        
        return lambdas
    
    def _phi_features(self, X, lambdas, dim_idx):
        """Compute Fourier features Φ(X) = [λ₀, λ₁·cos(2πω₁ᵀx), λ₂·sin(2πω₁ᵀx), ..., λ₂ₑ₋₁·sin(2πωₑᵀx)]
        
        Creates features using multivariate Fourier basis where each frequency
        ω is a vector in ℝ^D and features are cos(2πωᵀx) and sin(2πωᵀx).
        
        Parameters
        ----------
        X : ndarray [N × D]
            Input data with D dimensions
        lambdas : ndarray [E]
            Spectral coefficients for each feature
        
        Returns
        -------
        Phi : ndarray [N × 2E - 1]
            Fourier feature matrix
        """
        N = X.shape[0]
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]
        
        Phi = np.zeros((N, self.n_features))

        Phi[:, 0] = lambdas[0]

        inner_products = 2 * np.pi * (omegas @ X.T)
        if E > 1:
            phases = inner_products[1:, :]
            cos_block = (lambdas[1:, None] * np.cos(phases)).T
            sin_block = (lambdas[1:, None] * np.sin(phases)).T
            Phi[:, 1::2] = cos_block
            Phi[:, 2::2] = sin_block
        
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
            self._compute_posterior_params(X, y[:, i], i)
        self.gp_trained = True
    
    def _optimize_hyperparameters(self, X, y, dim_idx, max_iter=1000):
        """Optimize hyperparameters for a single output dimension
        
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
        initial_noise_var = self.noise_var[dim_idx]
        
        if self.kern_types[dim_idx] == "rbf":
            initial_factor = self.hyp[dim_idx]["factor"]
            initial_decay_rates = self.hyp[dim_idx]["exponential_decay_rates"]
            initial_params = np.concatenate([[initial_factor], initial_decay_rates, [initial_noise_var]])
        elif self.kern_types[dim_idx] == "individual":
            initial_lambdas = self.hyp[dim_idx]["lambdas"]
            initial_params = np.concatenate([initial_lambdas, [initial_noise_var]])
        else:
            raise NotImplementedError(f"Optimization for {self.kern_types[dim_idx]} not implemented")
        
        initial_params = np.log(initial_params)

        result = minimize(
            self._neg_log_marginal_likelihood,
            initial_params,
            args=(X, y, dim_idx),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success:
            params = np.exp(result.x)
            if self.kern_types[dim_idx] == "rbf":
                self.hyp[dim_idx]["factor"] = params[0]
                self.hyp[dim_idx]["exponential_decay_rates"] = params[1:-1]
                self.noise_var[dim_idx] = params[-1]
            elif self.kern_types[dim_idx] == "individual":
                self.hyp[dim_idx]["lambdas"] = params[:-1]
                self.noise_var[dim_idx] = params[-1]
            self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
            print(f"Optimized hyperparameters for dimension {dim_idx}: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]}")
        else:
            warnings.warn(
                f"Hyperparameter optimization failed for dimension {dim_idx}: {result.message}"
            )
            self.lambdas[dim_idx] = self._compute_lambdas(dim_idx)
            self.noise_var[dim_idx] = 0.04
    
    def _neg_log_marginal_likelihood(self, params, X, y, dim_idx):
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
        
        Returns
        -------
        nll : float
            Negative log marginal likelihood
        """
        params = np.exp(params)
        noise_var = params[-1]
        
        if self.kern_types[dim_idx] == "rbf":
            factor = params[0]
            decay_rates = params[1:-1]
            self.hyp[dim_idx]["factor"] = factor
            self.hyp[dim_idx]["exponential_decay_rates"] = decay_rates
            lambdas = self._compute_lambdas(dim_idx)
        elif self.kern_types[dim_idx] == "individual":
            lambdas = params[:-1]
        else:
            raise NotImplementedError(f"Optimization for {self.kern_types[dim_idx]} not implemented")
        
        # Compute features
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
        
        except (np.linalg.LinAlgError, ValueError):
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
        Phi_t = self._phi_features(X, self.lambdas[dim_idx], dim_idx)
        
        # A = ΦᵀΦ + σ²I
        PhiT_Phi = Phi_t.T @ Phi_t + self.noise_var[dim_idx] * np.eye(self.n_features)
        
        self.L_PhiT_Phi[dim_idx] = np.linalg.cholesky(PhiT_Phi)
        
        self.PhiT_y[dim_idx] = Phi_t.T @ y
        
        # Solve (ΦᵀΦ + σ²I)⁻¹ Φᵀy
        temp = np.linalg.solve(self.L_PhiT_Phi[dim_idx], self.PhiT_y[dim_idx])
        self.posterior_mean_coeffs[dim_idx] = np.linalg.solve(self.L_PhiT_Phi[dim_idx].T, temp)
    
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
            Phi = self._phi_features(x, self.lambdas[i], i)
            PhiTPhi = Phi.T @ Phi
            inf_gain_x_f[i] = np.log(
                np.linalg.det(np.eye(self.n_features) + (1 / noise_var_i) * PhiTPhi))

        return inf_gain_x_f

    def get_bounds(self, delta=0.05, R_subgaussian=1.0, projection_error=None):
        """Return a helper object for computing scalable GP confidence bounds."""

        return ScalableGPBounds(
            self,
            delta=delta,
            R_subgaussian=R_subgaussian,
            projection_error=projection_error,
        )
    
    def compute_bounds(self, delta=0.05, R_subgaussian=1.0, projection_error=None):
        """Compute β-values and projection-error offsets from current data."""

        if not self.gp_trained:
            raise ValueError("GP must be trained before integrating bounds")

        bounds = self.get_bounds(
            delta=delta,
            R_subgaussian=R_subgaussian,
            projection_error=projection_error,
        )

        self.beta_safety_per_dim = np.array([bounds.beta(dim_idx) for dim_idx in range(self.n_s_out)])
        self.projection_error_per_dim = np.array([bounds.projection_error(dim_idx) for dim_idx in range(self.n_s_out)])
    
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
        
        Parameters
        ----------
        X : casadi.SX or casadi.MX [1 × D]
            Input data (single point)
        lambdas : ndarray [E]
            Spectral coefficients for the current output dimension
        
        Returns
        -------
        Phi : casadi expression [1 × (2E-1)]
            Fourier feature vector
        """
        omegas = self.omegas[dim_idx]
        E = omegas.shape[0]

        inner_products = 2 * np.pi * mtimes(omegas, X.T)

        features = [lambdas[0]]
        for e in range(1, E):
            features.append(lambdas[e] * cos(inner_products[e]))
            features.append(lambdas[e] * sin(inner_products[e]))
        
        return horzcat(*features)
    
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
