# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import warnings
from ..ssm_gp_base import GPModelBase


class ScalableGPModel(GPModelBase):
    """Scalable GP implementation using spectral approximation
    
    Uses Fourier features to approximate stationary kernels, reducing
    computational complexity from O(N³) to O(NE²) where E is the number
    of features.
    
    Attributes:
        gp_trained (bool): Is set to TRUE once the train() method was called
        n_s (int): number of state dimensions of the dynamic system
        n_u (int): number of action/control dimensions of the dynamic system
        beta (np.ndarray): Posterior coefficients [E × n_s_out]
        inv_K (list): Not used in scalable GP (kept for API compatibility)
        z (np.ndarray): Training inputs
        hyp (list[dict]): List of hyperparameter dictionaries
        n_frequencies (int): Number of Fourier frequencies to use
        decay_type (str): Type of spectral density decay ('polynomial', 'exponential', 'individual')
    """
    
    def __init__(self, n_s_out, n_s_in, n_u, X=None, y=None, kern_types=None,
                 hyp=None, train=False, n_frequencies=25, period=10.0, decay_types=None):
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
            Number of Fourier frequencies (default: 25)
        period : float, optional
            Period for periodization (default: 10.0)
        decay_types : str, optional
            Spectral decay type: 'polynomial', 'exponential', or 'individual' (default: 'polynomial')
        """

        self.n_s_out = n_s_out
        self.n_s_in = n_s_in
        self.n_u = n_u
        self.input_dim = n_s_in + n_u
        self.gp_trained = False
        
        # Scalable GP specific attributes
        self.n_frequencies = n_frequencies
        self.period = period
        self.decay_types = decay_types
        
        # Spectral parameters
        self.k_values = None
        self.n_features = None
        self.omega_frequencies = None
        self.lambdas = None
        
        # GP parameters (for compatibility with base class)
        self.beta = None
        self.inv_K = None  # Not used in scalable GP, kept for API compatibility
        self.z = None
        self._init_kernel_function(kern_types, hyp)
        self.hyp = self._create_hyp_dict(self.kern_types)
        self.noise_var = np.zeros((self.n_s_out,))
        
        self.L_PhiT_Phi = [None] * n_s_out
        self.PhiT_y = [None] * n_s_out
        self.posterior_mean_coeffs = [None] * n_s_out
        
        self._init_frequencies()
        
        if X is None or y is None:
            train = False
        
        if train:
            self.train(X, y)
        
        super(ScalableGPModel, self).__init__(n_s_out, n_u)
    
    def _init_frequencies(self):
        """Initialize Fourier frequencies for spectral approximation"""
        self.k_values = np.arange(0, self.n_frequencies + 1)
        self.n_features = 2 * len(self.k_values) - 1
        self.omega_frequencies = self.k_values / self.period
    
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
            if kern_type not in ["rbf", "polynomial_decay"]:
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
                hyp_i["exponential_decay_rate"] = 0.5
                hyp_i["exponential_exponent"] = 2.0
                hyp_i["factor"] = 1.0
            elif kern_types[i] == "polynomial_decay":
                hyp_i["polynomial_exponent"] = 2.0
                hyp_i["factor"] = 1.0
            else:
                raise ValueError("kernel type not supported")
            hyp[i] = hyp_i
        
        return hyp
    
    def _compute_lambdas(self, dim_idx, duplicates=True):
        """Compute spectral decay coefficients (lambdas)
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index
        duplicates : bool, optional
            Whether to duplicate coefficients for sine/cosine pairs (default: True)
        
        Returns
        -------
        lambdas : ndarray
            Spectral decay coefficients
        """
        Q = len(self.omega_frequencies)
        
        lambdas = np.zeros(Q)
        q_vals = np.arange(1, Q)
        if self.kern_types[dim_idx] == "rbf":
            exponential_decay = self.hyp[dim_idx]["exponential_decay_rate"]
            exponential_exponent = self.hyp[dim_idx]["exponential_exponent"]
            factor = self.hyp[dim_idx]["factor"]
            lambdas[0] = factor
            lambdas[1:] = factor * np.exp(-exponential_decay * q_vals ** exponential_exponent)
        elif self.kern_types[dim_idx] == "polynomial_decay":
            polynomial_decay = self.hyp[dim_idx]["polynomial_exponent"]
            factor = self.hyp[dim_idx]["factor"]
            lambdas[0] = factor
            lambdas[1:] = factor / (q_vals**polynomial_decay)
        
        if duplicates:
            dublicated_lambdas = np.zeros(2 * Q - 1)
            dublicated_lambdas[0] = lambdas[0]
            dublicated_lambdas[1::2] = lambdas[1:]
            dublicated_lambdas[2::2] = lambdas[1:]
            return dublicated_lambdas
        return lambdas
    
    @staticmethod
    def _phi_features(X, omega_frequencies, lambdas):
        """Compute Fourier features Φ(X)
        
        Parameters
        ----------
        X : ndarray [N × D]
            Input data
        omega_frequencies : ndarray [Q]
            Fourier frequencies
        lambdas : ndarray [2Q-1] or [Q]
            Spectral coefficients
        
        Returns
        -------
        Phi : ndarray [N × (2Q-1)]
            Fourier feature matrix
        """
        N = X.shape[0]
        Q = len(omega_frequencies)
        n_features = 2 * Q - 1
        
        Phi = np.zeros((N, n_features))
        
        Phi[:, 0] = lambdas[0]
        
        for q in range(1, Q):
            omega_q = 2 * np.pi * omega_frequencies[q]
            phase = np.sum(X * omega_q, axis=1)
            Phi[:, 2*q-1] = lambdas[2*q-1] * np.cos(phase)
            Phi[:, 2*q] = lambdas[2*q] * np.sin(phase)
        
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
        
        # Store training data
        self.z = X
        self.x_train = X
        self.y_train = y
        
        if opt_hyp:
            for i in range(self.n_s_out):
                self._optimize_hyperparameters(X, y[:, i], i)
        else:
            raise NotImplementedError("Default hyperparameters not implemented")
        
        n_beta = self.n_features
        beta = np.empty((n_beta, self.n_s_out))
        
        for i in range(self.n_s_out):
            self._compute_posterior_params(X, y[:, i], i)
            beta[:, i] = self.posterior_mean_coeffs[i].reshape(-1)
        
        self.beta = beta
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
        # Initial parameters
        initial_noise_var = 0.04
        
        if self.decay_type == "polynomial":
            initial_factor = 1.0
            initial_params = np.array([initial_factor, initial_noise_var])
        elif self.decay_type == "exponential":
            initial_factor = 1.0
            initial_decay = 1.0
            initial_params = np.array([initial_factor, initial_noise_var, initial_decay])
        elif self.decay_type == "individual":
            initial_lambdas = np.ones(len(self.omega_frequencies))
            initial_params = np.concatenate([initial_lambdas, [initial_noise_var]])
        
        # Optimize in log space for positivity
        initial_params = np.log(initial_params)
        
        result = minimize(
            self._neg_log_marginal_likelihood,
            initial_params,
            args=(X, y, dim_idx),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success:
            # Extract optimized parameters
            params = np.exp(result.x)
            
            if self.decay_type == "polynomial":
                self.polynomial_factor[dim_idx] = params[0]
                self.lambdas[dim_idx] = self._compute_lambdas(dim_idx, factor=params[0])
                self.noise_var[dim_idx] = params[1]
            
            elif self.decay_type == "exponential":
                self.exponential_factor[dim_idx] = params[0]
                self.exponential_decay[dim_idx] = params[2]
                self.lambdas[dim_idx] = self._compute_lambdas(
                    dim_idx, factor=params[0], exponential_decay=params[2]
                )
                self.noise_var[dim_idx] = params[1]
                
                # Approximate lengthscale
                lengthscale_approx = np.sqrt(params[2] / (2 * np.pi**2))
                self.hyp[dim_idx]['lengthscale'] = lengthscale_approx * np.ones(self.input_dim)
            
            elif self.decay_type == "individual":
                self.lambdas[dim_idx] = params[:-1]
                self.noise_var[dim_idx] = params[-1]
        else:
            warnings.warn(
                f"Hyperparameter optimization failed for dimension {dim_idx}: {result.message}"
            )
            # Use default values
            self.lambdas[dim_idx] = self._compute_lambdas(dim_idx, factor=1.0)
            self.noise_var[dim_idx] = 0.04
    
    def _neg_log_marginal_likelihood(self, params, X, y, dim_idx):
        """Negative log marginal likelihood for scalable GP
        
        Uses Woodbury identity for efficient computation:
        log|K| = log|ΦᵀΦ + σ²I| + N*log(σ²)
        
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
        
        # Extract lambdas based on decay type
        if self.decay_type == "polynomial":
            factor = params[0]
            lambdas = self._compute_lambdas(dim_idx, factor=factor)
        elif self.decay_type == "exponential":
            factor = params[0]
            decay = params[2]
            lambdas = self._compute_lambdas(dim_idx, factor=factor, exponential_decay=decay)
        elif self.decay_type == "individual":
            lambdas = params[:-1]
        
        # Compute features
        Phi_train = self._phi_features(X, self.omega_frequencies, lambdas)
        E = Phi_train.shape[1]
        N = len(X)
        
        # A = ΦᵀΦ + σ²I
        A = Phi_train.T @ Phi_train + noise_var * np.eye(E)
        
        try:
            L = np.linalg.cholesky(A)
            
            # Log determinant: log|ΦᵀΦ + σ²I| = 2 * ∑ log(diag(L))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Quadratic form using Woodbury identity
            Phi_T_y = Phi_train.T @ y
            A_inv_Phi_T_y = np.linalg.solve(L, Phi_T_y)
            A_inv_Phi_T_y = np.linalg.solve(L.T, A_inv_Phi_T_y)
            
            y_T_y = np.sum(y**2)
            quadratic_term = (1/noise_var) * (y_T_y - (Phi_T_y.T @ A_inv_Phi_T_y))
            
            # Negative log marginal likelihood
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
        Phi_t = self._phi_features(X, self.omega_frequencies, self.lambdas[dim_idx])
        
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
            Phi_test = self._phi_features(x_new, self.omega_frequencies, self.lambdas[i])
            
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
        
        # Retrain with new data
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
            x = self.z
        
        n_data = np.shape(x)[0]
        inf_gain_x_f = [None] * self.n_s_out
        
        for i in range(self.n_s_out):
            # For scalable GP, compute information gain using feature representation
            Phi = self._phi_features(x, self.omega_frequencies, self.lambdas[i])
            K_approx = Phi @ Phi.T
            
            inf_gain_x_f[i] = np.log(
                np.linalg.det(np.eye(n_data) + (1 / self.noise_var[i]) * K_approx) + 1e-10
            )
        
        return inf_gain_x_f
    
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
        decay_type = gp_dict.get("decay_type", "polynomial")
        
        return cls(n_s_out, n_s_in, n_u, x, y, kern_types, hyp, train,
                   n_frequencies, period, decay_type)
    
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
        gp_dict["beta"] = self.beta
        gp_dict["inv_K"] = self.inv_K
        gp_dict["n_frequencies"] = self.n_frequencies
        gp_dict["period"] = self.period
        gp_dict["decay_type"] = self.decay_type
        gp_dict["lambdas"] = self.lambdas
        gp_dict["noise_var"] = self.noise_var
        
        return gp_dict
