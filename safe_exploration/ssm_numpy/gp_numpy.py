# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
from ..ssm_gp_base import KernelGPModel
from .gp_bounds import NumpyGPBounds


class NumpyGPModel(KernelGPModel):
    """ Pure NumPy GP implementation (drop-in replacement for SimpleGPModel)

    Pure NumPy implementation that mirrors SimpleGPModel structure
    but uses NumPy instead of GPy for fair algorithmic comparison to scalable GP implementation.

    Attributes:
        gp_trained (bool): Is set to TRUE once the train() method
            was called.
        n_s (int): number of state dimensions of the dynamic system
        n_u (int): number of action/control dimensions of the dynamic system
        beta (np.ndarray): Posterior coefficients [N × n_s_out]
        L_chol (list): List of Cholesky factors L where L @ L.T = K + σ²I, one per dimension
        z (np.ndarray): Training inputs
        hyp (list[dict]): List of hyperparameter dictionaries
    """

    def __init__(self, n_s_out, n_s_in, n_u, X=None, y=None, kern_types=None,
                 hyp=None, domain_lengths=None, train=False, use_global_opt_first=False, seed=None):
        """ Initialize GP Model (possibly without training set)

        Parameters
        ----------
            X (np.ndarray[float], optional): Training inputs
            y (np.ndarray[float], optional): Training targets
            kern_types (list[str]): a list of pre-specified covariance function types
            hyp (list[dict], optional): hyperparameters for each kernel
            use_global_opt_first (bool): Use differential evolution on first training
            seed (int, optional): Seed for random number generator

        """
        self.n_s_out = n_s_out
        self.n_s_in = n_s_in
        self.n_u = n_u
        self.input_dim = n_s_in + n_u
        self.gp_trained = False
        self.rng = np.random.default_rng(seed)
        self.domain_lengths = domain_lengths
        
        # Optimization settings
        self.use_global_opt_first = use_global_opt_first
        self.hyp_optimized = False

        self.beta = None
        self.L_chol = None
        self.z = None
        self.kern_types = self._init_kernel_function(kern_types, hyp)
        self.hyp = self._create_hyp_dict(self.kern_types)
        self.noise_var = 1e-7 * np.ones((self.n_s_out,))
        self.beta_safety_per_dim = None

        if X is None or y is None:  # initialize without training (no data available)
            train = False

        if train:
            self.train(X, y)

        super(NumpyGPModel, self).__init__(n_s_out, n_u)

    @classmethod
    def from_dict(cls, gp_dict):
        """ Initialize GP using data from a dict

        Initialized the NumpyGPModel from a dictionary containing
        the necessary information.

        Parameters
        ----------
        gp_dict: dict
            The dictionary containing the following entries:

        """
        data_available = False
        y = None
        x = None

        if "data_path" in gp_dict and not gp_dict["data_path"] is None:
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
            warnings.warn("""In order to be trained, GP either needs a data_path or
            the data itself (key 'data_path' or keys 'x' and 'y') -> we just instantiate the GP class, no training""")

        if "prior_model" in gp_dict:
            prior_model = gp_dict["prior_model"]
            if data_available:
                y = y - prior_model(x)

        n_s_in = gp_dict["n_s_in"]
        n_s_out = gp_dict["n_s_out"]
        n_u = gp_dict["n_u"]

        kern_types = None
        if "kern_types" in gp_dict:
            kern_types = gp_dict["kern_types"]

        train = False
        if "train" in gp_dict:
            train = gp_dict["train"]
        train = train and data_available

        hyp = None
        if "hyp" in gp_dict:
            hyp = gp_dict["hyp"]
        
        n_restarts = gp_dict.get("n_restarts", 5)
        use_global_opt_first = gp_dict.get("use_global_opt_first", True)

        return cls(n_s_out, n_s_in, n_u, x, y, kern_types, hyp, train, n_restarts, use_global_opt_first)

    def to_dict(self):
        """ return a dict summarizing the object """
        gp_dict = dict()
        gp_dict["x"] = self.x_train
        gp_dict["y"] = self.y_train
        gp_dict["kern_types"] = self.kern_types
        gp_dict["hyp"] = self.hyp
        gp_dict["beta"] = self.beta
        gp_dict["L_chol"] = self.L_chol

        return gp_dict

    def _init_kernel_function(self, kern_types=None, hyp=None):
        """ Initialize kernel functions based on name. Check if supported.

        Utility function to set up kernels based on their type name.
        Checks if the kernel type is supported.

        Parameters
        ----------
        kern_types: n_s x 0 array_like[str]
            The names of the kernels for each dimension
        hyp: list[dict], optional
            Hyperparameters for each kernel

        """
        if kern_types is None:
            kern_types = [None] * self.n_s_out
            for i in range(self.n_s_out):
                kern_types[i] = "rbf"
        
        for kern_type in kern_types:
            if kern_type not in ["rbf", "mat52", "prod_lin_rbf", "sum_lin_rbf", "lin_mat52"]:
                raise ValueError(
                    "kernel type '{}' not supported".format(kern_type))
        
        return kern_types

    def _create_hyp_dict(self, kern_types):
        """ Create a hyperparameter dict for the kernels

        Parameters
        ----------
        kern_types: list[str]
            The kernel identifiers

        Returns
        -------
        hyp: list[dict]
            A list of dictionaries containing the hyperparameters of the kernel type
            for each dimension.
        """
        hyp = [None] * self.n_s_out

        for i in range(self.n_s_out):
            hyp_i = dict()
            if kern_types[i] == "rbf":
                hyp_i["lengthscale"] = np.ones(self.input_dim)
                hyp_i["variance"] = 0.1
            elif kern_types[i] == "mat52":
                hyp_i["lengthscale"] = np.ones(self.input_dim)
                hyp_i["variance"] = 0.1
            elif kern_types[i] == "sum_lin_rbf":
                hyp_i["rbf.lengthscale"] = np.ones(self.input_dim)
                hyp_i["rbf.variance"] = 0.1
                hyp_i["linear.variances"] = 0.01 * np.ones(self.input_dim)
            elif kern_types[i] == "prod_lin_rbf":
                hyp_i["prod.rbf.lengthscale"] = np.ones(self.input_dim)
                hyp_i["prod.rbf.variance"] = 0.1
                hyp_i["prod.linear.variances"] = 0.01 * np.ones(self.input_dim)
                hyp_i["linear.variances"] = 0.01 * np.ones(self.input_dim)
            elif kern_types[i] == "lin_mat52":
                hyp_i["prod.mat52.lengthscale"] = np.ones(self.input_dim)
                hyp_i["prod.mat52.variance"] = 0.1
                hyp_i["prod.linear.variances"] = 0.01 * np.ones(self.input_dim)
                hyp_i["linear.variances"] = 0.01 * np.ones(self.input_dim)
            else:
                raise ValueError("kernel type not supported")
            hyp[i] = hyp_i
        return hyp

    def _squared_distances(self, X1, X2, lengthscale):
        """Compute scaled squared distances.
        
        Parameters
        ----------
        X1 : ndarray [N × D]
        X2 : ndarray [M × D]
        lengthscale : ndarray [D]
        
        Returns
        -------
        sq_dists : ndarray [N × M]
            Squared distances scaled by lengthscale
        """
        X1_scaled = X1 / lengthscale
        X2_scaled = X2 / lengthscale
        
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1^T*x2
        sq_norms1 = np.sum(X1_scaled**2, axis=1, keepdims=True)
        sq_norms2 = np.sum(X2_scaled**2, axis=1)
        sq_dists = sq_norms1 + sq_norms2 - 2 * X1_scaled @ X2_scaled.T
        
        # Numerical stability: ensure non-negative
        sq_dists = np.maximum(sq_dists, 0.0)
        
        return sq_dists
    
    def _rbf_kernel(self, X1, X2, hyp):
        """RBF (Gaussian) kernel: k(x,x') = σ² exp(-0.5 ||x-x'||²/ℓ²)"""
        sq_dists = self._squared_distances(X1, X2, hyp['lengthscale'])
        return hyp['variance'] * np.exp(-0.5 * sq_dists)
    
    def _matern52_kernel(self, X1, X2, hyp):
        """Matérn 5/2 kernel: k(x,x') = σ²(1 + √5*r + 5*r²/3)exp(-√5*r)"""
        sq_dists = self._squared_distances(X1, X2, hyp['lengthscale'])
        r = np.sqrt(sq_dists + 1e-12)  # Add small epsilon for numerical stability
        sqrt5_r = np.sqrt(5) * r
        
        return hyp['variance'] * (1.0 + sqrt5_r + 5.0 * sq_dists / 3.0) * np.exp(-sqrt5_r)
    
    def _linear_kernel(self, X1, X2, hyp):
        """Linear kernel: k(x,x') = x^T Σ x'"""
        variances = hyp.get('variances', np.ones(X1.shape[1]))
        X1_scaled = X1 * np.sqrt(variances)
        X2_scaled = X2 * np.sqrt(variances)
        return X1_scaled @ X2_scaled.T
    
    def compute_kernel(self, X1, X2, kern_type, hyp):
        """Compute kernel matrix based on type.
        
        Parameters
        ----------
        X1 : ndarray [N × D]
        X2 : ndarray [M × D]
        kern_type : str
            One of 'rbf', 'mat52', 'sum_lin_rbf', 'prod_lin_rbf', 'lin_mat52'
        hyp : dict
            Hyperparameters for the kernel
        
        Returns
        -------
        K : ndarray [N × M]
        """
        if kern_type == 'rbf':
            return self._rbf_kernel(X1, X2, hyp)
        elif kern_type == 'mat52':
            return self._matern52_kernel(X1, X2, hyp)
        elif kern_type == 'sum_lin_rbf':
            hyp_rbf = {'lengthscale': hyp['rbf.lengthscale'], 
                       'variance': hyp['rbf.variance']}
            hyp_lin = {'variances': hyp['linear.variances']}
            return (self._rbf_kernel(X1, X2, hyp_rbf) + 
                    self._linear_kernel(X1, X2, hyp_lin))
        elif kern_type == 'prod_lin_rbf':
            hyp_rbf = {'lengthscale': hyp['prod.rbf.lengthscale'], 
                       'variance': hyp['prod.rbf.variance']}
            hyp_lin1 = {'variances': hyp['prod.linear.variances']}
            hyp_lin2 = {'variances': hyp['linear.variances']}
            return (self._linear_kernel(X1, X2, hyp_lin1) * 
                    self._rbf_kernel(X1, X2, hyp_rbf) + 
                    self._linear_kernel(X1, X2, hyp_lin2))
        elif kern_type == 'lin_mat52':
            hyp_mat52 = {'lengthscale': hyp['prod.mat52.lengthscale'], 
                         'variance': hyp['prod.mat52.variance']}
            hyp_lin1 = {'variances': hyp['prod.linear.variances']}
            hyp_lin2 = {'variances': hyp['linear.variances']}
            return (self._linear_kernel(X1, X2, hyp_lin1) * 
                    self._matern52_kernel(X1, X2, hyp_mat52) + 
                    self._linear_kernel(X1, X2, hyp_lin2))
        else:
            raise ValueError(f"Unsupported kernel type: {kern_type}")


    def train(self, X, y, opt_hyp=True):
        """ Train a GP for each state dimension

        Args:
            X: Training inputs of size [N, n_s_in + n_u]
            y: Training targets of size [N, n_s_out]
            opt_hyp: bool, optional. If True, optimize hyperparameters
        """
        n_data, _ = np.shape(X)

        if opt_hyp:
            for i in range(self.n_s_out):
                self._optimize_hyperparameters(X, y[:, i], i)
            self.hyp_optimized = True

        n_beta = n_data
        beta = np.empty((n_beta, self.n_s_out))

        L_chol = [None] * self.n_s_out

        for i in range(self.n_s_out):
            y_i = y[:, i].reshape(-1, 1)
            
            K = self.compute_kernel(X, X, self.kern_types[i], self.hyp[i])
            K += self.noise_var[i] * np.eye(n_beta)
            
            # Add extra jitter
            casadi_jitter = 1e-5 * np.trace(K) / len(K)
            K += casadi_jitter * np.eye(n_beta)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                warnings.warn(f"Cholesky failed for dimension {i}, adding adaptive jitter")
                # Add jitter proportional to trace for better conditioning
                jitter = 1e-6 * np.trace(K) / len(K)
                K += jitter * np.eye(n_beta)
                L = np.linalg.cholesky(K)
            
            L_chol[i] = L
            
            beta[:, i] = np.linalg.solve(L.T, np.linalg.solve(L, y_i)).reshape(-1, )

        self.z = X
        self.L_chol = L_chol
        self.beta = beta
        self.gp_trained = True
        self.x_train = X
        self.y_train = y

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
            # Stage 1: Optimize linear component first
            print(f"[Dim {dim_idx}] Stage 1: Optimizing linear component...")
            initial_params = self._pack_hyperparameters(self.hyp[dim_idx], "sum_lin_rbf_linear_only", dim_idx)
            initial_params = np.log(initial_params)
            bounds = self._get_parameter_bounds("sum_lin_rbf_linear_only")
            
            best_params, best_nll = self._multi_start_optimize(X, y, dim_idx, bounds, "sum_lin_rbf_linear_only", max_iter)
            
            if best_params is not None:
                optimized_params = np.exp(best_params)
                partial_hyp = self._unpack_hyperparameters(optimized_params[:-1], "sum_lin_rbf_linear_only")
                self.hyp[dim_idx].update(partial_hyp)
                self.noise_var[dim_idx] = optimized_params[-1]
                print(f"[Dim {dim_idx}] Stage 1 succeeded, NLL: {best_nll:.4f}")
            else:
                warnings.warn(f"[Dim {dim_idx}] Linear optimization failed. Using initial values.")
            
            # Stage 2: Optimize RBF component with global opt and restarts
            print(f"[Dim {dim_idx}] Stage 2: Optimizing RBF component...")
            bounds = self._get_parameter_bounds("sum_lin_rbf_rbf_only")
            
            if self.use_global_opt_first and not self.hyp_optimized:
                best_params, best_nll = self._global_optimize(X, y, dim_idx, bounds, "sum_lin_rbf_rbf_only", max_iter)
            else:
                best_params, best_nll = self._multi_start_optimize(X, y, dim_idx, bounds, "sum_lin_rbf_rbf_only", max_iter)
            
            if best_params is not None:
                optimized_params = np.exp(best_params)
                partial_hyp = self._unpack_hyperparameters(optimized_params[:-1], "sum_lin_rbf_rbf_only")
                self.hyp[dim_idx].update(partial_hyp)
                self.noise_var[dim_idx] = optimized_params[-1]
                print(f"[Dim {dim_idx}] Final hyperparameters: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]:.2e}, NLL: {best_nll:.4f}")
            else:
                warnings.warn(f"[Dim {dim_idx}] RBF optimization failed. Using initial values.")
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
                print(f"[Dim {dim_idx}] Optimized hyperparameters: {self.hyp[dim_idx]}, noise_var: {self.noise_var[dim_idx]:.2e}, NLL: {best_nll:.4f}")
            else:
                warnings.warn(f"Hyperparameter optimization failed for dimension {dim_idx}. Using initial values.")
    
    def _multi_start_optimize(self, X, y, dim_idx, bounds, kern_type, max_iter):
        """Local-then-global optimization strategy
        
        First attempts local L-BFGS-B optimization from current hyperparameters.
        If that fails, falls back to global differential evolution optimization.
        
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
        initial_params = self._pack_hyperparameters(self.hyp[dim_idx], kern_type, dim_idx)
        initial_params_log = np.log(initial_params)
                
        result = minimize(
            self._neg_log_marginal_likelihood,
            initial_params_log,
            args=(X, y, dim_idx, kern_type),
            method='L-BFGS-B',
            # bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        if result.success or result.status == 1:
            print(f"[Dim {dim_idx}] Local optimization succeeded: {result.message}, NLL: {result.fun:.4f}")
            return result.x, result.fun
        
        print(f"[Dim {dim_idx}] Local optimization failed, trying global optimization...")
        return self._global_optimize(X, y, dim_idx, bounds, kern_type, max_iter)
        
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
            # bounds,
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
            # bounds=bounds,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success or result.status == 1:
            print(f"[Dim {dim_idx}] Refined NLL: {result.fun:.4f}")
            return result.x, result.fun
        else:
            return result_de.x, result_de.fun

    def _neg_log_marginal_likelihood(self, hyp_array_log, X, y, dim_idx, kern_type_override=None):
        """Negative log marginal likelihood
        
        Parameters
        ----------
        hyp_array_log : ndarray
            Hyperparameters in log space
        X : ndarray
            Training inputs
        y : ndarray
            Training targets
        dim_idx : int
            Output dimension index
        kern_type_override : str, optional
            Override kernel type for virtual types in staged optimization
            
        Returns
        -------
        nll : float
            Negative log marginal likelihood (or penalty for invalid parameters)
        """
        hyp_array = np.exp(hyp_array_log)
        
        kern_type = kern_type_override if kern_type_override is not None else self.kern_types[dim_idx]
        partial_hyp = self._unpack_hyperparameters(hyp_array[:-1], kern_type)
        
        if kern_type in ["sum_lin_rbf_linear_only", "sum_lin_rbf_rbf_only"]:
            hyp_dict = self.hyp[dim_idx].copy()
            hyp_dict.update(partial_hyp)
        else:
            hyp_dict = partial_hyp
        
        K = self.compute_kernel(X, X, self.kern_types[dim_idx], hyp_dict)
        noise_var = hyp_array[-1]
        K += noise_var * np.eye(X.shape[0])
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            warnings.warn(f"Cholesky failed for dimension {dim_idx}, adding adaptive jitter")
            # Add jitter proportional to trace for better conditioning
            jitter = 1e-6 * np.trace(K) / len(K)
            K += jitter * np.eye(X.shape[0])
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
        
        # Compute log marginal likelihood
        # log p(y|X,θ) = -0.5*y^T*K^{-1}*y - sum(log(diag(L))) - n/2*log(2π)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_likelihood = -0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2 * np.pi)
        
        return -log_likelihood
        
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
        if kern_type == 'rbf' or kern_type == 'mat52':
            hyp_array = np.concatenate([
                hyp_dict['lengthscale'],
                [hyp_dict['variance']],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == 'sum_lin_rbf':
            hyp_array = np.concatenate([
                hyp_dict['rbf.lengthscale'],
                [hyp_dict['rbf.variance']],
                hyp_dict['linear.variances'],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == 'sum_lin_rbf_linear_only':
            hyp_array = np.concatenate([
                hyp_dict['linear.variances'],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == 'sum_lin_rbf_rbf_only':
            hyp_array = np.concatenate([
                hyp_dict['rbf.lengthscale'],
                [hyp_dict['rbf.variance']],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == 'prod_lin_rbf':
            hyp_array = np.concatenate([
                hyp_dict['prod.rbf.lengthscale'],
                [hyp_dict['prod.rbf.variance']],
                hyp_dict['prod.linear.variances'],
                hyp_dict['linear.variances'],
                [self.noise_var[dim_idx]]
            ])
        elif kern_type == 'lin_mat52':
            hyp_array = np.concatenate([
                hyp_dict['prod.mat52.lengthscale'],
                [hyp_dict['prod.mat52.variance']],
                hyp_dict['prod.linear.variances'],
                hyp_dict['linear.variances'],
                [self.noise_var[dim_idx]]
            ])
        else:
            raise ValueError(f"Unsupported kernel type: {kern_type}")
        
        return hyp_array
    
    def _get_parameter_bounds(self, kern_type):
        """Get parameter-specific bounds to prevent degenerate solutions
        
        Parameters
        ----------
        kern_type : str
            Kernel type
        
        Returns
        -------
        bounds : list of tuples
            [(lower, upper), ...] in log-space for each parameter
        """
        bounds = []
        
        if kern_type == 'rbf' or kern_type == 'mat52':
            # Lengthscales: [1e-3, 1e3]
            bounds.extend([(-6.9, 6.9)] * self.input_dim)
            # Variance: [1e-6, 1e2]
            bounds.append((-13.8, 4.6))
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
        elif kern_type == 'sum_lin_rbf':
            # RBF lengthscales: [1e-6, 1e6]
            bounds.extend([(-13.8, 13.8)] * self.input_dim)
            # RBF variance: [1e-8, 1e6]
            bounds.append((-18.4, 13.8))
            # Linear variances: [1e-8, 1e1]
            bounds.extend([(-18.4, 2.3)] * self.input_dim)
            # Noise: [1e-14, 1e0]
            bounds.append((-32.2, 0.0))
        
        elif kern_type == 'sum_lin_rbf_linear_only':
            # Linear variances: [1e-8, 1e0]
            bounds.extend([(-18.4, 0.0)] * self.input_dim)
            # Noise: [1e-14, 1e0]
            bounds.append((-32.2, 0.0))
        
        elif kern_type == 'sum_lin_rbf_rbf_only':
            # RBF lengthscales: [1e-6, 1e6]
            bounds.extend([(-13.8, 13.8)] * self.input_dim)
            # RBF variance: [1e-8, 1e6]
            bounds.append((-18.4, 13.8))
            # Noise: [1e-14, 1e-3]
            bounds.append((-32.2, -6.9))
        
        elif kern_type == 'prod_lin_rbf' or kern_type == 'lin_mat52':
            # Lengthscales: [1e-3, 1e3]
            bounds.extend([(-6.9, 6.9)] * self.input_dim)
            # Variance: [1e-6, 1e2]
            bounds.append((-13.8, 4.6))
            # Product linear variances: [1e-2, 1e1]
            bounds.extend([(-4.6, 2.3)] * self.input_dim)
            # Additive linear variances: [1e-6, 1e1]
            bounds.extend([(-13.8, 2.3)] * self.input_dim)
            # Noise: [1e-10, 1e0]
            bounds.append((-23.0, 0.0))
        
        return bounds

    def _unpack_hyperparameters(self, hyp_array, kern_type):
        """Unpack 1D array into hyperparameter dict
        
        Parameters
        ----------
        hyp_array : ndarray
            1D array of hyperparameters (without noise variance)
        kern_type : str
            Kernel type
            
        Returns
        -------
        hyp_dict : dict
            Hyperparameter dictionary
        """
        hyp_dict = {}
        
        if kern_type == 'rbf' or kern_type == 'mat52':
            n_lengthscales = self.input_dim
            hyp_dict['lengthscale'] = hyp_array[:n_lengthscales]
            hyp_dict['variance'] = hyp_array[n_lengthscales]
        elif kern_type == 'sum_lin_rbf':
            n = self.input_dim
            idx = 0
            hyp_dict['rbf.lengthscale'] = hyp_array[idx:idx+n]
            idx += n
            hyp_dict['rbf.variance'] = hyp_array[idx]
            idx += 1
            hyp_dict['linear.variances'] = hyp_array[idx:idx+n]
        elif kern_type == 'sum_lin_rbf_linear_only':
            hyp_dict['linear.variances'] = hyp_array
        elif kern_type == 'sum_lin_rbf_rbf_only':
            n = self.input_dim
            hyp_dict['rbf.lengthscale'] = hyp_array[:n]
            hyp_dict['rbf.variance'] = hyp_array[n]
        elif kern_type == 'prod_lin_rbf':
            n = self.input_dim
            idx = 0
            hyp_dict['prod.rbf.lengthscale'] = hyp_array[idx:idx+n]
            idx += n
            hyp_dict['prod.rbf.variance'] = hyp_array[idx]
            idx += 1
            hyp_dict['prod.linear.variances'] = hyp_array[idx:idx+n]
            idx += n
            hyp_dict['linear.variances'] = hyp_array[idx:idx+n]
        elif kern_type == 'lin_mat52':
            n = self.input_dim
            idx = 0
            hyp_dict['prod.mat52.lengthscale'] = hyp_array[idx:idx+n]
            idx += n
            hyp_dict['prod.mat52.variance'] = hyp_array[idx]
            idx += 1
            hyp_dict['prod.linear.variances'] = hyp_array[idx:idx+n]
            idx += n
            hyp_dict['linear.variances'] = hyp_array[idx:idx+n]
        else:
            raise ValueError(f"Unsupported kernel type: {kern_type}")
        
        return hyp_dict


    def predict(self, x_new, quantiles=None, compute_gradients=False):
        """ Compute the predictive mean and variance at test points

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
            K_star = self.compute_kernel(x_new, self.z, 
                                        self.kern_types[i], self.hyp[i])
            
            y_mu_pred[:, i] = K_star @ self.beta[:, i]
            
            K_ss = self.compute_kernel(x_new, x_new, 
                                      self.kern_types[i], self.hyp[i])
            
            # Variance: k_** - k_* (K + σ²I)^{-1} k_*^T = k_** - k_* L^{-T} L^{-1} k_*^T
            V = np.linalg.solve(self.L_chol[i], K_star.T)
            
            var = np.diag(K_ss) - np.sum(V * V, axis=0)
            
            y_sigm_pred[:, i] = np.sqrt(np.maximum(var, 1e-10))

        if quantiles is not None:
            raise NotImplementedError()

        if compute_gradients:
            grad_mu = self.predictive_gradients(x_new)
            return y_mu_pred, y_sigm_pred, grad_mu

        return y_mu_pred, y_sigm_pred


    def predictive_gradients(self, x_new, grad_sigma=False):
        """ Compute the gradients of the predictive mean/variance w.r.t. inputs

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

        grad_mu_pred = np.empty([T, self.n_s_out, self.input_dim])

        for i in range(self.n_s_out):
            if self.kern_types[i] == 'rbf':
                grad_mu_pred[:, i, :] = self._rbf_gradient(x_new, i)
            elif self.kern_types[i] == 'mat52':
                grad_mu_pred[:, i, :] = self._matern52_gradient(x_new, i)
            else:
                warnings.warn(f"Gradient not implemented for kernel type {self.kern_types[i]}")
                grad_mu_pred[:, i, :] = 0.0

        return grad_mu_pred
    
    def _rbf_gradient(self, x_new, dim_idx):
        """Compute gradient of RBF GP mean w.r.t. inputs.
        
        ∂μ/∂x* = ∂k(x*,X)/∂x* @ beta
        """
        T = x_new.shape[0]
        N = self.z.shape[0]
        
        K_star = self.compute_kernel(x_new, self.z, 
                                     self.kern_types[dim_idx], 
                                     self.hyp[dim_idx])
        
        # Compute gradient of kernel
        lengthscale = self.hyp[dim_idx]['lengthscale']
        grad_K = np.zeros((T, N, self.input_dim))
        
        for d in range(self.input_dim):
            # ∂k/∂x*_d = k(x*,X) * (X_d - x*_d) / ℓ_d²
            diff = (self.z[:, d] - x_new[:, d, np.newaxis]) / (lengthscale[d]**2)
            grad_K[:, :, d] = K_star * diff
        
        # Chain rule: ∂μ/∂x* = ∂k/∂x* @ beta
        grad_mu = np.einsum('tnd,n->td', grad_K, self.beta[:, dim_idx])
        
        # TODO: Verify gradient implemention
        warnings.warn("RBF gradient implementation needs testing")

        return grad_mu
    
    def _matern52_gradient(self, x_new, dim_idx):
        """Compute gradient of Matérn 5/2 GP mean w.r.t. inputs."""
        warnings.warn("Matérn 5/2 gradient not implemented, returning zeros")
        return np.zeros((x_new.shape[0], self.input_dim))


    def update_model(self, x, y, opt_hyp=False, replace_old=True):
        """ Update the model based on the current settings and new data

        Parameters
        ----------
        x: n x (n_s + n_u) array[float]
            The training set
        y: n x n_s
            The training targets
        opt_hyp: bool, optional
            If this is set to TRUE the hyperparameters are re-optimized
        replace_old: bool, optional
            If True, replace old data; if False, append to old data
        """
        if replace_old:
            x_new = x
            y_new = y
        else:
            x_new = np.vstack((self.x_train, x))
            y_new = np.vstack((self.y_train, y))

        # Efficient online update without retraining from scratch
        if not replace_old and self.gp_trained and not opt_hyp:
            self._update_online(x, y)
        else:
            # Full retraining required
            self.train(x_new, y_new, opt_hyp=opt_hyp)
    
    def _update_online(self, x_new, y_new):
        """Efficient online GP update using incremental Cholesky factorization
        
        For each new point (x_N+1, y_N+1), extends the Cholesky factor:
        - Compute k = k(X_old, x_N+1)
        - Solve v = L^{-1} k  (forward triangular solve)
        - Compute α = √(k_++ - v^T v)  where k_++ = k(x_N+1, x_N+1) + σ²
        - Extend L_new = [L, 0; v^T, α]
        
        This is O(N²) per new point and numerically stable.
        
        Parameters
        ----------
        x_new : ndarray [n_new × (n_s + n_u)]
            New training inputs
        y_new : ndarray [n_new × n_s]
            New training targets
        """
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_2d(y_new)
        n_new = x_new.shape[0]
        
        if x_new.shape[0] != y_new.shape[0]:
            raise ValueError("x_new and y_new must have same number of samples")
        
        for i in range(self.n_s_out):
            L = self.L_chol[i]
            N_old = L.shape[0]
            
            # Extend Cholesky factor for each new point
            for j in range(n_new):
                x_j = x_new[j:j+1, :]
                
                # Compute cross-covariance
                if j == 0:
                    X_current = self.z
                else:
                    X_current = np.vstack([self.z, x_new[:j]])
                
                k = self.compute_kernel(X_current, x_j, self.kern_types[i], self.hyp[i]).ravel()
                
                # Self-covariance k_++ = k(x_j, x_j) + σ² + jitter
                k_plus_plus = self.compute_kernel(x_j, x_j, self.kern_types[i], self.hyp[i])[0, 0]
                k_plus_plus += self.noise_var[i]
                # Adaptive jitter based on kernel scale
                adaptive_jitter = 1e-5 * k_plus_plus
                k_plus_plus += adaptive_jitter
                
                # Solve v = L^{-1} k
                v = np.linalg.solve(L, k)
                
                # Compute α = √(k_++ - v^T v)
                alpha_sq = k_plus_plus - np.dot(v, v)
                
                if alpha_sq <= 0:
                    # Numerical issue: add adaptive jitter and retry
                    warnings.warn(f"Online update: negative α² = {alpha_sq:.2e} for dim {i}, point {j}. Adding jitter.")
                    jitter = max(1e-6, -alpha_sq + 1e-6)
                    alpha_sq = k_plus_plus + jitter - np.dot(v, v)
                    
                    if alpha_sq <= 0:
                        # Still negative, fall back to retraining
                        warnings.warn(f"Online update failed for dimension {i}, retraining from scratch")
                        x_all = np.vstack((self.x_train, x_new))
                        y_all = np.vstack((self.y_train, y_new))
                        self.train(x_all, y_all, opt_hyp=False)
                        return
                
                alpha = np.sqrt(alpha_sq)
                
                # Extend Cholesky factor: L_new = [L, 0; v^T, α]
                N_current = L.shape[0]
                L_new = np.zeros((N_current + 1, N_current + 1))
                L_new[:N_current, :N_current] = L
                L_new[N_current, :N_current] = v
                L_new[N_current, N_current] = alpha
                
                L = L_new
            
            self.L_chol[i] = L
            
            # Update posterior coefficients: β = (L L^T)^{-1} y = L^{-T} L^{-1} y
            y_all = np.vstack((self.y_train[:, i:i+1], y_new[:, i:i+1]))
            w = np.linalg.solve(L, y_all)
            beta_new = np.linalg.solve(L.T, w).ravel()
            
            # Resize beta array if needed
            N_total = N_old + n_new
            if self.beta.shape[0] < N_total:
                beta_resized = np.zeros((N_total, self.n_s_out))
                beta_resized[:self.beta.shape[0], :] = self.beta
                self.beta = beta_resized
            
            self.beta[:N_total, i] = beta_new
        
        # Update training data
        self.x_train = np.vstack((self.x_train, x_new))
        self.y_train = np.vstack((self.y_train, y_new))
        self.z = self.x_train

    def sample_from_gp(self, inp, size=10):
        """ Sample from GP predictive distribution


        Args:
            inp (numpy.ndarray[float]): array of shape n x (n_s + n_u); the
                test inputs
            size (int, optional): number of samples per test point

        Returns:
            S (numpy.ndarray[float]): array of shape n x size x n_s; array of size samples
            of the posterior distribution per test input

        """

        n = np.shape(inp)[0]
        S = np.empty((n, size, self.n_s_out))

        mu, sigma = self.predict(inp)
        
        for i in range(self.n_s_out):
            for j in range(size):
                S[:, j, i] = mu[:, i] + sigma[:, i] * self.rng.standard_normal(n)

        return S

    def get_bounds(self, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0):
        """Return a helper object for computing Abbasi-Yadkori style bounds."""

        return NumpyGPBounds(self, delta=delta, rkhs_norm=rkhs_norm, R_subgaussian=R_subgaussian)

    def estimate_true_rkhs_norm(self, X=None, y=None):
        """Estimate the RKHS norm of the true function in the current kernel RKHS.
        
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
            K = self.compute_kernel(X, X, self.kern_types[i], self.hyp[i])

            # Regularization
            lambda_reg = 1e-6
            K_lambda = K + lambda_reg * np.eye(K.shape[0])
            
            y_i = y[:, i]
            
            try:
                L = np.linalg.cholesky(K_lambda)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_i))

                norms[i] = np.sqrt(alpha.T @ K @ alpha)
            except np.linalg.LinAlgError:
                warnings.warn(f"Failed to estimate RKHS norm for dim {i} (Cholesky failed)")
                norms[i] = np.nan
        return norms

    def compute_bounds(self, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0):
        """Compute and store β-values derived from the current GP posterior."""

        if not self.gp_trained:
            raise ValueError("GP must be trained before integrating bounds")

        bounds = self.get_bounds(delta=delta, rkhs_norm=rkhs_norm, R_subgaussian=R_subgaussian)
        self.beta_safety_per_dim = np.array([bounds.beta(dim_idx) for dim_idx in range(self.n_s_out)])

    def information_gain(self, x=None):
        """ Mutual information between samples and system 
        
        Uses Cholesky decomposition for numerical stability to avoid overflow
        when computing log-determinant of large matrices.
        """

        if x is None:
            x = self.z

        n_data = np.shape(x)[0]
        inf_gain_x_f = [None] * self.n_s_out
        for i in range(self.n_s_out):
            noise_var_i = self.noise_var[i]
            K = self.compute_kernel(x, x, self.kern_types[i], self.hyp[i])
            
            # Compute log(det(I + K/noise_var)) using Cholesky for numerical stability
            # log(det(A)) = 2 * sum(log(diag(L))) where A = L @ L.T
            try:
                K_scaled = np.eye(n_data) + (1 / noise_var_i) * K
                L = np.linalg.cholesky(K_scaled)
                inf_gain_x_f[i] = 2 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                warnings.warn(f"Cholesky decomposition failed for dimension {i} in information_gain. "
                             f"Matrix may be ill-conditioned. Returning NaN.")
                inf_gain_x_f[i] = np.nan

        return inf_gain_x_f
