# -*- coding: utf-8 -*-

import numpy as np
from abc import ABC, abstractmethod


class GPBounds(ABC):
    """Abstract base class with noise computation for GP confidence bounds."""

    def __init__(self, gp_model, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0):
        """Initialize GPBounds base class.
        
        Parameters
        ----------
        gp_model : object
            Trained GP model instance (must have gp_trained attribute).
        delta : float
            Confidence parameter for bounds (e.g., 0.05 for 95% confidence).
        rkhs_norm : float or array-like
            RKHS norm(s) for the GP prior or posterior mean.
        R_subgaussian : float
            Subgaussian parameter for the noise process.
        """
        if not getattr(gp_model, "gp_trained", False):
            raise ValueError("GP must be trained before computing bounds")
        self.gp = gp_model
        self.delta = delta
        self.gp = gp_model
        self.delta = delta
        self.R_subgaussian = self._initialize_R_subgaussian(R_subgaussian)
        self.rkhs_norms = self._initialize_rkhs_norms(rkhs_norm)

    def _initialize_R_subgaussian(self, R_subgaussian):
        """Initialize R_subgaussian for all output dimensions.

        Parameters
        ----------
        R_subgaussian : float or array-like
            User-provided subgaussian parameter(s).

        Returns
        -------
        ndarray
            Array of R_subgaussian values for each output dimension.
        """
        arr = np.asarray(R_subgaussian, dtype=float)
        if arr.ndim == 0:
            return np.full(self.gp.n_s_out, float(arr))
        elif arr.ndim == 1 and arr.size == self.gp.n_s_out:
            return arr
        else:
            raise ValueError("R_subgaussian must be scalar or length n_s_out")

    def _initialize_rkhs_norms(self, rkhs_norm):
        """Initialize RKHS norms for all output dimensions.
        
        Parameters
        ----------
        rkhs_norm : float or array-like or None
            User-provided RKHS norms, or None to compute automatically.
        
        Returns
        -------
        ndarray
            Array of RKHS norms for each output dimension.
        """
        """Set RKHS norms for all output dimensions.
        
        Parameters
        ----------
        rkhs_norm : float or array-like
            User-provided RKHS norms or None to compute automatically
            
        Returns
        -------
        ndarray
            Array of RKHS norms for each output dimension
        """
        if rkhs_norm is None:
            return np.array([
                self.rkhs_norm_posterior_mean(dim_idx) 
                for dim_idx in range(self.gp.n_s_out)
            ])
        else:
            arr = np.asarray(rkhs_norm, dtype=float)
            if arr.ndim == 0:
                return np.full(self.gp.n_s_out, float(arr))
            elif arr.ndim == 1 and arr.size == self.gp.n_s_out:
                return arr
            else:
                raise ValueError("rkhs_norm must be scalar or length n_s_out")

    def _noise_term(self, gram_matrix, lambda_noise, dim_idx):
        """Compute the Abbasi-Yadkori noise contribution for confidence bounds.
        The formula is R / sqrt(lambda) * sqrt(2 * ln(det(I + K/lambda) / delta)).

        Parameters
        ----------
        gram_matrix : ndarray
            Gram matrix (kernel matrix or feature Gram matrix).
        lambda_noise : float
            Regularization/noise parameter.
        dim_idx : int
            Dimension index to retrieve the correct R_subgaussian.
        
        Returns
        -------
        float
            Noise term for the confidence bound.
        """
        if gram_matrix.size == 0:
            return 0.0

        lambda_noise = float(max(lambda_noise, 1e-12))
        identity = np.eye(gram_matrix.shape[0], dtype=gram_matrix.dtype)
        scaled = identity + (gram_matrix / lambda_noise)

        try:
            chol = np.linalg.cholesky(scaled)
            log_det = 2.0 * np.sum(np.log(np.clip(np.diag(chol), 1e-12, None)))
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(scaled)
            if sign <= 0:
                log_det = np.log(1e-12)

        log_term = max(0.0, log_det - np.log(self.delta))
        return (self.R_subgaussian[dim_idx] / np.sqrt(lambda_noise) * np.sqrt(2.0 * log_term))

    @abstractmethod
    def beta(self, dim_idx):
        """Compute the confidence bound parameter βₜ for one output dimension.

        Parameters
        ----------
        dim_idx : int
            Output dimension index.

        Returns
        -------
        float
            Confidence bound parameter βₜ for the specified output dimension.
        """
        pass

    @abstractmethod
    def confidence_bounds(self, x_new, dim_idx):
        """Return predictive mean, standard deviation, and confidence bounds for one output dimension.

        Parameters
        ----------
        x_new : ndarray
            Test input points.
        dim_idx : int
            Output dimension index.

        Returns
        -------
        tuple
            (mean, std, lower_bound, upper_bound) arrays for the specified dimension.
        """
        pass

    @abstractmethod
    def rkhs_norm_posterior_mean(self, dim_idx):
        """Compute RKHS norm of the posterior mean for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            RKHS norm of the posterior mean for the specified output dimension.
        """
        """Compute RKHS norm of the posterior mean for one output dimension."""
        pass

class NumpyGPBounds(GPBounds):
    """Confidence bounds for the exact NumPy GP implementation."""

    def __init__(self, gp_model, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0):
        """Initialize NumpyGPBounds for exact NumPy GP implementation.
        
        Parameters
        ----------
        gp_model : NumpyGPModel
            Trained NumPy GP model instance.
        delta : float
            Confidence parameter for bounds.
        rkhs_norm : float or array-like
            RKHS norm(s) for the GP prior or posterior mean.
        R_subgaussian : float
            Subgaussian parameter for the noise process.
        """
        super().__init__(gp_model, delta, rkhs_norm, R_subgaussian)

    def _gram_matrix(self, dim_idx):
        """Return the Gram (kernel) matrix for the specified output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        ndarray
            Gram matrix (kernel matrix) for training data.
        """
        """Return the Gram matrix."""
        X = self.gp.x_train
        return self.gp.compute_kernel(X, X, self.gp.kern_types[dim_idx], self.gp.hyp[dim_idx])

    def beta(self, dim_idx):
        """Compute the confidence bound parameter βₜ for the specified output dimension.
        Formula: βₜ = ||f|| + noise term

        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            Confidence bound parameter βₜ.
        """
        gram = self._gram_matrix(dim_idx)
        noise_term = self._noise_term(gram, self.gp.noise_var[dim_idx], dim_idx)
        return self.rkhs_norms[dim_idx] + noise_term

    def confidence_bounds(self, x_new, dim_idx):
        """Return predictive mean, standard deviation, and confidence intervals for one output dimension.
        
        Parameters
        ----------
        x_new : ndarray
            Test input points.
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        tuple
            (mean, std, lower_bound, upper_bound) arrays for the specified dimension.
        """
        mu, sigma = self.gp.predict(x_new)
        mu_dim = mu[:, dim_idx]
        sigma_dim = sigma[:, dim_idx]
        beta_val = self.beta(dim_idx)
        lower = mu_dim - beta_val * sigma_dim
        upper = mu_dim + beta_val * sigma_dim
        return mu_dim, sigma_dim, lower, upper

    def rkhs_norm_posterior_mean(self, dim_idx):
        """Compute RKHS norm of the posterior mean for the full GP for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            RKHS norm of the posterior mean for the specified output dimension.
        """
        beta = self.gp.beta[:, dim_idx]
        K = self._gram_matrix(dim_idx)
        rkhs_norm = np.sqrt(beta.T @ K @ beta)
        print(f"Computed RKHS norm for dim {dim_idx}: {rkhs_norm}")
        return rkhs_norm


class ScalableGPBounds(GPBounds):
    """Bounds for the scalable GP using Fourier features."""

    def __init__(self, gp_model, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0,
                 projection_error=None):
        """Initialize ScalableGPBounds for scalable GP using Fourier features.
        
        Parameters
        ----------
        gp_model : ScalableGPModel
            Trained scalable GP model instance.
        delta : float
            Confidence parameter for bounds.
        rkhs_norm : float or array-like
            RKHS norm(s) for the GP prior or posterior mean.
        R_subgaussian : float
            Subgaussian parameter for the noise process.
        projection_error : float or array-like, optional
            Projection error(s) for each output dimension.
        """
        super().__init__(gp_model, delta, rkhs_norm, R_subgaussian)
        self.projection_errors = self._initialize_projection_errors(projection_error)

    def _initialize_projection_errors(self, projection_error):
        """Initialize projection errors for all output dimensions.
        
        Parameters
        ----------
        projection_error : float, array-like, or None
            User-provided projection errors or None to compute automatically.
        
        Returns
        -------
        ndarray
            Array of projection errors for each output dimension.
        """
        """Initialize projection errors for all dimensions.
        
        Parameters
        ----------
        projection_error : float, array-like, or None
            User-provided projection errors or None to compute automatically
            
        Returns
        -------
        ndarray
            Array of projection errors for each output dimension
        """
        if projection_error is None:
            return np.array([
                self.compute_theoretical_projection_error(dim_idx) 
                for dim_idx in range(self.gp.n_s_out)
            ])
        else:
            arr = np.asarray(projection_error, dtype=float)
            if arr.ndim == 0:
                return np.full(self.gp.n_s_out, float(arr))
            elif arr.ndim == 1 and arr.size == self.gp.n_s_out:
                return arr
            else:
                raise ValueError("projection_error must be scalar or length n_s_out")

    def _feature_gram(self, dim_idx):
        """Return the feature Gram matrix ΦᵀΦ for the scalable GP.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        ndarray
            Feature Gram matrix for training data.
        """
        Phi = self.gp._phi_features(self.gp.x_train, self.gp.lambdas[dim_idx], dim_idx)
        # TODO verify dimensions
        return Phi.T @ Phi

    def compute_projection_error(self, dim_idx):
        """Compute an upper bound for the projection error ||f - P(f)|| for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            Upper bound of the projection error for the specified output dimension.
        """
        kern_type = self.gp.kern_types[dim_idx]
        
        if kern_type == 'rbf':
            C = self.gp.hyp[dim_idx]["factor"]
            decay_rates = self.gp.hyp[dim_idx]["exponential_decay_rates"]
        elif kern_type == 'sum_lin_rbf':
            C = self.gp.hyp[dim_idx]["rbf.factor"]
            decay_rates = self.gp.hyp[dim_idx]["rbf.exponential_decay_rates"]
        else:
            raise NotImplementedError(f"Projection error bound only implemented for 'rbf' and 'sum_lin_rbf' kernels, got '{kern_type}'.")
        
        M = self.gp.n_frequencies_per_dim[dim_idx] - 1
        
        full_sums = 1 + 0.5 * np.sqrt(np.pi / decay_rates)
        tails = np.exp(-decay_rates * (M**2)) / (decay_rates * M)
        
        # Calculating Sum_{k} [ Tail_k * Product_{j!=k} (Full_j) ]
        # Using Product_{j!=k} = (Total_Product / Full_k)
        total_product = np.prod(full_sums)
        contributions_per_dimension = tails * (total_product / full_sums)
        total_tail_mass = np.sum(contributions_per_dimension)
        projection_error = self.rkhs_norms[dim_idx] * np.sqrt(2 * C * total_tail_mass)

        print(f"Computed projection error for dim {dim_idx}: {projection_error}")
        
        return projection_error

    def compute_theoretical_projection_error(self, dim_idx):
        """Compute theoretical projection error term for scalable GP for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            Theoretical projection error term for the specified output dimension.
        """
        all_lambdas = self.gp._compute_lambdas(dim_idx, Q=100)
        used_lambdas = self.gp.lambdas[dim_idx]
        lambda_sum = np.sum(all_lambdas) - np.sum(used_lambdas)
        projection_error = self.rkhs_norms[dim_idx] * np.sqrt(2 * lambda_sum)

        print(f"Computed theoretical projection error for dim {dim_idx}: {projection_error}")
        
        return projection_error

    def projection_error_term(self, dim_idx):
        """Compute the projection error term added to β for scalable GP for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            Projection error term for the specified output dimension.
        """
        X_train = self.gp.x_train
        lambda_reg = float(max(self.gp.noise_var[dim_idx], 1e-12))
        projection_error_scalar = self.projection_errors[dim_idx]
        projection_error_vector = np.full(X_train.shape[0], projection_error_scalar)

        Phi_t = self.gp._phi_features(X_train, self.gp.lambdas[dim_idx], dim_idx)
        Phi_T_Phi_plus_lambda = Phi_t.T @ Phi_t + lambda_reg * np.eye(Phi_t.shape[1])

        try:
            L = np.linalg.cholesky(Phi_T_Phi_plus_lambda)
            inv_term = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(Phi_t.shape[1])))
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(Phi_T_Phi_plus_lambda)

        Phi_t_Pf_minus_f = Phi_t.T @ projection_error_vector
        temp = inv_term @ Phi_t_Pf_minus_f
        quadratic_term = float(np.real(Phi_t_Pf_minus_f.T @ temp))
        quadratic_term = max(0.0, quadratic_term)

        return np.sqrt(quadratic_term) / np.sqrt(lambda_reg)

    def beta(self, dim_idx):
        """Compute the scalable confidence bound parameter βₜ for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            Scalable confidence bound parameter βₜ.
        """
        PhiTPhi = self._feature_gram(dim_idx)
        noise_term = self._noise_term(PhiTPhi, self.gp.noise_var[dim_idx], dim_idx)
        projection_error_term = self.projection_error_term(dim_idx)
        return self.rkhs_norms[dim_idx] + noise_term + projection_error_term

    def confidence_bounds(self, x_new, dim_idx):
        """Return scalable GP predictive mean, standard deviation, and confidence intervals for one output dimension.
        
        Parameters
        ----------
        x_new : ndarray
            Test input points.
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        tuple
            (mean, std, lower_bound, upper_bound) arrays for the specified dimension.
        """
        mu, sigma = self.gp.predict(x_new)
        mu_dim = mu[:, dim_idx]
        sigma_dim = sigma[:, dim_idx]
        beta_val = self.beta(dim_idx)
        proj = self.projection_errors[dim_idx]
        lower = mu_dim - beta_val * sigma_dim - proj
        upper = mu_dim + beta_val * sigma_dim + proj
        return mu_dim, sigma_dim, lower, upper

    def rkhs_norm_posterior_mean(self, dim_idx):
        """Compute RKHS norm of the posterior mean for the scalable GP for one output dimension.
        
        Parameters
        ----------
        dim_idx : int
            Output dimension index.
        
        Returns
        -------
        float
            RKHS norm of the posterior mean for the specified output dimension.
        """
        w = self.gp.posterior_mean_coeffs[dim_idx]
        rkhs_norm = np.sqrt(w.T @ w)
        print(f"Computed RKHS norm for dim {dim_idx}: {rkhs_norm}")
        return rkhs_norm