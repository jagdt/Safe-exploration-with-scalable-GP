# -*- coding: utf-8 -*-

import numpy as np

class GPBounds:
    """Base class with noise computation for GP confidence bounds."""

    def __init__(self, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0):
        self.delta = delta
        self.rkhs_norm = rkhs_norm
        self.R_subgaussian = R_subgaussian

    def _noise_term(self, gram_matrix, lambda_noise):
        """Abbasi-Yadkori noise contribution.

        The formula is R / sqrt(lambda) * sqrt(2 * ln(det(I + K/lambda) / delta)).
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
        return (self.R_subgaussian / np.sqrt(lambda_noise) * np.sqrt(2.0 * log_term))


class NumpyGPBounds(GPBounds):
    """Confidence bounds for the exact NumPy GP implementation."""

    def __init__(self, gp_model, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0):
        super().__init__(delta, rkhs_norm, R_subgaussian)
        if not gp_model.gp_trained:
            raise ValueError("GP must be trained before computing bounds")
        self.gp = gp_model

    def _gram_matrix(self, dim_idx):
        """Return the Gram matrix."""
        X = self.gp.x_train
        return self.gp.compute_kernel(X, X, self.gp.kern_types[dim_idx], self.gp.hyp[dim_idx])

    def beta(self, dim_idx):
        """Compute βₜ = ||f|| + noise term."""
        gram = self._gram_matrix(dim_idx)
        noise_term = self._noise_term(gram, self.gp.noise_var[dim_idx])
        return self.rkhs_norm + noise_term

    def confidence_bounds(self, x_new, dim_idx):
        """Return predictive mean, std, and ±beta intervals for one dimension."""
        mu, sigma = self.gp.predict(x_new)
        mu_dim = mu[:, dim_idx]
        sigma_dim = sigma[:, dim_idx]
        beta_val = self.beta(dim_idx)
        lower = mu_dim - beta_val * sigma_dim
        upper = mu_dim + beta_val * sigma_dim
        return mu_dim, sigma_dim, lower, upper


class ScalableGPBounds(GPBounds):
    """Bounds for the scalable GP using Fourier features."""

    def __init__(self, gp_model, delta=0.05, rkhs_norm=1.0, R_subgaussian=1.0,
                 projection_error=None):
        super().__init__(delta, rkhs_norm, R_subgaussian)
        if not gp_model.gp_trained:
            raise ValueError("GP must be trained before computing bounds")
        self.gp = gp_model
        self.projection_errors = self._initialize_projection_errors(projection_error)

    def _initialize_projection_errors(self, projection_error):
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
                self.compute_projection_error(dim_idx) 
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
        """Return ΦᵀΦ for the scalable gp."""
        Phi = self.gp._phi_features(self.gp.x_train, self.gp.lambdas[dim_idx], dim_idx)
        # TODO verify dimensions
        return Phi.T @ Phi

    def compute_projection_error(self, dim_idx):
        """
        Computes an upper bound for the projection error.

        Returns:
            float: The upper bound of the projection error ||f - P(f)||.
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
        projection_error = self.rkhs_norm * np.sqrt(C * total_tail_mass)

        print(f"Computed projection error for dim {dim_idx}: {projection_error}")
        
        return projection_error

    def projection_error_term(self, dim_idx):
        """Compute σ̃-dependent projection error term added to β for scalable GP."""

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
        """Compute scalable βₜ."""
        PhiTPhi = self._feature_gram(dim_idx)
        noise_term = self._noise_term(PhiTPhi, self.gp.noise_var[dim_idx])
        projection_error_term = self.projection_error_term(dim_idx)
        return self.rkhs_norm + noise_term + projection_error_term

    def confidence_bounds(self, x_new, dim_idx):
        """Return scalable GP mean, std, and confidence band."""
        mu, sigma = self.gp.predict(x_new)
        mu_dim = mu[:, dim_idx]
        sigma_dim = sigma[:, dim_idx]
        beta_val = self.beta(dim_idx)
        proj = self.projection_errors[dim_idx]
        lower = mu_dim - beta_val * sigma_dim - proj
        upper = mu_dim + beta_val * sigma_dim + proj
        return mu_dim, sigma_dim, lower, upper
