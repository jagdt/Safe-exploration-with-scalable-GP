# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
from casadi import horzcat
from .gp_models_utils_casadi import gp_pred_function, _get_kernel_function
from ..state_space_models import StateSpaceModel


class GPModelBase(StateSpaceModel, ABC):
    """Base class for all GP models with CasADi symbolic functions
    
    Provides common CasADi interface methods that delegate to the abstract
    predict_casadi_symbolic method. Subclasses implement their own version
    of predict_casadi_symbolic based on their internal representation
    (kernel-based or feature-based).
    
    Required attributes in subclasses:
    - self.n_s_out: number of output dimensions
    - self.gp_trained: whether the GP has been trained
    - self.hyp: list of hyperparameter dicts
    - self.kern_types: list of kernel type strings
    
    The base class provides:
    - __call__: Single input prediction via CasADi
    - get_forward_model_casadi: Get symbolic forward model
    
    Subclasses must implement:
    - predict_casadi_symbolic: Symbolic prediction (implementation-specific)
    """
    
    def __call__(self, states, actions):
        """Single input predictions via CasADi symbolic computation
        
        Parameters
        ----------
        states : ndarray [1 × n_s]
            State vector (single state)
        actions : ndarray [1 × n_u]
            Action vector (single action)
            
        Returns
        -------
        mu : casadi expression
            Predictive mean
        sigma : casadi expression
            Predictive standard deviation
        """
        N, n = np.shape(states)
        if N > 1:
            raise NotImplementedError(
                "Currently do not support multiple state-action pairs to evaluate on.")
        return self.predict_casadi_symbolic(horzcat(states, actions), True)

    def get_forward_model_casadi(self, compute_grads=False):
        """Return a symbolic CasADi function representing predictive mean/variance
        
        Parameters
        ----------
        compute_grads : bool, optional
            Whether to compute gradients (default: False)
            
        Returns
        -------
        forward_model : callable
            Lambda function that takes (x_new, u_new) and returns predictions
        """
        return lambda x_new, u_new: self.predict_casadi_symbolic(
            horzcat(x_new, u_new), compute_grads)

    @abstractmethod
    def predict_casadi_symbolic(self, x_new, compute_grads=False):
        """Return symbolic CasADi expressions for predictive mean/variance
        
        Subclasses must implement this method based on their internal representation.
        Kernel-based GPs use kernel functions and inv_K matrices.
        Feature-based GPs use Fourier features and posterior coefficients.
        
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
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the GP model on data
        
        Parameters
        ----------
        X : ndarray [N × (n_s_in + n_u)]
            Training inputs
        y : ndarray [N × n_s_out]
            Training targets
        **kwargs
            Additional keyword arguments specific to each implementation
        """
        pass
    
    @abstractmethod
    def predict(self, x_new, **kwargs):
        """Compute predictive mean and variance at test inputs
        
        Parameters
        ----------
        x_new : ndarray [T × (n_s_in + n_u)]
            Test inputs
        **kwargs
            Additional keyword arguments specific to each implementation
            
        Returns
        -------
        mu : ndarray [T × n_s_out]
            Predictive mean
        sigma : ndarray [T × n_s_out]
            Predictive standard deviation
        """
        pass
    
    @abstractmethod
    def update_model(self, x, y, **kwargs):
        """Update the model with new data
        
        Parameters
        ----------
        x : ndarray [n × (n_s_in + n_u)]
            New training inputs
        y : ndarray [n × n_s_out]
            New training targets
        **kwargs
            Additional keyword arguments specific to each implementation
        """
        pass

class KernelGPModel(GPModelBase):
    """Base class for kernel-based GP models
    
    Provides the predict_casadi_symbolic implementation for traditional GPs
    that use kernel functions and kernel matrix representations.
    
    Required attributes in subclasses:
    - self.z: training inputs
    - self.beta: posterior coefficients
    - self.L_chol: list of Cholesky factors of kernel matrices
    - self.hyp: list of hyperparameter dicts
    - self.kern_types: list of kernel type strings
    """
    
    def predict_casadi_symbolic(self, x_new, compute_grads=False):
        """Return symbolic CasADi expressions for predictive mean/variance
        
        This method uses the pre-computed GP parameters (hyp, z, beta, L_chol)
        to create symbolic CasADi expressions for prediction. This is used
        for gradient-based optimization in MPC.
        
        Computes inverse Cholesky factors inv_L = L^{-1} for variance computation.
        This is done via triangular solve with identity, which is numerically stable.
        
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
        assert np.shape(x_new)[0] == 1, \
            "We only support this for a single input vector right now"

        inv_L = [np.linalg.solve(L, np.eye(L.shape[0])) for L in self.L_chol]
        
        out_dict = gp_pred_function(x_new, self.hyp, self.kern_types, self.z, 
                                    self.beta, inv_L, True, compute_grads)
        mu_new = out_dict["pred_mu"]
        sigma_new = out_dict["pred_sigma"]
        
        if compute_grads:
            jac_mu = out_dict["jac_mu"]
            return mu_new.T, sigma_new.T, jac_mu

        return mu_new.T, sigma_new.T

    def get_kern_func_casadi(self):
        """Return CasADi kernel functions for each output dimension
        
        Returns
        -------
        kern_funcs : list
            List of CasADi kernel functions, one per output dimension
        """
        return [_get_kernel_function(kern_type, hyp) 
                for (kern_type, hyp) in zip(self.kern_types, self.hyp)]
