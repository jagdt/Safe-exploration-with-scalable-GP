# -*- coding: utf-8 -*-


from .gp_numpy import NumpyGPModel
from .scalable_gp import ScalableGPModel
from .gp_bounds import NumpyGPBounds, ScalableGPBounds

__all__ = ['NumpyGPModel', 'ScalableGPModel', 'NumpyGPBounds', 'ScalableGPBounds']