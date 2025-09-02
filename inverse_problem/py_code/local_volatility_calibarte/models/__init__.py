# __init__.py
"""
The models package for the scientific research project on inverse problems.

This package provides a unified interface for calibrating a local volatility surface
using various machine learning and optimization methods, including:
- Neural Networks (NN)
- Gaussian Processes (GP)
- Reinforcement Learning (RL) with Evolution Strategies
"""


from .helper import black_scholes_call, prepare_calibration_data
from .model_calibration import VolatilityCalibrator

__all__ =[
    "black_scholes_call", 
    "prepare_calibration_data",
    "VolatilityCalibrator"
]