
import numpy as np
import pandas as pd
import time

# ========================================================
# Helper Functions
# ========================================================

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call option.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Avoid numerical issues with very small sigma or T
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    T = np.maximum(T, eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def prepare_calibration_data(df, data_type="synthetic", use_pricing_model=False):
    """
    Prepares the features (X) and targets (y) for calibration.

    Mode 1: Direct volatility calibration (synthetic data).
    Mode 2: Indirect calibration via option prices (real data).
    """
    X = df[['Strike', 'Maturity']].values

    if data_type == "synthetic":
        y = df['LocalVolatility'].values
        mode = 1
        notes = (
            "Direct Volatility Calibration:\n"
            "✅ Fast and ideal for testing against known synthetic surfaces.\n"
            "❌ Not applicable to real market data; ignores pricing dynamics."
        )
    elif data_type == "real":
        if use_pricing_model:
            y = df['OptionPrice'].values
            mode = 2
            notes = (
                "Indirect Calibration via Option Prices:\n"
                "✅ Matches real-world calibration; learns volatility from prices.\n"
                "✅ Extendable with pricing models (e.g., Dupire PDE / MC).\n"
                "❌ Requires pricing engine; more computationally intensive."
            )
        else:
            raise ValueError("Real market data requires a pricing model for calibration.")
    else:
        raise ValueError("data_type must be 'synthetic' or 'real'.")

    return X, y, mode, notes

