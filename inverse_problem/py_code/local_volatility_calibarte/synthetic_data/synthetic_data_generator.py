import numpy as np
import pandas as pd
from scipy.stats import norm

class SyntheticDataGenerator:
    def __init__(self, S0=100, r=0.01, q=0.005, n_strikes=20, n_maturities=10):
        """
        Initialize the synthetic data generator.

        Parameters:
        - S0: Initial asset price
        - r: Risk-free interest rate
        - q: Dividend yield
        - n_strikes: Number of strike prices
        - n_maturities: Number of maturities
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.n_strikes = n_strikes
        self.n_maturities = n_maturities
        self.strikes = np.linspace(80, 170, n_strikes)
        self.maturities = np.linspace(0.1, 2.0, n_maturities)

    def local_volatility(self, K, T):
        """
        Define a synthetic local volatility surface.
        """
        base_vol = 0.2
        smile = 0.1 * np.exp(-((K - self.S0)**2) / 500)
        term_structure = 0.05 * np.sin(2 * np.pi * T)
        return base_vol + smile + term_structure

    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Compute the Black-Scholes price of a European call option.
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def generate(self):
        """
        Generate the synthetic dataset.

        Returns:
        - DataFrame with columns: Strike, Maturity, LocalVolatility, OptionPrice
        """
        data = []
        for T in self.maturities:
            for K in self.strikes:
                sigma = self.local_volatility(K, T)
                price = self.black_scholes_call(self.S0, K, T, self.r, sigma)
                data.append({
                    'Strike': K,
                    'Maturity': T,
                    'LocalVolatility': sigma,
                    'OptionPrice': price
                })
        return pd.DataFrame(data)
