import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import griddata
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


# ========================================================
# RealMarketDataFetcher Class (as provided by user)
# ========================================================
class RealMarketDataFetcher:
    def __init__(self, ticker="AAPL", maturities_to_fetch=5, n_strikes=100,
                 r=0.01, q=0.0):
        self.ticker = ticker
        self.r = r
        self.q = q
        self.maturities_to_fetch = maturities_to_fetch
        self.n_strikes = n_strikes
        self.spot = self._fetch_spot_price()

    def _fetch_spot_price(self):
        asset = yf.Ticker(self.ticker)
        spot = asset.history(period="1d")["Close"].iloc[-1]
        return float(spot)

    def _black_scholes_iv(self, market_price, S, K, T, r):
        return np.nan

    def _get_option_data(self):
        try:
            asset = yf.Ticker(self.ticker)
            maturities = asset.options
            if not maturities:
                raise ValueError(f"No option maturities found for {self.ticker}.")
            maturities = maturities[:self.maturities_to_fetch]
            all_data = []
            for maturity_date_str in maturities:
                option_chain = asset.option_chain(maturity_date_str)
                calls = option_chain.calls
                calls = calls[(calls['strike'] >= 0.7 * self.spot) & (calls['strike'] <= 1.3 * self.spot)]
                calls['strike_dist'] = np.abs(calls['strike'] - self.spot)
                calls = calls.sort_values(by='strike_dist').head(self.n_strikes)
                maturity_datetime = pd.to_datetime(maturity_date_str)
                time_to_maturity = (maturity_datetime - pd.Timestamp.now()).days / 365.25
                calls['Maturity'] = time_to_maturity
                calls['Strike'] = calls['strike']
                calls['OptionPrice'] = (calls['bid'] + calls['ask']) / 2
                calls['ImpliedVolatility'] = calls['impliedVolatility']
                calls = calls.dropna(subset=['OptionPrice', 'ImpliedVolatility'])
                calls = calls[calls['OptionPrice'] > 0]
                all_data.append(calls[['Strike', 'Maturity', 'OptionPrice', 'ImpliedVolatility']].copy())
            if not all_data:
                raise ValueError("No valid option data found after filtering.")
            df = pd.concat(all_data, ignore_index=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return pd.DataFrame()

    # ========================================================
    # Helper methods with corrected signature
    # ========================================================

    def _build_interpolated_iv_surface(self, df):
        """
        Interpolates a sparse DataFrame of Implied Volatility (IV)
        into a smooth, dense surface.
        """
        strikes = np.linspace(df['Strike'].min(), df['Strike'].max(), 50)
        maturities = np.linspace(df['Maturity'].min(), df['Maturity'].max(), 50)
        grid_K, grid_T = np.meshgrid(strikes, maturities)
        points = df[['Strike', 'Maturity']].values
        values = df['ImpliedVolatility'].values
        grid_IV = griddata(points, values, (grid_K, grid_T), method='cubic')
        iv_surface = pd.DataFrame(grid_IV, index=maturities, columns=strikes)
        return iv_surface

    def _compute_derivatives_from_iv(self, iv_surface, strikes, maturities, i, j):
        """
        Computes the first and second partial derivatives of the IV surface
        at a specific grid point using finite differences.
        """
        T = maturities[i]
        K = strikes[j]

        # Finite differences for derivatives of IV
        dT = maturities[i+1] - maturities[i-1]
        dK = strikes[j+1] - strikes[j-1]

        C = iv_surface.loc[T, K]
        dC_dT = (iv_surface.loc[maturities[i+1], K] - iv_surface.loc[maturities[i-1], K]) / dT
        dC_dK = (iv_surface.loc[T, strikes[j+1]] - iv_surface.loc[T, strikes[j-1]]) / dK
        d2C_dK2 = (iv_surface.loc[T, strikes[j+1]] - 2 * C + iv_surface.loc[T, strikes[j-1]]) / (dK ** 2)

        return C, dC_dT, dC_dK, d2C_dK2

    def _apply_dupire_formula(self, T, K, C, dC_dT, dC_dK, d2C_dK2):
        """
        Calculates the local volatility squared using Dupire's formula.
        """
        try:
            C_price = self._black_scholes_call(self.spot, K, T, self.r, C)

            # Derivatives of option price with respect to strike
            # Using a more robust numerical approach
            dK_small = 1e-6
            dC_dK_price = (
                self._black_scholes_call(self.spot, K + dK_small, T, self.r, C) -
                self._black_scholes_call(self.spot, K - dK_small, T, self.r, C)
            ) / (2 * dK_small)

            d2C_dK2_price = (
                self._black_scholes_call(self.spot, K + dK_small, T, self.r, C) -
                2 * C_price +
                self._black_scholes_call(self.spot, K - dK_small, T, self.r, C)
            ) / (dK_small ** 2)

            numerator = dC_dT * C_price + self.r * K * dC_dK_price
            denominator = 0.5 * K**2 * d2C_dK2_price

            if denominator > 1e-12:
                sigma_sq = numerator / denominator
                return np.sqrt(max(sigma_sq, 0)) if sigma_sq > 0 else np.nan
        except Exception:
            return np.nan

    # ========================================================
    # Refactored _get_local_volatility method
    # ========================================================

    def _get_local_volatility(self, df):
        """
        Computes the local volatility surface by applying Dupire's formula
        to a smoothly interpolated implied volatility surface.
        """
        if df.empty or "Strike" not in df.columns or "Maturity" not in df.columns or "ImpliedVolatility" not in df.columns:
            print("⚠️ Input DataFrame is missing required columns or is empty.")
            return pd.DataFrame(columns=["Maturity", "Strike", "LocalVolatility"])

        # 1. Build a smooth IV surface
        iv_surface = self._build_interpolated_iv_surface(df)
        maturities, strikes = iv_surface.index, iv_surface.columns
        local_vol_surface = pd.DataFrame(index=maturities, columns=strikes)

        # 2. Iterate through the grid and compute local volatility
        for i in range(1, len(maturities) - 1):
            for j in range(1, len(strikes) - 1):
                K = strikes[j]
                T = maturities[i]

                # Get derivatives of the IV surface
                C, dC_dT, dC_dK, d2C_dK2 = self._compute_derivatives_from_iv(iv_surface, strikes, maturities, i, j)

                # Apply Dupire's formula to get local vol
                local_vol = self._apply_dupire_formula(T, K, C, dC_dT, dC_dK, d2C_dK2)
                local_vol_surface.loc[T, K] = local_vol

        # 3. Return clean DataFrame
        stacked = local_vol_surface.stack().reset_index()
        stacked.columns = ["Maturity", "Strike", "LocalVolatility"]
        return stacked


    def _black_scholes_call(self, S, K, T, r, sigma):
        eps = 1e-12
        sigma = np.maximum(sigma, eps)
        T = np.maximum(T, eps)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def get_data(self):
        df_options = self._get_option_data()
        if df_options.empty:
            return pd.DataFrame(), self.spot, self.r
        df_local_vol = self._get_local_volatility(df_options)
        df_final = pd.merge(df_options, df_local_vol, on=["Strike", "Maturity"], how="left")
        df_final['LocalVolatility'] = df_final['LocalVolatility'].interpolate(method='linear')
        df_final.fillna(method='bfill', inplace=True)
        df_final.fillna(method='ffill', inplace=True)
        return df_final, self.spot, self.r

