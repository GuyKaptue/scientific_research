# real_market_data_processor.py
import pandas as pd
import numpy as np
from .real_market_data_fetcher import RealMarketDataFetcher  # adjust import as needed


# ========================================================
# RealMarketDataProcessor Class
# ========================================================
class RealMarketDataProcessor:
    """
    Processes, aggregates, and inspects real market option data for multiple tickers.

    Attributes
    ----------
    tickers : dict
        Dictionary of ticker symbols, e.g., {"aapl": "AAPL"}
    maturities_to_fetch : int
        Number of option maturities to fetch
    n_strikes : int
        Number of strikes per maturity
    r : float
        Risk-free interest rate
    market_data : dict
        Loaded market data per ticker
    """

    def __init__(self, tickers=None, maturities_to_fetch=5, n_strikes=50, r=0.01):
        self.tickers = tickers or {
            "aapl": "AAPL",
            "goog": "GOOGL",
            "msft": "MSFT",
            "amzn": "AMZN"
        }
        self.maturities_to_fetch = maturities_to_fetch
        self.n_strikes = n_strikes
        self.r = r
        self.market_data = {}

    # ========================================================
    # Load Market Data
    # ========================================================
    def load_data(self):
        """Fetches option data, local vol surface, and spot price for each ticker."""
        for key, symbol in self.tickers.items():
            print(f"Fetching data for {symbol}...")
            fetcher = RealMarketDataFetcher(
                ticker=symbol,
                maturities_to_fetch=self.maturities_to_fetch,
                n_strikes=self.n_strikes,
                r=self.r
            )
            df, spot_price, rate = fetcher.get_data()
            self.market_data[key] = {
                "data": df,
                "spot": spot_price,
                "rate": rate
            }
        return self.market_data

    # ========================================================
    # Inspect DataFrame
    # ========================================================
    @staticmethod
    def inspect_dataframe(df, name="DataFrame"):
        """Prints detailed information about a DataFrame."""
        print(f"\n📄 Inspecting {name}...\n")

        print("🔹 First 5 rows:")
        print(df.head())

        print("\n🔹 Last 5 rows:")
        print(df.tail())

        print("\n🔹 Summary statistics:")
        print(df.describe(include='all'))

        print("\n🔹 DataFrame info:")
        print(df.info())

        print("\n🔹 Column names:")
        print(df.columns.tolist())

        print("\n🔹 Shape (rows, columns):")
        print(df.shape)

        print("\n🔹 Data types:")
        print(df.dtypes)

        print("\n🔹 Missing values per column:")
        print(df.isnull().sum())

    # ========================================================
    # Inspect all loaded tickers
    # ========================================================
    def inspect_all(self):
        """Inspect all loaded tickers."""
        for ticker, data_dict in self.market_data.items():
            df = data_dict["data"]
            if df is not None and not df.empty:
                print(f"\n✅ {ticker.upper()} DataFrame columns:", df.columns.tolist())
                self.inspect_dataframe(df, name=f"{ticker.upper()} Option Data")
            else:
                print(f"\n⚠️ {ticker.upper()} data is missing or empty.")

    # ========================================================
    # Aggregation / Processing Methods
    # ========================================================
    @staticmethod
    def aggregate_by_maturity(df, agg_func=np.mean):
        """
        Aggregate option prices or volatilities by maturity.

        Returns a DataFrame with columns: Maturity, AggregatedPrice, AggregatedVol
        """
        agg_df = df.groupby("Maturity").agg({
            "OptionPrice": agg_func,
            "LocalVolatility": agg_func
        }).reset_index()
        agg_df.rename(columns={
            "OptionPrice": "AggregatedPrice",
            "LocalVolatility": "AggregatedVol"
        }, inplace=True)
        return agg_df

    @staticmethod
    def aggregate_by_strike(df, agg_func=np.mean):
        """
        Aggregate option prices or volatilities by strike.

        Returns a DataFrame with columns: Strike, AggregatedPrice, AggregatedVol
        """
        agg_df = df.groupby("Strike").agg({
            "OptionPrice": agg_func,
            "LocalVolatility": agg_func
        }).reset_index()
        agg_df.rename(columns={
            "OptionPrice": "AggregatedPrice",
            "LocalVolatility": "AggregatedVol"
        }, inplace=True)
        return agg_df

    @staticmethod
    def compute_summary_statistics(df):
        """
        Compute summary statistics for strikes, maturities, prices, and volatilities.
        """
        summary = pd.DataFrame({
            "Strike": [df["Strike"].min(), df["Strike"].max(), df["Strike"].mean()],
            "Maturity": [df["Maturity"].min(), df["Maturity"].max(), df["Maturity"].mean()],
            "OptionPrice": [df["OptionPrice"].min(), df["OptionPrice"].max(), df["OptionPrice"].mean()],
            "LocalVolatility": [df["LocalVolatility"].min(), df["LocalVolatility"].max(), df["LocalVolatility"].mean()]
        }, index=["min", "max", "mean"])
        return summary

    @staticmethod
    def normalize_columns(df, columns=None):
        """
        Min-max normalization for selected columns.
        """
        columns = columns or ["Strike", "Maturity", "OptionPrice", "LocalVolatility"]
        df_norm = df.copy()
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val > 0:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0
        return df_norm

    @staticmethod
    def filter_data(df, strike_range=None, maturity_range=None):
        """
        Filter DataFrame by strike and maturity ranges.
        """
        df_filtered = df.copy()
        if strike_range is not None:
            df_filtered = df_filtered[(df_filtered["Strike"] >= strike_range[0]) & 
                                      (df_filtered["Strike"] <= strike_range[1])]
        if maturity_range is not None:
            df_filtered = df_filtered[(df_filtered["Maturity"] >= maturity_range[0]) &
                                      (df_filtered["Maturity"] <= maturity_range[1])]
        return df_filtered

    
