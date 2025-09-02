import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, df):
        """
        Class to handle processing of synthetic option data.

        Parameters:
        - df: DataFrame with columns [Strike, Maturity, LocalVolatility, OptionPrice]
        """
        self.df = df.copy()

    # ========================================================
    # 1. Basic info / summary
    # ========================================================
    def summary(self):
        """
        Print summary statistics of the dataset.
        """
        print("Dataset Shape:", self.df.shape)
        print("\nHead of Data:\n", self.df.head())
        print("\nSummary Stats:\n", self.df.describe())

    # ========================================================
    # 2. Normalization / Scaling
    # ========================================================
    def normalize(self, cols=["Strike", "Maturity", "LocalVolatility", "OptionPrice"]):
        """
        Normalize selected columns to [0,1].
        """
        for col in cols:
            min_val, max_val = self.df[col].min(), self.df[col].max()
            self.df[col + "_norm"] = (self.df[col] - min_val) / (max_val - min_val)
        return self.df

    # ========================================================
    # 3. Filtering
    # ========================================================
    def filter_data(self, strike_range=None, maturity_range=None):
        """
        Filter the dataset by strike and/or maturity ranges.
        """
        df_filtered = self.df.copy()
        if strike_range:
            df_filtered = df_filtered[
                (df_filtered["Strike"] >= strike_range[0]) &
                (df_filtered["Strike"] <= strike_range[1])
            ]
        if maturity_range:
            df_filtered = df_filtered[
                (df_filtered["Maturity"] >= maturity_range[0]) &
                (df_filtered["Maturity"] <= maturity_range[1])
            ]
        return df_filtered

    # ========================================================
    # 4. Aggregation
    # ========================================================
    def aggregate(self, by="Maturity", agg_col="OptionPrice"):
        """
        Aggregate option prices or volatilities by maturity or strike.

        Parameters:
        - by: "Maturity" or "Strike"
        - agg_col: "OptionPrice" or "LocalVolatility"
        """
        return self.df.groupby(by)[agg_col].mean().reset_index()

    # ========================================================
    # 5. Export
    # ========================================================
    def export(self, filename="synthetic_data.csv"):
        """
        Save the processed dataset to CSV.
        """
        self.df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
