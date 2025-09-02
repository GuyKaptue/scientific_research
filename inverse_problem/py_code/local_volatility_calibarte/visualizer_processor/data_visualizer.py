import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

class DataVisualizer:
    def __init__(self, df, n_strikes=20, n_maturities=10):
        """
        Visualization class for synthetic option data.

        Parameters:
        - synthetic_data_generator: Instance of SyntheticDataGenerator
        - df: DataFrame with columns [Strike, Maturity, LocalVolatility, OptionPrice]
        - n_strikes: number of strike points (grid shape)
        - n_maturities: number of maturity points (grid shape)
        """
        self.df = df
        self.n_strikes = n_strikes
        self.n_maturities = n_maturities

        # Reshape into grids for surface/contour plots
        self.X = df["Strike"].values.reshape(n_maturities, n_strikes)
        self.Y = df["Maturity"].values.reshape(n_maturities, n_strikes)
        self.Z_vol = df["LocalVolatility"].values.reshape(n_maturities, n_strikes)
        self.Z_price = df["OptionPrice"].values.reshape(n_maturities, n_strikes)

    # ========================================================
    # 1. Heatmaps
    # ========================================================
    def plot_heatmaps(self):
        pivot_vol = self.df.pivot_table(index="Maturity", columns="Strike", values="LocalVolatility")
        pivot_price = self.df.pivot_table(index="Maturity", columns="Strike", values="OptionPrice")

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_vol, cmap="viridis")
        plt.title("Local Volatility Heatmap")
        plt.xlabel("Strike")
        plt.ylabel("Maturity")
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_price, cmap="plasma")
        plt.title("Option Price Heatmap")
        plt.xlabel("Strike")
        plt.ylabel("Maturity")
        plt.show()

    # ========================================================
    # 2. Contour plots
    # ========================================================
    def plot_contours(self):
        plt.figure(figsize=(10, 6))
        cp = plt.contourf(self.X, self.Y, self.Z_vol, cmap="viridis", levels=15)
        plt.colorbar(cp)
        plt.title("Local Volatility Contour")
        plt.xlabel("Strike")
        plt.ylabel("Maturity")
        plt.show()

        plt.figure(figsize=(10, 6))
        cp = plt.contourf(self.X, self.Y, self.Z_price, cmap="plasma", levels=15)
        plt.colorbar(cp)
        plt.title("Option Price Contour")
        plt.xlabel("Strike")
        plt.ylabel("Maturity")
        plt.show()

    # ========================================================
    # 3. 3D Surface plots
    # ========================================================
    def plot_surfaces(self):
        fig = plt.figure(figsize=(12, 6))

        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z_vol, cmap="viridis")
        ax.set_title("Local Volatility Surface")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturity")
        ax.set_zlabel("Volatility")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(self.X, self.Y, self.Z_price, cmap="plasma")
        ax2.set_title("Option Price Surface")
        ax2.set_xlabel("Strike")
        ax2.set_ylabel("Maturity")
        ax2.set_zlabel("Price")

        plt.show()

    # ========================================================
    # 4. Smiles and term structures
    # ========================================================
    def plot_slices(self, maturities_to_plot=[0.25, 1.0, 2.0], strikes_to_plot=[90, 100, 110]):
        # Smiles
        plt.figure(figsize=(10, 6))
        for T in maturities_to_plot:
            slice_df = self.df[np.isclose(self.df["Maturity"], T)]
            plt.plot(slice_df["Strike"], slice_df["LocalVolatility"], label=f"T={T}")
        plt.title("Volatility Smiles at Different Maturities")
        plt.xlabel("Strike")
        plt.ylabel("Local Volatility")
        plt.legend()
        plt.show()

        # Term structures
        plt.figure(figsize=(10, 6))
        for K in strikes_to_plot:
            slice_df = self.df[np.isclose(self.df["Strike"], K)]
            plt.plot(slice_df["Maturity"], slice_df["LocalVolatility"], label=f"K={K}")
        plt.title("Volatility Term Structure at Different Strikes")
        plt.xlabel("Maturity")
        plt.ylabel("Local Volatility")
        plt.legend()
        plt.show()

    # ========================================================
    # 5. Interactive 3D Plotly surfaces
    # ========================================================
    def plot_interactive_surfaces(self):
        fig_vol = px.surface(
            x=np.unique(self.df["Strike"]),
            y=np.unique(self.df["Maturity"]),
            z=self.Z_vol,
            title="Interactive Local Volatility Surface",
            labels={"x": "Strike", "y": "Maturity", "z": "Volatility"}
        )
        fig_vol.show()

        fig_price = px.surface(
            x=np.unique(self.df["Strike"]),
            y=np.unique(self.df["Maturity"]),
            z=self.Z_price,
            title="Interactive Option Price Surface",
            labels={"x": "Strike", "y": "Maturity", "z": "Price"}
        )
        fig_price.show()
