import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress


# ========================================================
# Calibration Evaluator Class
# ========================================================
class CalibrationEvaluator:
    def __init__(self, calibrators, df):
        """
        Evaluation and visualization of volatility calibration results.

        Parameters:
        - calibrators: dict of {"ModelName": VolatilityCalibrator instance}
        - df: DataFrame with calibration data
        """
        if not isinstance(calibrators, dict):
            raise ValueError("calibrators must be a dict of {name: VolatilityCalibrator}")

        self.calibrators = calibrators
        self.df = df
        self.results = {}

    # ========================================================
    # Metrics
    # ========================================================
    def evaluate(self, X, y_true, model_name):
        """
        Compute evaluation metrics for a given model.

        Parameters:
        - X: input features [[Strike, Maturity]]
        - y_true: ground truth volatility
        - model_name: name of the model to evaluate
        """
        calibrator = self.calibrators[model_name]
        preds = calibrator.predict(X)

        if isinstance(preds, tuple):  # Gaussian Process returns (y_pred, y_std)
            preds = preds[0]

        mse = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        self.results[model_name] = metrics

        return metrics, preds

    def compare_models(self, X, y_true):
        """
        Compare all calibration models stored in self.calibrators.
        """
        comparison = {}
        for name in self.calibrators.keys():
            metrics, _ = self.evaluate(X, y_true, model_name=name)
            comparison[name] = metrics
        return pd.DataFrame(comparison).T

    # ========================================================
    # Visualization
    # ========================================================
    def plot_predictions(self, X, y_true, model_name, title=None):
        """
        Scatter plot of predicted vs true volatility for one model,
        including a linear trend line.
        """
        _, preds = self.evaluate(X, y_true, model_name=model_name)

        # Plot scatter plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_true, y=preds, alpha=0.6, label="Predictions")

        # Plot perfect prediction line (y=x)
        min_val = min(y_true.min(), preds.min())
        max_val = max(y_true.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")

        # Plot linear trend line
        slope, intercept, r_value, p_value, std_err = linregress(y_true, preds)
        plt.plot(y_true, intercept + slope * y_true, 'g-', label=f"Linear Trend (R²={r_value**2:.2f})")

        plt.xlabel("True Volatility")
        plt.ylabel("Predicted Volatility")
        plt.title(title or f"{model_name}: Predictions vs True")
        plt.legend()
        plt.grid(True)
        plt.axis('square') # Ensures the plot is a square
        plt.show()


    def plot_surface(self, X, y_true, model_name, title=None):
        """
        Plot 3D volatility surface (true vs predicted) for one model.
        """
        from mpl_toolkits.mplot3d import Axes3D

        _, preds = self.evaluate(X, y_true, model_name=model_name)

        fig = plt.figure(figsize=(14, 6))

        # True Surface
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], y_true, c=y_true, cmap="viridis")
        ax1.set_title("True Local Volatility")
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Maturity")
        ax1.set_zlabel("Volatility")

        # Predicted Surface
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X[:, 0], X[:, 1], preds, c=preds, cmap="plasma")
        ax2.set_title(f"{model_name} Predicted Volatility")
        ax2.set_xlabel("Strike")
        ax2.set_ylabel("Maturity")
        ax2.set_zlabel("Volatility")

        plt.suptitle(title or f"{model_name}: Volatility Surface")
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self):
        """
        Barplot comparing stored model metrics.
        """
        if not self.results:
            raise ValueError("No model results stored. Run evaluate() or compare_models() first.")

        df_metrics = pd.DataFrame(self.results).T
        df_metrics.plot(kind="bar", figsize=(12, 7))
        plt.title("Model Performance Comparison")
        plt.ylabel("Error")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
