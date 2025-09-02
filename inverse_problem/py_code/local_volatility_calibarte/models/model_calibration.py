import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim

from .helper import black_scholes_call, prepare_calibration_datac


# ========================================================
# Volatility Calibrator Class
# ========================================================
class VolatilityCalibrator:
    """
    Calibrates a local volatility surface using various machine learning methods.
    """
    def __init__(self, df, method="neural_net", regularization="l2", alpha=0.5,
                 device=None, seed=42, data_type="synthetic", use_pricing_model=False,
                 S0=100.0, r=0.01):

        required_cols = {"Strike", "Maturity", "OptionPrice", "LocalVolatility"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain {required_cols}")

        self.df = df.copy()
        if isinstance(method, (list, tuple)):
            self.methods = list(method)
        else:
            self.methods = [method]
        self.regularization = regularization
        self.alpha = alpha

        # Underlyer and rates for pricing-based calibration (mode=2)
        self.S0 = S0
        self.r = r

        # Prepare data based on the specified type and mode
        self.X, self.y, self.mode, self.notes = prepare_calibration_data(
            df, data_type=data_type, use_pricing_model=use_pricing_model
        )

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Storage
        self.trained_models = {}
        self.calibration_time = {}

        # RL parameterization placeholders
        self.rl_params = None
        self.rl_basis_centers = None
        self.rl_basis_scales = None
        self.rl_n_features = None

    # --------------------------------------------------------
    # Neural Network Model
    # --------------------------------------------------------
    class VolNet(nn.Module):
        """A simple feed-forward neural network for volatility surface regression."""
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.model(x)

    def train_neural_net(self, epochs=1000, lr=0.001, lambda_reg=1e-4, patience=50):
        """Trains the neural network model with early stopping."""
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(self.device)

        model = self.VolNet().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)

            # Regularization
            reg = 0.0
            if self.regularization == "l2":
                reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            elif self.regularization == "l1":
                reg = sum(torch.norm(p, 1) for p in model.parameters())
            elif self.regularization == "elastic_net":
                reg = sum(self.alpha * torch.norm(p, 1) + (1 - self.alpha) * torch.norm(p) ** 2
                          for p in model.parameters())

            total_loss = loss + lambda_reg * reg
            total_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)

            if epoch % 100 == 0:
                print(f"[NN] Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

            # Early stopping
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("[NN] Early stopping triggered.")
                    model.load_state_dict(best_model_state)
                    break

        self.trained_models["neural_net"] = model
        return model

    def predict_neural_net(self, X_input):
        """Predicts volatility using the trained neural network."""
        model = self.trained_models.get("neural_net")
        if model is None:
            raise ValueError("Neural net not trained yet.")
        model.eval()
        X_scaled = self.scaler.transform(X_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return model(X_tensor).cpu().numpy().flatten()

    # --------------------------------------------------------
    # Gaussian Process Model
    # --------------------------------------------------------
    def train_gaussian_process(self, kernel="matern"):
        """Trains a Gaussian Process Regressor."""
        if kernel == "matern":
            k = C(1.0) * Matern(length_scale=1.0, nu=1.5)
        elif kernel == "rbf":
            k = C(1.0) * RBF(length_scale=1.0)
        else:
            raise ValueError("Unsupported kernel")

        gp = GaussianProcessRegressor(
            kernel=k, n_restarts_optimizer=10, alpha=1e-4, normalize_y=True, random_state=42
        )
        gp.fit(self.X_scaled, self.y)
        self.trained_models["gaussian_process"] = gp
        return gp

    def predict_gaussian_process(self, X_input):
        """Predicts volatility and standard deviation using the trained GP."""
        gp = self.trained_models.get("gaussian_process")
        if gp is None:
            raise ValueError("Gaussian process not trained yet.")
        X_scaled = self.scaler.transform(X_input)
        y_pred, y_std = gp.predict(X_scaled, return_std=True)
        return y_pred, y_std

    # --------------------------------------------------------
    # Reinforcement Learning Calibration (Evolution Strategies)
    # --------------------------------------------------------
    def _init_rl_basis(self, n_centers_K=8, n_centers_T=6, length_scale_K=10.0, length_scale_T=0.5):
        """Initializes the RBF basis for the RL model."""
        K_vals = np.unique(self.df["Strike"].values)
        T_vals = np.unique(self.df["Maturity"].values)

        Ks = np.linspace(K_vals.min(), K_vals.max(), n_centers_K)
        Ts = np.linspace(T_vals.min(), T_vals.max(), n_centers_T)

        centers = np.array([(k, t) for t in Ts for k in Ks])  # (n_features, 2)
        scales = np.array([length_scale_K, length_scale_T])  # anisotropic

        self.rl_basis_centers = centers
        self.rl_basis_scales = scales
        self.rl_n_features = centers.shape[0]

    def _rbf_features(self, X):
        """Computes RBF features for the given input data."""
        # X: (N,2) with columns [Strike, Maturity]
        if self.rl_basis_centers is None:
            raise RuntimeError("RL basis not initialized.")

        diff = X[:, None, :] - self.rl_basis_centers[None, :, :]  # (N, n_feat, 2)
        inv_ls2 = 1.0 / (self.rl_basis_scales**2)                   # (2,)
        quad = (diff**2) * inv_ls2
        dist2 = quad.sum(axis=2)                                 # (N, n_feat)
        Phi = np.exp(-0.5 * dist2)                               # RBF features

        # Add a bias term
        Phi = np.concatenate([Phi, np.ones((Phi.shape[0], 1))], axis=1) # (N, n_feat+1)
        return Phi

    def _sigma_from_params(self, X, w):
        """Maps RL parameters (w) to a volatility surface."""
        Phi = self._rbf_features(X)
        s = Phi @ w  # raw output
        # Use a positive mapping (e.g., softplus) to ensure volatility is non-negative
        return np.log1p(np.exp(s)) + 1e-6

    def _pricing_error_mse(self, X, target_prices, w):
        """Calculates the mean squared error of option prices."""
        sigma = self._sigma_from_params(X, w)
        prices = black_scholes_call(self.S0, X[:, 0], X[:, 1], self.r, sigma)
        return float(np.mean((prices - target_prices)**2))

    def _vol_error_mse(self, X, target_vol, w):
        """Calculates the mean squared error of volatility."""
        sigma = self._sigma_from_params(X, w)
        return float(np.mean((sigma - target_vol)**2))

    def train_reinforcement_learning(
        self,
        episodes=400,
        population=32,
        sigma_action=0.05,
        lr=0.1,
        n_centers_K=8,
        n_centers_T=6,
        length_scale_K=10.0,
        length_scale_T=0.5,
        val_split=0.2,
        early_stop=50,
        verbose=True
    ):
        """
        Evolution Strategies for volatility surface calibration.
        Optimizes a vector of parameters `w` that defines the volatility surface.
        """
        # Initialize basis and params
        self._init_rl_basis(n_centers_K=n_centers_K, n_centers_T=n_centers_T,
                            length_scale_K=length_scale_K, length_scale_T=length_scale_T)
        n_feat = self.rl_n_features + 1  # bias added
        w = np.zeros(n_feat, dtype=float)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=val_split, random_state=42
        )

        best_w = w.copy()

        if self.mode == 2:
            best_loss = self._pricing_error_mse(X_val, y_val, w)
        else:
            best_loss = self._vol_error_mse(X_val, y_val, w)

        no_improve = 0

        for ep in range(episodes):
            deltas = np.random.randn(population, n_feat) * sigma_action
            rewards = np.zeros(population, dtype=float)

            # Evaluate population
            for i in range(population):
                w_i = w + deltas[i]
                if self.mode == 2:
                    loss_i = self._pricing_error_mse(X_train, y_train, w_i)
                    rewards[i] = -loss_i
                else:
                    loss_i = self._vol_error_mse(X_train, y_train, w_i)
                    rewards[i] = -loss_i

            # Normalize rewards for NES update
            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-8
            norm_rewards = (rewards - r_mean) / r_std

            # NES gradient estimator update
            grad_est = (norm_rewards[:, None] * deltas).mean(axis=0) / sigma_action
            w = w + lr * grad_est

            # Validation loss
            if self.mode == 2:
                val_loss = self._pricing_error_mse(X_val, y_val, w)
            else:
                val_loss = self._vol_error_mse(X_val, y_val, w)

            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_w = w.copy()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (ep % 20 == 0 or ep == episodes - 1):
                msg_target = "Price MSE" if self.mode == 2 else "Vol MSE"
                print(f"[RL] Ep {ep:03d} | Train {msg_target}: {(-rewards).mean():.6f} | Val {msg_target}: {val_loss:.6f}")

            if no_improve >= early_stop:
                if verbose:
                    print("[RL] Early stopping: no improvement on validation.")
                break

        # Store best params
        self.rl_params = best_w
        self.trained_models["reinforcement_learning"] = {"w": best_w}
        return best_w

    def predict_reinforcement_learning(self, X_input):
        """Predicts volatility using the trained RL model."""
        if self.rl_params is None:
            raise RuntimeError("RL model not trained. Call train_reinforcement_learning() first.")
        return self._sigma_from_params(X_input, self.rl_params)

    # --------------------------------------------------------
    # Unified Interface
    # --------------------------------------------------------
    def calibrate(self):
        """
        Runs calibration for all selected methods.
        """
        for method in self.methods:
            start = time.time()
            if method == "neural_net":
                self.train_neural_net()
            elif method == "gaussian_process":
                self.train_gaussian_process()
            elif method == "reinforcement_learning":
                self.train_reinforcement_learning()
            else:
                raise ValueError(f"Unknown method: {method}")
            end = time.time()
            self.calibration_time[method] = end - start
            print(f"[{method}] Calibration completed in {self.calibration_time[method]:.2f} seconds")

    def predict(self, X_input, method=None):
        """
        Predicts volatility for a given input using the specified method.
        If no method is specified, defaults to the first one in `self.methods`.
        """
        if method is None:
            if not self.methods:
                raise ValueError("No calibration method was specified during initialization.")
            method = self.methods[0]

        if method not in self.trained_models:
            raise ValueError(f"Model for '{method}' has not been calibrated yet.")

        if method == "neural_net":
            return self.predict_neural_net(X_input)
        elif method == "gaussian_process":
            # This method returns (pred, std), we return just the prediction
            y_pred, _ = self.predict_gaussian_process(X_input)
            return y_pred
        elif method == "reinforcement_learning":
            return self.predict_reinforcement_learning(X_input)
        else:
            raise ValueError(f"Prediction not supported for method {method}")