## 📌 Project Description: Real Market Local Volatility Calibration

This project focuses on the calibration of local volatility surfaces using real-world European option data from major publicly traded companies: **AAPL**, **GOOG**, **MSFT**, and **AMZN**. The objective is to infer a smooth, arbitrage-free local volatility function \( \sigma(K, T) \) that accurately reflects market dynamics and can be used for pricing and risk management.

### 🔍 Data Acquisition
- Option chain data is retrieved via the Yahoo Finance API.
- Bid-ask midpoints are used to estimate market prices.
- Implied volatilities are extracted and interpolated to construct a smooth surface.
- The **Dupire equation** is applied to derive local volatility from the implied volatility grid.

### 🧠 Model Calibration
Three machine learning models are trained to learn the mapping from option features (strike and maturity) to local volatility:
- **Neural Network (NN):** A deep learning model optimized via backpropagation to minimize volatility error.
- **Gaussian Process (GP):** A non-parametric Bayesian model that provides both predictions and uncertainty estimates.
- **Reinforcement Learning (RL):** An agent-based model that iteratively adjusts the volatility surface to minimize pricing error.

Each model is trained independently on each ticker’s data, using a unified pipeline that includes feature scaling, validation splits, and early stopping.

### 📊 Evaluation
Model performance is assessed using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Coefficient of Determination (\( R^2 \))**

Visual diagnostics include:
- 3D volatility surface plots
- Scatter plots of predicted vs. true volatility
- Comparative bar charts across models and tickers

### 🧾 Key Findings
- **Gaussian Process** consistently delivers the most accurate and stable results, with \( R^2 \) values exceeding 0.9999 across all tickers.
- **Neural Networks** perform robustly, capturing non-linear patterns with high fidelity.
- **Reinforcement Learning**, while conceptually promising, underperforms in direct volatility calibration and may require architectural refinement or alternative reward structures.

### ✅ Outcome
This work demonstrates the feasibility and effectiveness of machine learning models—especially Gaussian Processes and Neural Networks—for calibrating local volatility surfaces from real market data. The pipeline is modular, extensible, and suitable for deployment in quantitative finance applications such as option pricing, hedging, and volatility forecasting.

---
