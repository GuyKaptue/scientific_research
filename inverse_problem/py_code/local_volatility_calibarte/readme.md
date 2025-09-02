#  Option Pricing, Local Volatility & Calibration Framework

This repository is a **research and experimentation framework** for working with **option pricing models**, **local volatility surfaces**, and **calibration techniques**. It supports both **synthetic data** and **real market data** workflows, allowing you to explore **machine learning-based calibration** in a controlled or realistic environment.

---

##  Why This Project Matters

In quantitative finance:

* **Volatility surfaces** describe how option volatility varies with strike and maturity. They are **key inputs** for option pricing, hedging, and risk management.
* **Market data limitations**:

  * Real market option data is often **sparse**, **noisy**, and sometimes **missing** for certain strikes/maturities.
  * Local volatility surfaces are an **ill-posed inverse problem**; recovering them from option prices is challenging.
* **Machine learning** provides **flexible, data-driven alternatives** to classical calibration techniques:

  * Neural Networks capture nonlinear relationships.
  * Gaussian Processes provide probabilistic uncertainty estimates.
  * Reinforcement Learning explores optimization strategies for calibration.

This project allows you to experiment with both **synthetic, clean data** and **real market, noisy data**, comparing ML models and evaluating performance.

---

##  Project Structure

```
scientific_research/
│── README.md
│── inverse_problem/
│   ├── pdf_doc/
│   ├── py_code/
│   │   └── local_volatility_calibarte/
│   │       ├── __init__.py
│   │       ├── evaluator_prediction/
│   │       │   └── evalution_prediction.py
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── helper.py
│   │       │   └── model_calibration.py
│   │       ├── synthetic_data/
│   │       │   └── synthetic_data_generator.py
│   │       ├── visualizer_processor/  
│   │       │   ├── __init__.py
│   │       │   ├── data_processor.py
│   │       │   └── data_visualizer.py
│   │       ├── real_market_data/
│   │       │   ├── real_market_data_fetcher.py
│   │       │   └── real_market_data_processor.py    
│   │       ├── synthetic_main.py
│   │       └── real_market_main.py
│   └── requirements.txt
```

---

##  1. Synthetic Data Generation

**Synthetic data** is used to **create controlled experiments** where we know the ground truth volatility surface.

### What it does:

* Generates option prices over a grid of strikes and maturities.
* Computes **arbitrage-free local volatility surfaces**.
* Option prices are computed using **Black–Scholes** or other pricing formulas.
* Data can be filtered, normalized, and visualized.

### Example

```python
from synthetic_data.synthetic_data_generator import SyntheticDataGenerator
from visualizer_processor.data_processor import DataProcessor
from visualizer_processor.data_visualizer import DataVisualizer

# Generate synthetic option data
gen = SyntheticDataGenerator(S0=100, r=0.01, q=0.005, n_strikes=20, n_maturities=10)
df = gen.generate()

# Process
processor = DataProcessor(df)
processor.summary()
df_norm = processor.normalize()
df_filtered = processor.filter_data(strike_range=(90, 120), maturity_range=(0.2, 1.5))

# Visualize
viz = DataVisualizer(df)
viz.plot_heatmaps()
viz.plot_surfaces()
viz.plot_slices()
```

**Outputs:**

* Heatmaps of volatilities and option prices
* Cross-sections (smiles at fixed maturities, term structures at fixed strikes)
* 3D volatility surfaces

---

##  2. Volatility Calibration (Synthetic Data)

The `VolatilityCalibrator` learns the mapping:

$$
(\text{Strike}, \text{Maturity}) \;\longrightarrow\; \text{Local Volatility}
$$

### Supported Methods:

* **Neural Networks (VolNet)** – Fully connected, flexible nonlinear regression
* **Gaussian Processes (GP)** – Probabilistic, provides uncertainty, good for small datasets
* **Reinforcement Learning (RL)** – Learns optimal basis coefficients via evolutionary strategies

### Example

```python
from models.model_calibration import VolatilityCalibrator

calib = VolatilityCalibrator(df, method="neural_net", data_type="synthetic")
calib.calibrate()
preds = calib.predict([[100, 1.0], [110, 0.5]])
```

---

##  3. Real Market Data Calibration

Unlike synthetic data, **real market data** often **does not provide local volatility surfaces**.

### Key differences:

* Only **option prices** and **implied volatilities** are available.
* Local Volatility may be **null or unreliable** due to noise or missing data.
* **Adaptation:** Use **OptionPrice as the target variable** for calibration instead of LocalVolatility.

### Workflow:

```python
from real_market_data.real_market_data_processor import RealMarketDataProcessor
from models.model_calibration import VolatilityCalibrator
from evaluator_prediction.evalution_prediction import CalibrationEvaluator

# Load real market option data
processor = RealMarketDataProcessor(
    tickers={"aapl": "AAPL", "goog": "GOOGL"},
    maturities_to_fetch=5,
    n_strikes=50,
    r=0.01
)
market_data = processor.load_data()

for ticker, data_dict in market_data.items():
    df = data_dict["data"]
    df_norm = processor.normalize_columns(df)
    df_filtered = processor.filter_data(df_norm)

    # Train models using OptionPrice as the target
    calibrators = {}
    methods = ["neural_net", "gaussian_process", "reinforcement_learning"]
    for method in methods:
        calib = VolatilityCalibrator(
            df_filtered,
            method=method,
            data_type="real",
            use_pricing_model=False,
            target_column="OptionPrice"
        )
        calib.calibrate()
        calibrators[method] = calib

    # Evaluate
    evaluator = CalibrationEvaluator(calibrators, df_filtered)
    for name, calib in calibrators.items():
        X, y_true = calib.X, calib.y
        metrics, preds = evaluator.evaluate(X, y_true, model_name=name)
        print(f"{name.upper()} metrics for {ticker.upper()}:", metrics)
        evaluator.plot_predictions(X, y_true, model_name=name)
        evaluator.plot_surface(X, y_true, model_name=name)

    evaluator.plot_model_comparison()
```

**Highlights:**

* Models **predict option prices** instead of local volatility.
* Evaluation metrics (MSE, RMSE, MAE, R²) are computed on **OptionPrice**.
* Provides a realistic scenario for **ML calibration in real markets**.

---

##  4. Evaluation & Model Comparison

`CalibrationEvaluator` supports:

* **Metrics**: MSE, RMSE, MAE, R²
* **Visual diagnostics**:

  * Predicted vs true scatter plots
  * 3D surface plots
  * Model comparison barplots

**Notes:**

* For **synthetic data**, ground truth local volatility is available.
* For **real data**, OptionPrice serves as the target; evaluation reflects price-fitting performance rather than volatility surface accuracy.

---

##  Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

* `numpy`, `pandas`, `scipy`
* `matplotlib`, `seaborn`, `plotly`
* `scikit-learn`, `torch`, `yfinance`

---

##  Future Extensions

* Add **market data loader** with more tickers and historical depth
* Implement **stochastic volatility models** (Heston, SABR)
* Extend RL calibration to **actor-critic methods**
* Build **interactive dashboards** (Streamlit / Plotly Dash) for visual exploration

---



