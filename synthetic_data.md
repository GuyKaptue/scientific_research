##  Project Description: Synthetic Local Volatility Calibration

This project establishes a controlled framework for calibrating local volatility surfaces using synthetically generated European option data. The synthetic setting allows for precise benchmarking of machine learning models against a known ground truth volatility function, enabling rigorous evaluation of calibration accuracy and robustness.

### 🧪 Synthetic Data Generation
- Option prices and local volatilities are generated from a predefined local volatility surface using the Dupire PDE or Black-Scholes framework.
- The synthetic dataset includes a dense grid of strikes and maturities, ensuring full coverage of the volatility surface.
- Both local volatility and option price targets are available, allowing for direct and indirect calibration strategies.

### 🧠 Model Calibration
Three machine learning models are trained to learn the mapping from option features (strike and maturity) to the true local volatility:
- **Neural Network (NN):** A feedforward architecture trained to minimize volatility error using backpropagation and regularization techniques.
- **Gaussian Process (GP):** A kernel-based probabilistic model that captures smoothness and uncertainty, ideal for interpolating structured surfaces.
- **Reinforcement Learning (RL):** An agent-based model that iteratively adjusts the volatility surface to minimize pricing error, receiving rewards based on accuracy improvements.

Each model is trained on the synthetic dataset using standardized preprocessing, validation splits, and early stopping to ensure fair comparison.

###  Evaluation
Model performance is assessed using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Coefficient of Determination (\( R^2 \))**

Visual diagnostics include:
- 3D plots of predicted vs. true volatility surfaces
- Scatter plots of predicted vs. true volatility values
- Comparative bar charts across models

###  Key Findings
- **Gaussian Process** delivers near-perfect calibration, accurately reconstructing the synthetic volatility surface with minimal error and high \( R^2 \).
- **Neural Networks** perform strongly, capturing non-linear patterns and generalizing well across the grid.
- **Reinforcement Learning** shows promise but struggles with direct volatility calibration, often requiring more episodes or refined reward structures to converge.

### ✅ Outcome
The synthetic calibration pipeline provides a reliable benchmark for testing and validating volatility models. It confirms that Gaussian Processes and Neural Networks are highly effective for surface reconstruction, while Reinforcement Learning may be better suited for indirect calibration or dynamic environments. These insights inform the design of real-market calibration strategies and highlight the importance of model selection in quantitative finance.

---
