# scientific_research/inverse_problem/py_code/local_volatility_calibarte/synthetic_main.py

"""
Synthetic Main Script

This script demonstrates the full pipeline:
1. Generate synthetic option data
2. Process & visualize the data
3. Calibrate local volatility models (NN, GP, RL)
4. Evaluate predictions and compare models
"""

from local_volatility_calibarte import (
    SyntheticDataGenerator,
    DataProcessor,
    DataVisualizer,
    VolatilityCalibrator,
    CalibrationEvaluator,
)


def main():
    # 1. Generate synthetic data
    print("📊 Generating synthetic option data...")
    generator = SyntheticDataGenerator(S0=100, r=0.01, q=0.005, n_strikes=20, n_maturities=10)
    df = generator.generate()

    # 2. Process the data
    print("🔧 Processing synthetic data...")
    processor = DataProcessor(df)
    processor.summary()
    df_norm = processor.normalize()

    # 3. Visualize the data
    print("📈 Visualizing volatility surfaces...")
    viz = DataVisualizer(df)
    viz.plot_heatmaps()
    viz.plot_surfaces()
    viz.plot_slices()

    # 4. Train multiple calibrators
    methods = ["neural_net", "gaussian_process", "reinforcement_learning"]
    calibrators = {}

    print("🤖 Calibrating models...")
    for method in methods:
        calib = VolatilityCalibrator(df, method=method, data_type="synthetic", use_pricing_model=False)
        calib.calibrate()
        calibrators[method] = calib

    # 5. Evaluate calibration performance
    print("📊 Evaluating model performance...")
    evaluator = CalibrationEvaluator(calibrators, df)

    for name, calib in calibrators.items():
        X, y_true = calib.X, calib.y
        metrics, preds = evaluator.evaluate(X, y_true, model_name=name)
        print(f"\n{name.upper()} metrics:", metrics)

        evaluator.plot_predictions(X, y_true, model_name=name)
        evaluator.plot_surface(X, y_true, model_name=name)

    # 6. Compare across models
    print("\n📊 Comparing models...")
    evaluator.plot_model_comparison()


if __name__ == "__main__":
    main()
