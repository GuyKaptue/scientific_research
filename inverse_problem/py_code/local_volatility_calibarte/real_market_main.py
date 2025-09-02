# scientific_research/inverse_problem/py_code/local_volatility_calibarte/real_market_main.py

"""
Real Market Main Script

This script demonstrates the full pipeline for real market data:
1. Load real market option data
2. Process & inspect the data
3. Calibrate local volatility models (NN, GP, RL)
4. Evaluate predictions and compare models
"""


from local_volatility_calibarte import (
    VolatilityCalibrator,
    CalibrationEvaluator,
    RealMarketDataProcessor
)


def main():
    # 1. Load real market data
    print("📊 Fetching real market option data...")
    processor = RealMarketDataProcessor(
        tickers={"aapl": "AAPL", "goog": "GOOGL", "msft": "MSFT", "amzn": "AMZN"},
        maturities_to_fetch=5,
        n_strikes=50,
        r=0.01
    )
    market_data = processor.load_data()

    # 2. Inspect and process data for each ticker
    calibrators_all = {}
    for ticker, data_dict in market_data.items():
        df = data_dict["data"]
        if df is None or df.empty:
            print(f"⚠️ {ticker.upper()} data is missing or empty. Skipping...")
            continue

        print(f"\n🔧 Processing {ticker.upper()} option data...")
        processor.inspect_dataframe(df, name=f"{ticker.upper()} Option Data")

        # Normalize and filter if necessary
        df_norm = processor.normalize_columns(df)
        df_filtered = processor.filter_data(df_norm)

        # 3. Train multiple calibrators
        methods = ["neural_net", "gaussian_process", "reinforcement_learning"]
        calibrators = {}
        print(f"🤖 Calibrating models for {ticker.upper()}...")
        for method in methods:
            calib = VolatilityCalibrator(df_filtered, method=method, data_type="real", use_pricing_model=False)
            calib.calibrate()
            calibrators[method] = calib

        calibrators_all[ticker] = calibrators

        # 4. Evaluate calibration performance
        print(f"📊 Evaluating model performance for {ticker.upper()}...")
        evaluator = CalibrationEvaluator(calibrators, df_filtered)

        for name, calib in calibrators.items():
            X, y_true = calib.X, calib.y
            metrics, preds = evaluator.evaluate(X, y_true, model_name=name)
            print(f"\n{name.upper()} metrics for {ticker.upper()}:", metrics)

            evaluator.plot_predictions(X, y_true, model_name=name)
            evaluator.plot_surface(X, y_true, model_name=name)

        # 5. Compare across models for this ticker
        print(f"\n📊 Comparing models for {ticker.upper()}...")
        evaluator.plot_model_comparison()

    print("\n✅ Real market pipeline completed for all tickers.")


if __name__ == "__main__":
    main()
