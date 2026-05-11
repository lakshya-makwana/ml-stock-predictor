from data_loader import load_stock_data
from feature_engineering import add_features
from model import (
    prepare_data,
    train_random_forest,
    train_logistic_regression,
    train_xgboost,
    evaluate_model,
    save_model,
    FEATURE_COLUMNS
)
from backtest import backtest_strategy
from visualization import (
    plot_stock_price,
    plot_feature_importance,
    plot_equity_curve,
    monte_carlo_simulation,
    plot_correlation_heatmap
)


def main():

    ticker = input("Enter stock ticker: ").upper()

    # Load Data
    df = load_stock_data(ticker)

    print(df.head())

    # Add Features
    df = add_features(df)

    # Visualize Data
    plot_stock_price(df)
    plot_correlation_heatmap(df)

    # Prepare Data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    split_index = int(len(df) * 0.8)

    # Train Random Forest
    print("\nTraining Random Forest...")

    rf_model = train_random_forest(X_train, y_train)

    rf_predictions = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")

    lr_model = train_logistic_regression(X_train, y_train)

    evaluate_model(
        lr_model,
        X_test,
        y_test
    )

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)

    if xgb_model:
        print("\nTraining XGBoost...")

        evaluate_model(
            xgb_model,
            X_test,
            y_test
        )

    # Save Model
    save_model(rf_model)

    # Backtest
    test_df = backtest_strategy(
        df,
        rf_predictions,
        split_index
    )

    # Visualizations
    plot_feature_importance(
        rf_model,
        FEATURE_COLUMNS
    )

    plot_equity_curve(test_df)

    monte_carlo_simulation(df)

    # Export Predictions
    test_df.to_csv('predictions.csv')

    print("\nPredictions exported to predictions.csv")


if __name__ == "__main__":
    main()