"""
Trading-Specific ML Examples

Demonstrates how to use the ML system for trading scenarios
including price prediction, signal classification, and market analysis.
"""

import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_trading_data(n_samples=1000, n_features=15):
    """Generate synthetic trading data with realistic patterns."""
    np.random.seed(42)

    # Generate base price movement
    price_change = np.cumsum(np.random.normal(0, 0.02, n_samples))

    # Technical indicators (features)
    features = {}

    # Moving averages
    features['ma_5'] = np.convolve(price_change, np.ones(5)/5, mode='same')
    features['ma_20'] = np.convolve(price_change, np.ones(20)/20, mode='same')
    features['ma_50'] = np.convolve(price_change, np.ones(50)/50, mode='same')

    # Price-based features
    features['price_momentum'] = np.gradient(price_change)
    features['price_volatility'] = pd.Series(price_change).rolling(20).std().fillna(0).values

    # Volume indicators
    features['volume'] = np.random.lognormal(10, 1, n_samples)
    features['volume_ma'] = np.convolve(features['volume'], np.ones(10)/10, mode='same')

    # RSI-like indicator
    price_deltas = np.diff(price_change, prepend=0)
    gains = np.where(price_deltas > 0, price_deltas, 0)
    losses = np.where(price_deltas < 0, -price_deltas, 0)
    avg_gains = pd.Series(gains).rolling(14).mean().fillna(0).values
    avg_losses = pd.Series(losses).rolling(14).mean().fillna(0).values
    rs = avg_gains / (avg_losses + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger bands
    price_std = pd.Series(price_change).rolling(20).std().fillna(0).values
    features['bb_upper'] = features['ma_20'] + (2 * price_std)
    features['bb_lower'] = features['ma_20'] - (2 * price_std)
    features['bb_position'] = (price_change - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)

    # MACD-like
    ema_12 = pd.Series(price_change).ewm(span=12).mean().fillna(0).values
    ema_26 = pd.Series(price_change).ewm(span=26).mean().fillna(0).values
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = pd.Series(features['macd']).ewm(span=9).mean().fillna(0).values

    # Market microstructure
    features['bid_ask_spread'] = np.random.gamma(2, 0.001, n_samples)
    features['order_flow'] = np.random.normal(0, 1, n_samples)

    # Create feature matrix
    feature_matrix = np.column_stack([features[key] for key in sorted(features.keys())])
    feature_names = sorted(features.keys())

    return feature_matrix, price_change, feature_names


def price_prediction_example():
    """Demonstrate price prediction for trading."""
    logger.info("=== Price Prediction Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate trading data
        X, price_series, feature_names = generate_trading_data(n_samples=2000)

        # Create price prediction targets (next period return)
        y = np.roll(price_series, -1) - price_series  # Next period price change
        y = y[:-1]  # Remove last element
        X = X[:-1]  # Align features

        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split data (time series aware)
        split_idx = int(0.8 * len(X_df))
        X_train, X_test = X_df[:split_idx], X_df[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Configure for trading regression
        config = MLConfig.for_trading_regression()
        config.hyperparameter_optimization = True
        config.optimization_method = 'random_search'
        config.optimization_iterations = 30

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train price prediction model
        logger.info("Training price prediction model...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        # Make predictions
        predictions = ml_system.predict(X_test)

        # Evaluate model
        test_results = ml_system.evaluate(X_test, y_test, detailed=True)

        logger.info(f"Price prediction R² score: {test_results['r2_score']:.4f}")
        logger.info(f"Price prediction RMSE: {test_results['rmse']:.6f}")

        # Calculate trading-specific metrics
        prediction_accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
        logger.info(f"Direction prediction accuracy: {prediction_accuracy:.4f}")

        # Feature importance for trading insights
        feature_importance = ml_system.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top 5 predictive features:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")

        return {
            'r2_score': test_results['r2_score'],
            'rmse': test_results['rmse'],
            'direction_accuracy': prediction_accuracy,
            'top_features': top_features if 'top_features' in locals() else None,
            'training_score': training_results['train_metrics']['r2_score']
        }

    except Exception as e:
        logger.error(f"Price prediction example failed: {str(e)}")
        return None


def trading_signal_classification_example():
    """Demonstrate trading signal classification."""
    logger.info("=== Trading Signal Classification Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate trading data
        X, price_series, feature_names = generate_trading_data(n_samples=1500)

        # Create trading signals (buy/hold/sell based on future returns)
        future_returns = np.roll(price_series, -5) - price_series  # 5-period forward return
        future_returns = future_returns[:-5]
        X = X[:-5]

        # Define signal thresholds
        buy_threshold = np.percentile(future_returns, 70)
        sell_threshold = np.percentile(future_returns, 30)

        # Create signals: 0=sell, 1=hold, 2=buy
        signals = np.where(future_returns > buy_threshold, 2,
                          np.where(future_returns < sell_threshold, 0, 1))

        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split data (time series aware)
        split_idx = int(0.8 * len(X_df))
        X_train, X_test = X_df[:split_idx], X_df[split_idx:]
        y_train, y_test = signals[:split_idx], signals[split_idx:]

        logger.info(f"Signal distribution - Buy: {np.sum(signals==2)}, "
                   f"Hold: {np.sum(signals==1)}, Sell: {np.sum(signals==0)}")

        # Configure for trading classification
        config = MLConfig.for_trading_classification()
        config.hyperparameter_optimization = True
        config.optimization_method = 'random_search'
        config.optimization_iterations = 25

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train signal classification model
        logger.info("Training trading signal classifier...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        # Make predictions with probabilities
        predictions, probabilities = ml_system.predict(X_test, return_probabilities=True)

        # Evaluate model
        test_results = ml_system.evaluate(X_test, y_test, detailed=True)

        logger.info(f"Signal classification accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Signal classification F1 score: {test_results['f1_score']:.4f}")

        # Calculate class-specific metrics
        from sklearn.metrics import classification_report
        report = classification_report(y_test, predictions, target_names=['Sell', 'Hold', 'Buy'])
        logger.info(f"Classification report:\n{report}")

        # Analyze prediction confidence
        max_probabilities = np.max(probabilities, axis=1)
        high_confidence_mask = max_probabilities > 0.7
        high_confidence_accuracy = np.mean(predictions[high_confidence_mask] == y_test[high_confidence_mask])

        logger.info(f"High confidence predictions: {np.sum(high_confidence_mask)}/{len(predictions)}")
        logger.info(f"High confidence accuracy: {high_confidence_accuracy:.4f}")

        return {
            'accuracy': test_results['accuracy'],
            'f1_score': test_results['f1_score'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'high_confidence_accuracy': high_confidence_accuracy,
            'signal_distribution': {
                'buy': int(np.sum(signals==2)),
                'hold': int(np.sum(signals==1)),
                'sell': int(np.sum(signals==0))
            }
        }

    except Exception as e:
        logger.error(f"Trading signal classification example failed: {str(e)}")
        return None


def market_regime_detection_example():
    """Demonstrate market regime detection."""
    logger.info("=== Market Regime Detection Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate trading data with regime changes
        n_samples = 1200
        X, price_series, feature_names = generate_trading_data(n_samples)

        # Create artificial regime labels based on volatility and trend
        volatility = pd.Series(price_series).rolling(20).std().fillna(0).values
        trend = pd.Series(price_series).rolling(50).apply(lambda x: (x[-1] - x[0]) / len(x), raw=False).fillna(0).values

        # Define regimes: 0=bear_low_vol, 1=bear_high_vol, 2=bull_low_vol, 3=bull_high_vol
        vol_threshold = np.median(volatility)
        trend_threshold = 0

        regimes = np.where(
            (trend > trend_threshold) & (volatility <= vol_threshold), 2,  # bull_low_vol
            np.where(
                (trend > trend_threshold) & (volatility > vol_threshold), 3,  # bull_high_vol
                np.where(
                    (trend <= trend_threshold) & (volatility <= vol_threshold), 0,  # bear_low_vol
                    1  # bear_high_vol
                )
            )
        )

        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split data
        split_idx = int(0.8 * len(X_df))
        X_train, X_test = X_df[:split_idx], X_df[split_idx:]
        y_train, y_test = regimes[:split_idx], regimes[split_idx:]

        regime_names = ['Bear Low Vol', 'Bear High Vol', 'Bull Low Vol', 'Bull High Vol']
        logger.info("Regime distribution:")
        for i, name in enumerate(regime_names):
            count = np.sum(regimes == i)
            logger.info(f"  {name}: {count} ({count/len(regimes)*100:.1f}%)")

        # Configure for regime classification
        config = MLConfig(
            algorithm='random_forest',
            target_type='classification',
            hyperparameter_optimization=True,
            optimization_method='random_search',
            optimization_iterations=20,
            scaling_enabled=True,
            cross_validation_folds=5
        )

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train regime detection model
        logger.info("Training market regime detector...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        # Make predictions
        predictions = ml_system.predict(X_test)

        # Evaluate model
        test_results = ml_system.evaluate(X_test, y_test, detailed=True)

        logger.info(f"Regime detection accuracy: {test_results['accuracy']:.4f}")

        # Analyze regime transitions
        regime_transitions = np.sum(np.diff(predictions) != 0)
        actual_transitions = np.sum(np.diff(y_test) != 0)

        logger.info(f"Predicted regime transitions: {regime_transitions}")
        logger.info(f"Actual regime transitions: {actual_transitions}")

        # Feature importance for regime detection
        feature_importance = ml_system.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top features for regime detection:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")

        return {
            'accuracy': test_results['accuracy'],
            'f1_score': test_results['f1_score'],
            'predicted_transitions': int(regime_transitions),
            'actual_transitions': int(actual_transitions),
            'regime_distribution': {
                regime_names[i]: int(np.sum(regimes == i)) for i in range(4)
            },
            'top_features': top_features if 'top_features' in locals() else None
        }

    except Exception as e:
        logger.error(f"Market regime detection example failed: {str(e)}")
        return None


def portfolio_optimization_ml_example():
    """Demonstrate ML-assisted portfolio optimization."""
    logger.info("=== Portfolio Optimization ML Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate multi-asset data
        n_assets = 5
        n_samples = 800

        # Generate correlated asset returns
        correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)

        # Generate returns
        mean_returns = np.random.uniform(0.0005, 0.002, n_assets)
        volatilities = np.random.uniform(0.01, 0.03, n_assets)

        # Simulate asset returns
        asset_returns = np.random.multivariate_normal(
            mean_returns,
            np.outer(volatilities, volatilities) * correlation_matrix,
            n_samples
        )

        # Create features from asset returns
        features = {}

        for i in range(n_assets):
            asset_name = f"asset_{i}"
            returns = asset_returns[:, i]

            # Technical features for each asset
            features[f'{asset_name}_return'] = returns
            features[f'{asset_name}_ma_5'] = pd.Series(returns).rolling(5).mean().fillna(0).values
            features[f'{asset_name}_ma_20'] = pd.Series(returns).rolling(20).mean().fillna(0).values
            features[f'{asset_name}_vol'] = pd.Series(returns).rolling(10).std().fillna(0).values
            features[f'{asset_name}_momentum'] = pd.Series(returns).rolling(5).sum().fillna(0).values

        # Portfolio-level features
        portfolio_returns = np.mean(asset_returns, axis=1)  # Equal weight portfolio
        features['portfolio_return'] = portfolio_returns
        features['portfolio_vol'] = pd.Series(portfolio_returns).rolling(20).std().fillna(0).values
        features['max_drawdown'] = pd.Series(np.cumsum(portfolio_returns)).rolling(50).apply(
            lambda x: (x.iloc[-1] - x.max()) / x.max() if x.max() != 0 else 0, raw=False
        ).fillna(0).values

        # Create feature matrix
        X = np.column_stack([features[key] for key in sorted(features.keys())])
        feature_names = sorted(features.keys())

        # Target: predict next period portfolio return
        y = np.roll(portfolio_returns, -1)[:-1]
        X = X[:-1]

        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split data
        split_idx = int(0.8 * len(X_df))
        X_train, X_test = X_df[:split_idx], X_df[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Portfolio optimization - Assets: {n_assets}, Samples: {len(X_df)}")

        # Configure for portfolio return prediction
        config = MLConfig.for_trading_regression()
        config.algorithm = 'gradient_boosting'  # Good for non-linear relationships
        config.hyperparameter_optimization = True
        config.optimization_iterations = 25

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train portfolio model
        logger.info("Training portfolio optimization model...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        # Make predictions
        predictions = ml_system.predict(X_test)

        # Evaluate model
        test_results = ml_system.evaluate(X_test, y_test, detailed=True)

        logger.info(f"Portfolio prediction R² score: {test_results['r2_score']:.4f}")

        # Calculate Sharpe ratio improvement
        actual_sharpe = np.mean(y_test) / np.std(y_test) if np.std(y_test) > 0 else 0
        predicted_sharpe = np.mean(predictions) / np.std(predictions) if np.std(predictions) > 0 else 0

        logger.info(f"Actual returns Sharpe ratio: {actual_sharpe:.4f}")
        logger.info(f"Predicted returns Sharpe ratio: {predicted_sharpe:.4f}")

        # Feature importance for portfolio insights
        feature_importance = ml_system.get_feature_importance()
        if feature_importance:
            # Group by asset
            asset_importance = {}
            for feature, importance in feature_importance.items():
                if 'asset_' in feature:
                    asset = feature.split('_')[1]
                    if asset not in asset_importance:
                        asset_importance[asset] = 0
                    asset_importance[asset] += importance

            logger.info("Asset importance for portfolio prediction:")
            for asset, importance in sorted(asset_importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  Asset {asset}: {importance:.4f}")

        return {
            'r2_score': test_results['r2_score'],
            'rmse': test_results['rmse'],
            'actual_sharpe': actual_sharpe,
            'predicted_sharpe': predicted_sharpe,
            'n_assets': n_assets,
            'asset_importance': asset_importance if 'asset_importance' in locals() else None
        }

    except Exception as e:
        logger.error(f"Portfolio optimization example failed: {str(e)}")
        return None


def run_trading_examples():
    """Run all trading-specific ML examples."""
    logger.info("Starting Trading-Specific ML Examples")
    logger.info("=" * 70)

    examples_results = {}

    # Run trading examples
    examples_results['price_prediction'] = price_prediction_example()
    examples_results['signal_classification'] = trading_signal_classification_example()
    examples_results['regime_detection'] = market_regime_detection_example()
    examples_results['portfolio_optimization'] = portfolio_optimization_ml_example()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRADING EXAMPLES SUMMARY")
    logger.info("=" * 70)

    successful_examples = 0
    for example_name, result in examples_results.items():
        if result:
            logger.info(f"✓ {example_name}: SUCCESS")
            successful_examples += 1
        else:
            logger.info(f"✗ {example_name}: FAILED")

    logger.info(f"\nSuccessful examples: {successful_examples}/{len(examples_results)}")

    # Print key insights
    if examples_results['price_prediction']:
        direction_acc = examples_results['price_prediction']['direction_accuracy']
        logger.info(f"Price direction prediction accuracy: {direction_acc:.1%}")

    if examples_results['signal_classification']:
        signal_acc = examples_results['signal_classification']['accuracy']
        logger.info(f"Trading signal classification accuracy: {signal_acc:.1%}")

    if examples_results['regime_detection']:
        regime_acc = examples_results['regime_detection']['accuracy']
        logger.info(f"Market regime detection accuracy: {regime_acc:.1%}")

    if examples_results['portfolio_optimization']:
        portfolio_r2 = examples_results['portfolio_optimization']['r2_score']
        logger.info(f"Portfolio return prediction R²: {portfolio_r2:.3f}")

    return examples_results


if __name__ == "__main__":
    # Run trading examples
    results = run_trading_examples()

    print("\nTrading-Specific ML Examples completed!")
    print("Check the logs above for detailed results and insights.")
