"""
Basic ML System Usage Example

Demonstrates the fundamental usage of the ML system with
different algorithms and configurations.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basic_regression_example():
    """Demonstrate basic regression workflow."""
    logger.info("=== Basic Regression Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate sample data
        X, y = make_regression(
            n_samples=1000,
            n_features=10,
            noise=0.1,
            random_state=42
        )

        # Convert to DataFrame for better handling
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )

        # Create configuration for regression
        config = MLConfig.for_trading_regression()
        config.hyperparameter_optimization = False  # Disable for quick demo

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train model
        logger.info("Training regression model...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        logger.info(f"Training completed. R² Score: {training_results['train_metrics']['r2_score']:.4f}")

        # Make predictions
        predictions = ml_system.predict(X_test)

        # Evaluate on test set
        test_results = ml_system.evaluate(X_test, y_test, detailed=True)
        logger.info(f"Test R² Score: {test_results['r2_score']:.4f}")
        logger.info(f"Test RMSE: {test_results['rmse']:.4f}")

        # Get feature importance
        feature_importance = ml_system.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Top 3 features: {top_features}")

        return {
            'training_results': training_results,
            'test_results': test_results,
            'predictions': predictions[:5],  # First 5 predictions
            'feature_importance': feature_importance
        }

    except Exception as e:
        logger.error(f"Regression example failed: {str(e)}")
        return None


def basic_classification_example():
    """Demonstrate basic classification workflow."""
    logger.info("=== Basic Classification Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate sample data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_classes=3,
            random_state=42
        )

        # Convert to DataFrame
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create configuration for classification
        config = MLConfig.for_trading_classification()
        config.hyperparameter_optimization = False  # Disable for quick demo

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train model
        logger.info("Training classification model...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        logger.info(f"Training completed. Accuracy: {training_results['train_metrics']['accuracy']:.4f}")

        # Make predictions
        predictions = ml_system.predict(X_test)

        # Get probabilities
        pred_with_proba = ml_system.predict(X_test, return_probabilities=True)
        if isinstance(pred_with_proba, tuple):
            predictions, probabilities = pred_with_proba
            logger.info(f"Prediction probabilities shape: {probabilities.shape}")

        # Evaluate on test set
        test_results = ml_system.evaluate(X_test, y_test, detailed=True)
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {test_results['f1_score']:.4f}")

        return {
            'training_results': training_results,
            'test_results': test_results,
            'predictions': predictions[:5],  # First 5 predictions
            'class_distribution': np.bincount(y_test)
        }

    except Exception as e:
        logger.error(f"Classification example failed: {str(e)}")
        return None


def hyperparameter_optimization_example():
    """Demonstrate hyperparameter optimization."""
    logger.info("=== Hyperparameter Optimization Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate sample data
        X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )

        # Create configuration with hyperparameter optimization
        config = MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=True,
            optimization_method='random_search',
            optimization_iterations=20,  # Reduced for demo
            cross_validation_folds=3,    # Reduced for demo
            scaling_enabled=True
        )

        # Initialize ML system
        ml_system = MLSystem(config)

        # Train with hyperparameter optimization
        logger.info("Training with hyperparameter optimization...")
        training_results = ml_system.train(
            X_train, y_train,
            validation_split=0.2,
            cross_validation=True,
            save_model=False
        )

        logger.info(f"Best CV Score: {training_results.get('cross_validation', {}).get('mean_score', 'N/A')}")
        logger.info(f"Training R² Score: {training_results['train_metrics']['r2_score']:.4f}")

        # Evaluate on test set
        test_results = ml_system.evaluate(X_test, y_test)
        logger.info(f"Test R² Score: {test_results['r2_score']:.4f}")

        return {
            'training_results': training_results,
            'test_results': test_results,
            'best_params': training_results.get('best_params', {}),
            'cv_score': training_results.get('cross_validation', {}).get('mean_score', 0)
        }

    except Exception as e:
        logger.error(f"Hyperparameter optimization example failed: {str(e)}")
        return None


def algorithm_comparison_example():
    """Demonstrate comparison of different algorithms."""
    logger.info("=== Algorithm Comparison Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem

        # Generate sample data
        X, y = make_regression(n_samples=400, n_features=8, noise=0.1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )

        # Test different algorithms
        algorithms = ['linear_regression', 'random_forest', 'gradient_boosting']
        results = {}

        for algorithm in algorithms:
            logger.info(f"Testing {algorithm}...")

            # Create configuration
            config = MLConfig(
                algorithm=algorithm,
                target_type='regression',
                hyperparameter_optimization=False,
                scaling_enabled=True,
                cross_validation_folds=3
            )

            # Initialize and train
            ml_system = MLSystem(config)
            training_results = ml_system.train(
                X_train, y_train,
                cross_validation=True,
                save_model=False
            )

            # Evaluate
            test_results = ml_system.evaluate(X_test, y_test, detailed=False)

            results[algorithm] = {
                'cv_score': training_results.get('cross_validation', {}).get('mean_score', 0),
                'test_r2': test_results['r2_score'],
                'test_rmse': test_results['rmse'],
                'training_time': training_results.get('training_time', 0)
            }

            logger.info(f"{algorithm} - Test R²: {test_results['r2_score']:.4f}, "
                       f"RMSE: {test_results['rmse']:.4f}")

        # Find best algorithm
        best_algorithm = max(results.keys(), key=lambda k: results[k]['test_r2'])
        logger.info(f"Best algorithm: {best_algorithm} with R² = {results[best_algorithm]['test_r2']:.4f}")

        return {
            'results': results,
            'best_algorithm': best_algorithm,
            'best_score': results[best_algorithm]['test_r2']
        }

    except Exception as e:
        logger.error(f"Algorithm comparison example failed: {str(e)}")
        return None


def run_all_examples():
    """Run all ML system examples."""
    logger.info("Starting ML System Examples")
    logger.info("=" * 50)

    examples_results = {}

    # Run examples
    examples_results['regression'] = basic_regression_example()
    examples_results['classification'] = basic_classification_example()
    examples_results['hyperparameter_opt'] = hyperparameter_optimization_example()
    examples_results['algorithm_comparison'] = algorithm_comparison_example()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("EXAMPLES SUMMARY")
    logger.info("=" * 50)

    for example_name, result in examples_results.items():
        if result:
            logger.info(f"✓ {example_name}: SUCCESS")
        else:
            logger.info(f"✗ {example_name}: FAILED")

    return examples_results


if __name__ == "__main__":
    # Run examples
    results = run_all_examples()

    print("\nML System Examples completed!")
    print("Check the logs above for detailed results.")
