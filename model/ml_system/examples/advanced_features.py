"""
Advanced ML System Features Example

Demonstrates advanced features like feature engineering,
model persistence, ensemble methods, and pipeline management.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def feature_engineering_example():
    """Demonstrate advanced feature engineering capabilities."""
    logger.info("=== Feature Engineering Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem
        from model.ml_system.utils.feature_engineering import FeatureEngineer

        # Generate sample data with some noise
        X, y = make_regression(
            n_samples=800,
            n_features=15,
            noise=0.2,
            random_state=42
        )

        # Add some correlated features and outliers
        X = np.hstack([X, X[:, :3] + np.random.normal(0, 0.1, (X.shape[0], 3))])
        X[10, :] = X[10, :] * 5  # Add outlier

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Initialize feature engineer
        engineer = FeatureEngineer()

        # Apply comprehensive feature engineering
        logger.info("Applying feature engineering...")
        X_engineered = engineer.fit_transform(
            X_df,
            feature_selection=True,
            n_features=10,  # Select top 10 features
            scaling=True,
            polynomial_features=True,
            polynomial_degree=2,
            interaction_features=True
        )

        logger.info(f"Original features: {X_df.shape[1]}")
        logger.info(f"Engineered features: {X_engineered.shape[1]}")

        # Get feature importance scores
        feature_scores = engineer.get_feature_scores()
        if feature_scores:
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 selected features: {top_features}")

        # Train model with engineered features
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42
        )

        config = MLConfig.for_trading_regression()
        config.scaling_enabled = False  # Already scaled by feature engineer

        ml_system = MLSystem(config)
        training_results = ml_system.train(X_train, y_train, save_model=False)
        test_results = ml_system.evaluate(X_test, y_test)

        logger.info(f"Model performance with engineered features - R²: {test_results['r2_score']:.4f}")

        return {
            'original_features': X_df.shape[1],
            'engineered_features': X_engineered.shape[1],
            'feature_scores': feature_scores,
            'model_performance': test_results['r2_score']
        }

    except Exception as e:
        logger.error(f"Feature engineering example failed: {str(e)}")
        return None


def model_persistence_example():
    """Demonstrate model saving, loading, and versioning."""
    logger.info("=== Model Persistence Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem
        from model.ml_system.training.model_persistence import ModelPersistence

        # Generate data
        X, y = make_classification(
            n_samples=600,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Create ML system
        config = MLConfig.for_trading_classification()
        ml_system = MLSystem(config)

        # Train model
        logger.info("Training model for persistence example...")
        training_results = ml_system.train(X_train, y_train, save_model=True)

        # Initialize persistence manager
        persistence = ModelPersistence()

        # Save model with metadata
        model_info = {
            'algorithm': config.algorithm,
            'target_type': config.target_type,
            'training_samples': len(X_train),
            'feature_count': X_train.shape[1],
            'training_score': training_results['train_metrics']['accuracy']
        }

        model_path = persistence.save_model(
            ml_system.model,
            'trading_classifier_v1',
            metadata=model_info,
            create_version=True
        )

        logger.info(f"Model saved to: {model_path}")

        # Load model and test
        loaded_model = persistence.load_model('trading_classifier_v1')
        if loaded_model:
            # Test loaded model
            predictions = loaded_model.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            logger.info(f"Loaded model accuracy: {accuracy:.4f}")

            # Load metadata
            metadata = persistence.load_metadata('trading_classifier_v1')
            if metadata:
                logger.info(f"Model metadata: {metadata}")

        # List available models
        available_models = persistence.list_saved_models()
        logger.info(f"Available models: {available_models}")

        # Version management
        versions = persistence.get_model_versions('trading_classifier_v1')
        logger.info(f"Model versions: {versions}")

        return {
            'model_path': model_path,
            'metadata': model_info,
            'loaded_accuracy': accuracy if 'accuracy' in locals() else None,
            'available_models': available_models,
            'versions': versions
        }

    except Exception as e:
        logger.error(f"Model persistence example failed: {str(e)}")
        return None


def pipeline_management_example():
    """Demonstrate advanced pipeline management."""
    logger.info("=== Pipeline Management Example ===")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        from model.ml_system import MLConfig, PipelineManager

        # Generate data
        X, y = make_regression(n_samples=500, n_features=12, noise=0.1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Create custom pipeline
        config = MLConfig(
            algorithm='random_forest',
            target_type='regression',
            scaling_enabled=True
        )

        pipeline_manager = PipelineManager(config)

        # Build custom pipeline with multiple steps
        logger.info("Building custom pipeline...")
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('feature_selection', SelectKBest(score_func=f_regression, k=15)),
            ('estimator', RandomForestRegressor(n_estimators=50, random_state=42))
        ]

        custom_pipeline = pipeline_manager.create_custom_pipeline(pipeline_steps)

        # Fit pipeline
        custom_pipeline.fit(X_train, y_train)

        # Make predictions
        predictions = custom_pipeline.predict(X_test)
        r2_score = custom_pipeline.score(X_test, y_test)

        logger.info(f"Custom pipeline R² score: {r2_score:.4f}")

        # Pipeline introspection
        feature_names = custom_pipeline.named_steps['feature_selection'].get_feature_names_out()
        logger.info(f"Selected features count: {len(feature_names)}")

        # Get feature importance from the final estimator
        if hasattr(custom_pipeline.named_steps['estimator'], 'feature_importances_'):
            importances = custom_pipeline.named_steps['estimator'].feature_importances_
            top_features = np.argsort(importances)[-5:][::-1]
            logger.info(f"Top 5 feature indices: {top_features}")

        # Create and compare standard pipeline
        standard_pipeline = pipeline_manager.create_standard_pipeline()
        standard_pipeline.fit(X_train, y_train)
        standard_r2 = standard_pipeline.score(X_test, y_test)

        logger.info(f"Standard pipeline R² score: {standard_r2:.4f}")
        logger.info(f"Custom pipeline improvement: {r2_score - standard_r2:.4f}")

        return {
            'custom_pipeline_score': r2_score,
            'standard_pipeline_score': standard_r2,
            'improvement': r2_score - standard_r2,
            'selected_features_count': len(feature_names),
            'pipeline_steps': len(pipeline_steps)
        }

    except Exception as e:
        logger.error(f"Pipeline management example failed: {str(e)}")
        return None


def ensemble_methods_example():
    """Demonstrate ensemble methods and model combination."""
    logger.info("=== Ensemble Methods Example ===")

    try:
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
            VotingRegressor,
        )
        from sklearn.linear_model import LinearRegression

        from model.ml_system import MLConfig, MLSystem

        # Generate data
        X, y = make_regression(n_samples=600, n_features=10, noise=0.15, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Train individual models
        models = {}
        algorithms = ['linear_regression', 'random_forest', 'gradient_boosting']

        for algorithm in algorithms:
            logger.info(f"Training {algorithm}...")

            config = MLConfig(
                algorithm=algorithm,
                target_type='regression',
                scaling_enabled=True
            )

            ml_system = MLSystem(config)
            ml_system.train(X_train, y_train, save_model=False)

            # Evaluate individual model
            test_score = ml_system.evaluate(X_test, y_test)['r2_score']
            models[algorithm] = {
                'model': ml_system.model,
                'score': test_score
            }

            logger.info(f"{algorithm} R² score: {test_score:.4f}")

        # Create ensemble
        logger.info("Creating ensemble model...")

        # Prepare models for voting
        voting_models = [
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ]

        # Create voting ensemble
        ensemble = VotingRegressor(estimators=voting_models)

        # Apply same preprocessing as individual models
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit ensemble
        ensemble.fit(X_train_scaled, y_train)

        # Evaluate ensemble
        ensemble_score = ensemble.score(X_test_scaled, y_test)
        ensemble_predictions = ensemble.predict(X_test_scaled)

        logger.info(f"Ensemble R² score: {ensemble_score:.4f}")

        # Compare with best individual model
        best_individual = max(models.values(), key=lambda x: x['score'])['score']
        improvement = ensemble_score - best_individual

        logger.info(f"Best individual model: {best_individual:.4f}")
        logger.info(f"Ensemble improvement: {improvement:.4f}")

        return {
            'individual_scores': {k: v['score'] for k, v in models.items()},
            'ensemble_score': ensemble_score,
            'best_individual': best_individual,
            'ensemble_improvement': improvement,
            'ensemble_predictions': ensemble_predictions[:5].tolist()
        }

    except Exception as e:
        logger.error(f"Ensemble methods example failed: {str(e)}")
        return None


def performance_monitoring_example():
    """Demonstrate performance monitoring and tracking."""
    logger.info("=== Performance Monitoring Example ===")

    try:
        from model.ml_system import MLConfig, MLSystem
        from model.ml_system.utils.performance_tracker import PerformanceTracker

        # Generate data
        X, y = make_classification(n_samples=400, n_features=8, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Initialize performance tracker
        tracker = PerformanceTracker()

        # Test multiple algorithms and track performance
        algorithms = ['logistic_regression', 'random_forest', 'gradient_boosting']

        for algorithm in algorithms:
            logger.info(f"Training and tracking {algorithm}...")

            config = MLConfig(
                algorithm=algorithm,
                target_type='classification',
                scaling_enabled=True
            )

            ml_system = MLSystem(config)

            # Track training time
            import time
            start_time = time.time()

            training_results = ml_system.train(X_train, y_train, save_model=False)

            training_time = time.time() - start_time

            # Evaluate model
            test_results = ml_system.evaluate(X_test, y_test, detailed=True)

            # Track performance
            performance_data = {
                'algorithm': algorithm,
                'training_time': training_time,
                'training_samples': len(X_train),
                'feature_count': X_train.shape[1],
                'accuracy': test_results['accuracy'],
                'precision': test_results['precision'],
                'recall': test_results['recall'],
                'f1_score': test_results['f1_score'],
                'roc_auc': test_results.get('roc_auc', 0)
            }

            tracker.track_performance('classification_comparison', performance_data)

            logger.info(f"{algorithm} - Accuracy: {test_results['accuracy']:.4f}, "
                       f"Training time: {training_time:.2f}s")

        # Get performance summary
        summary = tracker.get_performance_summary('classification_comparison')
        logger.info(f"Performance tracking summary: {summary}")

        # Get best performing model
        if summary and 'metrics' in summary:
            best_accuracy = max(summary['metrics'], key=lambda x: x['accuracy'])
            logger.info(f"Best model: {best_accuracy['algorithm']} with "
                       f"accuracy {best_accuracy['accuracy']:.4f}")

        # Export tracking data
        tracking_data = tracker.export_data('classification_comparison')
        logger.info(f"Exported {len(tracking_data)} performance records")

        return {
            'summary': summary,
            'tracking_data': tracking_data,
            'algorithms_tested': len(algorithms),
            'best_model': best_accuracy if 'best_accuracy' in locals() else None
        }

    except Exception as e:
        logger.error(f"Performance monitoring example failed: {str(e)}")
        return None


def run_advanced_examples():
    """Run all advanced ML system examples."""
    logger.info("Starting Advanced ML System Examples")
    logger.info("=" * 60)

    examples_results = {}

    # Run advanced examples
    examples_results['feature_engineering'] = feature_engineering_example()
    examples_results['model_persistence'] = model_persistence_example()
    examples_results['pipeline_management'] = pipeline_management_example()
    examples_results['ensemble_methods'] = ensemble_methods_example()
    examples_results['performance_monitoring'] = performance_monitoring_example()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ADVANCED EXAMPLES SUMMARY")
    logger.info("=" * 60)

    successful_examples = 0
    for example_name, result in examples_results.items():
        if result:
            logger.info(f"✓ {example_name}: SUCCESS")
            successful_examples += 1
        else:
            logger.info(f"✗ {example_name}: FAILED")

    logger.info(f"\nSuccessful examples: {successful_examples}/{len(examples_results)}")

    return examples_results


if __name__ == "__main__":
    # Run advanced examples
    results = run_advanced_examples()

    print("\nAdvanced ML System Examples completed!")
    print("Check the logs above for detailed results.")
