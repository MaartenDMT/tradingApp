"""
Model Evaluation for ML System

Provides comprehensive model evaluation metrics and analysis
for both regression and classification tasks.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..config.ml_config import MLConfig

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for ML algorithms.

    Provides detailed metrics and analysis for both regression
    and classification tasks with visualization capabilities.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize model evaluator.

        Args:
            config: ML configuration object
        """
        self.config = config

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_type: str,
        detailed: bool = True,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            target_type: 'regression' or 'classification'
            detailed: Whether to include detailed metrics
            class_names: Names of classes for classification

        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if target_type == 'regression':
                return self._evaluate_regression(y_true, y_pred, detailed)
            elif target_type == 'classification':
                return self._evaluate_classification(y_true, y_pred, detailed, class_names)
            else:
                raise ValueError(f"Unknown target_type: {target_type}")

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """Evaluate regression model performance."""
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'explained_variance': float(explained_variance_score(y_true, y_pred))
        }

        if detailed:
            # Additional regression metrics
            residuals = y_true - y_pred

            metrics.update({
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'max_error': float(np.max(np.abs(residuals))),
                'mean_absolute_percentage_error': float(self._calculate_mape(y_true, y_pred)),
                'median_absolute_error': float(np.median(np.abs(residuals))),
                'residual_stats': {
                    'min': float(np.min(residuals)),
                    'max': float(np.max(residuals)),
                    'q25': float(np.percentile(residuals, 25)),
                    'q75': float(np.percentile(residuals, 75))
                }
            })

            # Performance categories
            metrics['performance_summary'] = self._categorize_regression_performance(r2, mae, rmse)

        return metrics

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        detailed: bool = True,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate classification model performance."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Handle binary vs multiclass
        average_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'

        precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_pred)
                metrics['roc_auc'] = float(auc)
            except ValueError:
                logger.warning("Could not calculate ROC AUC")

        if detailed:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            # Per-class metrics
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report

            # Additional metrics for multiclass
            if len(np.unique(y_true)) > 2:
                metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
                metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
                metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
                metrics['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
                metrics['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
                metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))

            # Performance summary
            metrics['performance_summary'] = self._categorize_classification_performance(accuracy, f1)

        return metrics

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _categorize_regression_performance(self, r2: float, mae: float, rmse: float) -> Dict[str, str]:
        """Categorize regression model performance."""
        if r2 >= 0.9:
            r2_category = "Excellent"
        elif r2 >= 0.8:
            r2_category = "Very Good"
        elif r2 >= 0.7:
            r2_category = "Good"
        elif r2 >= 0.5:
            r2_category = "Fair"
        else:
            r2_category = "Poor"

        return {
            'r2_category': r2_category,
            'overall_assessment': r2_category,
            'recommendations': self._get_regression_recommendations(r2, mae, rmse)
        }

    def _categorize_classification_performance(self, accuracy: float, f1: float) -> Dict[str, str]:
        """Categorize classification model performance."""
        avg_score = (accuracy + f1) / 2

        if avg_score >= 0.95:
            category = "Excellent"
        elif avg_score >= 0.9:
            category = "Very Good"
        elif avg_score >= 0.8:
            category = "Good"
        elif avg_score >= 0.7:
            category = "Fair"
        else:
            category = "Poor"

        return {
            'accuracy_category': category,
            'overall_assessment': category,
            'recommendations': self._get_classification_recommendations(accuracy, f1)
        }

    def _get_regression_recommendations(self, r2: float, mae: float, rmse: float) -> List[str]:
        """Get recommendations for improving regression performance."""
        recommendations = []

        if r2 < 0.7:
            recommendations.append("Consider feature engineering or different algorithms")
            recommendations.append("Check for overfitting or underfitting")

        if rmse > mae * 2:
            recommendations.append("Model may be sensitive to outliers")
            recommendations.append("Consider robust regression methods")

        if r2 < 0.5:
            recommendations.append("Model performance is poor - consider:")
            recommendations.append("- More relevant features")
            recommendations.append("- Different algorithm families")
            recommendations.append("- Better hyperparameter tuning")

        return recommendations

    def _get_classification_recommendations(self, accuracy: float, f1: float) -> List[str]:
        """Get recommendations for improving classification performance."""
        recommendations = []

        if accuracy < 0.8:
            recommendations.append("Consider feature engineering or different algorithms")

        if abs(accuracy - f1) > 0.1:
            recommendations.append("Class imbalance may be affecting performance")
            recommendations.append("Consider class balancing techniques")

        if accuracy < 0.7:
            recommendations.append("Model performance is poor - consider:")
            recommendations.append("- More training data")
            recommendations.append("- Better feature selection")
            recommendations.append("- Different algorithm families")

        return recommendations

    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.

        Args:
            evaluation_results: List of evaluation result dictionaries

        Returns:
            Dictionary containing model comparison
        """
        if not evaluation_results:
            return {}

        # Determine target type from first result
        target_type = 'regression' if 'r2_score' in evaluation_results[0] else 'classification'

        comparison = {
            'target_type': target_type,
            'n_models': len(evaluation_results),
            'models': []
        }

        if target_type == 'regression':
            key_metrics = ['r2_score', 'rmse', 'mae']
            best_metric = 'r2_score'
            higher_better = True
        else:
            key_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            best_metric = 'f1_score'
            higher_better = True

        # Extract key metrics for each model
        for i, result in enumerate(evaluation_results):
            model_summary = {'model_index': i}
            for metric in key_metrics:
                if metric in result:
                    model_summary[metric] = result[metric]
            comparison['models'].append(model_summary)

        # Find best model
        if higher_better:
            best_idx = max(range(len(evaluation_results)),
                          key=lambda i: evaluation_results[i].get(best_metric, 0))
        else:
            best_idx = min(range(len(evaluation_results)),
                          key=lambda i: evaluation_results[i].get(best_metric, float('inf')))

        comparison['best_model_index'] = best_idx
        comparison['best_model_metric'] = evaluation_results[best_idx].get(best_metric)

        # Calculate improvement over baseline (first model)
        if len(evaluation_results) > 1:
            baseline_score = evaluation_results[0].get(best_metric, 0)
            best_score = evaluation_results[best_idx].get(best_metric, 0)
            improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score != 0 else 0
            comparison['improvement_over_baseline'] = float(improvement)

        return comparison

    def generate_evaluation_report(self, results: Dict[str, Any], model_name: str = "Model") -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            results: Evaluation results dictionary
            model_name: Name of the model being evaluated

        Returns:
            Formatted evaluation report string
        """
        target_type = 'regression' if 'r2_score' in results else 'classification'

        report = [f"\n{model_name} Evaluation Report"]
        report.append("=" * (len(model_name) + 18))

        if target_type == 'regression':
            report.append("\nRegression Metrics:")
            report.append(f"  R² Score: {results.get('r2_score', 0):.4f}")
            report.append(f"  RMSE: {results.get('rmse', 0):.4f}")
            report.append(f"  MAE: {results.get('mae', 0):.4f}")
            report.append(f"  Explained Variance: {results.get('explained_variance', 0):.4f}")

            if 'performance_summary' in results:
                summary = results['performance_summary']
                report.append(f"\nPerformance Assessment: {summary.get('overall_assessment', 'Unknown')}")

                if 'recommendations' in summary and summary['recommendations']:
                    report.append("\nRecommendations:")
                    for rec in summary['recommendations']:
                        report.append(f"  • {rec}")

        else:  # classification
            report.append("\nClassification Metrics:")
            report.append(f"  Accuracy: {results.get('accuracy', 0):.4f}")
            report.append(f"  Precision: {results.get('precision', 0):.4f}")
            report.append(f"  Recall: {results.get('recall', 0):.4f}")
            report.append(f"  F1 Score: {results.get('f1_score', 0):.4f}")

            if 'roc_auc' in results:
                report.append(f"  ROC AUC: {results['roc_auc']:.4f}")

            if 'performance_summary' in results:
                summary = results['performance_summary']
                report.append(f"\nPerformance Assessment: {summary.get('overall_assessment', 'Unknown')}")

                if 'recommendations' in summary and summary['recommendations']:
                    report.append("\nRecommendations:")
                    for rec in summary['recommendations']:
                        report.append(f"  • {rec}")

        return "\n".join(report)
