"""
Performance Tracking for ML System

Provides comprehensive performance monitoring and logging
for ML training sessions, evaluations, and predictions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Performance tracking for ML workflows.

    Tracks and logs performance metrics for training, evaluation,
    and prediction sessions with persistent storage and analysis.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize performance tracker.

        Args:
            log_dir: Directory for storing performance logs
        """
        if log_dir is None:
            log_dir = Path("data/logs/performance")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_history = []
        self.evaluation_history = []
        self.performance_metrics = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # For test compatibility
        self.experiment_data = {}

    def track_performance(self, experiment_name: str, performance_data: Dict[str, Any]) -> None:
        """
        Track performance data for an experiment (for test compatibility).

        Args:
            experiment_name: Name of the experiment
            performance_data: Performance data to track
        """
        if experiment_name not in self.experiment_data:
            self.experiment_data[experiment_name] = {'metrics': []}

        self.experiment_data[experiment_name]['metrics'].append({
            'timestamp': datetime.now().isoformat(),
            **performance_data
        })

        # Also log using standard logging
        self.log_evaluation(performance_data)

    def log_training(self, results: Dict[str, Any]) -> None:
        """
        Log training session results.

        Args:
            results: Training results dictionary
        """
        try:
            log_entry = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'event_type': 'training',
                **results
            }

            # Add to history
            self.training_history.append(log_entry)

            # Update metrics
            self._update_training_metrics(results)

            # Save to file
            self._save_log_entry(log_entry, 'training')

            logger.info(f"Training session logged: {results.get('algorithm', 'unknown')} - "
                       f"Score: {results.get('train_score', 'N/A')}")

        except Exception as e:
            logger.error(f"Failed to log training session: {str(e)}")

    def log_evaluation(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation results.

        Args:
            results: Evaluation results dictionary
        """
        try:
            log_entry = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'event_type': 'evaluation',
                **results
            }

            # Add to history
            self.evaluation_history.append(log_entry)

            # Update metrics
            self._update_evaluation_metrics(results)

            # Save to file
            self._save_log_entry(log_entry, 'evaluation')

            logger.info(f"Evaluation logged: Score: {results.get('score', 'N/A')}")

        except Exception as e:
            logger.error(f"Failed to log evaluation: {str(e)}")

    def log_prediction(
        self,
        n_predictions: int,
        prediction_time: float,
        model_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log prediction session performance.

        Args:
            n_predictions: Number of predictions made
            prediction_time: Total prediction time in seconds
            model_info: Optional model information
        """
        try:
            log_entry = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'event_type': 'prediction',
                'n_predictions': n_predictions,
                'prediction_time': prediction_time,
                'predictions_per_second': n_predictions / prediction_time if prediction_time > 0 else 0,
                'avg_time_per_prediction': prediction_time / n_predictions if n_predictions > 0 else 0
            }

            if model_info:
                log_entry.update(model_info)

            # Save to file
            self._save_log_entry(log_entry, 'prediction')

            logger.info(f"Prediction session logged: {n_predictions} predictions in {prediction_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to log prediction session: {str(e)}")

    def get_performance_summary(self, experiment_name: str = None) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Args:
            experiment_name: Optional experiment name for test compatibility

        Returns:
            Dictionary containing performance summary
        """
        # For test compatibility
        if experiment_name and experiment_name in self.experiment_data:
            return self.experiment_data[experiment_name]

        # Original functionality
        summary = {
            'session_id': self.session_id,
            'summary_timestamp': datetime.now().isoformat(),
            'training_summary': self._summarize_training_metrics(),
            'evaluation_summary': self._summarize_evaluation_metrics(),
            'performance_trends': self._analyze_performance_trends(),
            'best_models': self._identify_best_models(),
            'session_count': {
                'training_sessions': len(self.training_history),
                'evaluation_sessions': len(self.evaluation_history)
            }
        }

        return summary

    def export_performance_data(self) -> Dict[str, Any]:
        """
        Export all performance data for external analysis.

        Returns:
            Dictionary containing all tracked performance data
        """
        return {
            'experiment_data': self.experiment_data,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'performance_metrics': self.performance_metrics
        }

    def _save_log_entry(self, entry: Dict[str, Any], log_type: str) -> None:
        """Save log entry to file."""
        try:
            log_file = self.log_dir / f"{log_type}_{self.session_id}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to save log entry: {str(e)}")

    def _update_training_metrics(self, results: Dict[str, Any]) -> None:
        """Update training metrics tracking."""
        try:
            algorithm = results.get('algorithm', 'unknown')
            score = results.get('train_score')

            if score is not None:
                self._track_metric(f"{algorithm}_train_score", score)

        except Exception as e:
            logger.error(f"Failed to update training metrics: {str(e)}")

    def _update_evaluation_metrics(self, results: Dict[str, Any]) -> None:
        """Update evaluation metrics tracking."""
        try:
            score = results.get('score')
            if score is not None:
                self._track_metric('evaluation_score', score)

        except Exception as e:
            logger.error(f"Failed to update evaluation metrics: {str(e)}")

    def _track_metric(self, metric_name: str, value: float) -> None:
        """Track a metric value."""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now().isoformat()
        })

    def _summarize_training_metrics(self) -> Dict[str, Any]:
        """Summarize training metrics."""
        if not self.training_history:
            return {}

        algorithms = {}
        total_sessions = len(self.training_history)

        for session in self.training_history:
            algo = session.get('algorithm', 'unknown')
            if algo not in algorithms:
                algorithms[algo] = {
                    'sessions': 0,
                    'scores': [],
                    'training_times': []
                }

            algorithms[algo]['sessions'] += 1

            if 'train_score' in session and session['train_score'] is not None:
                algorithms[algo]['scores'].append(session['train_score'])

            if 'training_time' in session and session['training_time'] is not None:
                algorithms[algo]['training_times'].append(session['training_time'])

        # Calculate statistics
        for algo_stats in algorithms.values():
            if algo_stats['scores']:
                algo_stats['avg_score'] = sum(algo_stats['scores']) / len(algo_stats['scores'])
                algo_stats['best_score'] = max(algo_stats['scores'])
                algo_stats['worst_score'] = min(algo_stats['scores'])

            if algo_stats['training_times']:
                algo_stats['avg_training_time'] = sum(algo_stats['training_times']) / len(algo_stats['training_times'])

        return {
            'total_sessions': total_sessions,
            'algorithms': algorithms,
            'summary_timestamp': datetime.now().isoformat()
        }

    def _summarize_evaluation_metrics(self) -> Dict[str, Any]:
        """Summarize evaluation metrics."""
        if not self.evaluation_history:
            return {}

        scores = [session.get('score') for session in self.evaluation_history
                 if session.get('score') is not None]

        summary = {
            'total_evaluations': len(self.evaluation_history),
            'summary_timestamp': datetime.now().isoformat()
        }

        if scores:
            summary.update({
                'avg_score': sum(scores) / len(scores),
                'best_score': max(scores),
                'worst_score': min(scores),
                'score_count': len(scores)
            })

        return summary

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            'training_trend': 'stable',
            'evaluation_trend': 'stable',
            'summary_timestamp': datetime.now().isoformat()
        }

        # Analyze training trends
        if len(self.training_history) >= 3:
            recent_scores = []
            for session in self.training_history[-3:]:
                score = session.get('train_score')
                if score is not None:
                    recent_scores.append(score)

            if len(recent_scores) >= 2:
                if recent_scores[-1] > recent_scores[0]:
                    trends['training_trend'] = 'improving'
                elif recent_scores[-1] < recent_scores[0]:
                    trends['training_trend'] = 'declining'

        return trends

    def _identify_best_models(self) -> Dict[str, Any]:
        """Identify best performing models."""
        best_models = {
            'best_training': None,
            'best_evaluation': None,
            'summary_timestamp': datetime.now().isoformat()
        }

        # Find best training session
        best_train_score = float('-inf')
        for session in self.training_history:
            score = session.get('train_score')
            if score is not None and score > best_train_score:
                best_train_score = score
                best_models['best_training'] = {
                    'algorithm': session.get('algorithm'),
                    'score': score,
                    'timestamp': session.get('timestamp')
                }

        # Find best evaluation session
        best_eval_score = float('-inf')
        for session in self.evaluation_history:
            score = session.get('score')
            if score is not None and score > best_eval_score:
                best_eval_score = score
                best_models['best_evaluation'] = {
                    'score': score,
                    'timestamp': session.get('timestamp')
                }

        return best_models

    def export_performance_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Export comprehensive performance report.

        Args:
            output_path: Optional output file path

        Returns:
            Path to the exported report
        """
        if output_path is None:
            output_path = self.log_dir / f"performance_report_{self.session_id}.json"

        report = self.get_performance_summary()

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Performance report exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export performance report: {str(e)}")
            raise

    def load_performance_history(self, log_dir: Path) -> None:
        """
        Load performance history from log directory.

        Args:
            log_dir: Directory containing performance logs
        """
        try:
            for log_file in log_dir.glob("*.jsonl"):
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        event_type = entry.get('event_type')

                        if event_type == 'training':
                            self.training_history.append(entry)
                        elif event_type == 'evaluation':
                            self.evaluation_history.append(entry)

            logger.info(f"Loaded performance history from: {log_dir}")

        except Exception as e:
            logger.error(f"Failed to load performance history: {str(e)}")
