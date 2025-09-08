"""
Model Persistence for ML System

Provides comprehensive model saving, loading, and versioning capabilities
with metadata tracking and validation.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline

from ..config.ml_config import MLConfig

logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Comprehensive model persistence with versioning and metadata.

    Handles saving and loading of trained models with proper versioning,
    metadata tracking, and validation to ensure model reproducibility.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize model persistence manager.

        Args:
            config: ML configuration object
        """
        self.config = config
        self.base_dir = config.model_save_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Pipeline,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_version: bool = False
    ) -> Path:
        """
        Save a trained model with metadata.

        Args:
            model: Trained scikit-learn pipeline
            model_name: Optional model name for the save path
            metadata: Optional metadata dictionary
            create_version: Whether to create a versioned filename

        Returns:
            Path where model was saved
        """
        try:
            # Check available disk space before saving
            try:
                import shutil
                free_space = shutil.disk_usage(self.base_dir).free
                # Require at least 1GB free space for model saving
                if free_space < 1024 * 1024 * 1024:  # 1GB
                    logger.warning(f"Low disk space: {free_space / (1024**3):.2f} GB available")
            except Exception:
                pass  # Continue if disk space check fails

            # Generate save path
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                model_name = f"{self.config.algorithm}_{self.config.target_type}_{timestamp}"
            else:
                if create_version or self.config.model_versioning:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    model_name = f"{model_name}_{timestamp}"

            path = self.base_dir / model_name

            # Create model directory
            path.mkdir(parents=True, exist_ok=True)

            # Save model with compression to reduce disk usage
            model_path = path / "model.joblib"
            try:
                # Use compression to reduce disk space usage
                joblib.dump(model, model_path, compress=3)
            except OSError as e:
                if e.errno == 28:  # No space left on device
                    logger.error("No space left on device. Attempting cleanup and retry...")
                    # Try to clean up old models if disk is full
                    self._cleanup_old_models(keep_last=5)
                    # Retry without compression
                    joblib.dump(model, model_path, compress=0)
                else:
                    raise

            # Prepare metadata
            model_metadata = {
                'algorithm': self.config.algorithm,
                'target_type': self.config.target_type,
                'timestamp': datetime.now().isoformat(),
                'sklearn_version': self._get_sklearn_version(),
                'config': self._serialize_config()
            }

            # Handle both Pipeline and bare model objects
            if hasattr(model, 'named_steps'):
                # This is a Pipeline
                model_metadata.update({
                    'model_type': type(model.named_steps.get('model', model)).__name__,
                    'pipeline_steps': [step[0] for step in model.steps],
                    'model_params': model.named_steps.get('model', model).get_params()
                })
            else:
                # This is a bare model
                model_metadata.update({
                    'model_type': type(model).__name__,
                    'pipeline_steps': ['model'],
                    'model_params': model.get_params()
                })

            # Add custom metadata
            if metadata:
                model_metadata.update(metadata)

            # Save metadata
            metadata_path = path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)

            # Save model info summary
            info_path = path / "model_info.txt"
            self._save_model_info(model, model_metadata, info_path)

            logger.info(f"Model saved to {path}")
            return path

        except OSError as e:
            if e.errno == 28:  # No space left on device
                logger.error("No space left on device. Cannot save model.")
                raise OSError("Insufficient disk space to save model. Please free up disk space and try again.")
            else:
                logger.error(f"Failed to save model: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, path_or_name) -> Pipeline:
        """
        Load a previously saved model.

        Args:
            path_or_name: Path to the saved model directory/file or model name string

        Returns:
            Loaded scikit-learn pipeline
        """
        try:
            # Handle string model names vs Path objects
            if isinstance(path_or_name, str):
                path = self.base_dir / path_or_name
            else:
                path = Path(path_or_name)

            # Handle both directory and file paths
            if path.is_file() and path.suffix == '.joblib':
                model_path = path
                metadata_path = path.parent / "metadata.json"
            else:
                model_path = path / "model.joblib"
                metadata_path = path / "metadata.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model
            model = joblib.load(model_path)

            # Load and validate metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self._validate_model_compatibility(metadata)
                logger.info(f"Model loaded from {path} (saved: {metadata.get('timestamp', 'unknown')})")
            else:
                logger.warning(f"No metadata found for model at {path}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model from {path_or_name}: {str(e)}")
            raise

    def load_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Load metadata for a saved model.

        Args:
            model_name: Name of the model

        Returns:
            Metadata dictionary
        """
        try:
            metadata_path = self.base_dir / model_name / "metadata.json"

            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            with open(metadata_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to load metadata for {model_name}: {str(e)}")
            raise

    def get_model_versions(self, model_name: str) -> List[str]:
        """
        Get all versions of a specific model.

        Args:
            model_name: Base name of the model

        Returns:
            List of model version names
        """
        try:
            versions = []
            if not self.base_dir.exists():
                return versions

            # Look for directories that start with the model name
            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith(model_name):
                    versions.append(model_dir.name)

            # Sort by name (which should include timestamp for versioned models)
            versions.sort()
            return versions

        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {str(e)}")
            return []

    def list_saved_models(self) -> List[str]:
        """
        List all saved model names.

        Returns:
            List of model names
        """
        models = []

        try:
            # Check if base directory exists before iterating
            if not self.base_dir.exists():
                logger.warning(f"Model directory {self.base_dir} does not exist")
                return models

            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir():
                    model_path = model_dir / "model.joblib"
                    if model_path.exists():
                        models.append(model_dir.name)

            # Sort alphabetically
            models.sort()

        except OSError as e:
            if e.errno == 28:  # No space left on device
                logger.error("No space left on device - cannot list models")
                return []
            else:
                logger.error(f"OS error while listing models: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")

        return models

    def list_saved_models_detailed(self) -> List[Dict[str, Any]]:
        """
        List all saved models with detailed metadata.

        Returns:
            List of dictionaries containing model information
        """
        models = []

        try:
            # Check if base directory exists before iterating
            if not self.base_dir.exists():
                logger.warning(f"Model directory {self.base_dir} does not exist")
                return models

            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    model_path = model_dir / "model.joblib"

                    if metadata_path.exists() and model_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)

                            model_info = {
                                'path': str(model_dir),
                                'name': model_dir.name,
                                'algorithm': metadata.get('algorithm'),
                                'target_type': metadata.get('target_type'),
                                'timestamp': metadata.get('timestamp'),
                                'model_type': metadata.get('model_type'),
                                'file_size': model_path.stat().st_size,
                                'pipeline_steps': metadata.get('pipeline_steps', [])
                            }
                            models.append(model_info)
                        except (OSError, IOError) as e:
                            # Handle file read errors including disk space issues
                            logger.warning(f"Could not read model {model_dir}: {str(e)}")
                            continue

            # Sort by timestamp (newest first)
            models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        except OSError as e:
            if e.errno == 28:  # No space left on device
                logger.error("No space left on device - cannot list models")
                # Return empty list rather than raising
                return []
            else:
                logger.error(f"OS error while listing models: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")

        return models

    def delete_model(self, path: Path) -> bool:
        """
        Delete a saved model and its metadata.

        Args:
            path: Path to the model directory

        Returns:
            True if deletion was successful
        """
        try:
            if path.exists() and path.is_dir():
                import shutil
                shutil.rmtree(path)
                logger.info(f"Model deleted: {path}")
                return True
            else:
                logger.warning(f"Model path does not exist: {path}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return False

    def export_model(self, model_path: Path, export_path: Path, format: str = 'joblib') -> Path:
        """
        Export model to different format.

        Args:
            model_path: Path to the saved model
            export_path: Path for exported model
            format: Export format ('joblib', 'pickle', 'onnx')

        Returns:
            Path to exported model
        """
        try:
            model = self.load_model(model_path)

            if format == 'joblib':
                export_file = export_path.with_suffix('.joblib')
                joblib.dump(model, export_file)
            elif format == 'pickle':
                export_file = export_path.with_suffix('.pkl')
                with open(export_file, 'wb') as f:
                    pickle.dump(model, f)
            elif format == 'onnx':
                # ONNX export would require skl2onnx
                logger.warning("ONNX export not implemented")
                raise NotImplementedError("ONNX export requires skl2onnx")
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Model exported to {export_file}")
            return export_file

        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            raise

    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except Exception:
            return "unknown"

    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize ML config for storage."""
        try:
            return {
                'algorithm': self.config.algorithm,
                'target_type': self.config.target_type,
                'random_state': self.config.random_state,
                'hyperparameter_optimization': self.config.hyperparameter_optimization,
                'optimization_method': self.config.optimization_method,
                'scaling_enabled': self.config.scaling_enabled,
                'feature_engineering_enabled': self.config.feature_engineering_enabled,
                'scoring_metric': self.config.scoring_metric
            }
        except Exception:
            return {}

    def _validate_model_compatibility(self, metadata: Dict[str, Any]) -> None:
        """Validate model compatibility with current environment."""
        warnings = []

        # Check sklearn version
        current_version = self._get_sklearn_version()
        saved_version = metadata.get('sklearn_version', 'unknown')

        if saved_version != 'unknown' and saved_version != current_version:
            warnings.append(f"Model saved with sklearn {saved_version}, "
                          f"current version is {current_version}")

        # Check algorithm compatibility
        if metadata.get('algorithm') != self.config.algorithm:
            warnings.append(f"Model algorithm ({metadata.get('algorithm')}) "
                          f"differs from config ({self.config.algorithm})")

        if warnings:
            for warning in warnings:
                logger.warning(warning)

    def _save_model_info(self, model: Pipeline, metadata: Dict[str, Any], path: Path) -> None:
        """Save human-readable model information."""
        try:
            with open(path, 'w') as f:
                f.write("Model Information\n")
                f.write("=================\n\n")

                f.write(f"Algorithm: {metadata.get('algorithm', 'Unknown')}\n")
                f.write(f"Target Type: {metadata.get('target_type', 'Unknown')}\n")
                f.write(f"Model Type: {metadata.get('model_type', 'Unknown')}\n")
                f.write(f"Saved: {metadata.get('timestamp', 'Unknown')}\n")
                f.write(f"Scikit-learn Version: {metadata.get('sklearn_version', 'Unknown')}\n\n")

                f.write("Pipeline Steps:\n")
                for step in metadata.get('pipeline_steps', []):
                    f.write(f"  - {step}\n")

                f.write("\nModel Parameters:\n")
                params = metadata.get('model_params', {})
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")

                # Add feature importance if available
                if hasattr(model, 'named_steps'):
                    # Pipeline model
                    model_step = model.named_steps.get('model', model)
                else:
                    # Bare model
                    model_step = model

                if hasattr(model_step, 'feature_importances_'):
                    f.write("\nFeature Importance Available: Yes\n")
                elif hasattr(model_step, 'coef_'):
                    f.write("\nModel Coefficients Available: Yes\n")
                else:
                    f.write("\nFeature Importance Available: No\n")

        except Exception as e:
            logger.warning(f"Could not save model info: {str(e)}")

    def get_model_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a saved model.

        Args:
            path: Path to the model directory

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")

        return None

    def _cleanup_old_models(self, keep_last: int = 5) -> None:
        """
        Internal cleanup method for handling disk space issues.

        Args:
            keep_last: Number of latest models to keep
        """
        try:
            logger.info(f"Attempting cleanup - keeping last {keep_last} models")
            self.cleanup_old_models(keep_latest=keep_last)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def cleanup_old_models(self, keep_latest: int = 5) -> List[str]:
        """
        Clean up old model versions, keeping only the latest ones.

        Args:
            keep_latest: Number of latest models to keep

        Returns:
            List of deleted model paths
        """
        try:
            models = self.list_saved_models_detailed()
            deleted = []

            if len(models) > keep_latest:
                models_to_delete = models[keep_latest:]

                for model_info in models_to_delete:
                    model_path = Path(model_info['path'])
                    if self.delete_model(model_path):
                        deleted.append(str(model_path))

                logger.info(f"Cleaned up {len(deleted)} old models")

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old models: {str(e)}")
            return []
