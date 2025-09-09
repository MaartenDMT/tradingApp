"""
Production Configuration Module

This module provides production-ready configurations and settings
for deploying the trading application in live environments.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import timedelta

@dataclass
class ProductionConfig:
    """Production configuration settings."""
    
    # Environment settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Security settings
    enable_authentication: bool = True
    session_timeout: timedelta = timedelta(hours=8)
    max_login_attempts: int = 3
    password_min_length: int = 8
    require_2fa: bool = False
    
    # Performance settings
    max_concurrent_operations: int = 10
    cache_ttl_seconds: int = 300
    ui_update_interval_ms: int = 1000
    
    # Trading settings
    enable_live_trading: bool = False
    max_position_size: float = 10000.0
    risk_management: bool = True
    stop_loss_percentage: float = 0.02
    take_profit_percentage: float = 0.04
    
    # RL System settings
    max_training_episodes: int = 10000
    model_checkpoint_interval: int = 100
    enable_distributed_training: bool = False
    gpu_acceleration: bool = True
    
    # Data settings
    data_refresh_interval: timedelta = timedelta(minutes=1)
    historical_data_days: int = 365
    backup_interval: timedelta = timedelta(hours=6)
    
    # API settings
    rate_limit_requests_per_minute: int = 100
    api_timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_alerting: bool = True
    health_check_interval: timedelta = timedelta(minutes=5)
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Create configuration from environment variables."""
        return cls(
            environment=os.getenv('TRADING_ENV', 'production'),
            debug_mode=os.getenv('DEBUG', 'False').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            enable_live_trading=os.getenv('ENABLE_LIVE_TRADING', 'False').lower() == 'true',
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '10000.0')),
            gpu_acceleration=os.getenv('GPU_ACCELERATION', 'True').lower() == 'true',
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, timedelta):
                result[field_name] = value.total_seconds()
            else:
                result[field_name] = value
        return result

class ProductionManager:
    """Manager for production deployment and monitoring."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self._health_status = {}
        self._metrics = {}
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate production environment readiness."""
        checks = {
            'python_version': self._check_python_version(),
            'required_packages': self._check_required_packages(),
            'system_resources': self._check_system_resources(),
            'network_connectivity': self._check_network_connectivity(),
            'gpu_availability': self._check_gpu_availability() if self.config.gpu_acceleration else True,
            'storage_space': self._check_storage_space(),
        }
        return checks
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        import sys
        return sys.version_info >= (3, 8)
    
    def _check_required_packages(self) -> bool:
        """Check if all required packages are installed."""
        required_packages = [
            'torch', 'numpy', 'pandas', 'scikit-learn',
            'ttkbootstrap', 'asyncio', 'psycopg2-binary'
        ]
        try:
            for package in required_packages:
                __import__(package.replace('-', '_'))
            return True
        except ImportError:
            return False
    
    def _check_system_resources(self) -> bool:
        """Check if system has adequate resources."""
        import psutil
        # Check RAM (minimum 4GB)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        # Check CPU cores (minimum 2)
        cpu_cores = psutil.cpu_count()
        return memory_gb >= 4 and cpu_cores >= 2
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    def _check_gpu_availability(self) -> bool:
        """Check GPU availability for acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_storage_space(self) -> bool:
        """Check available storage space (minimum 1GB)."""
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        return free_gb >= 1
    
    def get_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment readiness report."""
        validation_results = self.validate_environment()
        
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'config': self.config.to_dict(),
            'environment_checks': validation_results,
            'readiness_score': sum(validation_results.values()) / len(validation_results),
            'recommendations': self._generate_recommendations(validation_results),
            'deployment_ready': all(validation_results.values())
        }
        
        return report
    
    def _generate_recommendations(self, checks: Dict[str, bool]) -> list:
        """Generate deployment recommendations based on checks."""
        recommendations = []
        
        if not checks.get('python_version', True):
            recommendations.append("Upgrade to Python 3.8 or higher")
        
        if not checks.get('required_packages', True):
            recommendations.append("Install missing packages: pip install -r requirements.txt")
        
        if not checks.get('system_resources', True):
            recommendations.append("Upgrade system: minimum 4GB RAM and 2 CPU cores required")
        
        if not checks.get('network_connectivity', True):
            recommendations.append("Check network connectivity for market data access")
        
        if not checks.get('gpu_availability', True) and self.config.gpu_acceleration:
            recommendations.append("Install CUDA drivers for GPU acceleration or disable GPU in config")
        
        if not checks.get('storage_space', True):
            recommendations.append("Free up disk space: minimum 1GB required for operation")
        
        return recommendations

# Global production configuration
PRODUCTION_CONFIG = ProductionConfig.from_env()
