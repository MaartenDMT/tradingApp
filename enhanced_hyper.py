"""
Enhanced Hyperparameter Tuning Module

This module provides enhanced hyperparameter tuning capabilities for both 
reinforcement learning and machine learning models using Optuna and other 
optimization techniques.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, List, Optional, Tuple
import json
import os
from datetime import datetime
import logging

import util.loggers as loggers
from util.config_manager import get_config

# Logger setup
logger_dict = loggers.setup_loggers()
hyper_logger = logger_dict['app']  # Using app logger for hyperparameter tuning

# Get configuration
config = get_config()

class HyperparameterTuner:
    """Enhanced hyperparameter tuner with advanced optimization capabilities."""
    
    def __init__(self, study_name: str = "hyperparameter_optimization"):
        """Initialize the hyperparameter tuner.
        
        Args:
            study_name: Name for the Optuna study
        """
        self.study_name = study_name
        self.study = None
        self.results = {}
        self.best_params = {}
        self.history = []
        
    def create_study(self, 
                    direction: str = "maximize",
                    storage: Optional[str] = None,
                    load_if_exists: bool = True) -> optuna.Study:
        """Create or load an Optuna study.
        
        Args:
            direction: Optimization direction ("maximize" or "minimize")
            storage: Database URL for persistent storage (optional)
            load_if_exists: Whether to load existing study if it exists
            
        Returns:
            Optuna study instance
        """
        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=direction,
                storage=storage,
                load_if_exists=load_if_exists
            )
            hyper_logger.info(f"Created/loaded study: {self.study_name}")
            return self.study
        except Exception as e:
            hyper_logger.error(f"Failed to create study: {e}")
            raise
            
    def optimize_rl_agent(self, 
                         objective_function: Callable,
                         n_trials: int = 100,
                         timeout: Optional[int] = None,
                         n_jobs: int = 1) -> Dict[str, Any]:
        """Optimize reinforcement learning agent hyperparameters.
        
        Args:
            objective_function: Function to optimize (takes trial as parameter)
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with optimization results
        """
        if self.study is None:
            self.create_study()
            
        try:
            # Run optimization
            self.study.optimize(
                objective_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs
            )
            
            # Store results
            self.best_params = self.study.best_params
            self.results = {
                'best_value': self.study.best_value,
                'best_params': self.best_params,
                'n_trials': len(self.study.trials)
            }
            
            # Store trial history
            self.history = [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime_start': trial.datetime_start
                }
                for trial in self.study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            
            hyper_logger.info(f"RL optimization completed with {n_trials} trials")
            hyper_logger.info(f"Best value: {self.study.best_value}")
            
            return self.results
            
        except Exception as e:
            hyper_logger.error(f"RL optimization failed: {e}")
            raise
            
    def optimize_ml_model(self,
                         objective_function: Callable,
                         param_space: Dict[str, Any],
                         n_trials: int = 50,
                         cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize machine learning model hyperparameters.
        
        Args:
            objective_function: Function to optimize
            param_space: Dictionary defining parameter search space
            n_trials: Number of trials to run
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with optimization results
        """
        if self.study is None:
            self.create_study()
            
        try:
            # Define search space
            def objective(trial):
                # Sample parameters based on param_space
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step=param_config.get('step', 1)
                        )
                
                # Call objective function with parameters
                return objective_function(params)
            
            # Run optimization
            self.study.optimize(objective, n_trials=n_trials)
            
            # Store results
            self.best_params = self.study.best_params
            self.results = {
                'best_value': self.study.best_value,
                'best_params': self.best_params,
                'n_trials': len(self.study.trials)
            }
            
            hyper_logger.info(f"ML optimization completed with {n_trials} trials")
            hyper_logger.info(f"Best value: {self.study.best_value}")
            
            return self.results
            
        except Exception as e:
            hyper_logger.error(f"ML optimization failed: {e}")
            raise
            
    def get_top_trials(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top n trials from the study.
        
        Args:
            n: Number of top trials to return
            
        Returns:
            List of top trial information
        """
        if self.study is None:
            return []
            
        try:
            # Get completed trials sorted by value
            trials = sorted(
                [trial for trial in self.study.trials 
                 if trial.state == optuna.trial.TrialState.COMPLETE],
                key=lambda x: x.value,
                reverse=(self.study.direction == optuna.study.StudyDirection.MAXIMIZE)
            )
            
            # Return top n trials
            top_trials = []
            for trial in trials[:n]:
                top_trials.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime_start': trial.datetime_start
                })
                
            return top_trials
            
        except Exception as e:
            hyper_logger.error(f"Failed to get top trials: {e}")
            return []
            
    def save_results(self, filepath: str = None) -> str:
        """Save optimization results to file.
        
        Args:
            filepath: File path to save results (optional)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"hyperparameter_results_{timestamp}.json"
            
        try:
            # Prepare results for saving
            save_data = {
                'study_name': self.study_name,
                'best_params': self.best_params,
                'best_value': self.results.get('best_value'),
                'n_trials': self.results.get('n_trials'),
                'history': self.history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
                
            hyper_logger.info(f"Results saved to {filepath}")
            return filepath
            
        except Exception as e:
            hyper_logger.error(f"Failed to save results: {e}")
            raise
            
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded results dictionary
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
                
            # Update instance variables
            self.best_params = results.get('best_params', {})
            self.results = {
                'best_value': results.get('best_value'),
                'best_params': self.best_params,
                'n_trials': results.get('n_trials')
            }
            self.history = results.get('history', [])
            
            hyper_logger.info(f"Results loaded from {filepath}")
            return results
            
        except Exception as e:
            hyper_logger.error(f"Failed to load results: {e}")
            raise
            
    def plot_optimization_history(self, filepath: str = None) -> Optional[str]:
        """Plot optimization history.
        
        Args:
            filepath: File path to save plot (optional)
            
        Returns:
            Path to saved plot or None if plotting failed
        """
        try:
            # Only import matplotlib if needed
            import matplotlib.pyplot as plt
            
            if not self.history:
                hyper_logger.warning("No history to plot")
                return None
                
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract values and trial numbers
            trial_numbers = [trial['number'] for trial in self.history]
            values = [trial['value'] for trial in self.history]
            
            # Plot
            ax.plot(trial_numbers, values, 'o-', markersize=4)
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Objective Value')
            ax.set_title(f'Hyperparameter Optimization History - {self.study_name}')
            ax.grid(True, alpha=0.3)
            
            # Save or show plot
            if filepath:
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                hyper_logger.info(f"Optimization history plot saved to {filepath}")
                return filepath
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            hyper_logger.error(f"Failed to plot optimization history: {e}")
            return None

# Predefined parameter spaces for common models
RL_PARAM_SPACE = {
    'gamma': {'type': 'float', 'low': 0.9, 'high': 0.999},
    'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
    'epsilon': {'type': 'float', 'low': 0.01, 'high': 1.0},
    'epsilon_decay': {'type': 'float', 'low': 0.995, 'high': 0.9999},
    'hidden_units': {'type': 'int', 'low': 16, 'high': 256},
    'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
    'batch_size': {'type': 'int', 'low': 16, 'high': 256},
    'modelname': {'type': 'categorical', 'choices': [
        "Standard_Model", "Dense_Model", "LSTM_Model", 
        "CONV1D_LSTM_Model", "build_resnet_model", 
        "base_conv1d_model", "base_transformer_model"
    ]},
    'act': {'type': 'categorical', 'choices': ['argmax', 'softmax']},
    'm_activation': {'type': 'categorical', 'choices': ['linear', 'tanh', 'sigmoid']}
}

ML_PARAM_SPACE = {
    'n_estimators': {'type': 'int', 'low': 10, 'high': 1000},
    'max_depth': {'type': 'int', 'low': 1, 'high': 20},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
    'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True},
    'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0}
}

def create_enhanced_rl_objective(env, agent_class, test_episodes: int = 10):
    """Create an enhanced objective function for RL agent optimization.
    
    Args:
        env: Trading environment
        agent_class: RL agent class
        test_episodes: Number of episodes for testing
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        # Sample hyperparameters
        hyperparameters = {
            'modelname': trial.suggest_categorical('modelname', RL_PARAM_SPACE['modelname']['choices']),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.9, 1.1),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.9999),
            'hidden_units': trial.suggest_int('hidden_units', 16, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
            'batch_size': trial.suggest_int('batch_size', 16, 256),
            'act': trial.suggest_categorical('act', ['argmax', 'softmax']),
            'm_activation': trial.suggest_categorical('m_activation', ['linear', 'tanh', 'sigmoid'])
        }
        
        try:
            # Create and train agent
            agent = agent_class(env=env, **hyperparameters)
            agent.learn(episodes=5)  # Quick training for optimization
            
            # Test agent
            test_rewards = agent.test(episodes=test_episodes)
            avg_reward = sum(test_rewards) / len(test_rewards)
            
            return avg_reward
            
        except Exception as e:
            hyper_logger.error(f"Trial failed: {e}")
            return float('-inf')  # Return worst possible score
            
    return objective

# Example usage function
def run_hyperparameter_tuning():
    """Example function demonstrating hyperparameter tuning."""
    print("Enhanced Hyperparameter Tuning Module")
    print("=" * 40)
    
    # Create tuner
    tuner = HyperparameterTuner("example_study")
    
    # For RL optimization, you would typically do:
    # 1. Create environment
    # 2. Define objective function
    # 3. Run optimization
    # 4. Save results
    
    print("Hyperparameter tuner initialized successfully")
    print("Ready for optimization tasks")

if __name__ == "__main__":
    run_hyperparameter_tuning()