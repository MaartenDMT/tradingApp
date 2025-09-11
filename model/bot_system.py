"""
Advanced Bot System

A comprehensive bot management system that integrates with trained ML and RL models
to create intelligent trading bots with various strategies and risk management capabilities.
"""

import asyncio
import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import json
import traceback

import numpy as np
import pandas as pd

# Import ML and RL systems
from model.ml_system.core.ml_system import MLSystem
from model.rl_system.integration.integration_manager import IntegrationManager
from model.rl_system.factory import RLAlgorithmFactory
from model.manualtrading.trading import Trading

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
bot_logger = logger_dict['manual']


class BotStatus(Enum):
    """Bot status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    TRAINING = "training"


class BotType(Enum):
    """Bot type enumeration."""
    ML_BASED = "ml_based"
    RL_BASED = "rl_based"
    HYBRID = "hybrid"
    MANUAL = "manual"
    RULE_BASED = "rule_based"


class RiskLevel(Enum):
    """Risk level enumeration."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class TradingBot:
    """
    Advanced trading bot that can use ML/RL models for decision making.
    """
    
    def __init__(self, 
                 bot_id: str,
                 name: str,
                 bot_type: BotType,
                 symbol: str = "BTC/USD:USD",
                 exchange=None,
                 config: Optional[Dict] = None):
        
        self.bot_id = bot_id
        self.name = name
        self.bot_type = bot_type
        self.symbol = symbol
        self.exchange = exchange
        self.config = config or {}
        
        # Initialize status
        self.status = BotStatus.CREATED
        self.created_at = datetime.now()
        self.started_at = None
        self.stopped_at = None
        
        # Performance tracking
        self.trades_executed = 0
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.current_position = None
        self.equity_curve = []
        
        # AI models
        self.ml_model = None
        self.rl_model = None
        
        # Trading engine
        self.trading_engine = Trading(exchange, symbol) if exchange else None
        
        # Risk management
        self.risk_config = self._initialize_risk_config()
        
        # Threading
        self._stop_event = threading.Event()
        self._thread = None
        
        bot_logger.info(f"TradingBot {self.name} ({self.bot_id}) created")
    
    def _initialize_risk_config(self) -> Dict:
        """Initialize risk management configuration."""
        risk_level = self.config.get('risk_level', RiskLevel.MODERATE)
        
        if risk_level == RiskLevel.CONSERVATIVE:
            return {
                'max_position_size': 0.01,  # 1% of portfolio
                'stop_loss_pct': 0.02,      # 2% stop loss
                'take_profit_pct': 0.04,    # 4% take profit
                'max_daily_trades': 5,
                'max_drawdown': 0.05        # 5% max drawdown
            }
        elif risk_level == RiskLevel.MODERATE:
            return {
                'max_position_size': 0.05,  # 5% of portfolio
                'stop_loss_pct': 0.03,      # 3% stop loss
                'take_profit_pct': 0.06,    # 6% take profit
                'max_daily_trades': 10,
                'max_drawdown': 0.10        # 10% max drawdown
            }
        elif risk_level == RiskLevel.AGGRESSIVE:
            return {
                'max_position_size': 0.10,  # 10% of portfolio
                'stop_loss_pct': 0.05,      # 5% stop loss
                'take_profit_pct': 0.10,    # 10% take profit
                'max_daily_trades': 20,
                'max_drawdown': 0.20        # 20% max drawdown
            }
        else:  # CUSTOM
            return self.config.get('custom_risk', {
                'max_position_size': 0.05,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06,
                'max_daily_trades': 10,
                'max_drawdown': 0.10
            })
    
    def load_ml_model(self, model_path: str) -> bool:
        """Load a trained ML model."""
        try:
            if not Path(model_path).exists():
                bot_logger.error(f"ML model not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            
            bot_logger.info(f"ML model loaded successfully: {model_path}")
            return True
        except Exception as e:
            bot_logger.error(f"Error loading ML model: {e}")
            return False
    
    def load_rl_model(self, model_path: str, algorithm: str = "DQN") -> bool:
        """Load a trained RL model."""
        try:
            if not Path(model_path).exists():
                bot_logger.error(f"RL model not found: {model_path}")
                return False
            
            # Create RL agent using factory
            factory = RLAlgorithmFactory()
            self.rl_model = factory.create_algorithm(algorithm)
            
            # Load model weights
            self.rl_model.load_model(model_path)
            
            bot_logger.info(f"RL model loaded successfully: {model_path}")
            return True
        except Exception as e:
            bot_logger.error(f"Error loading RL model: {e}")
            return False
    
    def get_ml_prediction(self, market_data: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from ML model."""
        if not self.ml_model:
            return None
        
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(market_data)
            prediction = self.ml_model.predict(features)
            
            return {
                'signal': prediction[0],
                'confidence': getattr(self.ml_model, 'predict_proba', lambda x: [0.5])(features)[0].max(),
                'timestamp': datetime.now()
            }
        except Exception as e:
            bot_logger.error(f"Error getting ML prediction: {e}")
            return None
    
    def get_rl_action(self, state: np.ndarray) -> Optional[Dict]:
        """Get action from RL model."""
        if not self.rl_model:
            return None
        
        try:
            action = self.rl_model.get_action(state, training=False)
            
            return {
                'action': action,
                'q_values': getattr(self.rl_model, 'q_values', None),
                'timestamp': datetime.now()
            }
        except Exception as e:
            bot_logger.error(f"Error getting RL action: {e}")
            return None
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model prediction."""
        # This would include technical indicators, price features, etc.
        # Simplified version for now
        if len(data) < 20:
            return np.zeros((1, 10))  # Default feature size
        
        # Calculate basic features
        features = []
        
        # Price features
        features.append(data['close'].iloc[-1] / data['close'].iloc[-2] - 1)  # Return
        features.append(data['close'].rolling(5).mean().iloc[-1] / data['close'].iloc[-1] - 1)  # MA5 ratio
        features.append(data['close'].rolling(20).mean().iloc[-1] / data['close'].iloc[-1] - 1)  # MA20 ratio
        
        # Volume features
        if 'volume' in data.columns:
            features.append(data['volume'].rolling(5).mean().iloc[-1] / data['volume'].iloc[-1] - 1)
        else:
            features.append(0.0)
        
        # Volatility features
        returns = data['close'].pct_change()
        features.append(returns.rolling(20).std().iloc[-1])
        
        # RSI-like feature
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100.0)
        
        # Add more features to reach expected size
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10]).reshape(1, -1)
    
    def _prepare_rl_state(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare state for RL model."""
        # This should match the state space used during training
        if len(data) < 50:
            return np.zeros(50)  # Default state size
        
        # Use recent price data as state
        prices = data['close'].iloc[-50:].values
        normalized_prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        return normalized_prices
    
    def make_trading_decision(self, market_data: pd.DataFrame) -> Optional[Dict]:
        """Make trading decision based on bot type and loaded models."""
        try:
            if self.bot_type == BotType.ML_BASED:
                return self.get_ml_prediction(market_data)
            
            elif self.bot_type == BotType.RL_BASED:
                state = self._prepare_rl_state(market_data)
                return self.get_rl_action(state)
            
            elif self.bot_type == BotType.HYBRID:
                # Combine ML and RL predictions
                ml_pred = self.get_ml_prediction(market_data)
                rl_pred = self.get_rl_action(self._prepare_rl_state(market_data))
                
                if ml_pred and rl_pred:
                    # Simple ensemble - average the signals
                    combined_signal = (ml_pred.get('signal', 0) + rl_pred.get('action', 0)) / 2
                    return {
                        'signal': combined_signal,
                        'ml_confidence': ml_pred.get('confidence', 0),
                        'rl_action': rl_pred.get('action', 0),
                        'timestamp': datetime.now()
                    }
            
            elif self.bot_type == BotType.RULE_BASED:
                return self._rule_based_decision(market_data)
            
            return None
        
        except Exception as e:
            bot_logger.error(f"Error making trading decision: {e}")
            return None
    
    def _rule_based_decision(self, data: pd.DataFrame) -> Dict:
        """Simple rule-based trading decision."""
        if len(data) < 20:
            return {'signal': 0, 'reason': 'insufficient_data'}
        
        # Simple moving average crossover strategy
        ma_short = data['close'].rolling(5).mean().iloc[-1]
        ma_long = data['close'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if ma_short > ma_long and current_price > ma_short:
            return {'signal': 1, 'reason': 'bullish_crossover', 'confidence': 0.7}
        elif ma_short < ma_long and current_price < ma_short:
            return {'signal': -1, 'reason': 'bearish_crossover', 'confidence': 0.7}
        else:
            return {'signal': 0, 'reason': 'no_clear_signal', 'confidence': 0.5}
    
    def execute_trade(self, decision: Dict) -> bool:
        """Execute trade based on decision."""
        if not self.trading_engine:
            bot_logger.warning("No trading engine available")
            return False
        
        try:
            signal = decision.get('signal', 0)
            confidence = decision.get('confidence', 0.5)
            
            # Only trade if confidence is above threshold
            min_confidence = self.config.get('min_confidence', 0.6)
            if confidence < min_confidence:
                return False
            
            # Determine position size based on risk management
            position_size = self._calculate_position_size(confidence)
            
            if signal > 0.5:  # Buy signal
                result = self.trading_engine.place_trade(
                    symbol=self.symbol,
                    side='buy',
                    order_type='market',
                    quantity=position_size
                )
                if result:
                    self.trades_executed += 1
                    bot_logger.info(f"Bot {self.name} executed BUY order: {position_size}")
                    return True
            
            elif signal < -0.5:  # Sell signal
                result = self.trading_engine.place_trade(
                    symbol=self.symbol,
                    side='sell',
                    order_type='market',
                    quantity=position_size
                )
                if result:
                    self.trades_executed += 1
                    bot_logger.info(f"Bot {self.name} executed SELL order: {position_size}")
                    return True
            
            return False
        
        except Exception as e:
            bot_logger.error(f"Error executing trade: {e}")
            return False
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on risk management and confidence."""
        base_size = self.risk_config['max_position_size']
        
        # Adjust size based on confidence
        adjusted_size = base_size * confidence
        
        # Apply portfolio balance constraints
        if self.trading_engine and hasattr(self.trading_engine, 'balance'):
            max_dollar_amount = self.trading_engine.balance * adjusted_size
            # Convert to position size based on current price
            # This is simplified - should use actual market data
            return max_dollar_amount * 0.001  # Assuming BTC around $1000 for calculation
        
        return adjusted_size
    
    def start(self):
        """Start the bot."""
        if self.status == BotStatus.RUNNING:
            bot_logger.warning(f"Bot {self.name} is already running")
            return
        
        try:
            self.status = BotStatus.RUNNING
            self.started_at = datetime.now()
            self._stop_event.clear()
            
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            
            bot_logger.info(f"Bot {self.name} started successfully")
        
        except Exception as e:
            self.status = BotStatus.ERROR
            bot_logger.error(f"Error starting bot {self.name}: {e}")
    
    def stop(self):
        """Stop the bot."""
        try:
            self._stop_event.set()
            self.status = BotStatus.STOPPED
            self.stopped_at = datetime.now()
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            
            bot_logger.info(f"Bot {self.name} stopped successfully")
        
        except Exception as e:
            bot_logger.error(f"Error stopping bot {self.name}: {e}")
    
    def pause(self):
        """Pause the bot."""
        self.status = BotStatus.PAUSED
        bot_logger.info(f"Bot {self.name} paused")
    
    def resume(self):
        """Resume the bot."""
        if self.status == BotStatus.PAUSED:
            self.status = BotStatus.RUNNING
            bot_logger.info(f"Bot {self.name} resumed")
    
    def _run_loop(self):
        """Main bot execution loop."""
        bot_logger.info(f"Bot {self.name} entering main loop")
        
        while not self._stop_event.is_set() and self.status != BotStatus.STOPPED:
            try:
                if self.status == BotStatus.PAUSED:
                    time.sleep(1)
                    continue
                
                # Get market data
                market_data = self._get_market_data()
                if market_data is None or len(market_data) < 20:
                    time.sleep(self.config.get('sleep_interval', 60))
                    continue
                
                # Make trading decision
                decision = self.make_trading_decision(market_data)
                if decision:
                    # Execute trade if decision is made
                    self.execute_trade(decision)
                
                # Update performance metrics
                self._update_performance()
                
                # Sleep before next iteration
                sleep_time = self.config.get('sleep_interval', 60)
                time.sleep(sleep_time)
            
            except Exception as e:
                bot_logger.error(f"Error in bot {self.name} main loop: {e}")
                bot_logger.error(traceback.format_exc())
                time.sleep(30)  # Wait before retrying
        
        bot_logger.info(f"Bot {self.name} exiting main loop")
    
    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get market data for decision making."""
        if not self.trading_engine:
            return None
        
        try:
            # Get historical data (simplified)
            data = self.trading_engine.fetch_historical_data(
                timeframe='1h',
                limit=100
            )
            
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            
            return None
        
        except Exception as e:
            bot_logger.error(f"Error getting market data: {e}")
            return None
    
    def _update_performance(self):
        """Update performance metrics."""
        try:
            # This would calculate actual performance metrics
            # Simplified for now
            if self.trading_engine and hasattr(self.trading_engine, 'balance'):
                current_equity = self.trading_engine.balance
                self.equity_curve.append({
                    'timestamp': datetime.now(),
                    'equity': current_equity
                })
                
                # Keep only last 1000 entries
                if len(self.equity_curve) > 1000:
                    self.equity_curve = self.equity_curve[-1000:]
        
        except Exception as e:
            bot_logger.error(f"Error updating performance: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        return {
            'bot_id': self.bot_id,
            'name': self.name,
            'status': self.status.value,
            'trades_executed': self.trades_executed,
            'total_profit': self.total_profit,
            'win_rate': self.win_rate,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'equity_points': len(self.equity_curve)
        }
    
    def save_config(self, filepath: str):
        """Save bot configuration."""
        config_data = {
            'bot_id': self.bot_id,
            'name': self.name,
            'bot_type': self.bot_type.value,
            'symbol': self.symbol,
            'config': self.config,
            'risk_config': self.risk_config,
            'performance': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        bot_logger.info(f"Bot config saved: {filepath}")


class BotManager:
    """
    Advanced bot manager that handles multiple trading bots.
    """
    
    def __init__(self):
        self.bots: Dict[str, TradingBot] = {}
        self.ml_engine = None
        self.rl_engine = None
        
        # Initialize AI engines
        try:
            self.ml_engine = MLSystem()
            self.rl_engine = IntegrationManager()
        except Exception as e:
            bot_logger.warning(f"Could not initialize AI engines: {e}")
    
    def create_bot(self, 
                   name: str,
                   bot_type: BotType,
                   symbol: str = "BTC/USD:USD",
                   exchange=None,
                   config: Optional[Dict] = None) -> str:
        """Create a new trading bot."""
        try:
            bot_id = f"bot_{len(self.bots)}_{int(time.time())}"
            
            bot = TradingBot(
                bot_id=bot_id,
                name=name,
                bot_type=bot_type,
                symbol=symbol,
                exchange=exchange,
                config=config or {}
            )
            
            self.bots[bot_id] = bot
            bot_logger.info(f"Created bot: {name} ({bot_id})")
            
            return bot_id
        
        except Exception as e:
            bot_logger.error(f"Error creating bot: {e}")
            return None
    
    def delete_bot(self, bot_id: str) -> bool:
        """Delete a bot."""
        try:
            if bot_id in self.bots:
                bot = self.bots[bot_id]
                bot.stop()  # Ensure bot is stopped
                del self.bots[bot_id]
                bot_logger.info(f"Deleted bot: {bot_id}")
                return True
            else:
                bot_logger.warning(f"Bot not found: {bot_id}")
                return False
        
        except Exception as e:
            bot_logger.error(f"Error deleting bot: {e}")
            return False
    
    def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        if bot_id in self.bots:
            self.bots[bot_id].start()
            return True
        return False
    
    def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        if bot_id in self.bots:
            self.bots[bot_id].stop()
            return True
        return False
    
    def pause_bot(self, bot_id: str) -> bool:
        """Pause a bot."""
        if bot_id in self.bots:
            self.bots[bot_id].pause()
            return True
        return False
    
    def resume_bot(self, bot_id: str) -> bool:
        """Resume a bot."""
        if bot_id in self.bots:
            self.bots[bot_id].resume()
            return True
        return False
    
    def get_bot(self, bot_id: str) -> Optional[TradingBot]:
        """Get a bot by ID."""
        return self.bots.get(bot_id)
    
    def list_bots(self) -> List[Dict]:
        """List all bots with their status."""
        return [bot.get_performance_summary() for bot in self.bots.values()]
    
    def stop_all_bots(self):
        """Stop all running bots."""
        for bot in self.bots.values():
            if bot.status == BotStatus.RUNNING:
                bot.stop()
        
        bot_logger.info("All bots stopped")
    
    def get_system_performance(self) -> Dict:
        """Get overall system performance."""
        total_bots = len(self.bots)
        running_bots = sum(1 for bot in self.bots.values() if bot.status == BotStatus.RUNNING)
        total_trades = sum(bot.trades_executed for bot in self.bots.values())
        total_profit = sum(bot.total_profit for bot in self.bots.values())
        
        return {
            'total_bots': total_bots,
            'running_bots': running_bots,
            'total_trades': total_trades,
            'total_profit': total_profit,
            'average_profit_per_bot': total_profit / total_bots if total_bots > 0 else 0
        }
    
    def load_trained_models(self, models_config: Dict):
        """Load trained ML and RL models for bots."""
        try:
            # Load ML models
            if 'ml_models' in models_config:
                for model_name, model_path in models_config['ml_models'].items():
                    # Associate models with bots that need them
                    for bot in self.bots.values():
                        if bot.bot_type in [BotType.ML_BASED, BotType.HYBRID]:
                            bot.load_ml_model(model_path)
            
            # Load RL models
            if 'rl_models' in models_config:
                for model_name, model_info in models_config['rl_models'].items():
                    model_path = model_info.get('path')
                    algorithm = model_info.get('algorithm', 'DQN')
                    
                    for bot in self.bots.values():
                        if bot.bot_type in [BotType.RL_BASED, BotType.HYBRID]:
                            bot.load_rl_model(model_path, algorithm)
            
            bot_logger.info("Trained models loaded successfully")
            
        except Exception as e:
            bot_logger.error(f"Error loading trained models: {e}")
    
    def save_all_configs(self, directory: str):
        """Save all bot configurations."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for bot_id, bot in self.bots.items():
            filepath = Path(directory) / f"{bot_id}_config.json"
            bot.save_config(str(filepath))
        
        bot_logger.info(f"All bot configs saved to {directory}")


# Global bot manager instance
bot_manager = BotManager()
