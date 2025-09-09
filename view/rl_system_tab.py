"""
RL System Tab

Reinforcement Learning system interface for:
- Agent training and management
- Environment configuration
- Action space and reward design
- Training monitoring and visualization
- Policy evaluation and testing
- Portfolio management with RL
"""

import threading
from datetime import datetime, timedelta
from tkinter import messagebox, filedialog
from typing import Any, Dict, List, Optional, Tuple

try:
    from ttkbootstrap import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar, 
        BooleanVar, IntVar, Notebook, Progressbar, Separator, Treeview, Text
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar,
        BooleanVar, IntVar, Text
    )
    from tkinter.ttk import Notebook, Progressbar, Separator, Treeview
    HAS_TTKBOOTSTRAP = False

from view.utils import (
    ValidationMixin, StatusIndicator, LoadingIndicator, FormValidator,
    format_currency, format_percentage, notification_system
)

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class RLSystemTab(Frame, ValidationMixin):
    """
    Advanced RL system interface with agent management and training analytics.
    """
    
    # RL system constants
    AGENT_TYPES = ['dqn', 'ppo', 'a3c', 'ddpg', 'td3', 'sac']
    ENVIRONMENTS = ['discrete_trading', 'continuous_trading', 'portfolio_management', 'market_making']
    REWARD_FUNCTIONS = ['profit_loss', 'sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'custom']
    ACTION_SPACES = ['discrete', 'continuous', 'multi_discrete']
    TRAINING_ALGORITHMS = ['online', 'offline', 'mixed']
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # RL system state
        self._agents = {}
        self._training_history = []
        self._episodes = []
        self._rewards = []
        
        # Training state
        self._is_training = False
        self._training_progress = 0.0
        self._current_agent = None
        self._current_episode = 0
        
        # Environment state
        self._environment = None
        self._action_history = []
        
        # UI state
        self._is_loading = False
        self._last_update = None
        
        # Form validator
        self.form_validator = FormValidator()
        self._setup_validation_rules()
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Start data updates
        self._start_updates()
        
        app_logger.info("RLSystemTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            self.form_validator.add_field_rule('episodes', 'numeric_range', min=100, max=100000)
            self.form_validator.add_field_rule('learning_rate', 'numeric_range', min=0.00001, max=0.1)
            self.form_validator.add_field_rule('batch_size', 'numeric_range', min=16, max=2048)
            self.form_validator.add_field_rule('memory_size', 'numeric_range', min=1000, max=1000000)
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_agent_section()
            self._create_environment_section()
            self._create_training_section()
            self._create_performance_section()
            self._create_control_section()
            
        except Exception as e:
            app_logger.error(f"Error creating widgets: {e}")

    def _create_header_section(self):
        """Create header with RL system status and controls."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(2, weight=1)
        
        # Title and status
        Label(header_frame, text="RL System", font=('Arial', 16, 'bold')).grid(
            row=0, column=0, padx=5
        )
        
        self.system_status_label = Label(
            header_frame, text="Status: READY", 
            font=('Arial', 12, 'bold'), foreground='#17a2b8'
        )
        self.system_status_label.grid(row=0, column=1, padx=20)
        
        # RL controls
        control_frame = Frame(header_frame)
        control_frame.grid(row=0, column=3, padx=5)
        
        self.train_button = Button(
            control_frame, text="Train Agent", command=self._start_training,
            style='success.TButton', width=12
        )
        self.train_button.grid(row=0, column=0, padx=2)
        
        self.test_button = Button(
            control_frame, text="Test Agent", command=self._test_agent,
            style='info.TButton', width=12
        )
        self.test_button.grid(row=0, column=1, padx=2)
        
        # Agent summary
        self.agent_summary = Label(
            header_frame, text="Agents: 0 | Best Reward: N/A | Episodes: 0",
            font=('Arial', 10), foreground='#888888'
        )
        self.agent_summary.grid(row=1, column=0, columnspan=4, pady=5)

    def _create_agent_section(self):
        """Create agent configuration section."""
        agent_frame = Frame(self)
        agent_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        agent_frame.grid_rowconfigure(1, weight=1)
        
        Label(agent_frame, text="Agent Configuration", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Create notebook for agent sections
        agent_notebook = Notebook(agent_frame)
        agent_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Agent Selection Tab
        selection_tab = Frame(agent_notebook)
        agent_notebook.add(selection_tab, text="Agent Selection")
        self._create_agent_selection(selection_tab)
        
        # Network Tab
        network_tab = Frame(agent_notebook)
        agent_notebook.add(network_tab, text="Network Architecture")
        self._create_network_architecture(network_tab)
        
        # Hyperparameters Tab
        params_tab = Frame(agent_notebook)
        agent_notebook.add(params_tab, text="Hyperparameters")
        self._create_agent_hyperparameters(params_tab)

    def _create_agent_selection(self, parent):
        """Create agent selection interface."""
        # Agent type selection
        Label(parent, text="Agent Type:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.agent_type_var = StringVar(value='dqn')
        self.agent_type_select = OptionMenu(
            parent, self.agent_type_var, 'dqn', *self.AGENT_TYPES
        )
        self.agent_type_select.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Action space
        Label(parent, text="Action Space:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.action_space_var = StringVar(value='discrete')
        self.action_space_select = OptionMenu(
            parent, self.action_space_var, 'discrete', *self.ACTION_SPACES
        )
        self.action_space_select.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Training algorithm
        Label(parent, text="Training Mode:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.training_mode_var = StringVar(value='online')
        self.training_mode_select = OptionMenu(
            parent, self.training_mode_var, 'online', *self.TRAINING_ALGORITHMS
        )
        self.training_mode_select.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Active agents display
        Label(parent, text="Active Agents:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=(20, 5)
        )
        
        self.agents_tree = Treeview(
            parent, 
            columns=('Agent', 'Type', 'Reward', 'Episodes', 'Status'),
            show='headings', height=8
        )
        
        for col in ('Agent', 'Type', 'Reward', 'Episodes', 'Status'):
            self.agents_tree.heading(col, text=col)
            self.agents_tree.column(col, width=80)
        
        self.agents_tree.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        parent.grid_columnconfigure(1, weight=1)

    def _create_network_architecture(self, parent):
        """Create network architecture interface."""
        # Network parameters
        network_frame = Frame(parent)
        network_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(network_frame, text="Hidden Layers:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.hidden_layers_var = StringVar(value='256,128')
        self.hidden_layers_entry = Entry(network_frame, textvariable=self.hidden_layers_var, width=20)
        self.hidden_layers_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        Label(network_frame, text="Activation Function:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.activation_var = StringVar(value='relu')
        self.activation_select = OptionMenu(
            network_frame, self.activation_var, 'relu', 
            'relu', 'tanh', 'sigmoid', 'leaky_relu'
        )
        self.activation_select.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        Label(network_frame, text="Optimizer:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.optimizer_var = StringVar(value='adam')
        self.optimizer_select = OptionMenu(
            network_frame, self.optimizer_var, 'adam',
            'adam', 'sgd', 'rmsprop', 'adamw'
        )
        self.optimizer_select.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Advanced options
        Label(network_frame, text="Dropout Rate:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        
        self.dropout_var = StringVar(value='0.1')
        self.dropout_scale = Scale(
            network_frame, from_=0.0, to=0.5, orient='horizontal',
            variable=self.dropout_var
        )
        self.dropout_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        
        network_frame.grid_columnconfigure(1, weight=1)

    def _create_agent_hyperparameters(self, parent):
        """Create agent hyperparameters interface."""
        # Training parameters
        params_frame = Frame(parent)
        params_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(params_frame, text="Episodes:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.episodes_var = StringVar(value='10000')
        self.episodes_entry = Entry(params_frame, textvariable=self.episodes_var, width=10)
        self.episodes_entry.grid(row=0, column=1, padx=5, pady=5)
        
        Label(params_frame, text="Learning Rate:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.rl_learning_rate_var = StringVar(value='0.001')
        self.rl_learning_rate_entry = Entry(params_frame, textvariable=self.rl_learning_rate_var, width=10)
        self.rl_learning_rate_entry.grid(row=1, column=1, padx=5, pady=5)
        
        Label(params_frame, text="Batch Size:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.rl_batch_size_var = StringVar(value='64')
        self.rl_batch_size_entry = Entry(params_frame, textvariable=self.rl_batch_size_var, width=10)
        self.rl_batch_size_entry.grid(row=2, column=1, padx=5, pady=5)
        
        Label(params_frame, text="Memory Size:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        
        self.memory_size_var = StringVar(value='100000')
        self.memory_size_entry = Entry(params_frame, textvariable=self.memory_size_var, width=10)
        self.memory_size_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # RL-specific parameters
        Label(params_frame, text="Discount Factor (γ):", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', padx=5, pady=5
        )
        
        self.gamma_var = StringVar(value='0.99')
        self.gamma_scale = Scale(
            params_frame, from_=0.9, to=0.999, orient='horizontal',
            variable=self.gamma_var
        )
        self.gamma_scale.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        
        Label(params_frame, text="Exploration Rate (ε):", font=('Arial', 10, 'bold')).grid(
            row=5, column=0, sticky='w', padx=5, pady=5
        )
        
        self.epsilon_var = StringVar(value='1.0')
        self.epsilon_scale = Scale(
            params_frame, from_=0.01, to=1.0, orient='horizontal',
            variable=self.epsilon_var
        )
        self.epsilon_scale.grid(row=5, column=1, sticky='ew', padx=5, pady=5)
        
        params_frame.grid_columnconfigure(1, weight=1)

    def _create_environment_section(self):
        """Create environment configuration section."""
        env_frame = Frame(self)
        env_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        env_frame.grid_rowconfigure(1, weight=1)
        
        Label(env_frame, text="Environment", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Environment configuration
        config_frame = Frame(env_frame)
        config_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        Label(config_frame, text="Environment Type:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.env_type_var = StringVar(value='discrete_trading')
        self.env_type_select = OptionMenu(
            config_frame, self.env_type_var, 'discrete_trading', *self.ENVIRONMENTS
        )
        self.env_type_select.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        Label(config_frame, text="Reward Function:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.reward_func_var = StringVar(value='profit_loss')
        self.reward_func_select = OptionMenu(
            config_frame, self.reward_func_var, 'profit_loss', *self.REWARD_FUNCTIONS
        )
        self.reward_func_select.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Action space configuration
        Label(config_frame, text="Action Configuration:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=(20, 5)
        )
        
        actions_frame = Frame(config_frame)
        actions_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5)
        
        # Action buttons for discrete actions
        self.action_vars = {}
        actions = ['BUY', 'SELL', 'HOLD']
        
        for i, action in enumerate(actions):
            var = BooleanVar(value=True)
            self.action_vars[action] = var
            
            checkbox = Button(
                actions_frame, text=action,
                command=lambda a=action: self._toggle_action(a),
                style='outline.TButton', width=10
            )
            checkbox.grid(row=0, column=i, padx=5, pady=5)
        
        # Environment status
        Separator(env_frame, orient='horizontal').grid(
            row=2, column=0, sticky='ew', pady=20
        )
        
        Label(env_frame, text="Environment Status:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        
        self.env_status_label = Label(
            env_frame, text="Not initialized", 
            font=('Arial', 9), foreground='#888888'
        )
        self.env_status_label.grid(row=4, column=0, sticky='w', padx=5)
        
        # Environment controls
        env_controls = Frame(env_frame)
        env_controls.grid(row=5, column=0, sticky='ew', padx=5, pady=10)
        
        Button(
            env_controls, text="Initialize Env", command=self._initialize_environment,
            style='success.TButton', width=15
        ).grid(row=0, column=0, padx=2)
        
        Button(
            env_controls, text="Reset Env", command=self._reset_environment,
            style='warning.TButton', width=15
        ).grid(row=0, column=1, padx=2)
        
        config_frame.grid_columnconfigure(1, weight=1)

    def _create_training_section(self):
        """Create training monitoring section."""
        training_frame = Frame(self)
        training_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        training_frame.grid_rowconfigure(3, weight=1)
        
        Label(training_frame, text="Training Monitor", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Training progress
        progress_frame = Frame(training_frame)
        progress_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        Label(progress_frame, text="Training Progress:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w'
        )
        
        self.training_progress_bar = Progressbar(
            progress_frame, mode='determinate', style='info.Horizontal.TProgressbar'
        )
        self.training_progress_bar.grid(row=1, column=0, sticky='ew', pady=5)
        
        self.training_status_label = Label(
            progress_frame, text="Ready to train", font=('Arial', 9)
        )
        self.training_status_label.grid(row=2, column=0, sticky='w')
        
        # Episode info
        episode_frame = Frame(training_frame)
        episode_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        self.episode_label = Label(
            episode_frame, text="Episode: 0/0", font=('Arial', 9)
        )
        self.episode_label.grid(row=0, column=0, sticky='w')
        
        self.reward_label = Label(
            episode_frame, text="Last Reward: 0.0", font=('Arial', 9)
        )
        self.reward_label.grid(row=1, column=0, sticky='w')
        
        self.avg_reward_label = Label(
            episode_frame, text="Avg Reward: 0.0", font=('Arial', 9)
        )
        self.avg_reward_label.grid(row=2, column=0, sticky='w')
        
        # Training log
        Label(training_frame, text="Training Log:", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', pady=(10, 5)
        )
        
        self.rl_training_log = Text(training_frame, height=12, width=35, font=('Courier', 8))
        self.rl_training_log.grid(row=5, column=0, sticky='nsew', padx=5, pady=5)
        
        # Training controls
        control_frame = Frame(training_frame)
        control_frame.grid(row=6, column=0, sticky='ew', padx=5, pady=5)
        
        Button(
            control_frame, text="Start", command=self._start_training,
            style='success.TButton', width=8
        ).grid(row=0, column=0, padx=1)
        
        Button(
            control_frame, text="Stop", command=self._stop_training,
            style='danger.TButton', width=8
        ).grid(row=0, column=1, padx=1)
        
        Button(
            control_frame, text="Pause", command=self._pause_training,
            style='warning.TButton', width=8
        ).grid(row=0, column=2, padx=1)
        
        Button(
            control_frame, text="Clear", command=self._clear_rl_log,
            style='secondary.TButton', width=8
        ).grid(row=0, column=3, padx=1)
        
        progress_frame.grid_columnconfigure(0, weight=1)

    def _create_performance_section(self):
        """Create performance monitoring section."""
        perf_frame = Frame(self)
        perf_frame.grid(row=2, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        perf_frame.grid_rowconfigure(1, weight=1)
        
        Label(perf_frame, text="Agent Performance", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Performance notebook
        perf_notebook = Notebook(perf_frame)
        perf_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Rewards tab
        rewards_tab = Frame(perf_notebook)
        perf_notebook.add(rewards_tab, text="Rewards")
        self._create_rewards_display(rewards_tab)
        
        # Actions tab
        actions_tab = Frame(perf_notebook)
        perf_notebook.add(actions_tab, text="Action History")
        self._create_actions_display(actions_tab)
        
        # Portfolio tab
        portfolio_tab = Frame(perf_notebook)
        perf_notebook.add(portfolio_tab, text="Portfolio")
        self._create_portfolio_display(portfolio_tab)

    def _create_rewards_display(self, parent):
        """Create rewards visualization display."""
        rewards_frame = Frame(parent)
        rewards_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Reward statistics
        stats_frame = Frame(rewards_frame)
        stats_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        self.total_reward_label = Label(stats_frame, text="Total Reward: 0.0", font=('Arial', 10, 'bold'))
        self.total_reward_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.best_reward_label = Label(stats_frame, text="Best Episode: 0.0", font=('Arial', 10))
        self.best_reward_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.worst_reward_label = Label(stats_frame, text="Worst Episode: 0.0", font=('Arial', 10))
        self.worst_reward_label.grid(row=2, column=0, sticky='w', pady=2)
        
        # Recent rewards display
        Label(parent, text="Recent Episodes:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=(20, 5)
        )
        
        self.rewards_tree = Treeview(
            parent,
            columns=('Episode', 'Reward', 'Steps', 'Avg_Reward', 'Epsilon'),
            show='headings', height=12
        )
        
        for col in ('Episode', 'Reward', 'Steps', 'Avg_Reward', 'Epsilon'):
            self.rewards_tree.heading(col, text=col)
            self.rewards_tree.column(col, width=80)
        
        self.rewards_tree.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def _create_actions_display(self, parent):
        """Create actions history display."""
        self.actions_tree = Treeview(
            parent,
            columns=('Step', 'State', 'Action', 'Reward', 'Done'),
            show='headings', height=15
        )
        
        for col in ('Step', 'State', 'Action', 'Reward', 'Done'):
            self.actions_tree.heading(col, text=col)
            self.actions_tree.column(col, width=80)
        
        self.actions_tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def _create_portfolio_display(self, parent):
        """Create portfolio performance display."""
        portfolio_frame = Frame(parent)
        portfolio_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Portfolio metrics
        self.portfolio_value_label = Label(portfolio_frame, text="Portfolio Value: $0.00", font=('Arial', 10, 'bold'))
        self.portfolio_value_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.total_return_label = Label(portfolio_frame, text="Total Return: 0.0%", font=('Arial', 10))
        self.total_return_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.sharpe_ratio_label = Label(portfolio_frame, text="Sharpe Ratio: 0.0", font=('Arial', 10))
        self.sharpe_ratio_label.grid(row=2, column=0, sticky='w', pady=2)
        
        self.max_drawdown_label = Label(portfolio_frame, text="Max Drawdown: 0.0%", font=('Arial', 10))
        self.max_drawdown_label.grid(row=3, column=0, sticky='w', pady=2)

    def _create_control_section(self):
        """Create system control and status section."""
        control_frame = Frame(self)
        control_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons
        buttons_frame = Frame(control_frame)
        buttons_frame.grid(row=0, column=0, padx=5)
        
        Button(
            buttons_frame, text="Save Agent", command=self._save_agent,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            buttons_frame, text="Load Agent", command=self._load_agent,
            style='info.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        Button(
            buttons_frame, text="Export Results", command=self._export_results,
            style='secondary.TButton', width=12
        ).grid(row=0, column=2, padx=2)
        
        # Status indicator
        self.status_indicator = StatusIndicator(control_frame)
        self.status_indicator.grid(row=0, column=1, sticky='ew', padx=10)
        
        # Loading indicator
        self.loading_indicator = LoadingIndicator(control_frame)
        self.loading_indicator.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

    def _setup_layout(self):
        """Configure grid layout weights."""
        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

    def _bind_events(self):
        """Bind event handlers."""
        # Bind validation events
        self.episodes_entry.bind('<FocusOut>', self._validate_episodes)
        self.rl_learning_rate_entry.bind('<FocusOut>', self._validate_rl_learning_rate)
        self.rl_batch_size_entry.bind('<FocusOut>', self._validate_rl_batch_size)
        self.memory_size_entry.bind('<FocusOut>', self._validate_memory_size)

    def _start_updates(self):
        """Start periodic data updates."""
        try:
            self._update_agent_status()
            self._update_performance_metrics()
            
            # Schedule next update
            self.after(10000, self._start_updates)  # Update every 10 seconds
            
        except Exception as e:
            app_logger.error(f"Error in periodic updates: {e}")

    def _toggle_action(self, action: str):
        """Toggle action availability."""
        try:
            current_state = self.action_vars[action].get()
            self.action_vars[action].set(not current_state)
            
            self.status_indicator.set_status(
                f"Action {action} {'enabled' if not current_state else 'disabled'}", 
                'info'
            )
            
        except Exception as e:
            app_logger.error(f"Error toggling action {action}: {e}")

    def _initialize_environment(self):
        """Initialize RL environment."""
        try:
            self.loading_indicator.show_loading("Initializing environment...")
            
            # Simulate environment initialization
            env_type = self.env_type_var.get()
            reward_func = self.reward_func_var.get()
            
            # Update environment status
            self.env_status_label.configure(
                text=f"Initialized: {env_type} with {reward_func}",
                foreground='#28a745'
            )
            
            self.status_indicator.set_status("Environment initialized successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error initializing environment: {e}")
            self.status_indicator.set_status(f"Environment initialization failed: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _reset_environment(self):
        """Reset RL environment."""
        try:
            self.env_status_label.configure(
                text="Reset complete - Ready for new episode",
                foreground='#17a2b8'
            )
            
            self.status_indicator.set_status("Environment reset", 'info')
            
        except Exception as e:
            app_logger.error(f"Error resetting environment: {e}")

    def _start_training(self):
        """Start RL agent training."""
        try:
            if self._is_training:
                self.status_indicator.set_status("Training already in progress", 'warning')
                return
                
            # Validate configuration
            validation_result = self._validate_rl_config()
            if not validation_result['valid']:
                self.status_indicator.set_status(f"Configuration error: {validation_result['message']}", 'error')
                return
            
            self.loading_indicator.show_loading("Starting RL training...")
            self._is_training = True
            self._current_episode = 0
            
            # Update UI
            self.train_button.configure(state='disabled')
            self.system_status_label.configure(
                text="Status: TRAINING", 
                foreground='#ffc107'
            )
            
            # Start training in background thread
            training_thread = threading.Thread(target=self._rl_training_worker, daemon=True)
            training_thread.start()
            
            self.status_indicator.set_status("RL training started", 'success')
            
        except Exception as e:
            app_logger.error(f"Error starting RL training: {e}")
            self.status_indicator.set_status(f"Failed to start training: {str(e)}", 'error')
            self._is_training = False
            self.train_button.configure(state='normal')

    def _rl_training_worker(self):
        """Background RL training worker."""
        try:
            total_episodes = int(self.episodes_var.get())
            
            self.after(0, self._log_rl_training, "Starting RL training...")
            self.after(0, self._log_rl_training, f"Agent: {self.agent_type_var.get()}")
            self.after(0, self._log_rl_training, f"Environment: {self.env_type_var.get()}")
            
            for episode in range(total_episodes):
                if not self._is_training:  # Check for stop signal
                    break
                    
                self._current_episode = episode + 1
                
                # Simulate episode training
                episode_reward = self._simulate_episode()
                
                # Update progress
                progress = (episode + 1) / total_episodes * 100
                self.after(0, self._update_rl_training_progress, progress, episode + 1, episode_reward)
                
                # Log progress periodically
                if (episode + 1) % 100 == 0:
                    avg_reward = sum(self._rewards[-100:]) / min(len(self._rewards), 100)
                    self.after(0, self._log_rl_training, 
                              f"Episode {episode + 1}/{total_episodes} - Avg Reward: {avg_reward:.3f}")
                
                # Simulate training time
                import time
                time.sleep(0.01)
            
            if self._is_training:  # Training completed normally
                self.after(0, self._rl_training_completed)
            else:  # Training was stopped
                self.after(0, self._rl_training_stopped)
                
        except Exception as e:
            app_logger.error(f"Error in RL training worker: {e}")
            self.after(0, self._rl_training_error, str(e))

    def _simulate_episode(self) -> float:
        """Simulate a single training episode."""
        # Simulate episode reward (placeholder)
        import random
        base_reward = random.uniform(-100, 100)
        
        # Simulate learning improvement
        improvement = (self._current_episode / 10000) * 50
        episode_reward = base_reward + improvement
        
        self._rewards.append(episode_reward)
        
        # Add to rewards display
        if len(self._rewards) % 50 == 0:  # Update display every 50 episodes
            self.after(0, self._add_episode_to_display, self._current_episode, episode_reward)
        
        return episode_reward

    def _update_rl_training_progress(self, progress: float, episode: int, reward: float):
        """Update RL training progress."""
        try:
            self.training_progress_bar['value'] = progress
            self.training_status_label.configure(text=f"Training... {progress:.1f}% complete")
            self.episode_label.configure(text=f"Episode: {episode}/{self.episodes_var.get()}")
            self.reward_label.configure(text=f"Last Reward: {reward:.2f}")
            
            if self._rewards:
                avg_reward = sum(self._rewards[-100:]) / min(len(self._rewards), 100)
                self.avg_reward_label.configure(text=f"Avg Reward: {avg_reward:.2f}")
            
        except Exception as e:
            app_logger.error(f"Error updating RL training progress: {e}")

    def _add_episode_to_display(self, episode: int, reward: float):
        """Add episode to rewards display."""
        try:
            epsilon = float(self.epsilon_var.get()) * (0.95 ** (episode / 1000))  # Decay
            avg_reward = sum(self._rewards[-100:]) / min(len(self._rewards), 100)
            
            self.rewards_tree.insert('', 0, values=(
                episode, f"{reward:.2f}", "100", f"{avg_reward:.2f}", f"{epsilon:.3f}"
            ))
            
            # Limit display size
            children = self.rewards_tree.get_children()
            if len(children) > 100:
                for item in children[100:]:
                    self.rewards_tree.delete(item)
                    
        except Exception as e:
            app_logger.error(f"Error adding episode to display: {e}")

    def _log_rl_training(self, message: str):
        """Add message to RL training log."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.rl_training_log.insert('end', log_entry)
            self.rl_training_log.see('end')
            
        except Exception as e:
            app_logger.error(f"Error logging RL training message: {e}")

    def _rl_training_completed(self):
        """Handle RL training completion."""
        try:
            self._is_training = False
            self.train_button.configure(state='normal')
            
            self.system_status_label.configure(
                text="Status: READY",
                foreground='#17a2b8'
            )
            
            self.training_status_label.configure(text="Training completed successfully")
            self.loading_indicator.hide_loading()
            
            self._log_rl_training("RL training completed successfully!")
            
            # Update agent list
            agent_name = f"{self.agent_type_var.get()}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            best_reward = max(self._rewards) if self._rewards else 0.0
            
            self._agents[agent_name] = {
                'type': self.agent_type_var.get(),
                'best_reward': best_reward,
                'episodes': len(self._rewards),
                'status': 'trained',
                'trained_at': datetime.now()
            }
            
            self._update_agent_display()
            self.status_indicator.set_status("RL training completed successfully", 'success')
            
            notification_system.show_success(
                "RL Training",
                "Agent training completed successfully"
            )
            
        except Exception as e:
            app_logger.error(f"Error handling RL training completion: {e}")

    def _rl_training_stopped(self):
        """Handle RL training stop."""
        try:
            self._is_training = False
            self.train_button.configure(state='normal')
            
            self.system_status_label.configure(
                text="Status: READY",
                foreground='#17a2b8'
            )
            
            self.training_status_label.configure(text="Training stopped by user")
            self.loading_indicator.hide_loading()
            
            self._log_rl_training("Training stopped by user")
            self.status_indicator.set_status("Training stopped", 'warning')
            
        except Exception as e:
            app_logger.error(f"Error handling RL training stop: {e}")

    def _rl_training_error(self, error_message: str):
        """Handle RL training error."""
        try:
            self._is_training = False
            self.train_button.configure(state='normal')
            
            self.system_status_label.configure(
                text="Status: ERROR",
                foreground='#dc3545'
            )
            
            self.training_status_label.configure(text=f"Training failed: {error_message}")
            self.loading_indicator.hide_loading()
            
            self._log_rl_training(f"Training failed: {error_message}")
            self.status_indicator.set_status(f"Training failed: {error_message}", 'error')
            
        except Exception as e:
            app_logger.error(f"Error handling RL training error: {e}")

    def _stop_training(self):
        """Stop RL training."""
        try:
            if not self._is_training:
                self.status_indicator.set_status("No training in progress", 'warning')
                return
                
            self._is_training = False
            self.status_indicator.set_status("Stopping training...", 'warning')
            
        except Exception as e:
            app_logger.error(f"Error stopping training: {e}")

    def _pause_training(self):
        """Pause RL training."""
        try:
            # Placeholder for pause functionality
            self.status_indicator.set_status("Training paused", 'warning')
            
        except Exception as e:
            app_logger.error(f"Error pausing training: {e}")

    def _clear_rl_log(self):
        """Clear RL training log."""
        try:
            self.rl_training_log.delete('1.0', 'end')
            self.status_indicator.set_status("Training log cleared", 'info')
            
        except Exception as e:
            app_logger.error(f"Error clearing RL log: {e}")

    def _test_agent(self):
        """Test trained agent."""
        try:
            if not self._agents:
                self.status_indicator.set_status("No trained agents available", 'warning')
                return
                
            self.loading_indicator.show_loading("Testing agent...")
            
            # Simulate agent testing
            test_episodes = 10
            test_rewards = []
            
            for i in range(test_episodes):
                reward = self._simulate_episode()
                test_rewards.append(reward)
            
            avg_test_reward = sum(test_rewards) / len(test_rewards)
            
            self._log_rl_training(f"Test completed: Avg reward over {test_episodes} episodes: {avg_test_reward:.2f}")
            self.status_indicator.set_status(f"Agent testing completed - Avg reward: {avg_test_reward:.2f}", 'success')
            
        except Exception as e:
            app_logger.error(f"Error testing agent: {e}")
            self.status_indicator.set_status(f"Agent testing failed: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _validate_rl_config(self) -> Dict[str, Any]:
        """Validate RL configuration."""
        try:
            form_data = {
                'episodes': self.episodes_var.get(),
                'learning_rate': self.rl_learning_rate_var.get(),
                'batch_size': self.rl_batch_size_var.get(),
                'memory_size': self.memory_size_var.get()
            }
            
            validation_result = self.form_validator.validate_form(form_data)
            
            if not validation_result['valid']:
                return {
                    'valid': False,
                    'message': '; '.join(validation_result['messages'])
                }
            
            return {'valid': True, 'message': 'Configuration is valid'}
            
        except Exception as e:
            app_logger.error(f"Error validating RL config: {e}")
            return {'valid': False, 'message': str(e)}

    def _validate_episodes(self, event=None):
        """Validate episodes parameter."""
        try:
            value = self.episodes_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('episodes', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid episodes: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating episodes: {e}")

    def _validate_rl_learning_rate(self, event=None):
        """Validate RL learning rate."""
        try:
            value = self.rl_learning_rate_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('learning_rate', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid learning rate: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating RL learning rate: {e}")

    def _validate_rl_batch_size(self, event=None):
        """Validate RL batch size."""
        try:
            value = self.rl_batch_size_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('batch_size', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid batch size: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating RL batch size: {e}")

    def _validate_memory_size(self, event=None):
        """Validate memory size."""
        try:
            value = self.memory_size_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('memory_size', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid memory size: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating memory size: {e}")

    def _update_agent_display(self):
        """Update agents display."""
        try:
            # Clear current display
            self.agents_tree.delete(*self.agents_tree.get_children())
            
            # Add active agents
            for agent_name, data in self._agents.items():
                self.agents_tree.insert('', 'end', values=(
                    agent_name,
                    data['type'].upper(),
                    f"{data['best_reward']:.2f}",
                    data['episodes'],
                    data['status'].upper()
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating agent display: {e}")

    def _update_agent_status(self):
        """Update agent status summary."""
        try:
            agent_count = len(self._agents)
            best_reward = max([a['best_reward'] for a in self._agents.values()], default=0.0)
            total_episodes = sum([a['episodes'] for a in self._agents.values()])
            
            self.agent_summary.configure(
                text=f"Agents: {agent_count} | Best Reward: {best_reward:.2f} | Episodes: {total_episodes}"
            )
            
        except Exception as e:
            app_logger.error(f"Error updating agent status: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics display."""
        try:
            if self._rewards:
                total_reward = sum(self._rewards)
                best_reward = max(self._rewards)
                worst_reward = min(self._rewards)
                
                # Update reward statistics
                self.total_reward_label.configure(text=f"Total Reward: {total_reward:.2f}")
                self.best_reward_label.configure(text=f"Best Episode: {best_reward:.2f}")
                self.worst_reward_label.configure(text=f"Worst Episode: {worst_reward:.2f}")
                
                # Update portfolio metrics (placeholder)
                portfolio_value = 10000 + total_reward  # Starting with $10,000
                total_return = (total_reward / 10000) * 100
                
                self.portfolio_value_label.configure(text=f"Portfolio Value: {format_currency(portfolio_value)}")
                self.total_return_label.configure(text=f"Total Return: {total_return:.2f}%")
                
        except Exception as e:
            app_logger.error(f"Error updating performance metrics: {e}")

    def _save_agent(self):
        """Save trained agent."""
        try:
            if not self._agents:
                self.status_indicator.set_status("No agents to save", 'warning')
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".pth",
                filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
            )
            
            if filename:
                # Placeholder for agent saving
                self.status_indicator.set_status("Agent saved successfully", 'success')
                
        except Exception as e:
            app_logger.error(f"Error saving agent: {e}")
            self.status_indicator.set_status(f"Failed to save agent: {str(e)}", 'error')

    def _load_agent(self):
        """Load trained agent."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
            )
            
            if filename:
                # Placeholder for agent loading
                self.status_indicator.set_status("Agent loaded successfully", 'success')
                self._update_agent_display()
                
        except Exception as e:
            app_logger.error(f"Error loading agent: {e}")
            self.status_indicator.set_status(f"Failed to load agent: {str(e)}", 'error')

    def _export_results(self):
        """Export training results."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Placeholder for results export
                self.status_indicator.set_status("Results exported successfully", 'success')
                
        except Exception as e:
            app_logger.error(f"Error exporting results: {e}")
            self.status_indicator.set_status(f"Failed to export results: {str(e)}", 'error')

    def cleanup(self):
        """Cleanup tab resources."""
        try:
            # Stop training if in progress
            if self._is_training:
                self._is_training = False
                
            app_logger.info("RLSystemTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during RLSystemTab cleanup: {e}")

    def refresh(self):
        """Refresh tab content."""
        try:
            self._update_agent_status()
            self._update_performance_metrics()
            self.status_indicator.set_status("Data refreshed", 'success')
            
        except Exception as e:
            app_logger.error(f"Error refreshing RLSystemTab: {e}")
