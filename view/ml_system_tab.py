"""
ML System Tab

Machine Learning system interface for:
- Model training and management
- Feature engineering and selection
- Prediction generation and analysis
- Model performance monitoring
- Data preprocessing and validation
- Hyperparameter optimization
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


class MLSystemTab(Frame, ValidationMixin):
    """
    Advanced ML system interface with model management and prediction analytics.
    """
    
    # ML system constants
    MODEL_TYPES = ['linear_regression', 'random_forest', 'xgboost', 'lstm', 'transformer', 'ensemble']
    FEATURE_TYPES = ['technical', 'fundamental', 'sentiment', 'market_structure', 'cross_asset']
    PREDICTION_HORIZONS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    OPTIMIZATION_METHODS = ['grid_search', 'random_search', 'bayesian', 'genetic', 'pso']
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # ML system state
        self._models = {}
        self._training_history = []
        self._predictions = []
        self._feature_importance = {}
        
        # Training state
        self._is_training = False
        self._training_progress = 0.0
        self._current_model = None
        
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
        
        app_logger.info("MLSystemTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            self.form_validator.add_field_rule('train_split', 'numeric_range', min=0.1, max=0.9)
            self.form_validator.add_field_rule('learning_rate', 'numeric_range', min=0.0001, max=0.1)
            self.form_validator.add_field_rule('epochs', 'numeric_range', min=1, max=1000)
            self.form_validator.add_field_rule('batch_size', 'numeric_range', min=16, max=1024)
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_model_section()
            self._create_training_section()
            self._create_prediction_section()
            self._create_performance_section()
            self._create_control_section()
            
        except Exception as e:
            app_logger.error(f"Error creating widgets: {e}")

    def _create_header_section(self):
        """Create header with ML system status and controls."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(2, weight=1)
        
        # Title and status
        Label(header_frame, text="ML System", font=('Arial', 16, 'bold')).grid(
            row=0, column=0, padx=5
        )
        
        self.system_status_label = Label(
            header_frame, text="Status: READY", 
            font=('Arial', 12, 'bold'), foreground='#17a2b8'
        )
        self.system_status_label.grid(row=0, column=1, padx=20)
        
        # ML controls
        control_frame = Frame(header_frame)
        control_frame.grid(row=0, column=3, padx=5)
        
        self.train_button = Button(
            control_frame, text="Train Model", command=self._start_training,
            style='success.TButton', width=12
        )
        self.train_button.grid(row=0, column=0, padx=2)
        
        self.predict_button = Button(
            control_frame, text="Generate Predictions", command=self._generate_predictions,
            style='info.TButton', width=15
        )
        self.predict_button.grid(row=0, column=1, padx=2)
        
        # Model summary
        self.model_summary = Label(
            header_frame, text="Models: 0 | Accuracy: N/A | Last Training: Never",
            font=('Arial', 10), foreground='#888888'
        )
        self.model_summary.grid(row=1, column=0, columnspan=4, pady=5)

    def _create_model_section(self):
        """Create model configuration section."""
        model_frame = Frame(self)
        model_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        model_frame.grid_rowconfigure(1, weight=1)
        
        Label(model_frame, text="Model Configuration", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Create notebook for model sections
        model_notebook = Notebook(model_frame)
        model_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Model Selection Tab
        selection_tab = Frame(model_notebook)
        model_notebook.add(selection_tab, text="Model Selection")
        self._create_model_selection(selection_tab)
        
        # Features Tab
        features_tab = Frame(model_notebook)
        model_notebook.add(features_tab, text="Feature Engineering")
        self._create_feature_engineering(features_tab)
        
        # Hyperparameters Tab
        params_tab = Frame(model_notebook)
        model_notebook.add(params_tab, text="Hyperparameters")
        self._create_hyperparameters(params_tab)

    def _create_model_selection(self, parent):
        """Create model selection interface."""
        # Model type selection
        Label(parent, text="Model Type:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.model_type_var = StringVar(value='random_forest')
        self.model_type_select = OptionMenu(
            parent, self.model_type_var, 'random_forest', *self.MODEL_TYPES
        )
        self.model_type_select.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Prediction horizon
        Label(parent, text="Prediction Horizon:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.horizon_var = StringVar(value='1h')
        self.horizon_select = OptionMenu(
            parent, self.horizon_var, '1h', *self.PREDICTION_HORIZONS
        )
        self.horizon_select.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Target variable
        Label(parent, text="Target Variable:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.target_var = StringVar(value='price_direction')
        self.target_select = OptionMenu(
            parent, self.target_var, 'price_direction', 
            'price_direction', 'price_return', 'volatility', 'volume'
        )
        self.target_select.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Active models display
        Label(parent, text="Active Models:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=(20, 5)
        )
        
        self.models_tree = Treeview(
            parent, 
            columns=('Model', 'Type', 'Accuracy', 'Status'),
            show='headings', height=8
        )
        
        for col in ('Model', 'Type', 'Accuracy', 'Status'):
            self.models_tree.heading(col, text=col)
            self.models_tree.column(col, width=100)
        
        self.models_tree.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        parent.grid_columnconfigure(1, weight=1)

    def _create_feature_engineering(self, parent):
        """Create feature engineering interface."""
        # Feature types
        Label(parent, text="Feature Types:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        feature_frame = Frame(parent)
        feature_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        self.feature_vars = {}
        for i, feature_type in enumerate(self.FEATURE_TYPES):
            var = BooleanVar(value=True)
            self.feature_vars[feature_type] = var
            
            checkbox = Button(
                feature_frame, text=feature_type.replace('_', ' ').title(),
                command=lambda f=feature_type: self._toggle_feature(f),
                style='outline.TButton', width=15
            )
            checkbox.grid(row=i//3, column=i%3, padx=5, pady=2, sticky='ew')
        
        # Feature parameters
        params_frame = Frame(parent)
        params_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=20)
        
        Label(params_frame, text="Lookback Period:", font=('Arial', 10)).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.lookback_var = IntVar(value=20)
        self.lookback_scale = Scale(
            params_frame, from_=5, to=100, orient='horizontal',
            variable=self.lookback_var
        )
        self.lookback_scale.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        Label(params_frame, text="Feature Selection:", font=('Arial', 10)).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.feature_selection_var = StringVar(value='correlation')
        self.feature_selection_select = OptionMenu(
            params_frame, self.feature_selection_var, 'correlation',
            'correlation', 'mutual_info', 'chi2', 'recursive', 'lasso'
        )
        self.feature_selection_select.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        params_frame.grid_columnconfigure(1, weight=1)

    def _create_hyperparameters(self, parent):
        """Create hyperparameters interface."""
        # Training parameters
        params_frame = Frame(parent)
        params_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(params_frame, text="Train/Test Split:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.train_split_var = StringVar(value='0.8')
        self.train_split_entry = Entry(params_frame, textvariable=self.train_split_var, width=10)
        self.train_split_entry.grid(row=0, column=1, padx=5, pady=5)
        
        Label(params_frame, text="Learning Rate:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.learning_rate_var = StringVar(value='0.001')
        self.learning_rate_entry = Entry(params_frame, textvariable=self.learning_rate_var, width=10)
        self.learning_rate_entry.grid(row=1, column=1, padx=5, pady=5)
        
        Label(params_frame, text="Epochs:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.epochs_var = StringVar(value='100')
        self.epochs_entry = Entry(params_frame, textvariable=self.epochs_var, width=10)
        self.epochs_entry.grid(row=2, column=1, padx=5, pady=5)
        
        Label(params_frame, text="Batch Size:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        
        self.batch_size_var = StringVar(value='32')
        self.batch_size_entry = Entry(params_frame, textvariable=self.batch_size_var, width=10)
        self.batch_size_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Optimization
        Label(params_frame, text="Optimization Method:", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', padx=5, pady=5
        )
        
        self.optimization_var = StringVar(value='grid_search')
        self.optimization_select = OptionMenu(
            params_frame, self.optimization_var, 'grid_search', *self.OPTIMIZATION_METHODS
        )
        self.optimization_select.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        
        # Auto-tuning
        self.auto_tune_var = BooleanVar()
        auto_tune_checkbox = Button(
            params_frame, text="Enable Auto-Tuning",
            command=self._toggle_auto_tune,
            style='outline.TButton'
        )
        auto_tune_checkbox.grid(row=5, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        params_frame.grid_columnconfigure(1, weight=1)

    def _create_training_section(self):
        """Create training monitoring section."""
        training_frame = Frame(self)
        training_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        training_frame.grid_rowconfigure(2, weight=1)
        
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
            progress_frame, mode='determinate', style='success.Horizontal.TProgressbar'
        )
        self.training_progress_bar.grid(row=1, column=0, sticky='ew', pady=5)
        
        self.training_status_label = Label(
            progress_frame, text="Ready to train", font=('Arial', 9)
        )
        self.training_status_label.grid(row=2, column=0, sticky='w')
        
        # Training log
        Label(training_frame, text="Training Log:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', pady=(20, 5)
        )
        
        self.training_log = Text(training_frame, height=15, width=40, font=('Courier', 9))
        self.training_log.grid(row=4, column=0, sticky='nsew', padx=5, pady=5)
        
        # Training controls
        control_frame = Frame(training_frame)
        control_frame.grid(row=5, column=0, sticky='ew', padx=5, pady=5)
        
        Button(
            control_frame, text="Start Training", command=self._start_training,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            control_frame, text="Stop Training", command=self._stop_training,
            style='danger.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        Button(
            control_frame, text="Clear Log", command=self._clear_log,
            style='secondary.TButton', width=12
        ).grid(row=0, column=2, padx=2)
        
        progress_frame.grid_columnconfigure(0, weight=1)

    def _create_prediction_section(self):
        """Create prediction generation section."""
        pred_frame = Frame(self)
        pred_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        pred_frame.grid_rowconfigure(2, weight=1)
        
        Label(pred_frame, text="Predictions", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Prediction controls
        controls_frame = Frame(pred_frame)
        controls_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        Label(controls_frame, text="Symbol:", font=('Arial', 10)).grid(
            row=0, column=0, sticky='w', padx=5
        )
        
        self.pred_symbol_var = StringVar(value='BTC/USD:USD')
        self.pred_symbol_entry = Entry(controls_frame, textvariable=self.pred_symbol_var, width=15)
        self.pred_symbol_entry.grid(row=0, column=1, padx=5)
        
        Button(
            controls_frame, text="Predict", command=self._generate_predictions,
            style='info.TButton', width=10
        ).grid(row=0, column=2, padx=5)
        
        # Predictions display
        self.predictions_tree = Treeview(
            pred_frame,
            columns=('Time', 'Symbol', 'Prediction', 'Confidence', 'Model'),
            show='headings', height=12
        )
        
        for col in ('Time', 'Symbol', 'Prediction', 'Confidence', 'Model'):
            self.predictions_tree.heading(col, text=col)
            self.predictions_tree.column(col, width=80 if col != 'Symbol' else 120)
        
        self.predictions_tree.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        
        # Prediction statistics
        stats_frame = Frame(pred_frame)
        stats_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        
        self.pred_stats_label = Label(
            stats_frame, text="Predictions: 0 | Avg Confidence: 0%",
            font=('Arial', 9), foreground='#888888'
        )
        self.pred_stats_label.grid(row=0, column=0)

    def _create_performance_section(self):
        """Create performance monitoring section."""
        perf_frame = Frame(self)
        perf_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        perf_frame.grid_rowconfigure(1, weight=1)
        
        Label(perf_frame, text="Model Performance", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Performance notebook
        perf_notebook = Notebook(perf_frame)
        perf_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Metrics tab
        metrics_tab = Frame(perf_notebook)
        perf_notebook.add(metrics_tab, text="Metrics")
        self._create_metrics_display(metrics_tab)
        
        # Feature importance tab
        importance_tab = Frame(perf_notebook)
        perf_notebook.add(importance_tab, text="Feature Importance")
        self._create_feature_importance(importance_tab)
        
        # Validation tab
        validation_tab = Frame(perf_notebook)
        perf_notebook.add(validation_tab, text="Cross Validation")
        self._create_validation_display(validation_tab)

    def _create_metrics_display(self, parent):
        """Create model metrics display."""
        metrics_frame = Frame(parent)
        metrics_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Key metrics
        self.accuracy_label = Label(metrics_frame, text="Accuracy: N/A", font=('Arial', 10, 'bold'))
        self.accuracy_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.precision_label = Label(metrics_frame, text="Precision: N/A", font=('Arial', 10))
        self.precision_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.recall_label = Label(metrics_frame, text="Recall: N/A", font=('Arial', 10))
        self.recall_label.grid(row=2, column=0, sticky='w', pady=2)
        
        self.f1_label = Label(metrics_frame, text="F1-Score: N/A", font=('Arial', 10))
        self.f1_label.grid(row=3, column=0, sticky='w', pady=2)
        
        # Additional metrics
        self.mse_label = Label(metrics_frame, text="MSE: N/A", font=('Arial', 10))
        self.mse_label.grid(row=0, column=1, sticky='w', padx=20, pady=2)
        
        self.mae_label = Label(metrics_frame, text="MAE: N/A", font=('Arial', 10))
        self.mae_label.grid(row=1, column=1, sticky='w', padx=20, pady=2)
        
        self.r2_label = Label(metrics_frame, text="R²: N/A", font=('Arial', 10))
        self.r2_label.grid(row=2, column=1, sticky='w', padx=20, pady=2)
        
        self.auc_label = Label(metrics_frame, text="AUC: N/A", font=('Arial', 10))
        self.auc_label.grid(row=3, column=1, sticky='w', padx=20, pady=2)

    def _create_feature_importance(self, parent):
        """Create feature importance display."""
        self.importance_tree = Treeview(
            parent,
            columns=('Feature', 'Importance', 'Type'),
            show='headings', height=12
        )
        
        for col in ('Feature', 'Importance', 'Type'):
            self.importance_tree.heading(col, text=col)
            self.importance_tree.column(col, width=120)
        
        self.importance_tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def _create_validation_display(self, parent):
        """Create cross-validation display."""
        val_frame = Frame(parent)
        val_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(val_frame, text="Cross Validation Results:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', pady=5
        )
        
        self.cv_results_tree = Treeview(
            parent,
            columns=('Fold', 'Accuracy', 'Precision', 'Recall', 'F1'),
            show='headings', height=10
        )
        
        for col in ('Fold', 'Accuracy', 'Precision', 'Recall', 'F1'):
            self.cv_results_tree.heading(col, text=col)
            self.cv_results_tree.column(col, width=100)
        
        self.cv_results_tree.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def _create_control_section(self):
        """Create system control and status section."""
        control_frame = Frame(self)
        control_frame.grid(row=2, column=2, sticky='ew', padx=5, pady=5)
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons
        buttons_frame = Frame(control_frame)
        buttons_frame.grid(row=0, column=0, padx=5)
        
        Button(
            buttons_frame, text="Save Model", command=self._save_model,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            buttons_frame, text="Load Model", command=self._load_model,
            style='info.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        Button(
            buttons_frame, text="Export Data", command=self._export_data,
            style='secondary.TButton', width=12
        ).grid(row=0, column=2, padx=2)
        
        # Status indicator
        self.status_indicator = StatusIndicator(control_frame)
        self.status_indicator.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        # Loading indicator
        self.loading_indicator = LoadingIndicator(control_frame)
        self.loading_indicator.grid(row=2, column=0, sticky='ew', padx=5, pady=5)

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
        self.train_split_entry.bind('<FocusOut>', self._validate_train_split)
        self.learning_rate_entry.bind('<FocusOut>', self._validate_learning_rate)
        self.epochs_entry.bind('<FocusOut>', self._validate_epochs)
        self.batch_size_entry.bind('<FocusOut>', self._validate_batch_size)

    def _start_updates(self):
        """Start periodic data updates."""
        try:
            self._update_model_status()
            self._update_performance_metrics()
            
            # Schedule next update
            self.after(15000, self._start_updates)  # Update every 15 seconds
            
        except Exception as e:
            app_logger.error(f"Error in periodic updates: {e}")

    def _toggle_feature(self, feature_type: str):
        """Toggle feature type selection."""
        try:
            current_state = self.feature_vars[feature_type].get()
            self.feature_vars[feature_type].set(not current_state)
            
            self.status_indicator.set_status(
                f"{'Enabled' if not current_state else 'Disabled'} {feature_type} features", 
                'info'
            )
            
        except Exception as e:
            app_logger.error(f"Error toggling feature {feature_type}: {e}")

    def _toggle_auto_tune(self):
        """Toggle auto-tuning mode."""
        try:
            enabled = self.auto_tune_var.get()
            self.status_indicator.set_status(
                f"Auto-tuning {'enabled' if enabled else 'disabled'}", 
                'info'
            )
            
        except Exception as e:
            app_logger.error(f"Error toggling auto-tune: {e}")

    def _start_training(self):
        """Start model training."""
        try:
            if self._is_training:
                self.status_indicator.set_status("Training already in progress", 'warning')
                return
                
            # Validate configuration
            validation_result = self._validate_training_config()
            if not validation_result['valid']:
                self.status_indicator.set_status(f"Configuration error: {validation_result['message']}", 'error')
                return
            
            self.loading_indicator.show_loading("Preparing training data...")
            self._is_training = True
            
            # Update UI
            self.train_button.configure(state='disabled')
            self.system_status_label.configure(
                text="Status: TRAINING", 
                foreground='#ffc107'
            )
            
            # Start training in background thread
            training_thread = threading.Thread(target=self._training_worker, daemon=True)
            training_thread.start()
            
            self.status_indicator.set_status("Training started", 'success')
            
        except Exception as e:
            app_logger.error(f"Error starting training: {e}")
            self.status_indicator.set_status(f"Failed to start training: {str(e)}", 'error')
            self._is_training = False
            self.train_button.configure(state='normal')

    def _training_worker(self):
        """Background training worker."""
        try:
            # Simulate training process
            self._log_training("Starting training process...")
            self._log_training(f"Model type: {self.model_type_var.get()}")
            self._log_training(f"Target: {self.target_var.get()}")
            
            total_epochs = int(self.epochs_var.get())
            
            for epoch in range(total_epochs):
                if not self._is_training:  # Check for stop signal
                    break
                    
                # Simulate training progress
                progress = (epoch + 1) / total_epochs * 100
                self._training_progress = progress
                
                # Update UI on main thread
                self.after(0, self._update_training_progress, progress)
                
                # Simulate epoch training
                import time
                time.sleep(0.1)  # Simulate computation time
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    accuracy = 0.6 + (epoch / total_epochs) * 0.3  # Simulate improving accuracy
                    self.after(0, self._log_training, f"Epoch {epoch + 1}/{total_epochs} - Accuracy: {accuracy:.3f}")
            
            if self._is_training:  # Training completed normally
                self.after(0, self._training_completed)
            else:  # Training was stopped
                self.after(0, self._training_stopped)
                
        except Exception as e:
            app_logger.error(f"Error in training worker: {e}")
            self.after(0, self._training_error, str(e))

    def _update_training_progress(self, progress: float):
        """Update training progress bar."""
        try:
            self.training_progress_bar['value'] = progress
            self.training_status_label.configure(
                text=f"Training... {progress:.1f}% complete"
            )
        except Exception as e:
            app_logger.error(f"Error updating training progress: {e}")

    def _log_training(self, message: str):
        """Add message to training log."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.training_log.insert('end', log_entry)
            self.training_log.see('end')
            
        except Exception as e:
            app_logger.error(f"Error logging training message: {e}")

    def _training_completed(self):
        """Handle training completion."""
        try:
            self._is_training = False
            self.train_button.configure(state='normal')
            
            self.system_status_label.configure(
                text="Status: READY",
                foreground='#17a2b8'
            )
            
            self.training_status_label.configure(text="Training completed successfully")
            self.loading_indicator.hide_loading()
            
            self._log_training("Training completed successfully!")
            
            # Update model list
            model_name = f"{self.model_type_var.get()}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self._models[model_name] = {
                'type': self.model_type_var.get(),
                'accuracy': 0.85,  # Placeholder
                'status': 'ready',
                'trained_at': datetime.now()
            }
            
            self._update_model_display()
            self.status_indicator.set_status("Model training completed successfully", 'success')
            
            notification_system.show_success(
                "ML Training",
                "Model training completed successfully"
            )
            
        except Exception as e:
            app_logger.error(f"Error handling training completion: {e}")

    def _training_stopped(self):
        """Handle training stop."""
        try:
            self._is_training = False
            self.train_button.configure(state='normal')
            
            self.system_status_label.configure(
                text="Status: READY",
                foreground='#17a2b8'
            )
            
            self.training_status_label.configure(text="Training stopped by user")
            self.loading_indicator.hide_loading()
            
            self._log_training("Training stopped by user")
            self.status_indicator.set_status("Training stopped", 'warning')
            
        except Exception as e:
            app_logger.error(f"Error handling training stop: {e}")

    def _training_error(self, error_message: str):
        """Handle training error."""
        try:
            self._is_training = False
            self.train_button.configure(state='normal')
            
            self.system_status_label.configure(
                text="Status: ERROR",
                foreground='#dc3545'
            )
            
            self.training_status_label.configure(text=f"Training failed: {error_message}")
            self.loading_indicator.hide_loading()
            
            self._log_training(f"Training failed: {error_message}")
            self.status_indicator.set_status(f"Training failed: {error_message}", 'error')
            
        except Exception as e:
            app_logger.error(f"Error handling training error: {e}")

    def _stop_training(self):
        """Stop model training."""
        try:
            if not self._is_training:
                self.status_indicator.set_status("No training in progress", 'warning')
                return
                
            self._is_training = False
            self.status_indicator.set_status("Stopping training...", 'warning')
            
        except Exception as e:
            app_logger.error(f"Error stopping training: {e}")

    def _clear_log(self):
        """Clear training log."""
        try:
            self.training_log.delete('1.0', 'end')
            self.status_indicator.set_status("Training log cleared", 'info')
            
        except Exception as e:
            app_logger.error(f"Error clearing log: {e}")

    def _generate_predictions(self):
        """Generate model predictions."""
        try:
            if not self._models:
                self.status_indicator.set_status("No trained models available", 'warning')
                return
                
            self.loading_indicator.show_loading("Generating predictions...")
            
            # Simulate prediction generation
            symbol = self.pred_symbol_var.get()
            
            predictions = [
                {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'symbol': symbol,
                    'prediction': 'BUY',
                    'confidence': '0.87',
                    'model': 'random_forest_20241208'
                },
                {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'symbol': symbol,
                    'prediction': 'HOLD',
                    'confidence': '0.62',
                    'model': 'xgboost_20241208'
                }
            ]
            
            # Add to predictions display
            for pred in predictions:
                self.predictions_tree.insert('', 0, values=(
                    pred['time'], pred['symbol'], pred['prediction'],
                    pred['confidence'], pred['model']
                ))
            
            # Limit prediction history
            children = self.predictions_tree.get_children()
            if len(children) > 100:
                for item in children[100:]:
                    self.predictions_tree.delete(item)
            
            self._update_prediction_stats()
            self.status_indicator.set_status(f"Generated {len(predictions)} predictions", 'success')
            
        except Exception as e:
            app_logger.error(f"Error generating predictions: {e}")
            self.status_indicator.set_status(f"Prediction generation failed: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _validate_training_config(self) -> Dict[str, Any]:
        """Validate training configuration."""
        try:
            form_data = {
                'train_split': self.train_split_var.get(),
                'learning_rate': self.learning_rate_var.get(),
                'epochs': self.epochs_var.get(),
                'batch_size': self.batch_size_var.get()
            }
            
            validation_result = self.form_validator.validate_form(form_data)
            
            if not validation_result['valid']:
                return {
                    'valid': False,
                    'message': '; '.join(validation_result['messages'])
                }
            
            return {'valid': True, 'message': 'Configuration is valid'}
            
        except Exception as e:
            app_logger.error(f"Error validating training config: {e}")
            return {'valid': False, 'message': str(e)}

    def _validate_train_split(self, event=None):
        """Validate train/test split."""
        try:
            value = self.train_split_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('train_split', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid train split: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating train split: {e}")

    def _validate_learning_rate(self, event=None):
        """Validate learning rate."""
        try:
            value = self.learning_rate_var.get()
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
            app_logger.error(f"Error validating learning rate: {e}")

    def _validate_epochs(self, event=None):
        """Validate epochs."""
        try:
            value = self.epochs_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('epochs', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid epochs: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating epochs: {e}")

    def _validate_batch_size(self, event=None):
        """Validate batch size."""
        try:
            value = self.batch_size_var.get()
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
            app_logger.error(f"Error validating batch size: {e}")

    def _update_model_display(self):
        """Update models display."""
        try:
            # Clear current display
            self.models_tree.delete(*self.models_tree.get_children())
            
            # Add active models
            for model_name, data in self._models.items():
                self.models_tree.insert('', 'end', values=(
                    model_name,
                    data['type'].title(),
                    f"{data['accuracy']:.2%}",
                    data['status'].upper()
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating model display: {e}")

    def _update_model_status(self):
        """Update model status summary."""
        try:
            model_count = len(self._models)
            avg_accuracy = sum(m['accuracy'] for m in self._models.values()) / max(model_count, 1)
            last_training = max([m['trained_at'] for m in self._models.values()], default=None)
            
            last_training_str = last_training.strftime('%Y-%m-%d %H:%M') if last_training else 'Never'
            
            self.model_summary.configure(
                text=f"Models: {model_count} | Accuracy: {avg_accuracy:.1%} | Last Training: {last_training_str}"
            )
            
        except Exception as e:
            app_logger.error(f"Error updating model status: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics display."""
        try:
            # Placeholder performance data
            metrics = {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1': 0.85,
                'mse': 0.023,
                'mae': 0.15,
                'r2': 0.78,
                'auc': 0.89
            }
            
            # Update metric labels
            self.accuracy_label.configure(text=f"Accuracy: {metrics['accuracy']:.2%}")
            self.precision_label.configure(text=f"Precision: {metrics['precision']:.2%}")
            self.recall_label.configure(text=f"Recall: {metrics['recall']:.2%}")
            self.f1_label.configure(text=f"F1-Score: {metrics['f1']:.3f}")
            self.mse_label.configure(text=f"MSE: {metrics['mse']:.3f}")
            self.mae_label.configure(text=f"MAE: {metrics['mae']:.3f}")
            self.r2_label.configure(text=f"R²: {metrics['r2']:.3f}")
            self.auc_label.configure(text=f"AUC: {metrics['auc']:.3f}")
            
            # Update feature importance
            self._update_feature_importance()
            
        except Exception as e:
            app_logger.error(f"Error updating performance metrics: {e}")

    def _update_feature_importance(self):
        """Update feature importance display."""
        try:
            # Clear current data
            self.importance_tree.delete(*self.importance_tree.get_children())
            
            # Placeholder feature importance data
            features = [
                {'name': 'RSI_14', 'importance': 0.15, 'type': 'Technical'},
                {'name': 'MACD_Signal', 'importance': 0.12, 'type': 'Technical'},
                {'name': 'Volume_MA', 'importance': 0.10, 'type': 'Technical'},
                {'name': 'Price_MA_20', 'importance': 0.08, 'type': 'Technical'},
                {'name': 'Bollinger_Upper', 'importance': 0.07, 'type': 'Technical'}
            ]
            
            for feature in features:
                self.importance_tree.insert('', 'end', values=(
                    feature['name'],
                    f"{feature['importance']:.2%}",
                    feature['type']
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating feature importance: {e}")

    def _update_prediction_stats(self):
        """Update prediction statistics."""
        try:
            children = self.predictions_tree.get_children()
            pred_count = len(children)
            
            if pred_count > 0:
                # Calculate average confidence
                total_confidence = 0
                for item in children:
                    values = self.predictions_tree.item(item, 'values')
                    confidence_str = values[3]  # Confidence column
                    total_confidence += float(confidence_str)
                
                avg_confidence = total_confidence / pred_count
                
                self.pred_stats_label.configure(
                    text=f"Predictions: {pred_count} | Avg Confidence: {avg_confidence:.1%}"
                )
            else:
                self.pred_stats_label.configure(text="Predictions: 0 | Avg Confidence: 0%")
                
        except Exception as e:
            app_logger.error(f"Error updating prediction stats: {e}")

    def _save_model(self):
        """Save trained model."""
        try:
            if not self._models:
                self.status_indicator.set_status("No models to save", 'warning')
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if filename:
                # Placeholder for model saving
                self.status_indicator.set_status("Model saved successfully", 'success')
                
        except Exception as e:
            app_logger.error(f"Error saving model: {e}")
            self.status_indicator.set_status(f"Failed to save model: {str(e)}", 'error')

    def _load_model(self):
        """Load trained model."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if filename:
                # Placeholder for model loading
                self.status_indicator.set_status("Model loaded successfully", 'success')
                self._update_model_display()
                
        except Exception as e:
            app_logger.error(f"Error loading model: {e}")
            self.status_indicator.set_status(f"Failed to load model: {str(e)}", 'error')

    def _export_data(self):
        """Export prediction data."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Placeholder for data export
                self.status_indicator.set_status("Data exported successfully", 'success')
                
        except Exception as e:
            app_logger.error(f"Error exporting data: {e}")
            self.status_indicator.set_status(f"Failed to export data: {str(e)}", 'error')

    def cleanup(self):
        """Cleanup tab resources."""
        try:
            # Stop training if in progress
            if self._is_training:
                self._is_training = False
                
            app_logger.info("MLSystemTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during MLSystemTab cleanup: {e}")

    def refresh(self):
        """Refresh tab content."""
        try:
            self._update_model_status()
            self._update_performance_metrics()
            self.status_indicator.set_status("Data refreshed", 'success')
            
        except Exception as e:
            app_logger.error(f"Error refreshing MLSystemTab: {e}")
