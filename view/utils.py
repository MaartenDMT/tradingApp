"""
View Utilities Module

Shared utilities, components, and mixins for the view layer:
- UI theme management and styling
- Common validation patterns
- Reusable UI components
- Status indicators and notifications
- Form validation helpers
"""

import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from tkinter import messagebox
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from ttkbootstrap import (
        Button, Entry, Frame, Label, Style, Progressbar, 
        StringVar, BooleanVar, IntVar
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import (
        Button, Entry, Frame, Label, StringVar, BooleanVar, IntVar
    )
    from tkinter.ttk import Style, Progressbar
    HAS_TTKBOOTSTRAP = False

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class UIThemeManager:
    """
    Centralized theme management for the application.
    """
    
    def __init__(self, root_window):
        self.root_window = root_window
        self._style = Style()
        self._current_theme = None
        self._theme_callbacks = []
        
        # Available themes with descriptions
        self.theme_info = {
            'superhero': {'name': 'Superhero', 'type': 'dark', 'description': 'Dark blue theme'},
            'darkly': {'name': 'Darkly', 'type': 'dark', 'description': 'Dark grey theme'},
            'solar': {'name': 'Solar', 'type': 'dark', 'description': 'Dark orange theme'},
            'cyborg': {'name': 'Cyborg', 'type': 'dark', 'description': 'Dark cyan theme'},
            'flatly': {'name': 'Flatly', 'type': 'light', 'description': 'Light blue theme'},
            'cosmo': {'name': 'Cosmo', 'type': 'light', 'description': 'Light grey theme'},
            'litera': {'name': 'Litera', 'type': 'light', 'description': 'Clean light theme'},
            'minty': {'name': 'Minty', 'type': 'light', 'description': 'Light green theme'},
        }
        
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        try:
            available = self._style.theme_names()
            return [theme for theme in available if theme in self.theme_info]
        except Exception as e:
            app_logger.error(f"Error getting available themes: {e}")
            return list(self.theme_info.keys())
    
    def set_theme(self, theme_name: str) -> bool:
        """Set the application theme."""
        try:
            if theme_name not in self.get_available_themes():
                app_logger.warning(f"Theme {theme_name} not available")
                return False
                
            self._style.theme_use(theme_name)
            self._current_theme = theme_name
            
            # Apply custom styling
            self._apply_custom_styles()
            
            # Notify callbacks
            for callback in self._theme_callbacks:
                try:
                    callback(theme_name)
                except Exception as e:
                    app_logger.error(f"Theme callback error: {e}")
            
            app_logger.info(f"Theme changed to: {theme_name}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error setting theme {theme_name}: {e}")
            return False
    
    def _apply_custom_styles(self):
        """Apply custom styling modifications."""
        try:
            # Configure custom styles for status indicators
            self._style.configure(
                'Success.TLabel',
                foreground='#28a745' if self.is_light_theme() else '#5cb85c'
            )
            
            self._style.configure(
                'Error.TLabel', 
                foreground='#dc3545' if self.is_light_theme() else '#d9534f'
            )
            
            self._style.configure(
                'Warning.TLabel',
                foreground='#ffc107' if self.is_light_theme() else '#f0ad4e'
            )
            
            self._style.configure(
                'Info.TLabel',
                foreground='#17a2b8' if self.is_light_theme() else '#5bc0de'
            )
            
        except Exception as e:
            app_logger.error(f"Error applying custom styles: {e}")
    
    def is_dark_theme(self) -> bool:
        """Check if current theme is dark."""
        if not self._current_theme:
            return True  # Default assumption
        return self.theme_info.get(self._current_theme, {}).get('type') == 'dark'
    
    def is_light_theme(self) -> bool:
        """Check if current theme is light."""
        return not self.is_dark_theme()
    
    def get_current_theme(self) -> Optional[str]:
        """Get current theme name."""
        return self._current_theme
    
    def register_theme_callback(self, callback: Callable[[str], None]):
        """Register a callback for theme changes."""
        self._theme_callbacks.append(callback)


class ValidationMixin:
    """
    Mixin class providing common validation methods for UI components.
    """
    
    # Common regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,30}$')
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        return bool(self.EMAIL_PATTERN.match(email.strip()))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format."""
        return bool(self.PHONE_PATTERN.match(phone.strip().replace(' ', '').replace('-', '')))
    
    def validate_username(self, username: str) -> bool:
        """Validate username format."""
        return bool(self.USERNAME_PATTERN.match(username.strip()))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength and return details."""
        result = {
            'valid': True,
            'score': 0,
            'messages': [],
            'requirements': {
                'length': len(password) >= 8,
                'uppercase': bool(re.search(r'[A-Z]', password)),
                'lowercase': bool(re.search(r'[a-z]', password)),
                'digit': bool(re.search(r'\d', password)),
                'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
            }
        }
        
        # Calculate score
        for requirement, met in result['requirements'].items():
            if met:
                result['score'] += 1
        
        # Add requirement messages
        if not result['requirements']['length']:
            result['messages'].append('At least 8 characters')
        if not result['requirements']['uppercase']:
            result['messages'].append('At least 1 uppercase letter')
        if not result['requirements']['lowercase']:
            result['messages'].append('At least 1 lowercase letter')
        if not result['requirements']['digit']:
            result['messages'].append('At least 1 number')
        if not result['requirements']['special']:
            result['messages'].append('At least 1 special character')
        
        # Determine if valid (at least 3 requirements met)
        result['valid'] = result['score'] >= 3
        
        return result
    
    def validate_numeric_range(self, value: str, min_val: float = None, max_val: float = None) -> Dict[str, Any]:
        """Validate numeric value within range."""
        result = {'valid': True, 'value': None, 'message': ''}
        
        try:
            numeric_value = float(value.strip())
            result['value'] = numeric_value
            
            if min_val is not None and numeric_value < min_val:
                result['valid'] = False
                result['message'] = f'Value must be at least {min_val}'
            elif max_val is not None and numeric_value > max_val:
                result['valid'] = False  
                result['message'] = f'Value must be at most {max_val}'
                
        except (ValueError, TypeError):
            result['valid'] = False
            result['message'] = 'Invalid numeric value'
        
        return result
    
    def validate_required_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that required fields are not empty."""
        result = {'valid': True, 'missing_fields': [], 'messages': []}
        
        for field_name, field_value in fields.items():
            if not field_value or (isinstance(field_value, str) and not field_value.strip()):
                result['valid'] = False
                result['missing_fields'].append(field_name)
                result['messages'].append(f'{field_name.replace("_", " ").title()} is required')
        
        return result


class StatusIndicator(Frame):
    """
    Reusable status indicator component with different severity levels.
    """
    
    STATUS_STYLES = {
        'success': {'fg': '#28a745', 'prefix': '✅'},
        'error': {'fg': '#dc3545', 'prefix': '❌'},
        'warning': {'fg': '#ffc107', 'prefix': '⚠️'},
        'info': {'fg': '#17a2b8', 'prefix': 'ℹ️'},
        'loading': {'fg': '#6c757d', 'prefix': '⏳'},
    }
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self._status_label = Label(self, text="", font=('Arial', 9))
        self._status_label.grid(row=0, column=0, sticky='ew')
        
        self.grid_columnconfigure(0, weight=1)
        
        # Auto-clear timer
        self._clear_timer = None
    
    def set_status(self, message: str, level: str = 'info', auto_clear: int = 0):
        """
        Set status message with specified level.
        
        Args:
            message: Status message to display
            level: Status level (success, error, warning, info, loading)
            auto_clear: Seconds after which to auto-clear (0 = no auto-clear)
        """
        try:
            # Cancel existing timer
            if self._clear_timer:
                self.after_cancel(self._clear_timer)
                self._clear_timer = None
            
            # Get style info
            style_info = self.STATUS_STYLES.get(level, self.STATUS_STYLES['info'])
            
            # Format message
            prefix = style_info.get('prefix', '')
            formatted_message = f"{prefix} {message}" if prefix else message
            
            # Update label
            self._status_label.configure(
                text=formatted_message,
                foreground=style_info.get('fg', '#000000')
            )
            
            # Set auto-clear timer if specified
            if auto_clear > 0:
                self._clear_timer = self.after(auto_clear * 1000, self.clear_status)
            
        except Exception as e:
            app_logger.error(f"Error setting status: {e}")
    
    def clear_status(self):
        """Clear the status message."""
        try:
            self._status_label.configure(text="")
            if self._clear_timer:
                self.after_cancel(self._clear_timer)
                self._clear_timer = None
        except Exception as e:
            app_logger.error(f"Error clearing status: {e}")
    
    def get_status(self) -> str:
        """Get current status message."""
        return self._status_label.cget('text')


class LoadingIndicator(Frame):
    """
    Reusable loading indicator with progress bar and message.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Message label
        self._message_label = Label(self, text="Loading...", font=('Arial', 10))
        self._message_label.grid(row=0, column=0, pady=(0, 5))
        
        # Progress bar
        self._progress_bar = Progressbar(
            self, 
            mode='indeterminate',
            style='success.Horizontal.TProgressbar'
        )
        self._progress_bar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Initially hidden
        self.grid_remove()
    
    def show_loading(self, message: str = "Loading..."):
        """Show loading indicator with message."""
        try:
            self._message_label.configure(text=message)
            self._progress_bar.start(10)
            self.grid()
        except Exception as e:
            app_logger.error(f"Error showing loading indicator: {e}")
    
    def hide_loading(self):
        """Hide loading indicator."""
        try:
            self._progress_bar.stop()
            self.grid_remove()
        except Exception as e:
            app_logger.error(f"Error hiding loading indicator: {e}")
    
    def update_message(self, message: str):
        """Update loading message."""
        try:
            self._message_label.configure(text=message)
        except Exception as e:
            app_logger.error(f"Error updating loading message: {e}")


class NotificationSystem:
    """
    Centralized notification system for showing user messages.
    """
    
    def __init__(self):
        self._notifications = []
        self._max_notifications = 5
        self._lock = threading.Lock()
    
    def show_success(self, title: str, message: str, auto_dismiss: bool = True):
        """Show success notification."""
        self._show_notification('success', title, message, auto_dismiss)
    
    def show_error(self, title: str, message: str, auto_dismiss: bool = False):
        """Show error notification."""
        self._show_notification('error', title, message, auto_dismiss)
    
    def show_warning(self, title: str, message: str, auto_dismiss: bool = True):
        """Show warning notification."""
        self._show_notification('warning', title, message, auto_dismiss)
    
    def show_info(self, title: str, message: str, auto_dismiss: bool = True):
        """Show info notification."""
        self._show_notification('info', title, message, auto_dismiss)
    
    def _show_notification(self, level: str, title: str, message: str, auto_dismiss: bool):
        """Internal method to show notification."""
        try:
            with self._lock:
                notification = {
                    'level': level,
                    'title': title,
                    'message': message,
                    'timestamp': datetime.now(),
                    'auto_dismiss': auto_dismiss
                }
                
                self._notifications.append(notification)
                
                # Limit number of notifications
                if len(self._notifications) > self._max_notifications:
                    self._notifications.pop(0)
                
                # Show using appropriate method
                self._display_notification(notification)
                
        except Exception as e:
            app_logger.error(f"Error showing notification: {e}")
    
    def _display_notification(self, notification: Dict[str, Any]):
        """Display notification using appropriate method."""
        try:
            level = notification['level']
            title = notification['title']
            message = notification['message']
            
            # Use messagebox for now (could be enhanced with custom popup)
            if level == 'error':
                messagebox.showerror(title, message)
            elif level == 'warning':
                messagebox.showwarning(title, message)
            elif level == 'success' or level == 'info':
                messagebox.showinfo(title, message)
                
        except Exception as e:
            app_logger.error(f"Error displaying notification: {e}")
    
    def get_recent_notifications(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent notifications."""
        with self._lock:
            return self._notifications[-count:] if self._notifications else []
    
    def clear_notifications(self):
        """Clear all notifications."""
        with self._lock:
            self._notifications.clear()


class FormValidator:
    """
    Enhanced form validation with field-specific rules and error tracking.
    """
    
    def __init__(self):
        self.field_rules = {}
        self.field_errors = {}
        self.custom_validators = {}
    
    def add_field_rule(self, field_name: str, rule_type: str, **kwargs):
        """
        Add validation rule for a field.
        
        Args:
            field_name: Name of the field
            rule_type: Type of rule (required, email, numeric_range, etc.)
            **kwargs: Rule-specific parameters
        """
        if field_name not in self.field_rules:
            self.field_rules[field_name] = []
        
        rule = {'type': rule_type, 'params': kwargs}
        self.field_rules[field_name].append(rule)
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """Add custom validator function."""
        self.custom_validators[name] = validator_func
    
    def validate_field(self, field_name: str, value: Any) -> Dict[str, Any]:
        """Validate a single field."""
        result = {'valid': True, 'messages': []}
        
        if field_name not in self.field_rules:
            return result
        
        for rule in self.field_rules[field_name]:
            rule_result = self._apply_rule(rule, value)
            
            if not rule_result['valid']:
                result['valid'] = False
                result['messages'].extend(rule_result['messages'])
        
        # Update field errors
        if result['valid']:
            self.field_errors.pop(field_name, None)
        else:
            self.field_errors[field_name] = result['messages']
        
        return result
    
    def validate_form(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire form."""
        result = {'valid': True, 'field_errors': {}, 'messages': []}
        
        # Validate each field with rules
        for field_name in self.field_rules:
            field_value = form_data.get(field_name, '')
            field_result = self.validate_field(field_name, field_value)
            
            if not field_result['valid']:
                result['valid'] = False
                result['field_errors'][field_name] = field_result['messages']
                result['messages'].extend([f"{field_name}: {msg}" for msg in field_result['messages']])
        
        return result
    
    def _apply_rule(self, rule: Dict[str, Any], value: Any) -> Dict[str, Any]:
        """Apply a validation rule to a value."""
        rule_type = rule['type']
        params = rule['params']
        
        if rule_type == 'required':
            return self._validate_required(value)
        elif rule_type == 'email':
            return self._validate_email_rule(value)
        elif rule_type == 'numeric_range':
            return self._validate_numeric_range_rule(value, params)
        elif rule_type == 'min_length':
            return self._validate_min_length(value, params.get('length', 0))
        elif rule_type == 'max_length':
            return self._validate_max_length(value, params.get('length', 100))
        elif rule_type == 'custom':
            validator_name = params.get('validator')
            if validator_name in self.custom_validators:
                return self.custom_validators[validator_name](value, params)
        
        return {'valid': True, 'messages': []}
    
    def _validate_required(self, value: Any) -> Dict[str, Any]:
        """Validate required field."""
        is_valid = value is not None and str(value).strip() != ''
        return {
            'valid': is_valid,
            'messages': [] if is_valid else ['This field is required']
        }
    
    def _validate_email_rule(self, value: str) -> Dict[str, Any]:
        """Validate email format."""
        if not value:  # Allow empty if not required
            return {'valid': True, 'messages': []}
            
        is_valid = ValidationMixin.EMAIL_PATTERN.match(str(value).strip()) is not None
        return {
            'valid': is_valid,
            'messages': [] if is_valid else ['Invalid email format']
        }
    
    def _validate_numeric_range_rule(self, value: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numeric range."""
        try:
            numeric_value = float(value)
            min_val = params.get('min')
            max_val = params.get('max')
            
            messages = []
            
            if min_val is not None and numeric_value < min_val:
                messages.append(f'Value must be at least {min_val}')
            if max_val is not None and numeric_value > max_val:
                messages.append(f'Value must be at most {max_val}')
            
            return {'valid': len(messages) == 0, 'messages': messages}
            
        except (ValueError, TypeError):
            return {'valid': False, 'messages': ['Invalid numeric value']}
    
    def _validate_min_length(self, value: Any, min_length: int) -> Dict[str, Any]:
        """Validate minimum length."""
        value_str = str(value) if value is not None else ''
        is_valid = len(value_str) >= min_length
        return {
            'valid': is_valid,
            'messages': [] if is_valid else [f'Minimum length is {min_length} characters']
        }
    
    def _validate_max_length(self, value: Any, max_length: int) -> Dict[str, Any]:
        """Validate maximum length."""
        value_str = str(value) if value is not None else ''
        is_valid = len(value_str) <= max_length
        return {
            'valid': is_valid,
            'messages': [] if is_valid else [f'Maximum length is {max_length} characters']
        }
    
    def has_errors(self) -> bool:
        """Check if form has any validation errors."""
        return len(self.field_errors) > 0
    
    def get_errors(self) -> Dict[str, List[str]]:
        """Get all field errors."""
        return self.field_errors.copy()
    
    def clear_errors(self):
        """Clear all validation errors."""
        self.field_errors.clear()


# Global instances
notification_system = NotificationSystem()


# Utility functions
def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency string."""
    try:
        return f"{currency} {amount:,.2f}"
    except (ValueError, TypeError):
        return f"{currency} 0.00"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    try:
        return f"{value:.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.00%"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_status_color(status: str, theme_type: str = 'dark') -> str:
    """Get color for status based on theme."""
    colors = {
        'light': {
            'success': '#28a745',
            'error': '#dc3545', 
            'warning': '#ffc107',
            'info': '#17a2b8',
            'default': '#000000'
        },
        'dark': {
            'success': '#5cb85c',
            'error': '#d9534f',
            'warning': '#f0ad4e', 
            'info': '#5bc0de',
            'default': '#ffffff'
        }
    }
    
    theme_colors = colors.get(theme_type, colors['dark'])
    return theme_colors.get(status.lower(), theme_colors['default'])
