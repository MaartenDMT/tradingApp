"""
Enhanced Security Module

This module provides enhanced security features for the trading application,
including improved credential handling, input validation, and security monitoring.
"""

import os
import base64
import hashlib
import secrets
import re
from typing import Dict, List, Optional, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field

from util.config_manager import get_config

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event log entry."""
    event_type: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

class SecurityManager:
    """Enhanced security manager with comprehensive security features."""
    
    def __init__(self):
        self.config = get_config()
        self.security_events: List[SecurityEvent] = []
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self._lock = None  # For thread safety if needed
        
        # Initialize encryption
        self._initialize_encryption()
        
    def _initialize_encryption(self):
        """Initialize encryption keys and settings."""
        try:
            # Get encryption key from environment or config
            encryption_key = os.environ.get('ENCRYPTION_KEY')
            if not encryption_key:
                # Generate a new key if none exists
                encryption_key = Fernet.generate_key().decode()
                logger.warning("Generated new encryption key. Store this securely!")
            
            self.fernet = Fernet(encryption_key.encode())
            logger.info("Security manager encryption initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def encrypt_credential(self, data: str, password: Optional[str] = None) -> str:
        """Encrypt sensitive credential data."""
        try:
            if password:
                # Use password-based encryption
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                f = Fernet(key)
                encrypted_data = f.encrypt(data.encode())
                # Combine salt and encrypted data
                result = base64.urlsafe_b64encode(salt + encrypted_data).decode()
            else:
                # Use default encryption
                encrypted_data = self.fernet.encrypt(data.encode())
                result = encrypted_data.decode()
            
            self._log_security_event(
                event_type="credential_encrypted",
                severity="low",
                description="Credential encrypted successfully"
            )
            
            return result
        except Exception as e:
            self._log_security_event(
                event_type="encryption_error",
                severity="high",
                description=f"Failed to encrypt credential: {e}"
            )
            raise
    
    def decrypt_credential(self, encrypted_data: str, password: Optional[str] = None) -> str:
        """Decrypt sensitive credential data."""
        try:
            if password:
                # Use password-based decryption
                data = base64.urlsafe_b64decode(encrypted_data.encode())
                salt = data[:16]
                encrypted = data[16:]
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                f = Fernet(key)
                decrypted_data = f.decrypt(encrypted)
            else:
                # Use default decryption
                decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            
            result = decrypted_data.decode()
            
            self._log_security_event(
                event_type="credential_decrypted",
                severity="low",
                description="Credential decrypted successfully"
            )
            
            return result
        except Exception as e:
            self._log_security_event(
                event_type="decryption_error",
                severity="high",
                description=f"Failed to decrypt credential: {e}"
            )
            raise
    
    def hash_password(self, password: str, method: str = "scrypt") -> str:
        """Hash a password using a secure hashing algorithm."""
        try:
            if method == "scrypt":
                salt = os.urandom(16)
                kdf = Scrypt(
                    salt=salt,
                    length=32,
                    n=2**14,
                    r=8,
                    p=1,
                )
                hashed = kdf.derive(password.encode())
                # Combine salt and hash
                result = base64.urlsafe_b64encode(salt + hashed).decode()
            elif method == "pbkdf2":
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                hashed = kdf.derive(password.encode())
                # Combine salt and hash
                result = base64.urlsafe_b64encode(salt + hashed).decode()
            else:
                raise ValueError(f"Unsupported hashing method: {method}")
            
            self._log_security_event(
                event_type="password_hashed",
                severity="low",
                description="Password hashed successfully"
            )
            
            return result
        except Exception as e:
            self._log_security_event(
                event_type="hashing_error",
                severity="high",
                description=f"Failed to hash password: {e}"
            )
            raise
    
    def verify_password(self, password: str, hashed: str, method: str = "scrypt") -> bool:
        """Verify a password against its hash."""
        try:
            data = base64.urlsafe_b64decode(hashed.encode())
            salt = data[:16]
            hash_data = data[16:]
            
            if method == "scrypt":
                kdf = Scrypt(
                    salt=salt,
                    length=32,
                    n=2**14,
                    r=8,
                    p=1,
                )
                kdf.verify(password.encode(), hash_data)
            elif method == "pbkdf2":
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                kdf.verify(password.encode(), hash_data)
            else:
                raise ValueError(f"Unsupported hashing method: {method}")
            
            self._log_security_event(
                event_type="password_verified",
                severity="low",
                description="Password verified successfully"
            )
            
            return True
        except Exception as e:
            self._log_security_event(
                event_type="password_verification_failed",
                severity="medium",
                description=f"Password verification failed: {e}"
            )
            return False
    
    def mask_sensitive_data(self, data: str, show_chars: int = 4) -> str:
        """Mask sensitive data for logging and display."""
        if not data:
            return ""
        
        if len(data) <= show_chars:
            return "*" * len(data)
        
        # Show first and last few characters, mask the rest
        masked = data[:show_chars] + "*" * (len(data) - show_chars * 2) + data[-show_chars:]
        return masked
    
    def validate_input(self, data: Any, validation_type: str) -> bool:
        """Validate user input based on type."""
        try:
            if validation_type == "username":
                # Username validation: alphanumeric, underscore, hyphen, 3-20 chars
                if not isinstance(data, str):
                    return False
                return bool(re.match(r'^[a-zA-Z0-9_-]{3,20}$', data))
            
            elif validation_type == "password":
                # Password validation: at least 8 chars, contains letters and numbers
                if not isinstance(data, str):
                    return False
                if len(data) < 8:
                    return False
                if not re.search(r'[A-Za-z]', data):
                    return False
                if not re.search(r'[0-9]', data):
                    return False
                return True
            
            elif validation_type == "email":
                # Email validation
                if not isinstance(data, str):
                    return False
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                return bool(re.match(email_pattern, data))
            
            elif validation_type == "api_key":
                # API key validation: alphanumeric, 32-128 chars
                if not isinstance(data, str):
                    return False
                return bool(re.match(r'^[a-zA-Z0-9]{32,128}$', data))
            
            elif validation_type == "symbol":
                # Trading symbol validation: uppercase letters and slashes
                if not isinstance(data, str):
                    return False
                return bool(re.match(r'^[A-Z0-9/]{3,20}$', data))
            
            else:
                logger.warning(f"Unknown validation type: {validation_type}")
                return True  # Unknown type, allow by default
                
        except Exception as e:
            self._log_security_event(
                event_type="input_validation_error",
                severity="medium",
                description=f"Input validation failed for {validation_type}: {e}"
            )
            return False
    
    def rate_limit_check(self, identifier: str, max_attempts: int = 5, 
                        window_minutes: int = 15) -> bool:
        """Check if an identifier has exceeded rate limits."""
        try:
            now = datetime.now()
            
            # Remove old attempts
            if identifier in self.failed_login_attempts:
                self.failed_login_attempts[identifier] = [
                    attempt for attempt in self.failed_login_attempts[identifier]
                    if attempt > now - timedelta(minutes=window_minutes)
                ]
            
            # Check if blocked
            if identifier in self.blocked_ips:
                if now < self.blocked_ips[identifier]:
                    return False  # Still blocked
                else:
                    # Unblock
                    del self.blocked_ips[identifier]
            
            # Check attempt count
            attempts = len(self.failed_login_attempts.get(identifier, []))
            if attempts >= max_attempts:
                # Block for 1 hour
                self.blocked_ips[identifier] = now + timedelta(hours=1)
                self._log_security_event(
                    event_type="rate_limit_exceeded",
                    severity="high",
                    description=f"Rate limit exceeded for {identifier}",
                    additional_data={"attempts": attempts}
                )
                return False
            
            return True
            
        except Exception as e:
            self._log_security_event(
                event_type="rate_limit_error",
                severity="medium",
                description=f"Rate limit check failed: {e}"
            )
            return True  # Allow on error
    
    def record_failed_attempt(self, identifier: str):
        """Record a failed login attempt."""
        try:
            now = datetime.now()
            if identifier not in self.failed_login_attempts:
                self.failed_login_attempts[identifier] = []
            self.failed_login_attempts[identifier].append(now)
            
            self._log_security_event(
                event_type="failed_login_attempt",
                severity="medium",
                description=f"Failed login attempt for {identifier}",
                additional_data={"attempt_count": len(self.failed_login_attempts[identifier])}
            )
        except Exception as e:
            self._log_security_event(
                event_type="security_event_error",
                severity="low",
                description=f"Failed to record failed attempt: {e}"
            )
    
    def reset_failed_attempts(self, identifier: str):
        """Reset failed login attempts for an identifier."""
        try:
            if identifier in self.failed_login_attempts:
                del self.failed_login_attempts[identifier]
            
            if identifier in self.blocked_ips:
                del self.blocked_ips[identifier]
                
            self._log_security_event(
                event_type="attempts_reset",
                severity="low",
                description=f"Failed attempts reset for {identifier}"
            )
        except Exception as e:
            self._log_security_event(
                event_type="security_event_error",
                severity="low",
                description=f"Failed to reset attempts: {e}"
            )
    
    def _log_security_event(self, event_type: str, severity: str, description: str,
                           user_id: Optional[str] = None, ip_address: Optional[str] = None,
                           additional_data: Optional[Dict[str, Any]] = None):
        """Log a security event."""
        try:
            event = SecurityEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                severity=severity,
                description=description,
                user_id=user_id,
                ip_address=ip_address,
                additional_data=additional_data or {}
            )
            
            self.security_events.append(event)
            
            # Log to file or external system if configured
            if severity in ["high", "critical"]:
                logger.warning(f"Security Event - {event_type}: {description}")
            elif severity == "medium":
                logger.info(f"Security Event - {event_type}: {description}")
                
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def get_security_events(self, severity: Optional[str] = None, 
                           limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events."""
        try:
            events = self.security_events
            
            if severity:
                events = [event for event in events if event.severity == severity]
            
            # Return most recent events
            return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Failed to get security events: {e}")
            return []
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        try:
            token = secrets.token_urlsafe(length)
            self._log_security_event(
                event_type="token_generated",
                severity="low",
                description="Secure token generated"
            )
            return token
        except Exception as e:
            self._log_security_event(
                event_type="token_generation_error",
                severity="high",
                description=f"Failed to generate secure token: {e}"
            )
            raise
    
    def validate_api_key(self, api_key: str, required_permissions: Optional[List[str]] = None) -> bool:
        """Validate an API key and check permissions."""
        try:
            # Basic validation
            if not self.validate_input(api_key, "api_key"):
                self._log_security_event(
                    event_type="api_key_invalid",
                    severity="medium",
                    description="Invalid API key format"
                )
                return False
            
            # Check if key exists and is active (this would typically query a database)
            # For demo purposes, we'll just check if it's a valid format
            # In a real implementation, you'd check against stored keys
            
            self._log_security_event(
                event_type="api_key_validated",
                severity="low",
                description="API key validated successfully"
            )
            
            return True
        except Exception as e:
            self._log_security_event(
                event_type="api_key_validation_error",
                severity="high",
                description=f"API key validation failed: {e}"
            )
            return False

# Global security manager instance
security_manager = SecurityManager()

# Decorator for securing functions
def secure_function(required_permissions: Optional[List[str]] = None):
    """Decorator to add security checks to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if security manager is available
            if not hasattr(security_manager, 'validate_api_key'):
                logger.warning("Security manager not available, skipping security checks")
                return func(*args, **kwargs)
            
            # Check for API key in kwargs or args
            api_key = kwargs.get('api_key') or next((arg for arg in args if isinstance(arg, str) and len(arg) > 32), None)
            
            if api_key:
                if not security_manager.validate_api_key(api_key, required_permissions):
                    raise PermissionError("Invalid or insufficient API key permissions")
            
            # Execute function
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Input validation decorator
def validate_input(validation_rules: Dict[str, str]):
    """Decorator to validate function arguments."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature to map arguments
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument according to rules
            for arg_name, validation_type in validation_rules.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    if not security_manager.validate_input(value, validation_type):
                        raise ValueError(f"Invalid value for {arg_name}: {value}")
            
            # Execute function
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing function
def run_security_demo():
    """Run a demonstration of the enhanced security features."""
    print("Enhanced Security Module")
    print("=" * 30)
    
    # Create security manager
    manager = security_manager
    
    # Test credential encryption/decryption
    print("Testing credential encryption...")
    original_credential = "my_secret_api_key_12345"
    encrypted = manager.encrypt_credential(original_credential)
    decrypted = manager.decrypt_credential(encrypted)
    print(f"  Original: {original_credential}")
    print(f"  Encrypted: {manager.mask_sensitive_data(encrypted)}")
    print(f"  Decrypted: {decrypted}")
    print(f"  Match: {original_credential == decrypted}")
    
    # Test password hashing
    print("\nTesting password hashing...")
    password = "MySecurePassword123!"
    hashed = manager.hash_password(password)
    is_valid = manager.verify_password(password, hashed)
    print(f"  Password: {manager.mask_sensitive_data(password)}")
    print(f"  Hashed: {manager.mask_sensitive_data(hashed)}")
    print(f"  Valid: {is_valid}")
    
    # Test input validation
    print("\nTesting input validation...")
    test_inputs = [
        ("valid_username", "username"),
        ("invalid@username", "username"),
        ("short", "username"),
        ("valid@email.com", "email"),
        ("invalid-email", "email"),
        ("valid_symbol", "symbol"),
    ]
    
    for test_input, validation_type in test_inputs:
        is_valid = manager.validate_input(test_input, validation_type)
        print(f"  {validation_type}: {test_input} -> {'Valid' if is_valid else 'Invalid'}")
    
    # Test rate limiting
    print("\nTesting rate limiting...")
    identifier = "test_user"
    for i in range(7):
        allowed = manager.rate_limit_check(identifier, max_attempts=5)
        if not allowed:
            print(f"  Attempt {i+1}: Blocked")
            break
        else:
            print(f"  Attempt {i+1}: Allowed")
            manager.record_failed_attempt(identifier)
    
    # Test secure token generation
    print("\nTesting secure token generation...")
    token = manager.generate_secure_token()
    print(f"  Generated token: {manager.mask_sensitive_data(token, 8)}")
    
    # Get security events
    events = manager.get_security_events()
    print(f"\nSecurity events logged: {len(events)}")
    
    print("\nSecurity module demonstration completed!")

if __name__ == "__main__":
    run_security_demo()