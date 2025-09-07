"""
Secure credentials handling utilities for the Trading Application.

This module provides functions for securely handling API keys and other sensitive
credentials, including encryption/decryption capabilities.
"""

import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_key_from_password(password: str, salt: bytes) -> bytes:
    """
    Generate a key from a password using PBKDF2.
    
    Args:
        password (str): The password to derive the key from
        salt (bytes): The salt to use for key derivation
        
    Returns:
        bytes: The derived key
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_data(data: str, password: str) -> str:
    """
    Encrypt data using a password.
    
    Args:
        data (str): The data to encrypt
        password (str): The password to use for encryption
        
    Returns:
        str: The encrypted data as a base64-encoded string
    """
    # Generate a random salt
    salt = os.urandom(16)
    
    # Generate key from password
    key = generate_key_from_password(password, salt)
    
    # Create Fernet instance
    f = Fernet(key)
    
    # Encrypt the data
    encrypted_data = f.encrypt(data.encode())
    
    # Combine salt and encrypted data
    result = base64.urlsafe_b64encode(salt + encrypted_data).decode()
    
    return result

def decrypt_data(encrypted_data: str, password: str) -> str:
    """
    Decrypt data using a password.
    
    Args:
        encrypted_data (str): The encrypted data as a base64-encoded string
        password (str): The password to use for decryption
        
    Returns:
        str: The decrypted data
    """
    try:
        # Decode the base64-encoded data
        data = base64.urlsafe_b64decode(encrypted_data.encode())
        
        # Extract salt and encrypted data
        salt = data[:16]
        encrypted = data[16:]
        
        # Generate key from password
        key = generate_key_from_password(password, salt)
        
        # Create Fernet instance
        f = Fernet(key)
        
        # Decrypt the data
        decrypted_data = f.decrypt(encrypted)
        
        return decrypted_data.decode()
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")

def mask_sensitive_data(data: str, show_chars: int = 4) -> str:
    """
    Mask sensitive data for logging purposes.
    
    Args:
        data (str): The sensitive data to mask
        show_chars (int): Number of characters to show at the beginning and end
        
    Returns:
        str: The masked data
    """
    if not data:
        return "None"
    
    if len(data) <= show_chars * 2:
        return "*" * len(data)
    
    return data[:show_chars] + "*" * (len(data) - show_chars * 2) + data[-show_chars:]

def get_secure_env_var(var_name: str, default: str = None) -> str:
    """
    Get an environment variable value, masking it if it's sensitive.
    
    Args:
        var_name (str): The name of the environment variable
        default (str): The default value if the variable is not set
        
    Returns:
        str: The environment variable value
    """
    value = os.environ.get(var_name, default)
    if value and ('KEY' in var_name or 'SECRET' in var_name):
        return mask_sensitive_data(value)
    return value

def secure_store_credential(credential_name: str, credential_value: str, master_password: str) -> None:
    """
    Securely store a credential in a file using encryption.
    
    Args:
        credential_name (str): The name of the credential
        credential_value (str): The value of the credential
        master_password (str): The master password for encryption
    """
    try:
        # Encrypt the credential
        encrypted = encrypt_data(credential_value, master_password)
        
        # Store in a secure file
        credentials_dir = "data/credentials"
        if not os.path.exists(credentials_dir):
            os.makedirs(credentials_dir)
            
        file_path = os.path.join(credentials_dir, f"{credential_name}.enc")
        with open(file_path, 'w') as f:
            f.write(encrypted)
            
    except Exception as e:
        raise RuntimeError(f"Failed to store credential securely: {e}")

def secure_retrieve_credential(credential_name: str, master_password: str) -> str:
    """
    Retrieve and decrypt a securely stored credential.
    
    Args:
        credential_name (str): The name of the credential
        master_password (str): The master password for decryption
        
    Returns:
        str: The decrypted credential value
    """
    try:
        # Read the encrypted credential
        file_path = os.path.join("data/credentials", f"{credential_name}.enc")
        with open(file_path, 'r') as f:
            encrypted = f.read()
            
        # Decrypt the credential
        decrypted = decrypt_data(encrypted, master_password)
        
        return decrypted
    except FileNotFoundError:
        raise ValueError(f"Credential '{credential_name}' not found")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve credential: {e}")