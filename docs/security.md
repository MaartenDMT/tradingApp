# Security Guide for Trading Application

## Overview

This document provides guidance on securing the Trading Application, protecting sensitive data like API keys, passwords, and other credentials.

## 1. Environment Variables

### Best Practices
- Store all API keys, secrets, and passwords in environment variables, not in code or config files
- Never commit environment files (like `.env`) to version control
- Use descriptive variable names that don't reveal the purpose of the keys

### Example .env File
```bash
# Phemex Testnet API Keys
API_KEY_PHE_TEST=your_phemex_testnet_api_key_here
API_SECRET_PHE_TEST=your_phemex_testnet_api_secret_here

# Database Configuration (PostgreSQL)
PGHOST=localhost
PGPORT=5432
PGDATABASE=trading_app_db
PGUSER=your_database_username
PGPASSWORD=your_database_password

# Add other exchange API keys as needed
# API_KEY_BINANCE=your_binance_api_key
# API_SECRET_BINANCE=your_binance_api_secret
```

## 2. File Permissions

### Secure Directory Structure
- Set restrictive permissions on the application directory
- Ensure the `data/credentials/` directory (if used) has restricted access
- Protect log files from unauthorized access

### Recommended Permissions (Linux/Mac)
```bash
# Set owner-only read/write permissions
chmod 700 /path/to/tradingApp
chmod 600 .env
chmod 700 data/
chmod 700 data/logs/
```

## 3. Encryption

### Credential Encryption
The application supports encrypting sensitive credentials using a master password:
- Use `secure_store_credential()` to store encrypted credentials
- Use `secure_retrieve_credential()` to retrieve decrypted credentials
- Never store the master password in plain text

### Implementation Example
```python
from util.secure_credentials import secure_store_credential, secure_retrieve_credential

# Store a credential securely
secure_store_credential('phemex_api_key', 'actual_api_key_here', 'master_password')

# Retrieve a credential
api_key = secure_retrieve_credential('phemex_api_key', 'master_password')
```

## 4. Configuration Security

### Config File Best Practices
- Avoid storing sensitive data in `config.ini`
- Use the secure config loader which masks sensitive values
- Regularly audit configuration files for sensitive data

### Validation
Run the configuration validator to check for security issues:
```python
from util.secure_config import validate_config

issues = validate_config()
for issue in issues:
    print(f"Security issue: {issue}")
```

## 5. Logging Security

### Log File Protection
- Application logs are stored in `data/logs/`
- Sensitive data is masked in logs automatically
- Regularly rotate and archive old log files
- Restrict access to log directories

### Masking Example
API keys and secrets are automatically masked in logs:
```
INFO: Using API key (masked): abcd************wxyz
```

## 6. Network Security

### Exchange Connections
- Use sandbox/testnet modes during development
- Enable rate limiting to prevent abuse
- Use secure connections (HTTPS/WebSocket) whenever possible

### Database Connections
- Use connection pooling to limit the number of connections
- Implement proper authentication for database access
- Use encrypted connections when possible

## 7. Application Security

### User Authentication
- Passwords are stored securely in the database
- Implement proper session management
- Consider adding two-factor authentication (2FA)

### Input Validation
- Validate all user inputs
- Sanitize data before database queries
- Use parameterized queries to prevent SQL injection

## 8. Deployment Security

### Production Environment
- Never use test credentials in production
- Rotate API keys regularly
- Monitor for unauthorized access
- Implement proper backup and recovery procedures

### Security Audits
- Regularly review code for security vulnerabilities
- Update dependencies to address known security issues
- Monitor logs for suspicious activity

## 9. Additional Recommendations

### Master Password
- Use a strong, unique master password for credential encryption
- Never store the master password in files or code
- Consider using a password manager

### Regular Maintenance
- Update the application and dependencies regularly
- Review and update security measures periodically
- Monitor exchange and database security advisories

### Incident Response
- Have a plan for responding to security incidents
- Know how to revoke API keys quickly
- Maintain backups of critical data