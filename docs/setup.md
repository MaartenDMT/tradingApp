# Trading Application Setup Guide

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation Steps

1. Clone or download the repository

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Configuration

Create a `.env` file in the root directory with the following variables:

```
API_KEY_PHE_TEST=your_phemex_testnet_api_key
API_SECRET_PHE_TEST=your_phemex_testnet_api_secret
PGHOST=your_postgresql_host
PGPORT=your_postgresql_port
PGDATABASE=your_database_name
PGUSER=your_database_user
PGPASSWORD=your_database_password
```

## Database Setup

The application uses PostgreSQL for user authentication. Create a database table with the following structure:

```sql
CREATE TABLE "User" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL
);
```

## Configuration

Review and modify `config.ini` as needed for your specific requirements:
- Adjust window size
- Modify WebSocket parameters
- Update data timeframes
- Tune trading parameters
- Configure model hyperparameters

## Running the Application

```bash
python main.py
```

## Default Login Credentials

For testing purposes, you can use:
- Username: test
- Password: t

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Check that your environment variables are properly set

3. Verify database connectivity

4. Confirm API keys are valid and have necessary permissions