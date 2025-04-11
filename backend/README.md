# TradeIQ Assessment Platform Backend

This directory contains the backend API for the TradeIQ assessment platform, which provides trading assessments and educational content focused on technical analysis, market psychology, and fundamentals.

## Application Structure

The backend follows a modular design pattern with the following structure:

```
backend/
├── __init__.py         # Package initialization and app factory
├── main.py             # Central entry point for the application
├── api.py              # Shared API utilities and central router
├── assessments/        # Assessment modules
│   ├── candlestick_patterns/  # Candlestick pattern assessment module
│   ├── market_psychology/     # Market psychology assessment module
│   └── market_fundamentals/   # Market fundamentals assessment module
├── cache/              # Cache utilities
├── common/             # Shared utilities and components
└── tests/              # Test suite
```

Each assessment module is self-contained and includes:
- API definitions
- Database models and repositories
- AI components for question generation and evaluation
- Configuration and utilities

## Development Environment Setup

### Option 1: Using the Setup Script

The easiest way to set up the development environment is to use the provided setup script:

```bash
# Make the script executable if needed
chmod +x setup-dev.sh

# Run the setup script
./setup-dev.sh
```

This will:
1. Create a virtual environment (`tradeiq-env`)
2. Install all required dependencies
3. Check if PostgreSQL and Redis are running
4. Provide instructions for activating the environment and running the application

### Option 2: Manual Setup

If you prefer to set up manually:

```bash
# Create a virtual environment
python -m venv tradeiq-env

# Activate the virtual environment
source tradeiq-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Development Mode

To run the application in development mode with auto-reload:

```bash
# Activate the virtual environment if not already active
source tradeiq-env/bin/activate

# Run with default settings
python -m backend.main

# Or using uvicorn directly
uvicorn backend.main:app --reload
```

### Production Mode

For production deployment:

```bash
# Set environment variables
export ENV=production
export RELOAD=false
export PORT=8000
export HOST=0.0.0.0
export AUTO_DB_INIT=true

# Run the application
uvicorn backend.main:app --host $HOST --port $PORT
```

## Testing

### Using the Test Runner Script

The easiest way to run tests is using the provided script:

```bash
# Make the script executable if needed
chmod +x run-tests.sh

# Run all tests
./run-tests.sh

# Run a specific test file
./run-tests.sh --test backend/tests/test_api.py

# Run with coverage
./run-tests.sh --coverage

# Run with verbose output
./run-tests.sh -v
```

### Manual Testing

If you prefer to run tests manually:

```bash
# Activate the virtual environment
source tradeiq-env/bin/activate

# Run all tests
pytest backend/tests/

# Run specific test files
pytest backend/tests/test_api.py
pytest backend/tests/test_question_generation.py

# Run with coverage
pytest backend/tests/ --cov=backend

# Run comprehensive question generation tests
python backend/tests/run_question_generation_tests.py
```

## Adding New Assessment Modules

To add a new assessment module:

1. Create a new directory under `assessments/`
2. Implement the necessary components (API, database, AI, etc.)
3. Create a FastAPI router in your module's API file
4. Update `backend/__init__.py` to include your module in the `register_assessment_modules()` function

## API Documentation

When the application is running, API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 