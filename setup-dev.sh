#!/bin/bash
# Setup script for TradeIQ development environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up TradeIQ development environment...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "tradeiq-env" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv tradeiq-env
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source tradeiq-env/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if PostgreSQL is running
echo -e "${YELLOW}Checking PostgreSQL status...${NC}"
if command -v pg_isready &> /dev/null; then
    if pg_isready &> /dev/null; then
        echo -e "${GREEN}PostgreSQL is running.${NC}"
    else
        echo -e "${RED}PostgreSQL is not running. Please start it before running the application.${NC}"
    fi
else
    echo -e "${YELLOW}Cannot check PostgreSQL status. Make sure it's running before starting the application.${NC}"
fi

# Check if Redis is running
echo -e "${YELLOW}Checking Redis status...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}Redis is running.${NC}"
    else
        echo -e "${RED}Redis is not running. Please start it before running the application.${NC}"
    fi
else
    echo -e "${YELLOW}Cannot check Redis status. Make sure it's running before starting the application.${NC}"
fi

# Setup complete
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}To activate the virtual environment, run:${NC}"
echo -e "source tradeiq-env/bin/activate"
echo -e "${BLUE}To run the application, run:${NC}"
echo -e "python -m backend.main"
echo -e "${BLUE}Or with uvicorn directly:${NC}"
echo -e "uvicorn backend.main:app --reload"

# Keep the virtual environment active
exec $SHELL 