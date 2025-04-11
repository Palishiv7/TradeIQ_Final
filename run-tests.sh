#!/bin/bash
# Run tests for TradeIQ

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Running TradeIQ tests...${NC}"

# Check if virtual environment exists
if [ ! -d "tradeiq-env" ]; then
    echo -e "${RED}Virtual environment not found. Please run setup-dev.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source tradeiq-env/bin/activate

# Parse command line arguments
ALL_TESTS=true
SPECIFIC_TEST=""
COVERAGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            ALL_TESTS=false
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./run-tests.sh [--test TEST_FILE] [--coverage] [-v|--verbose]"
            exit 1
            ;;
    esac
done

# Set verbosity level
VERBOSITY=""
if $VERBOSE; then
    VERBOSITY="-v"
fi

# Build command
if $ALL_TESTS; then
    if $COVERAGE; then
        echo -e "${YELLOW}Running all tests with coverage...${NC}"
        pytest backend/tests/ $VERBOSITY --cov=backend
    else
        echo -e "${YELLOW}Running all tests...${NC}"
        pytest backend/tests/ $VERBOSITY
    fi
else
    if $COVERAGE; then
        echo -e "${YELLOW}Running $SPECIFIC_TEST with coverage...${NC}"
        pytest $SPECIFIC_TEST $VERBOSITY --cov=backend
    else
        echo -e "${YELLOW}Running $SPECIFIC_TEST...${NC}"
        pytest $SPECIFIC_TEST $VERBOSITY
    fi
fi

# Run question generation tests if requested
if $ALL_TESTS || [[ "$SPECIFIC_TEST" == *"question_generation"* ]]; then
    echo -e "${YELLOW}Running comprehensive question generation tests...${NC}"
    python backend/tests/run_question_generation_tests.py
fi

echo -e "${GREEN}Tests completed!${NC}" 