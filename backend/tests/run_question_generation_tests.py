#!/usr/bin/env python
"""
Comprehensive test runner for question generation system tests.

This script runs all four parts of the comprehensive database functional tests
for the question generation system in sequence:

1. Part 1: Basic functionality and template selection tests
2. Part 2: Question uniqueness and database persistence tests
3. Part 3: Error handling and recovery mechanism tests
4. Part 4: Concurrency and race condition tests

Usage:
    python run_question_generation_tests.py [--part PART_NUMBER]

    Optional arguments:
    --part PART_NUMBER: Run only the specified part (1-4)
"""

import os
import sys
import asyncio
import argparse
import pytest
import importlib
from datetime import datetime

# Import test modules
try:
    from backend.tests.test_question_generation import test_part1_basic_functionality
    from backend.tests.test_question_generation_part2 import test_part2_question_persistence
    from backend.tests.test_question_generation_part3 import test_part3_error_handling
    from backend.tests.test_question_generation_part4 import test_part4_concurrency
except ImportError as e:
    print(f"Error importing test modules: {str(e)}")
    print("Make sure all test modules are properly installed.")
    sys.exit(1)

async def run_all_tests():
    """Run all test parts in sequence"""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"Starting comprehensive question generation tests at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Run part 1: Basic functionality and template selection
    print("\n\n" + "="*30 + " PART 1: BASIC FUNCTIONALITY " + "="*30)
    try:
        await test_part1_basic_functionality()
        print("\n✓ Part 1 completed successfully")
    except Exception as e:
        print(f"\n✗ Part 1 failed: {str(e)}")
        raise
    
    # Run part 2: Question uniqueness and database persistence
    print("\n\n" + "="*30 + " PART 2: UNIQUENESS & PERSISTENCE " + "="*30)
    try:
        await test_part2_question_persistence()
        print("\n✓ Part 2 completed successfully")
    except Exception as e:
        print(f"\n✗ Part 2 failed: {str(e)}")
        raise
    
    # Run part 3: Error handling and recovery mechanisms
    print("\n\n" + "="*30 + " PART 3: ERROR HANDLING & RECOVERY " + "="*30)
    try:
        await test_part3_error_handling()
        print("\n✓ Part 3 completed successfully")
    except Exception as e:
        print(f"\n✗ Part 3 failed: {str(e)}")
        raise
    
    # Run part 4: Concurrency and race conditions
    print("\n\n" + "="*30 + " PART 4: CONCURRENCY & RACE CONDITIONS " + "="*30)
    try:
        await test_part4_concurrency()
        print("\n✓ Part 4 completed successfully")
    except Exception as e:
        print(f"\n✗ Part 4 failed: {str(e)}")
        raise
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n\n{'='*80}")
    print(f"All tests completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"{'='*80}")

async def run_part(part_number):
    """Run a specific test part"""
    if part_number == 1:
        print("\n" + "="*30 + " PART 1: BASIC FUNCTIONALITY " + "="*30)
        await test_part1_basic_functionality()
    elif part_number == 2:
        print("\n" + "="*30 + " PART 2: UNIQUENESS & PERSISTENCE " + "="*30)
        await test_part2_question_persistence()
    elif part_number == 3:
        print("\n" + "="*30 + " PART 3: ERROR HANDLING & RECOVERY " + "="*30)
        await test_part3_error_handling()
    elif part_number == 4:
        print("\n" + "="*30 + " PART 4: CONCURRENCY & RACE CONDITIONS " + "="*30)
        await test_part4_concurrency()
    else:
        print(f"Invalid part number: {part_number}. Choose from 1-4.")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run question generation system tests")
    parser.add_argument("--part", type=int, help="Run only a specific test part (1-4)")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run tests
    if args.part:
        asyncio.run(run_part(args.part))
    else:
        asyncio.run(run_all_tests()) 