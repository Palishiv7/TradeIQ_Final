# Comprehensive Question Generation Test Suite

This test suite provides exhaustive testing for the candlestick pattern question generation system, focusing on identifying edge cases and potential breaking points without using mocks. The tests interact with real database, cache, and question generation components to provide realistic validation of system behavior.

## Test Suite Structure

The test suite is divided into four major parts to manage complexity and token limits:

### Part 1: Basic Functionality and Template Selection
- Located in `test_question_generation.py`
- Tests template selection with various inputs
- Validates question generation fundamentals
- Tests input validation and basic error handling
- Ensures proper question formatting and diversity

### Part 2: Question Uniqueness and Database Persistence
- Located in `test_question_generation_part2.py`
- Tests uniqueness verification with various scenarios
- Validates database persistence with large or special content
- Tests concurrent database operations
- Ensures proper data integrity in the database

### Part 3: Error Handling and Recovery Mechanisms
- Located in `test_question_generation_part3.py`
- Tests system behavior during database failures
- Validates cache failure recovery
- Tests transaction isolation
- Ensures data integrity during failure scenarios
- Tests performance degradation recovery

### Part 4: Concurrency and Race Conditions
- Located in `test_question_generation_part4.py`
- Tests concurrent question generation for the same and different users
- Validates database behavior during concurrent updates
- Ensures proper connection pool management
- Tests resource management under high load

## Running the Tests

All tests can be run using the main test runner script:

```bash
python run_question_generation_tests.py
```

To run a specific test part only:

```bash
python run_question_generation_tests.py --part 1  # Run only Part 1
python run_question_generation_tests.py --part 2  # Run only Part 2
python run_question_generation_tests.py --part 3  # Run only Part 3
python run_question_generation_tests.py --part 4  # Run only Part 4
```

Alternatively, each test file can be run individually:

```bash
pytest test_question_generation.py -v
pytest test_question_generation_part2.py -v
pytest test_question_generation_part3.py -v
pytest test_question_generation_part4.py -v
```

## Test Prerequisites

Before running the tests, ensure the following prerequisites are met:

1. Database is properly configured and accessible
2. Redis cache service is running
3. All required Python packages are installed
4. The question generation system is properly configured

## Test Fixtures

The test suite uses several fixtures to prepare the testing environment:

- `database_connection`: Sets up a test database connection
- `db_session`: Creates a new database session for each test
- `repository`: Provides a repository instance for database operations
- `clear_cache`: Flushes the cache before each test
- `seed_pattern_data`: Seeds the database with pattern statistics
- `seed_user_data`: Seeds the database with user performance records
- `seed_assessment_data`: Seeds the database with assessment attempts
- `seed_question_history`: Seeds the database with question history records

These fixtures are defined in the first test part file and imported by subsequent parts.

## Best Practices Implemented

This test suite follows several best practices for comprehensive testing:

1. **Real Component Testing**: Tests interact with actual system components rather than using mocks.
2. **Edge Case Coverage**: Tests include extreme inputs, performance degradation, and error scenarios.
3. **Concurrency Testing**: Tests validate system behavior under concurrent load.
4. **Proper Cleanup**: Each test cleans up after itself to avoid interference between tests.
5. **Data Integrity Verification**: Tests validate that data remains consistent during failures.
6. **Performance Validation**: Tests include timing and performance metrics to ensure system efficiency.
7. **Recovery Testing**: Tests validate that the system can recover from various failure scenarios.

## Troubleshooting

If the tests fail, check the following:

1. **Database Connectivity**: Ensure the database is running and accessible.
2. **Redis Cache**: Verify the Redis cache service is running.
3. **Environment Configuration**: Check that all environment variables are properly set.
4. **Resource Limitations**: Some tests create high loads, ensure the system has sufficient resources.
5. **Test Dependencies**: Ensure all test fixtures are properly defined and accessible.

For detailed error messages, run the tests with increased verbosity:

```bash
pytest test_question_generation.py -vv
```

## Extending the Tests

To add new tests:

1. Identify the appropriate test part based on the functionality being tested.
2. Add a new test method to an existing test class or create a new test class.
3. Follow the pattern of existing tests, using fixtures and assertions as needed.
4. Update the main test runner if new test files are added.

## Performance Considerations

Some tests, particularly in Part 4, create high system loads to test concurrency. These tests might take longer to run and could impact system performance while running. Consider running these tests in a non-production environment. 