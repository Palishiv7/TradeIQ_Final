# TradeIQ Assessment Platform Architecture

## Overview

The TradeIQ Assessment Platform uses a modular, layered architecture based on domain-driven design principles. This document outlines the key architectural components and provides guidelines for developers.

## Architecture Layers

1. **Presentation Layer**: API controllers and endpoints
   - Handles HTTP requests/responses
   - Input validation and error handling
   - Route registration and documentation

2. **Business Logic Layer**: Services and domain operations
   - Implements core business logic
   - Coordinates between repositories and external services
   - Enforces business rules and validation

3. **Data Access Layer**: Repositories and data models
   - Abstracts database operations
   - Provides CRUD operations for entities
   - Ensures data consistency and integrity

4. **Common Infrastructure**: Shared components and utilities
   - Authentication and authorization
   - Caching and task scheduling
   - Logging and configuration

## Component Structure

### Assessment Architecture

All assessment functionality is built on a common base architecture:

- **Base Components**
  - Located in `backend/assessments/base/`
  - Defines interfaces and abstract classes for all assessment types
  - Provides common functionality for questions, sessions, and evaluation

- **Assessment Types**
  - Each assessment type (e.g., candlestick patterns) has its own directory
  - Extends the base components with type-specific functionality
  - Follows consistent patterns for controllers, services, and models

### Directory Structure

```
backend/
├── assessments/                # Assessment functionality
│   ├── base/                   # Base assessment architecture
│   │   ├── models.py           # Core data models
│   │   ├── repositories.py     # Data access interfaces
│   │   ├── services.py         # Service interfaces 
│   │   ├── controllers.py      # API controller base classes
│   │   └── tasks.py            # Background task definitions
│   ├── candlestick_patterns/   # Candlestick pattern assessment
│   │   ├── candlestick_models.py
│   │   ├── candlestick_service.py
│   │   ├── candlestick_controller.py
│   │   └── ...
│   └── market_fundamentals/    # Market fundamentals assessment
│       └── ...
├── common/                     # Shared infrastructure
│   ├── auth/                   # Authentication components
│   ├── cache/                  # Caching system
│   ├── db/                     # Database utilities
│   ├── finance/                # Financial data models
│   ├── tasks/                  # Task scheduling
│   └── ...
└── server/                     # Application server
    └── ...
```

## Architecture Migration

The platform is currently undergoing an architecture consolidation to simplify code and avoid duplication.

### Migration Status

The following duplicate components have been refactored:

1. Core assessment models:
   - `backend/common/base_assessment.py` is being deprecated in favor of `backend/assessments/base/models.py`
   - A compatibility layer is maintained to avoid breaking existing imports

2. Repository interfaces:
   - `backend/common/assessment_repository.py` is being deprecated in favor of `backend/assessments/base/repositories.py`
   - A compatibility layer is maintained to avoid breaking existing imports

3. Service interfaces:
   - `backend/common/assessment_service.py` is being deprecated in favor of `backend/assessments/base/services.py`
   - A compatibility layer is maintained to avoid breaking existing imports

### Migration Guidelines for Developers

When working with assessment code, follow these guidelines:

1. **For New Code**:
   - Import directly from `backend.assessments.base.models`, `backend.assessments.base.repositories`, etc.
   - Do not import from the deprecated `backend.common.base_assessment`

2. **For Existing Code**:
   - No immediate changes needed - compatibility layers ensure it continues to work
   - When making significant changes to a file, update its imports to use the canonical locations

3. **For Assessment Implementation**:
   - Always extend the classes in `backend.assessments.base.*`
   - Follow the standard naming conventions for controllers, services, models, etc.
   - Reference the candlestick pattern assessment (`backend/assessments/candlestick_patterns/`) as an example

## Best Practices

1. **Clean Architecture**:
   - Maintain strict separation of concerns between layers
   - Follow dependency inversion - depend on abstractions, not implementations
   - Keep domain entities with well-defined boundaries

2. **Domain-Driven Design**:
   - Use aggregate roots with clear relationships
   - Apply value objects for immutable concepts
   - Use domain events for cross-boundary communication

3. **SOLID Principles**:
   - **S**ingle Responsibility: Each class has one reason to change
   - **O**pen/Closed: Extend functionality without modifying existing code
   - **L**iskov Substitution: Derived classes don't break base functionality
   - **I**nterface Segregation: Many specific interfaces over one general interface
   - **D**ependency Inversion: High-level modules don't depend on low-level modules

4. **Repository Pattern**:
   - Generic interfaces with specific implementations
   - Abstract data access logic
   - Enable unit testability with mock repositories 