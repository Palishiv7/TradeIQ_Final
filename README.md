# TradeIQ: AI-Powered Financial Market Education Platform

## Overview

TradeIQ is a revolutionary AI-driven edtech platform transforming financial market education through automated, personalized assessment and learning experiences. Our system leverages state-of-the-art AI models to dynamically generate and evaluate assessments on candlestick patterns, market fundamentals, and market psychology without the need for manual data entry or dataset creation.

The platform's competitive advantage comes from its fully automated AI pipeline that:
- Generates diverse, contextually relevant questions tailored to each user's skill level
- Evaluates responses with sophisticated, multi-factor scoring algorithms
- Provides personalized feedback and recommendations for improvement
- Adapts difficulty levels in real-time based on performance metrics

## Vision & Mission

We believe financial education should be accessible, data-driven, and engaging for everyone. TradeIQ eliminates inefficiencies in traditional financial education by:

- **Full Automation**: Removing manual data entry and content creation through comprehensive AI automation
- **Personalization**: Providing real-time, tailored feedback based on individual learning patterns
- **Adaptive Learning**: Dynamically adjusting difficulty to keep users in their optimal learning zone
- **Engagement Engineering**: Implementing gamification and reward systems based on cognitive psychology research
- **Market Relevance**: Connecting theoretical concepts to real-world market conditions and scenarios
- **Psychological Training**: Developing mental resilience and reducing cognitive biases in trading decisions

## Core Technology Stack

### Frontend (Blazing-Fast UI & SEO-Optimized)

- **Framework**: React.js (Next.js)
  - Server-side rendering for optimal SEO and faster initial load
  - Static site generation for content-heavy pages
  - Dynamic routing for seamless user experience
  - API routes for backend communication

- **UI Library**: Tailwind CSS
  - Utility-first approach for rapid development
  - Custom design system with financial-specific components
  - Responsive design across all devices and screen sizes
  - Dark/light mode support for reduced eye strain during extended sessions

- **Real-Time Features**: WebSockets + React Query
  - Live updates for market data and assessments
  - Optimistic UI updates for responsive interfaces
  - Automatic revalidation and stale data management
  - Configurable polling intervals for different data types

- **State Management**: Zustand
  - Lightweight, hook-based store for global state
  - Middleware support for persistence and synchronization
  - Devtools integration for debugging and development
  - Atomic state updates for optimized rendering

- **Data Visualization**: Recharts.js
  - Interactive candlestick charts with pattern overlays
  - Performance analytics dashboards
  - Learning progress visualization
  - Customizable themes and styling

### Backend (High-Performance API & Business Logic)

- **Framework**: FastAPI
  - Asynchronous request handling for high throughput
  - Automatic OpenAPI documentation generation
  - Type hints and validation using Pydantic
  - Dependency injection system for clean architecture

- **AI Inference**: ONNX Runtime + Hugging Face Transformers
  - Model optimization for reduced inference time
  - Model versioning and runtime switching capabilities
  - Quantization for efficient resource utilization
  - Hardware acceleration (CPU/GPU/TPU) where available

- **Background Processing**: Celery + Redis
  - Task queuing for CPU-intensive operations
  - Scheduled and periodic tasks for data updates
  - Task prioritization and rate limiting
  - Distributed task execution across worker nodes

- **Authentication**: JWT (PyJWT)
  - Token-based authentication with refresh capabilities
  - Role-based access control for different user types
  - Session invalidation and blacklisting
  - Secure token storage and transmission

- **WebSockets**: FastAPI WebSockets
  - Bidirectional communication for real-time features
  - Connection pooling and management
  - Heartbeat mechanism for connection stability
  - Topic-based subscription model

- **Database ORM**: SQLAlchemy (PostgreSQL) + Motor (MongoDB)
  - Type-safe queries and model definitions
  - Migration management with Alembic
  - Connection pooling for efficient resource usage
  - Asynchronous query execution

### Database (Hybrid Storage for Speed & Scalability)

- **Primary DB**: PostgreSQL
  - Stores structured data with relational integrity
  - User accounts, authentication, and authorization
  - Assessment sessions, scores, and results
  - Leaderboards and progression tracking

- **NoSQL DB**: MongoDB
  - Stores semi-structured data with flexible schemas
  - AI-generated questions and explanations
  - Content variations and versioning
  - User interaction histories and patterns

- **Cache & Session Storage**: Redis
  - In-memory caching for frequently accessed data
  - Session management and temporary storage
  - Pub/sub for real-time notifications
  - Rate limiting and throttling implementation

## Detailed System Architecture

TradeIQ follows a modular, layered architecture based on domain-driven design principles, implementing:

### 1. Presentation Layer: API Controllers and Endpoints

- **REST API Controllers**
  - Implements RESTful endpoints for CRUD operations
  - Handles input validation and error responses
  - Maps domain objects to API-specific models
  - Manages authentication and authorization checks

- **WebSocket Controllers**
  - Handles real-time communication
  - Manages connection state and client tracking
  - Implements message serialization and deserialization
  - Provides event broadcasting capabilities

- **GraphQL Interface** (planned)
  - Schema-defined API for flexible data fetching
  - Resolvers for efficient data source integration
  - Subscriptions for real-time updates
  - Federation support for distributed services

- **API Documentation**
  - Auto-generated OpenAPI specifications
  - Interactive documentation with Swagger UI
  - Code examples and usage guides
  - Rate limit and authentication details

### 2. Business Logic Layer: Services and Domain Operations

- **Core Domain Services**
  - Implements business rules and validation logic
  - Provides transaction management and consistency
  - Coordinates between repositories and external services
  - Enforces domain-specific constraints and policies

- **Assessment Engines**
  - Question generation algorithms
  - Answer evaluation engines
  - Performance analysis and scoring
  - Difficulty adaptation mechanisms

- **AI Service Integrations**
  - Model inference services
  - Data preprocessing pipelines
  - Pattern recognition systems
  - NLP services for text analysis

- **Event System**
  - Domain events for state changes
  - Event handlers and subscribers
  - Event sourcing for audit trails
  - Message queue integration

### 3. Data Access Layer: Repositories and Data Models

- **Domain Models**
  - Rich domain entities with behavior
  - Value objects for immutable concepts
  - Aggregates for transactional boundaries
  - Domain events for state transitions

- **Repository Interfaces**
  - Generic interfaces with type parameters
  - CRUD operations for domain entities
  - Query specifications for filtering
  - Transaction support and unit of work

- **Repository Implementations**
  - Database-specific implementations
  - Caching strategies for performance
  - Query optimization techniques
  - Connection management and pooling

- **Data Transfer Objects (DTOs)**
  - Boundary-crossing data structures
  - Serialization/deserialization logic
  - Validation rules and constraints
  - Mapping services to domain models

### 4. Common Infrastructure: Shared Components and Utilities

- **Authentication & Authorization**
  - User identity management
  - Role-based access control (RBAC)
  - Permission verification
  - OAuth integration for third-party auth

- **Caching Infrastructure**
  - Multi-level caching strategy
  - Cache invalidation policies
  - Distributed cache synchronization
  - Memory-efficient cache implementations

- **Task Scheduling**
  - Background job processing
  - Periodic task execution
  - Distributed worker coordination
  - Failure handling and retries

- **Logging & Monitoring**
  - Structured logging with context
  - Log aggregation and search
  - Performance metrics collection
  - Alerting and notification systems

- **Error Handling**
  - Centralized error management
  - Custom exception types and hierarchies
  - Consistent error responses
  - Error tracking and analysis

## Assessment Modules

### 1. Candlestick Pattern Recognition 🕯️

The candlestick assessment module implements a comprehensive system for testing and improving users' ability to identify candlestick patterns from chart data.

#### Core Components:

- **Pattern Detection Engine**
  - Multi-strategy system combining rule-based and ML approaches
  - Ensemble detection with weighted consensus algorithms
  - Confidence scoring with statistical validation
  - Multiple detector implementations:
    - **Rule-Based**: Geometric and shape-based detection
    - **Statistical**: Using statistical rules and heuristics
    - **CNN-Based**: Using convolutional neural networks
    - **ResNet**: Using residual networks for deep feature extraction
    - **EfficientNet**: Using efficient scaled convolutions
    - **ConvNeXt**: Using state-of-the-art convolution designs

- **Question Generation System**
  - Dynamic chart generation with matplotlib/mplfinance
  - Pattern embedding at varying difficulty levels
  - Distractor option generation for multiple-choice questions
  - Contextual market data incorporation

- **Answer Evaluation System**
  - Pattern recognition accuracy assessment
  - Response time analysis and benchmarking
  - Partial credit for related pattern types
  - Confidence-based scoring adjustments

- **Learning Analysis**
  - Pattern-specific performance tracking
  - Confusion matrix generation for error analysis
  - Skill progression visualization
  - Targeted practice recommendations

- **Data Synchronization System** 
  - Robust market data synchronization with configurable schedules
  - Full and incremental sync modes for optimal performance
  - Pattern database maintenance with automatic updates
  - Self-healing error handling with comprehensive logging
  - Modular architecture for easy extension and customization
  - Automated cleanup of outdated data with archiving capabilities

#### Implementation Details:

- **Factory Pattern**: For detector creation and configuration
- **Strategy Pattern**: For interchangeable detection algorithms
- **Observer Pattern**: For notification of detection events
- **Builder Pattern**: For configurable chart generation
- **Repository Pattern**: For data access abstraction

#### Supported Pattern Types:

- **Single Candlestick Patterns**
  - Doji (Standard, Dragonfly, Gravestone, Long-Legged)
  - Hammer and Inverted Hammer
  - Shooting Star
  - Spinning Top
  - Marubozu (Bullish/Bearish)

- **Double Candlestick Patterns**
  - Bullish/Bearish Engulfing
  - Bullish/Bearish Harami
  - Tweezer Top/Bottom
  - Piercing Line/Dark Cloud Cover
  - Kicking Patterns

- **Triple Candlestick Patterns**
  - Morning Star/Evening Star
  - Three White Soldiers/Three Black Crows
  - Abandoned Baby (Bullish/Bearish)
  - Three Inside Up/Down
  - Three Outside Up/Down

- **Complex Patterns**
  - Head and Shoulders/Inverse Head and Shoulders
  - Double Top/Bottom
  - Triple Top/Bottom
  - Rising/Falling Wedge
  - Cup and Handle

### 2. Market Fundamentals & Technical Analysis 📊

This module tests and develops users' understanding of key financial concepts, indicators, and analysis techniques.

#### Core Components:

- **Question Generation Engine**
  - Uses DistilGPT-2 & Llama 2 for dynamic question creation
  - Integrates real-time market data for contextual relevance
  - Generates scenario-based problem-solving challenges
  - Creates comparative analysis questions across assets

- **Knowledge Assessment System**
  - Technical indicator calculation validation
  - Market analysis accuracy evaluation
  - Financial literacy assessment
  - Strategic decision-making evaluation

- **Content Adaptation System**
  - Adjusts complexity based on user performance
  - Targets specific knowledge gaps
  - Increases difficulty progressively
  - Provides scaffolded learning paths

- **Explanation Generator**
  - Creates detailed explanations for correct/incorrect answers
  - Provides visual aids and examples
  - Offers strategic application guidance
  - Includes market-specific context and insights

#### Topics Covered:

- **Technical Analysis**
  - Trend analysis and identification
  - Support and resistance levels
  - Volume analysis and interpretation
  - Chart patterns and formations
  - Technical indicators (Moving Averages, RSI, MACD, etc.)

- **Fundamental Analysis**
  - Financial statement analysis
  - Economic indicator interpretation
  - Company valuation methods
  - Industry and sector analysis
  - Risk assessment techniques

- **Market Structure**
  - Order types and execution
  - Market microstructure
  - Liquidity and volume analysis
  - Market participant behavior
  - Exchange mechanisms and operations

- **Trading Strategies**
  - Trend following strategies
  - Mean reversion approaches
  - Breakout trading techniques
  - Position sizing and risk management
  - Portfolio construction and diversification

### 3. Market Psychology & Bias Detection 🧠

This module helps users identify and overcome psychological biases affecting trading decisions.

#### Core Components:

- **Scenario Generation Engine**
  - Uses Mistral 7B / Llama 2 for creating realistic trading scenarios
  - Embeds specific cognitive biases within scenarios
  - Generates personalized challenges based on user history
  - Creates adaptive difficulty levels for psychological testing

- **Bias Detection System**
  - Identifies common trading biases in user responses
  - Measures susceptibility to different bias types
  - Tracks improvement in bias recognition over time
  - Analyzes decision patterns across market conditions

- **Psychological Profiling**
  - Creates trading psychology profiles
  - Identifies dominant decision-making styles
  - Maps emotional responses to market conditions
  - Suggests personalized improvement strategies

- **Intervention System**
  - Provides targeted exercises for bias reduction
  - Offers real-time trading decision checks
  - Implements cognitive restructuring techniques
  - Develops mindfulness practices for trading

#### Psychological Factors Assessed:

- **Cognitive Biases**
  - Confirmation bias
  - Recency bias
  - Anchoring effect
  - Availability heuristic
  - Overconfidence bias
  - Gambler's fallacy
  - Hindsight bias
  - Loss aversion

- **Emotional Factors**
  - Fear of missing out (FOMO)
  - Fear, uncertainty, and doubt (FUD)
  - Greed and euphoria
  - Panic and capitulation
  - Regret and disappointment
  - Overconfidence and complacency

- **Decision-Making Patterns**
  - Risk tolerance assessment
  - Decision consistency under pressure
  - Information processing style
  - Adaptation to changing market conditions
  - Response to gains and losses
  - Time horizon preferences

## Multi-Strategy Pattern Detection System

The pattern detection system is a cornerstone of the platform, providing sophisticated candlestick pattern identification capabilities through a well-structured, extensible, and robust architecture.

### Architecture Components

1. **Core Interfaces and Abstract Classes**
   - `PatternDetector`: Abstract base class defining the contract for all detectors with comprehensive error handling and async support
   - `PatternMatch`: Rich data class representing detection results with validation, normalization, and serialization capabilities
   - `DetectionStrategy`: Enum defining the detection approach types (RULE_BASED, GEOMETRIC, STATISTICAL, CNN, RESNET, EFFICIENTNET, ENSEMBLE, WEIGHTED_CONSENSUS, etc.)

2. **Strategy Implementations**
   - **Rule-Based Detection**:
     - `RuleBasedDetector`: Base class for rule-based approaches with pattern-specific validation
     - `GeometricPatternDetector`: Uses geometric measurements and relationships for shape-based detection
     - `StatisticalPatternDetector`: Incorporates statistical validation and filtering with trend analysis
     
   - **Model-Based Detection**:
     - `ModelBasedDetector`: Base class for all ML model detectors with model loading and inference
     - `CNNPatternDetector`: CNN implementation optimized for pattern feature extraction
     - `ResNetPatternDetector`: Implements ResNet architecture for deep pattern recognition
     - `EfficientNetPatternDetector`: Uses EfficientNet for improved efficiency and accuracy

3. **Ensemble Mechanism**
   - `EnsembleDetector`: Base class for ensemble approaches with concurrent execution
   - `WeightedConsensusDetector`: Advanced implementation with:
     - Weighted voting with confidence calibration
     - Dynamic detector weight adjustment based on performance analytics
     - Similarity-based pattern grouping with configurable thresholds
     - Intelligent metadata consolidation from multiple detectors
     - Consensus-based confidence boosting

4. **Factory System**
   - `create_detector()`: Async factory function with strategy-specific configuration
   - `get_default_detector()`: Returns the optimized detector configuration with caching
   - Fallback mechanisms with graceful degradation for unavailable models

### Key Implementation Features

1. **Robust Error Handling**
   - Comprehensive exception hierarchy for pattern-specific errors
   - Safe pattern detection methods with explicit error returns
   - Graceful degradation when components fail
   - Detailed error logging with context information
   - Recovery mechanisms for transient failures

2. **Asynchronous Architecture**
   - Full async/await pattern implementation throughout
   - Concurrent detector execution in ensembles
   - Non-blocking initialization and resource loading
   - Proper cancellation support with timeouts
   - Async-friendly interfaces with proper typing

3. **Pattern Validation & Confidence Scoring**
   - Multi-factor confidence scoring system with statistical validation
   - Pattern-specific verification rules with contextual awareness
   - Ensemble consensus scoring with dynamic weighting
   - Bullish/bearish classification with strength indicators
   - Detailed metadata capturing detection context and characteristics

4. **Data Handling & Conversion**
   - Robust serialization and deserialization with validation
   - Conversion between detection formats and domain models
   - Support for various input and output formats
   - Safe handling of invalid or incomplete data
   - Efficient data structures for pattern representation

5. **Performance Optimizations**
   - LRU caching for factory functions and expensive operations
   - Lazy loading of ML models and resources
   - Early termination for obvious pattern cases
   - Pattern filtering based on confidence thresholds
   - Efficient pattern similarity calculation for ensembles

### Using the Pattern Detection System

```python
# Basic async usage with default detector
import asyncio
from backend.assessments.candlestick_patterns.pattern_detection import get_default_detector
from backend.assessments.candlestick_patterns.candlestick_utils import Candle, CandlestickData

async def detect_patterns():
    # Create candlestick data
    candles = [
        Candle(time=1621123200, open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000),
        Candle(time=1621209600, open=103.0, high=110.0, low=102.0, close=108.0, volume=1200000),
        Candle(time=1621296000, open=108.0, high=112.0, low=107.0, close=109.0, volume=900000),
    ]

    # Create data container
    data = CandlestickData(symbol="AAPL", timeframe="1d", candles=candles)

    # Get detector and find patterns (with proper async handling)
    detector = await get_default_detector(config={"min_confidence": 0.7})
    patterns = await detector.detect_patterns(data)

    # Process results with improved error handling
    for pattern in patterns:
        print(f"Found {pattern.pattern_name} with confidence {pattern.confidence:.2f}")
        print(f"  Direction: {'Bullish' if pattern.bullish else 'Bearish' if pattern.bullish is False else 'Neutral'}")
        print(f"  Candle indices: {pattern.candle_indices}")
        print(f"  Detection strategy: {pattern.detection_strategy.value}")
        
        # Access rich metadata
        if pattern.metadata:
            for key, value in pattern.metadata.items():
                print(f"  Metadata - {key}: {value}")

# Advanced usage with custom ensemble and error handling
async def create_custom_detector():
    from backend.assessments.candlestick_patterns.pattern_detection import (
        DetectionStrategy, create_detector, WeightedConsensusDetector
    )
    
    try:
        # Create specific detectors with async initialization
        rule_detector = await create_detector(
            DetectionStrategy.RULE_BASED,
            config={"min_confidence": 0.6}
        )
        
        statistical_detector = await create_detector(
            DetectionStrategy.STATISTICAL,
            config={"min_confidence": 0.7}
        )
        
        # Try to create ML detector with fallback
        try:
            ml_detector = await create_detector(DetectionStrategy.ML_BASED)
            detectors = [rule_detector, statistical_detector, ml_detector]
        except Exception as e:
            print(f"ML detector unavailable, using rule-based only: {e}")
            detectors = [rule_detector, statistical_detector]
        
        # Create ensemble with custom configuration
        ensemble = WeightedConsensusDetector(
            detectors=detectors,
            name="CustomEnsembleDetector",
            min_confidence=0.65,
            max_patterns=10,
            config={
                "similarity_threshold": 0.8,
                "consensus_threshold": 0.6,
                "dynamic_weighting": True
            }
        )
        
        # Initialize the ensemble (this initializes all sub-detectors)
        await ensemble.initialize()
        return ensemble
        
    except Exception as e:
        print(f"Error creating detector: {e}")
        # Fallback to simple rule-based detector
        return await create_detector(DetectionStrategy.RULE_BASED)

# Safe pattern detection with error handling
async def detect_patterns_safely(data: CandlestickData):
    try:
        detector = await get_default_detector()
        patterns, error = await detector.detect_patterns_safe(data)
        
        if error:
            print(f"Warning: Detection completed with error: {error}")
            
        return patterns
    except Exception as e:
        print(f"Error in pattern detection: {e}")
        return []

### PatternMatch Class Details

The `PatternMatch` class provides a comprehensive representation of detected patterns with the following features:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from backend.assessments.candlestick_patterns.pattern_detection import DetectionStrategy

@dataclass
class PatternMatch:
    """Represents a detected candlestick pattern with metadata."""
    pattern_name: str
    confidence: float
    candle_indices: List[int]
    bullish: Optional[bool] = None
    description: Optional[str] = None
    detection_strategy: Optional[DetectionStrategy] = None
    detection_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize attributes after initialization."""
        # Validation and normalization code...
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Serialization code...
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternMatch':
        """Create a PatternMatch from a dictionary."""
        # Deserialization with validation...
        
    def to_candlestick_pattern(self, symbol: str, candles: List[Candle]) -> CandlestickPattern:
        """Convert to domain CandlestickPattern object."""
        # Conversion code with error handling...
```

## Extended Pattern Detection Support

The system has been extended with robust implementations for various pattern detection strategies:

1. **Single Candlestick Pattern Detection**
   - Improved doji detection with customizable body/shadow ratios
   - Enhanced hammer and shooting star recognition with statistical validation
   - Marubozu detection with configurable shadow thresholds
   - Spinning top identification with rotation-invariant algorithms

2. **Multi-Candlestick Pattern Detection**
   - Advanced engulfing pattern detection with volume confirmation
   - Harami pattern recognition with overlap calculation
   - Star pattern (morning/evening) detection with context awareness
   - Three candle pattern support (soldiers/crows) with trend validation
   - Complex pattern detection for rare formations

3. **Ensemble Methods**
   - Pattern similarity calculation using Jaccard index for candle overlap
   - Related pattern grouping with name-based similarity matching
   - Dynamic weight adjustment based on detector accuracy metrics
   - Consensus pattern formation with bullish/bearish voting
   - Metadata consolidation with statistical aggregation

4. **Model-Based Detection**
   - Support for multiple model formats and acceleration
   - Fallback mechanisms for unavailable models
   - Transfer learning support for specialized pattern recognition
   - Confidence calibration based on historical accuracy
   - Custom pattern extraction from model activations

## Integration Points

The pattern detection system integrates with other system components through:

1. **Assessment Generation**
   - Creating questions with embedded patterns of varied difficulty
   - Generating plausible distractors for multiple-choice questions
   - Producing explanations for detected patterns
   - Difficulty calibration based on pattern characteristics

2. **Performance Analysis**
   - Tracking user recognition accuracy for specific pattern types
   - Identifying pattern confusion patterns in user responses
   - Generating targeted practice recommendations
   - Providing pattern-specific difficulty metrics

3. **Learning Resources**
   - Linking detected patterns to educational content
   - Generating customized learning materials
   - Creating visual explanations of pattern characteristics
   - Providing real-world examples with detection analysis

## Core Domain Models

The system implements the following key domain models:

### Assessment Models

- **BaseQuestion**
  - Abstract base class for all question types
  - Properties: id, difficulty, created_at, updated_at
  - Methods: evaluate_answer(), get_explanation()

- **AssessmentSession**
  - Tracks a user's assessment session
  - Properties: id, user_id, start_time, end_time, status, questions, answers
  - Methods: submit_answer(), complete(), calculate_score()

- **UserAnswer**
  - Records user's answer to a question
  - Properties: question_id, answer_value, time_taken_ms, is_correct, score
  - Methods: calculate_score(), get_feedback()

- **Performance**
  - Tracks user's performance metrics
  - Properties: user_id, assessment_type, metrics, strengths, weaknesses
  - Methods: update_metrics(), get_recommendations()

### Candlestick Pattern Models

- **CandlestickQuestion** (extends BaseQuestion)
  - Properties: chart_data, correct_pattern, options, difficulty
  - Methods: generate_chart(), evaluate_answer(), get_explanation()

- **CandlestickData**
  - Container for candlestick series data
  - Properties: symbol, timeframe, candles, metadata
  - Methods: normalize(), to_array(), to_dataframe()

- **Candle**
  - Represents a single candlestick
  - Properties: time, open, high, low, close, volume
  - Methods: is_bullish(), body_size(), shadow_size()

- **PatternMatch**
  - Result of pattern detection
  - Properties: pattern_name, confidence, bullish, candle_indices, detection_strategy
  - Methods: to_dict(), to_json(), to_candlestick_pattern()

### Market Fundamentals Models

- **FundamentalsQuestion** (extends BaseQuestion)
  - Properties: question_text, correct_answer, options, context_data
  - Methods: evaluate_answer(), get_explanation(), get_difficulty()

- **TechnicalIndicator**
  - Represents a technical analysis indicator
  - Properties: name, parameters, values, metadata
  - Methods: calculate(), visualize(), get_signals()

- **MarketData**
  - Container for market information
  - Properties: symbol, timeframe, data_points, indicators
  - Methods: add_indicator(), get_trend(), normalize()

### Psychology Models

- **PsychologyQuestion** (extends BaseQuestion)
  - Properties: scenario, decision_options, biases, correct_response
  - Methods: evaluate_answer(), identify_bias(), get_explanation()

- **BiasProfile**
  - Tracks user's bias tendencies
  - Properties: user_id, bias_scores, dominant_biases, improvements
  - Methods: update(), get_recommendations(), compare_to_average()

- **DecisionScenario**
  - Represents a trading decision scenario
  - Properties: context, options, embedded_biases, optimal_choice
  - Methods: evaluate_decision(), explain_biases(), get_difficulty()

## Database Schema

### PostgreSQL Schema

```sql
-- Users and Authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(200) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE TABLE user_roles (
    user_id INTEGER REFERENCES users(id),
    role VARCHAR(20) NOT NULL,
    PRIMARY KEY (user_id, role)
);

-- Assessment Sessions
CREATE TABLE assessment_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    assessment_type VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,
    score NUMERIC(5,2),
    metadata JSONB
);

-- Questions and Answers
CREATE TABLE questions (
    id SERIAL PRIMARY KEY,
    assessment_type VARCHAR(50) NOT NULL,
    difficulty VARCHAR(20) NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_answers (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES assessment_sessions(id),
    question_id INTEGER REFERENCES questions(id),
    answer_value TEXT NOT NULL,
    time_taken_ms INTEGER NOT NULL,
    is_correct BOOLEAN NOT NULL,
    score NUMERIC(5,2) NOT NULL,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance Tracking
CREATE TABLE user_performance (
    user_id INTEGER REFERENCES users(id),
    assessment_type VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (user_id, assessment_type)
);

-- Leaderboards
CREATE TABLE leaderboard_entries (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    assessment_type VARCHAR(50) NOT NULL,
    score NUMERIC(7,2) NOT NULL,
    rank INTEGER,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### MongoDB Collections

```javascript
// AI-generated questions
db.createCollection("generated_questions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["assessment_type", "difficulty", "content", "created_at"],
      properties: {
        assessment_type: { bsonType: "string" },
        difficulty: { bsonType: "string" },
        content: { bsonType: "object" },
        metadata: { bsonType: "object" },
        created_at: { bsonType: "date" }
      }
    }
  }
});

// Pattern explanations
db.createCollection("pattern_explanations", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["pattern_name", "explanation_levels"],
      properties: {
        pattern_name: { bsonType: "string" },
        pattern_type: { bsonType: "string" },
        explanation_levels: { bsonType: "object" },
        examples: { bsonType: "array" },
        success_rate: { bsonType: "double" }
      }
    }
  }
});

// User interaction history
db.createCollection("user_interactions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "interaction_type", "timestamp"],
      properties: {
        user_id: { bsonType: "int" },
        interaction_type: { bsonType: "string" },
        details: { bsonType: "object" },
        timestamp: { bsonType: "date" }
      }
    }
  }
});
```

## Development Environment Setup

### Prerequisites

- Python 3.8+ (3.9 recommended)
- Node.js 14+ (16 recommended)
- PostgreSQL 13+
- MongoDB 4.4+
- Redis 6+
- Git
- Docker and Docker Compose (optional)

### Detailed Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-organization/tradeiq.git
cd tradeiq

# Create a virtual environment
python -m venv tradeiq-env

# Activate the virtual environment
# On Windows
tradeiq-env\Scripts\activate
# On macOS/Linux
source tradeiq-env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize the database
python -m backend.scripts.init_db

# Run database migrations
alembic upgrade head

# Generate initial test data (optional)
python -m backend.scripts.generate_test_data

# Run the backend
uvicorn backend.main:app --reload --port 8000
```

### Detailed Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Run development server
npm run dev

# Build for production
npm run build

# Run production build locally
npm run start
```

### Docker Setup (Optional)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Run migrations
docker-compose exec backend alembic upgrade head

# Stop all services
docker-compose down
```

### Testing

```bash
# Run backend tests
pytest backend/tests/

# Run with coverage
pytest backend/tests/ --cov=backend

# Run frontend tests
cd frontend
npm test

# End-to-end tests
cd e2e
npm test
```

## API Reference

### Authentication Endpoints

```
POST /api/auth/register
POST /api/auth/login
POST /api/auth/refresh
POST /api/auth/logout
GET /api/auth/me
```

### Candlestick Pattern Assessment Endpoints

```
POST /api/assessments/candlestick/start
GET /api/assessments/candlestick/sessions/{session_id}
GET /api/assessments/candlestick/sessions/{session_id}/questions/{question_id}
POST /api/assessments/candlestick/sessions/{session_id}/questions/{question_id}/submit
GET /api/assessments/candlestick/sessions/{session_id}/results
GET /api/assessments/candlestick/patterns/{pattern_name}/explanation
GET /api/assessments/candlestick/performance
```

### Market Fundamentals Assessment Endpoints

```
POST /api/assessments/fundamentals/start
GET /api/assessments/fundamentals/sessions/{session_id}
GET /api/assessments/fundamentals/sessions/{session_id}/questions/{question_id}
POST /api/assessments/fundamentals/sessions/{session_id}/questions/{question_id}/submit
GET /api/assessments/fundamentals/sessions/{session_id}/results
GET /api/assessments/fundamentals/topics/{topic_name}/explanation
GET /api/assessments/fundamentals/performance
```

### Market Psychology Assessment Endpoints

```
POST /api/assessments/psychology/start
GET /api/assessments/psychology/sessions/{session_id}
GET /api/assessments/psychology/sessions/{session_id}/questions/{question_id}
POST /api/assessments/psychology/sessions/{session_id}/questions/{question_id}/submit
GET /api/assessments/psychology/sessions/{session_id}/results
GET /api/assessments/psychology/biases/{bias_name}/explanation
GET /api/assessments/psychology/performance
```

## Best Practices

TradeIQ development follows these best practices:

### 1. Clean Architecture

- **Separation of Concerns**
  - Clear boundaries between layers
  - Well-defined interfaces between components
  - Dependency inversion for testability
  - Minimized coupling between modules

- **Domain-Centric Design**
  - Rich domain models with behavior
  - Business rules encapsulated in domain
  - Infrastructure concerns isolated
  - Persistence ignorance in domain layer

- **Testability**
  - Dependency injection for easy mocking
  - Interface-based design for component substitution
  - Clear seams between components
  - High unit test coverage of core logic

### 2. Domain-Driven Design

- **Ubiquitous Language**
  - Consistent terminology across codebase
  - Domain terms reflected in code
  - Clear glossary of terms in documentation
  - Business concepts mapped to code structures

- **Bounded Contexts**
  - Clear boundaries between assessment types
  - Context maps documenting relationships
  - Anti-corruption layers where needed
  - Consistent naming within contexts

- **Aggregates and Entities**
  - Well-defined aggregate roots (Sessions, Questions)
  - Transaction boundaries around aggregates
  - Consistent identity management
  - Entity lifecycle handling

- **Value Objects**
  - Immutable objects for conceptual wholes
  - Self-validation on creation
  - Equality by value, not reference
  - Rich domain behavior in value objects

### 3. SOLID Principles

- **Single Responsibility Principle**
  - Each class has one reason to change
  - Focused, cohesive components
  - Clear responsibilities documented
  - Avoidance of "god classes"

- **Open/Closed Principle**
  - Extension without modification
  - Strategy pattern for varying behaviors
  - Plugin architecture for assessment modules
  - Customization points for extension

- **Liskov Substitution Principle**
  - Derived classes don't break base functionality
  - Consistent behavior across inheritance hierarchy
  - Abstract base classes with contracts
  - Interface segregation for type safety

- **Interface Segregation Principle**
  - Focused interfaces with minimal methods
  - Client-specific interfaces
  - No forced dependency on unused methods
  - Role interfaces over header interfaces

- **Dependency Inversion Principle**
  - High-level modules don't depend on low-level modules
  - Abstraction over implementation
  - Consistent dependency injection
  - Factories for object creation

### 4. Performance Optimization

- **Caching Strategies**
  - Multi-level caching (memory, Redis)
  - TTL-based expiration policies
  - Cache invalidation on entity changes
  - Selective caching of expensive operations

- **Asynchronous Processing**
  - Non-blocking I/O operations
  - Background task processing
  - Asynchronous event handling
  - Parallel processing where appropriate

- **Database Optimization**
  - Efficient query patterns
  - Proper indexing strategy
  - Connection pooling
  - Batch operations for bulk data

- **Resource Management**
  - Proper connection cleanup
  - Memory leak prevention
  - Resource pooling
  - Timeout handling

## Deployment

### Production Environment Setup

```bash
# Set production environment variables
export ENV=production
export DEBUG=false
export LOG_LEVEL=info
export DATABASE_URL=postgresql://user:password@db:5432/tradeiq
export REDIS_URL=redis://redis:6379/0
export JWT_SECRET=your-production-secret
export PORT=8000

# Run with production settings
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:$PORT backend.main:app
```

### Deployment Options

1. **Docker Container Deployment**
   - Docker Compose for local deployment
   - Kubernetes for orchestration
   - Helm charts for deployment configuration
   - Horizontal pod autoscaling

2. **Serverless Deployment**
   - AWS Lambda with API Gateway
   - Azure Functions
   - Google Cloud Functions
   - Vercel for frontend

3. **Traditional VM Deployment**
   - Nginx as reverse proxy
   - Systemd service management
   - Automated deployment with Ansible
   - Load balancing with HAProxy

### CI/CD Pipeline

1. **Continuous Integration**
   - Automated testing on push/PR
   - Code quality checks
   - Security scanning
   - Documentation generation

2. **Continuous Deployment**
   - Automated deployment to staging
   - Manual promotion to production
   - Canary releases
   - Blue-green deployments

## Monitoring and Observability

1. **Logging**
   - Structured JSON logging
   - Log aggregation with ELK stack
   - Log correlation with request IDs
   - Log level management

2. **Metrics**
   - Application metrics with Prometheus
   - Business metrics tracking
   - SLA monitoring
   - Resource utilization tracking

3. **Tracing**
   - Distributed tracing with OpenTelemetry
   - Request flow visualization
   - Performance bottleneck identification
   - Error tracking and alerting

4. **Alerting**
   - Threshold-based alerts
   - Anomaly detection
   - On-call rotation
   - Incident management workflow

## Roadmap

### Short-Term (3-6 Months)
- Complete core assessments implementation for candlestick patterns
- Optimize AI-generated content for question variety and quality
- Launch beta version with limited user access
- Implement real-time market data integration with major providers
- Complete unit and integration test coverage
- Implement comprehensive error handling and recovery

### Mid-Term (6-12 Months)
- Expand AI-driven learning paths with personalized recommendations
- Introduce advanced market strategies assessment module
- Develop mobile applications for Android and iOS
- Launch community and social learning features
- Implement gamification system with achievements and rewards
- Enhance AI models with user feedback loop for continuous improvement
- Optimize infrastructure for scale and performance

### Long-Term (1-3 Years)
- Implement AI-powered virtual trading coach with real-time guidance
- Develop institutional and enterprise solutions with custom assessment creation
- Expand to multiple languages and international markets
- Create AI-powered automated trading simulator with performance analysis
- Build comprehensive data analytics platform for learning insights
- Develop API ecosystem for third-party integrations
- Implement advanced personalization using reinforcement learning

## Contributing

We welcome contributions from all team members. Please follow these guidelines:

1. **Fork the Repository**
   - Create your own fork of the code
   - Keep your fork updated with the main repository

2. **Create a Feature Branch**
   - Branch naming convention: `feature/description`, `bugfix/description`
   - One feature/fix per branch

3. **Follow Code Standards**
   - Follow PEP 8 for Python code
   - Use ESLint rules for JavaScript
   - Document all public APIs
   - Maintain test coverage

4. **Submit Pull Requests**
   - Provide clear description of changes
   - Reference any related issues
   - Include unit tests for new features
   - Update documentation as needed

5. **Code Review Process**
   - All PRs require at least one reviewer
   - Address review comments promptly
   - Automated tests must pass
   - Code quality gates must be satisfied

## License and Legal

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

© 2023 TradeIQ Technologies. All rights reserved.
