"""
Base Assessment Architecture

This package defines the core assessment architecture used by all assessment types.
It provides base classes for assessment components, including repositories, models,
services, and controllers.
"""

from backend.assessments.base.models import (
    AssessmentType,
    AssessmentSession,
    BaseQuestion,
    QuestionDifficulty,
    UserAnswer,
    AnswerEvaluation
)

from backend.assessments.base.repositories import (
    AssessmentRepository,
    QuestionRepository,
    SessionRepository
)

from backend.assessments.base.services import (
    AssessmentService,
    QuestionGenerator,
    AnswerEvaluator,
    ExplanationGenerator,
    PerformanceAnalyzer
)

from backend.assessments.base.controllers import (
    BaseAssessmentController
)

__all__ = [
    # Models
    'AssessmentType',
    'AssessmentSession',
    'BaseQuestion',
    'QuestionDifficulty',
    'UserAnswer',
    'AnswerEvaluation',
    
    # Repositories
    'AssessmentRepository',
    'QuestionRepository',
    'SessionRepository',
    
    # Services
    'AssessmentService',
    'QuestionGenerator',
    'AnswerEvaluator',
    'ExplanationGenerator',
    'PerformanceAnalyzer',
    
    # Controllers
    'BaseAssessmentController'
] 