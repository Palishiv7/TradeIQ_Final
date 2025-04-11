"""
Performance Tracker

This module provides classes and functions for tracking user performance metrics
across different assessment types, supporting adaptive difficulty adjustments
and personalized learning experiences.
"""

import time
import math
import enum
import datetime
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict

from backend.common.logger import app_logger
from backend.common.serialization import SerializableMixin

# Module logger
logger = app_logger.getChild("performance.tracker")


class SkillLevel(enum.Enum):
    """User skill levels for a particular domain or topic."""
    BEGINNER = "beginner"
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    
    @classmethod
    def from_score(cls, score: float) -> 'SkillLevel':
        """
        Determine skill level from a normalized score (0-100).
        
        Args:
            score: Normalized score between 0 and 100
            
        Returns:
            Corresponding skill level
        """
        if score < 20:
            return cls.BEGINNER
        elif score < 40:
            return cls.NOVICE
        elif score < 60:
            return cls.INTERMEDIATE
        elif score < 80:
            return cls.ADVANCED
        else:
            return cls.EXPERT
    
    def to_numeric(self) -> int:
        """Convert skill level to a numeric value (1-5)."""
        return {
            SkillLevel.BEGINNER: 1,
            SkillLevel.NOVICE: 2,
            SkillLevel.INTERMEDIATE: 3,
            SkillLevel.ADVANCED: 4,
            SkillLevel.EXPERT: 5
        }[self]


class LearningRate(enum.Enum):
    """User learning rate categorizations."""
    VERY_SLOW = "very_slow"
    SLOW = "slow"
    AVERAGE = "average"
    FAST = "fast"
    VERY_FAST = "very_fast"
    
    @classmethod
    def from_rate(cls, rate: float) -> 'LearningRate':
        """
        Determine learning rate category from a normalized rate.
        
        Args:
            rate: Normalized learning rate
            
        Returns:
            Corresponding learning rate category
        """
        if rate < 0.2:
            return cls.VERY_SLOW
        elif rate < 0.4:
            return cls.SLOW
        elif rate < 0.6:
            return cls.AVERAGE
        elif rate < 0.8:
            return cls.FAST
        else:
            return cls.VERY_FAST


@dataclass
class TopicPerformance:
    """Performance metrics for a specific topic or skill area."""
    
    topic: str
    attempts: int = 0
    correct: int = 0
    streak: int = 0
    max_streak: int = 0
    avg_time_ms: float = 0
    last_seen: Optional[datetime.datetime] = None
    skill_level: SkillLevel = SkillLevel.BEGINNER
    proficiency: float = 0.0  # 0.0 to 1.0
    
    def record_attempt(self, is_correct: bool, time_ms: float) -> None:
        """
        Record a single attempt at a question in this topic.
        
        Args:
            is_correct: Whether the attempt was correct
            time_ms: Time taken in milliseconds
        """
        self.attempts += 1
        
        # Update time metrics
        if self.avg_time_ms == 0:
            self.avg_time_ms = time_ms
        else:
            # Exponential moving average with alpha=0.3
            self.avg_time_ms = 0.7 * self.avg_time_ms + 0.3 * time_ms
        
        # Update accuracy metrics
        if is_correct:
            self.correct += 1
            self.streak += 1
            if self.streak > self.max_streak:
                self.max_streak = self.streak
        else:
            self.streak = 0
        
        # Update last seen timestamp
        self.last_seen = datetime.datetime.now()
        
        # Update proficiency
        self._update_proficiency()
        
        # Update skill level
        self._update_skill_level()
    
    def _update_proficiency(self) -> None:
        """Update proficiency score based on performance metrics."""
        if self.attempts == 0:
            self.proficiency = 0.0
            return
        
        # Base proficiency from accuracy
        accuracy = self.correct / self.attempts
        
        # Apply streak bonus
        streak_factor = min(self.streak / 10, 0.2)
        
        # Apply attempts penalty (diminishes as attempts increase)
        attempts_factor = min(self.attempts / 20, 0.2)
        
        # Combine factors
        self.proficiency = min(accuracy + streak_factor + attempts_factor, 1.0)
    
    def _update_skill_level(self) -> None:
        """Update skill level based on proficiency."""
        self.skill_level = SkillLevel.from_score(self.proficiency * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "attempts": self.attempts,
            "correct": self.correct,
            "streak": self.streak,
            "max_streak": self.max_streak,
            "avg_time_ms": self.avg_time_ms,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "skill_level": self.skill_level.value,
            "proficiency": self.proficiency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopicPerformance':
        """Create from dictionary."""
        return cls(
            topic=data["topic"],
            attempts=data.get("attempts", 0),
            correct=data.get("correct", 0),
            streak=data.get("streak", 0),
            max_streak=data.get("max_streak", 0),
            avg_time_ms=data.get("avg_time_ms", 0.0),
            last_seen=datetime.datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
            skill_level=SkillLevel(data.get("skill_level", SkillLevel.BEGINNER.value)),
            proficiency=data.get("proficiency", 0.0)
        )


@dataclass
class SessionStats:
    """Statistics for a single assessment session."""
    
    session_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    questions_total: int = 0
    questions_answered: int = 0
    questions_correct: int = 0
    avg_time_per_question_ms: float = 0
    topics: Set[str] = field(default_factory=set)
    score: float = 0.0
    
    def record_question(self, topic: str, is_correct: bool, time_ms: float) -> None:
        """
        Record a question answer in this session.
        
        Args:
            topic: Question topic
            is_correct: Whether the answer was correct
            time_ms: Time taken in milliseconds
        """
        self.questions_answered += 1
        
        if is_correct:
            self.questions_correct += 1
        
        self.topics.add(topic)
        
        # Update average time
        if self.avg_time_per_question_ms == 0:
            self.avg_time_per_question_ms = time_ms
        else:
            self.avg_time_per_question_ms = (
                (self.avg_time_per_question_ms * (self.questions_answered - 1) + time_ms) / 
                self.questions_answered
            )
    
    def complete(self) -> None:
        """Mark the session as completed."""
        self.end_time = datetime.datetime.now()
        
        # Calculate score
        if self.questions_answered > 0:
            accuracy = self.questions_correct / self.questions_answered
            self.score = accuracy * 100
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        if not self.end_time:
            return (datetime.datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "questions_total": self.questions_total,
            "questions_answered": self.questions_answered,
            "questions_correct": self.questions_correct,
            "avg_time_per_question_ms": self.avg_time_per_question_ms,
            "topics": list(self.topics),
            "score": self.score,
            "duration_seconds": self.duration_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionStats':
        """Create from dictionary."""
        session = cls(
            session_id=data["session_id"],
            start_time=datetime.datetime.fromisoformat(data["start_time"]),
            end_time=datetime.datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            questions_total=data.get("questions_total", 0),
            questions_answered=data.get("questions_answered", 0),
            questions_correct=data.get("questions_correct", 0),
            avg_time_per_question_ms=data.get("avg_time_per_question_ms", 0.0),
            score=data.get("score", 0.0)
        )
        
        session.topics = set(data.get("topics", []))
        return session


@dataclass
class PerformanceMetrics:
    """Overall performance metrics for a user in an assessment area."""
    
    total_sessions: int = 0
    total_questions: int = 0
    total_correct: int = 0
    avg_score: float = 0.0
    total_time_spent_sec: float = 0.0
    last_session_time: Optional[datetime.datetime] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def update_with_session(self, session: SessionStats) -> None:
        """
        Update metrics with a completed session.
        
        Args:
            session: Session statistics
        """
        if not session.end_time:
            raise ValueError("Cannot update metrics with incomplete session")
        
        self.total_sessions += 1
        self.total_questions += session.questions_answered
        self.total_correct += session.questions_correct
        
        # Update average score
        self.avg_score = (
            (self.avg_score * (self.total_sessions - 1) + session.score) / 
            self.total_sessions
        )
        
        # Update time spent
        self.total_time_spent_sec += session.duration_seconds
        
        # Update last session time
        self.last_session_time = session.end_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_sessions": self.total_sessions,
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "avg_score": self.avg_score,
            "total_time_spent_sec": self.total_time_spent_sec,
            "last_session_time": self.last_session_time.isoformat() if self.last_session_time else None,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(
            total_sessions=data.get("total_sessions", 0),
            total_questions=data.get("total_questions", 0),
            total_correct=data.get("total_correct", 0),
            avg_score=data.get("avg_score", 0.0),
            total_time_spent_sec=data.get("total_time_spent_sec", 0.0),
            last_session_time=datetime.datetime.fromisoformat(data["last_session_time"]) 
                if data.get("last_session_time") else None,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", [])
        )


@dataclass
class PerformanceTracker(SerializableMixin):
    """
    Tracks user performance across topics and sessions.
    
    This class collects and manages performance data for a user, providing insights
    into strengths, weaknesses, and skill progression over time.
    """
    
    __serializable_fields__ = [
        "user_id", "assessment_type", "topic_performance", 
        "recent_sessions", "metrics", "created_at", "updated_at", "pattern_metrics"
    ]
    
    __optional_fields__ = [
        "created_at", "updated_at", "recent_sessions"
    ]
    
    def __init__(
        self,
        user_id: str,
        assessment_type: str,
        topic_performance: Optional[Dict[str, TopicPerformance]] = None,
        recent_sessions: Optional[List[SessionStats]] = None,
        metrics: Optional[PerformanceMetrics] = None,
        created_at: Optional[datetime.datetime] = None,
        updated_at: Optional[datetime.datetime] = None,
        pattern_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the performance tracker.
        
        Args:
            user_id: User identifier
            assessment_type: Type of assessment (e.g., "candlestick", "fundamental")
            topic_performance: Optional dictionary mapping topics to performance data
            recent_sessions: Optional list of recent session statistics
            metrics: Optional overall performance metrics
            created_at: Optional creation timestamp
            updated_at: Optional update timestamp
            pattern_metrics: Optional dictionary mapping patterns to performance data
        """
        self.user_id = user_id
        self.assessment_type = assessment_type
        self.topic_performance = topic_performance or {}
        self.recent_sessions = recent_sessions or []
        self.metrics = metrics or PerformanceMetrics()
        self.created_at = created_at or datetime.datetime.now()
        self.updated_at = updated_at or datetime.datetime.now()
        
        # Initialize pattern metrics with default values
        self.pattern_metrics = pattern_metrics or defaultdict(lambda: {
            "attempts": 0,
            "correct": 0,
            "streak": 0,
            "max_streak": 0,
            "avg_time_ms": 0.0,
            "last_seen": None,
            "mastery": 0.0
        })
    
    def start_session(self, session_id: str, questions_total: int = 0) -> SessionStats:
        """
        Start a new assessment session.
        
        Args:
            session_id: Session identifier
            questions_total: Total number of questions in the session
            
        Returns:
            New session statistics
        """
        if self.current_session:
            # Complete previous session if not already completed
            if not self.current_session.end_time:
                self.complete_session()
        
        self.current_session = SessionStats(
            session_id=session_id,
            start_time=datetime.datetime.now(),
            questions_total=questions_total
        )
        
        logger.debug(f"Started session {session_id} for user {self.user_id}")
        return self.current_session
    
    def record_answer(
        self,
        topic: str,
        is_correct: bool,
        time_ms: float,
        subtopics: Optional[List[str]] = None
    ) -> None:
        """
        Record an answer to a question.
        
        Args:
            topic: Question topic
            is_correct: Whether the answer was correct
            time_ms: Time taken in milliseconds
            subtopics: Optional list of subtopics
        """
        if not self.current_session:
            raise ValueError("No active session")
        
        # Record in current session
        self.current_session.record_question(topic, is_correct, time_ms)
        
        # Record in topic performance
        if topic not in self.topic_performance:
            self.topic_performance[topic] = TopicPerformance(topic=topic)
        
        self.topic_performance[topic].record_attempt(is_correct, time_ms)
        
        # Record subtopics if provided
        if subtopics:
            for subtopic in subtopics:
                full_topic = f"{topic}.{subtopic}"
                if full_topic not in self.topic_performance:
                    self.topic_performance[full_topic] = TopicPerformance(topic=full_topic)
                
                self.topic_performance[full_topic].record_attempt(is_correct, time_ms)
        
        self.updated_at = datetime.datetime.now()
    
    def complete_session(self) -> SessionStats:
        """
        Complete the current session and update metrics.
        
        Returns:
            Completed session statistics
        """
        if not self.current_session:
            raise ValueError("No active session")
        
        if not self.current_session.end_time:
            self.current_session.complete()
        
        # Update performance metrics
        self.metrics.update_with_session(self.current_session)
        
        # Add to recent sessions, keeping only the last 10
        self.recent_sessions.append(self.current_session)
        if len(self.recent_sessions) > 10:
            self.recent_sessions = self.recent_sessions[-10:]
        
        # Update strengths and weaknesses
        self._update_strengths_weaknesses()
        
        self.updated_at = datetime.datetime.now()
        
        logger.debug(f"Completed session {self.current_session.session_id} for user {self.user_id}")
        return self.current_session
    
    def _update_strengths_weaknesses(self, max_items: int = 5) -> None:
        """
        Update strengths and weaknesses based on topic performance.
        
        Args:
            max_items: Maximum number of items to include
        """
        # Need at least 3 topics to determine strengths and weaknesses
        if len(self.topic_performance) < 3:
            return
        
        # Sort topics by proficiency
        sorted_topics = sorted(
            [tp for tp in self.topic_performance.values() if tp.attempts >= 3],
            key=lambda tp: tp.proficiency,
            reverse=True
        )
        
        # Update strengths (top performers)
        self.metrics.strengths = [tp.topic for tp in sorted_topics[:max_items]]
        
        # Update weaknesses (bottom performers)
        self.metrics.weaknesses = [tp.topic for tp in reversed(sorted_topics[:max_items])]
    
    def get_skill_level(self, topic: Optional[str] = None) -> SkillLevel:
        """
        Get the user's skill level for a topic or overall.
        
        Args:
            topic: Optional topic to get skill level for
            
        Returns:
            Skill level
        """
        if topic and topic in self.topic_performance:
            return self.topic_performance[topic].skill_level
        
        # Calculate overall skill level
        if not self.topic_performance:
            return SkillLevel.BEGINNER
        
        # Average proficiency across topics
        avg_proficiency = sum(
            tp.proficiency for tp in self.topic_performance.values()
        ) / len(self.topic_performance)
        
        return SkillLevel.from_score(avg_proficiency * 100)
    
    def get_proficiency(self, topic: str) -> float:
        """
        Get the user's proficiency in a specific topic.
        
        Args:
            topic: Topic to get proficiency for
            
        Returns:
            Proficiency score (0-1)
        """
        if topic in self.topic_performance:
            return self.topic_performance[topic].proficiency
        return 0.0
    
    def get_recommended_topics(self, count: int = 3) -> List[str]:
        """
        Get recommended topics for the user to focus on.
        
        Args:
            count: Number of topics to recommend
            
        Returns:
            List of recommended topics
        """
        # Prioritize topics with low proficiency but some exposure
        candidates = [
            tp for tp in self.topic_performance.values()
            if tp.attempts > 0 and tp.proficiency < 0.7
        ]
        
        # Sort by proficiency (ascending) and recency (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda tp: (tp.proficiency, -time.mktime(tp.last_seen.timetuple()) if tp.last_seen else 0)
        )
        
        return [tp.topic for tp in sorted_candidates[:count]]
    
    def record_pattern_attempt(self, pattern: str, is_correct: bool, time_ms: float) -> None:
        """
        Record an attempt at a specific pattern.
        
        Args:
            pattern: Pattern identifier
            is_correct: Whether the attempt was correct
            time_ms: Time taken in milliseconds
        """
        if pattern not in self.pattern_metrics:
            self.pattern_metrics[pattern] = {
                "attempts": 0,
                "correct": 0,
                "streak": 0,
                "max_streak": 0,
                "avg_time_ms": 0.0,
                "last_seen": None,
                "mastery": 0.0
            }
        
        metrics = self.pattern_metrics[pattern]
        metrics["attempts"] += 1
        
        # Update time metrics
        if metrics["avg_time_ms"] == 0:
            metrics["avg_time_ms"] = time_ms
        else:
            # Exponential moving average with alpha=0.3
            metrics["avg_time_ms"] = 0.7 * metrics["avg_time_ms"] + 0.3 * time_ms
        
        # Update accuracy metrics
        if is_correct:
            metrics["correct"] += 1
            metrics["streak"] += 1
            if metrics["streak"] > metrics["max_streak"]:
                metrics["max_streak"] = metrics["streak"]
        else:
            metrics["streak"] = 0
        
        # Update last seen timestamp
        metrics["last_seen"] = datetime.datetime.now().isoformat()
        
        # Update mastery (0.0 to 1.0)
        accuracy = metrics["correct"] / metrics["attempts"]
        streak_factor = min(metrics["streak"] / 10, 0.2)
        attempts_factor = min(metrics["attempts"] / 20, 0.2)
        metrics["mastery"] = min(accuracy + streak_factor + attempts_factor, 1.0)
        
        # Update tracker's last update time
        self.updated_at = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "assessment_type": self.assessment_type,
            "topic_performance": {
                topic: tp.to_dict() for topic, tp in self.topic_performance.items()
            },
            "recent_sessions": [
                session.to_dict() for session in self.recent_sessions
            ],
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pattern_metrics": {
                pattern: dict(metrics) for pattern, metrics in self.pattern_metrics.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceTracker':
        """Create from dictionary."""
        topic_performance = {
            topic: TopicPerformance.from_dict(tp_data)
            for topic, tp_data in data.get("topic_performance", {}).items()
        }
        
        recent_sessions = [
            SessionStats.from_dict(session_data)
            for session_data in data.get("recent_sessions", [])
        ]
        
        return cls(
            user_id=data["user_id"],
            assessment_type=data["assessment_type"],
            topic_performance=topic_performance,
            recent_sessions=recent_sessions,
            metrics=PerformanceMetrics.from_dict(data.get("metrics", {})),
            created_at=datetime.datetime.fromisoformat(data["created_at"]) 
                if "created_at" in data else None,
            updated_at=datetime.datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data else None,
            pattern_metrics={
                pattern: dict(metrics) for pattern, metrics in data.get("pattern_metrics", {}).items()
            }
        )


def create_tracker(
    user_id: str,
    assessment_type: str
) -> PerformanceTracker:
    """
    Create a new performance tracker for a user.
    
    Args:
        user_id: User identifier
        assessment_type: Type of assessment
        
    Returns:
        Performance tracker
    """
    return PerformanceTracker(
        user_id=user_id,
        assessment_type=assessment_type
    ) 