"""
SQLAlchemy ORM model for storing aggregated user metrics.
"""
import datetime
from typing import Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Index
)
from sqlalchemy.dialects.postgresql import JSONB
# Use a common Base if defined elsewhere, otherwise define locally
# Assuming a Base is defined and accessible, e.g., from backend.database.base_meta
# If not, you'll need:
# from sqlalchemy.ext.declarative import declarative_base
# Base = declarative_base()
from backend.database.base_meta import Base # Make sure this path is correct


class UserMetrics(Base):
    """
    Stores aggregated performance metrics for a user across assessments.
    """
    __tablename__ = 'user_metrics'

    # Using user_id as the primary key assumes one metrics record per user.
    user_id = Column(String(255), primary_key=True)

    # Overall aggregate stats
    total_sessions_completed = Column(Integer, nullable=False, default=0)
    total_questions_attempted = Column(Integer, nullable=False, default=0)
    total_correct_answers = Column(Integer, nullable=False, default=0)

    # Could store overall accuracy/score, or calculate on the fly when needed
    # overall_accuracy = Column(Float, nullable=True)
    # overall_average_score = Column(Float, nullable=True)

    # Example streak tracking
    current_correct_streak = Column(Integer, nullable=False, default=0)
    longest_correct_streak = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_updated_at = Column(DateTime, default=datetime.datetime.utcnow,
                           onupdate=datetime.datetime.utcnow, nullable=False)

    # Optional field for assessment-specific aggregate data (e.g., accuracy per topic)
    assessment_specific_metrics = Column(JSONB, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_user_metrics_last_updated', last_updated_at),
    )

    def __repr__(self):
        return (f"<UserMetrics(user_id='{self.user_id}', "
                f"total_attempted={self.total_questions_attempted}, "
                f"total_correct={self.total_correct_answers})>")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the model instance to a dictionary."""
        accuracy = 0.0
        if self.total_questions_attempted > 0:
            accuracy = (self.total_correct_answers / self.total_questions_attempted) * 100

        return {
            "user_id": self.user_id,
            "total_sessions_completed": self.total_sessions_completed,
            "total_questions_attempted": self.total_questions_attempted,
            "total_correct_answers": self.total_correct_answers,
            "overall_accuracy": round(accuracy, 2),
            "current_correct_streak": self.current_correct_streak,
            "longest_correct_streak": self.longest_correct_streak,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated_at": self.last_updated_at.isoformat() if self.last_updated_at else None,
            "assessment_specific_metrics": self.assessment_specific_metrics or {}
        }

    @classmethod
    def calculate_accuracy(cls, total_correct: int, total_attempted: int) -> float:
        """Helper method to calculate accuracy."""
        if total_attempted <= 0:
            return 0.0
        return round((total_correct / total_attempted) * 100, 2) 