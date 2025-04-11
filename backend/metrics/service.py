"""
Service layer for managing user metrics.
"""

import logging
from typing import Dict, Any, Optional

from backend.metrics.repository import UserMetricsRepository
from backend.metrics.models import UserMetrics
# Import assessment session model to get performance data
from backend.assessments.candlestick_patterns.candlestick_models import CandlestickSession
from backend.assessments.base.models import SessionStatus
from backend.common.logger import get_logger

logger = get_logger(__name__)

class UserMetricsService:
    """
    Provides business logic for calculating and updating user aggregate metrics.
    """

    def __init__(self, metrics_repository: UserMetricsRepository):
        """
        Initialize the service with the metrics repository.
        
        Args:
            metrics_repository: An instance of UserMetricsRepository.
        """
        self._repository = metrics_repository
        logger.info("Initialized UserMetricsService")

    async def get_user_metrics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest metrics for a user.

        Args:
            user_id: The ID of the user.

        Returns:
            A dictionary representation of the user's metrics, or None if not found.
        """
        try:
            metrics = await self._repository.get_metrics(user_id)
            if metrics:
                return metrics.to_dict()
            else:
                # Optionally create metrics if they don't exist upon fetch
                # Or just return None/empty dict
                logger.warning(f"No metrics found for user {user_id} during get request.")
                # metrics = await self._repository.get_or_create_metrics(user_id)
                # return metrics.to_dict()
                return None
        except Exception as e:
            logger.error(f"Error retrieving metrics for user {user_id}: {e}", exc_info=True)
            # Depending on API design, might return None or raise a service-level exception
            return None 

    async def update_metrics_from_session(self, session: CandlestickSession) -> bool:
        """
        Updates a user's aggregate metrics based on a completed assessment session.

        Args:
            session: The completed CandlestickSession domain model instance.

        Returns:
            True if the metrics were successfully updated, False otherwise.
        """
        if not session or not session.user_id or session.status != SessionStatus.COMPLETED:
            logger.warning(f"Skipping metrics update: Session invalid or not completed (ID: {session.id if session else 'N/A'})")
            return False

        try:
            # 1. Get or create the user's current metrics record
            user_metrics = await self._repository.get_or_create_metrics(session.user_id)

            # 2. Calculate performance metrics from the session
            session_performance = session.get_performance() # Assumes this method exists and works

            # 3. Update the aggregate metrics
            user_metrics.total_sessions_completed += 1
            user_metrics.total_questions_attempted += session_performance.total_answered
            user_metrics.total_correct_answers += session_performance.total_correct
            
            # Update streaks (simple example based on last answer - more complex logic possible)
            # This requires knowing the result of the *very last* answer in the session.
            # The current CandlestickSession.get_performance doesn't directly provide this.
            # Let's assume a simpler update for now:
            # If accuracy was 100%, increment streak, otherwise reset.
            # A more robust implementation would track attempt order.
            if session_performance.total_answered > 0:
                 if session_performance.accuracy >= 100.0:
                      user_metrics.current_correct_streak += session_performance.total_answered
                 else:
                      # Need logic to determine if the *last* answer was correct to continue a streak
                      # Resetting if any were wrong in the session is one approach, but not ideal.
                      # Placeholder: Reset streak if session accuracy < 100%
                      user_metrics.current_correct_streak = 0 
            
            # Update longest streak
            user_metrics.longest_correct_streak = max(
                user_metrics.longest_correct_streak, 
                user_metrics.current_correct_streak
            )

            # 4. Update assessment-specific metrics (example: topic accuracy)
            # This requires getting topic performance for the *session*, not overall.
            # The current CandlestickSession model might not store this granularity.
            # Placeholder: Store overall session accuracy for 'candlestick' assessment.
            if user_metrics.assessment_specific_metrics is None:
                user_metrics.assessment_specific_metrics = {}
            
            user_metrics.assessment_specific_metrics[session.assessment_type] = {
                "last_session_id": session.id,
                "last_session_accuracy": session_performance.accuracy,
                "last_session_score": session_performance.total_score,
                "last_session_completed_at": session.completed_at.isoformat() if session.completed_at else None
                # Add more assessment-specific aggregates if needed (e.g., topic breakdowns)
            }

            # 5. Save the updated metrics
            success = await self._repository.update_metrics(user_metrics)
            if success:
                logger.info(f"Successfully updated metrics for user {session.user_id} from session {session.id}")
            else:
                # Should not happen if merge is used correctly, but log anyway
                 logger.error(f"Metrics update reported failure for user {session.user_id} from session {session.id}")
            return success

        except Exception as e:
            logger.error(f"Failed to update metrics for user {session.user_id} from session {session.id}: {e}", exc_info=True)
            return False

# Potential additions:
# - Method to calculate/update assessment_specific_metrics (e.g., topic accuracy)
# - Methods for resetting metrics or handling specific updates 