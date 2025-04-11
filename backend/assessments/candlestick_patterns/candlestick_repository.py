# backend/assessments/candlestick_patterns/candlestick_repository.py

import logging
from datetime import datetime, timedelta
import os # Keep for potential config paths, but remove file IO
import json
# Remove json import - no longer needed for content mapping
# import json 
# Removed functools.lru_cache as it's not compatible with async
from collections import defaultdict
import asyncio # Added for gather in recommendations

# Async SQLAlchemy imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete as sql_delete, update, func, and_, or_, distinct, cast as sql_cast
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.sql import select
from sqlalchemy import and_, or_, desc

# Import types
from typing import Dict, List, Any, Optional, Set, Tuple, cast, TypeVar

from backend.assessments.base.repositories import (
    QuestionRepository,
    SessionRepository,
    AssessmentRepository
)
from backend.assessments.base.models import (
    AssessmentType,
    SessionStatus,
    QuestionDifficulty # Import for mapping
)
# Import the ORM models defined for the database
from backend.assessments.candlestick_patterns.database_models import (
    CandlestickQuestion as CandlestickQuestionORM, # Alias ORM model
    CandlestickSession as CandlestickSessionORM,   # Alias ORM model
    CandlestickAttempt as CandlestickAttemptORM    # Import Attempt ORM for joins
)
# Import the domain models used by the service layer
from backend.assessments.candlestick_patterns.candlestick_models import (
    CandlestickQuestion as CandlestickQuestionDomain, # Alias Domain model
    CandlestickSession as CandlestickSessionDomain   # Alias Domain model
)
from backend.common.logger import get_logger
from backend.database.init_db import get_engine

# Set up logger
logger = get_logger(__name__)

DB_BATCH_SIZE = 50

# Define Custom Repository Exception
class RepositoryError(Exception):
    """ Custom exception for repository layer errors. """
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
        logger.error(f"RepositoryError: {message}" + (f" - Original: {original_exception}" if original_exception else ""))

# Define Generic Types for Repositories
Q_Domain = TypeVar('Q_Domain', bound=CandlestickQuestionDomain)
S_Domain = TypeVar('S_Domain', bound=CandlestickSessionDomain)
Q_ORM = TypeVar('Q_ORM', bound=CandlestickQuestionORM)
S_ORM = TypeVar('S_ORM', bound=CandlestickSessionORM)

class CandlestickQuestionRepository(QuestionRepository[Q_Domain]):
    """
    Repository implementation for candlestick pattern questions using SQLAlchemy Async.

    This class extends the base QuestionRepository interface and provides
    async methods for storing and retrieving candlestick pattern questions from the database.
    """

    def __init__(self):
        """
        Initialize the repository.
        """
        self._session_factory = None
        logger.info("Initialized CandlestickQuestionRepository")

    @property
    def async_session(self):
        """Get the async session factory, creating it if necessary."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                get_engine(),
                class_=AsyncSession,
                expire_on_commit=False
            )
        return self._session_factory

    # --- Base Interface Properties ---
    @property
    def domain_type(self) -> str:
        return "candlestick_question"

    @property
    def table_name(self) -> str:
        return CandlestickQuestionORM.__tablename__

    # --- Internal Helper Methods (Mapping between Domain and ORM) ---
    def _map_orm_to_domain(self, orm_question: Q_ORM) -> Q_Domain:
        """ Maps SQLAlchemy ORM object to Domain object. """
        try:
            # Parse content JSON
            content_dict = {}
            if orm_question.content:
                try:
                    if isinstance(orm_question.content, str):
                        content_dict = json.loads(orm_question.content)
                    else:
                        content_dict = orm_question.content
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse content JSON for question {orm_question.question_id}: {e}")
                    content_dict = {}

            # Parse topics JSON
            topics_list = []
            if orm_question.topics:
                try:
                    if isinstance(orm_question.topics, str):
                        topics_list = json.loads(orm_question.topics)
                    else:
                        topics_list = orm_question.topics
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse topics JSON for question {orm_question.question_id}: {e}")
                    topics_list = []

            # Create domain object with all required fields
            return Q_Domain(
                id=orm_question.question_id,
                difficulty=QuestionDifficulty(orm_question.difficulty),
                pattern=orm_question.pattern_type,
                question_text=content_dict.get("text", "Identify the candlestick pattern:"),
                options=content_dict.get("options", []),
                chart_data=content_dict.get("chart_data", {}),
                chart_image=content_dict.get("chart_image"),
                timeframe=content_dict.get("timeframe"),
                symbol=content_dict.get("symbol"),
                topics=topics_list,
                metadata=content_dict.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"Error mapping ORM->Domain for question {orm_question.question_id}: {e}", exc_info=True)
            raise RepositoryError(f"Failed to map question {orm_question.question_id} from ORM", original_exception=e)

    def _map_domain_to_orm(self, domain_question: Q_Domain) -> Q_ORM:
        """ Maps Domain object to SQLAlchemy ORM object. """
        try:
            # Ensure topics is a list
            topics = domain_question.topics if domain_question.topics else []
            if not topics:
                # Default topic is the pattern type
                topics = [domain_question.pattern.lower().replace(" ", "_")]

            # Create content dictionary with all required fields
            # Try to use the content property if it exists, otherwise build from individual attributes
            try:
                if hasattr(domain_question, 'content'):
                    content_dict = domain_question.content
                else:
                    content_dict = {
                        "text": domain_question.question_text,
                        "options": domain_question.options,
                        "chart_data": domain_question.chart_data,
                        "chart_image": domain_question.chart_image,
                        "timeframe": domain_question.timeframe,
                        "symbol": domain_question.symbol,
                        "metadata": domain_question.metadata
                    }
            except Exception as content_err:
                logger.error(f"Error creating content dict for question {domain_question.id}: {content_err}")
                # Fallback with minimal content
                content_dict = {
                    "text": getattr(domain_question, 'question_text', "Identify the candlestick pattern"),
                    "options": getattr(domain_question, 'options', []),
                    "metadata": getattr(domain_question, 'metadata', {})
                }

            # Create ORM object with properly serialized JSON strings
            return Q_ORM(
                question_id=domain_question.id,
                difficulty=domain_question.difficulty.value,
                pattern_type=domain_question.pattern,
                content=json.dumps(content_dict),  # Serialize to JSON string for Text column
                topics=json.dumps(topics)  # Serialize to JSON string for Text column
            )
        except Exception as e:
            logger.error(f"Error mapping Domain->ORM for question {domain_question.id}: {e}", exc_info=True)
            raise RepositoryError(f"Failed to map question {domain_question.id} to ORM", original_exception=e)

    # --- Base Interface Method Implementations (Async with DB) ---
    async def get_by_id(self, question_id: str) -> Optional[Q_Domain]:
        """ Get a question by ID from the database. """
        if not question_id:
            logger.warning("Attempted to get question with empty ID")
            return None # Or raise RepositoryError based on requirements
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(CandlestickQuestionORM).where(CandlestickQuestionORM.question_id == question_id)
                )
                orm_question = result.scalar_one_or_none()
                if orm_question:
                     # Mapping errors are now caught inside _map_orm_to_domain and raise RepositoryError
                     return self._map_orm_to_domain(orm_question)
                return None # Return None if not found
        except RepositoryError:
            # Re-raise RepositoryError if it originated from mapping
            raise
        except Exception as e:
            # Raise specific RepositoryError for database/unexpected issues
            raise RepositoryError(f"Database error retrieving question {question_id}", original_exception=e)

    async def save(self, question: Q_Domain) -> Optional[Q_Domain]:
        """Save a question to the database."""
        if not question:
            logger.warning("Attempted to save None question")
            return None

        try:
            # Create an ORM model from the domain model
            topics = question.topics if question.topics else []
            if not topics:
                topics = [question.pattern.lower().replace(" ", "_")]

            # Build content dictionary
            try:
                if hasattr(question, 'content'):
                    content_dict = question.content
                else:
                    content_dict = {
                        "text": question.question_text,
                        "options": question.options,
                        "chart_data": question.chart_data,
                        "chart_image": question.chart_image,
                        "timeframe": question.timeframe,
                        "symbol": question.symbol,
                        "metadata": question.metadata
                    }
            except Exception as content_err:
                logger.error(f"Error creating content dict for question {question.id}: {content_err}")
                content_dict = {
                    "text": getattr(question, 'question_text', "Identify the candlestick pattern"),
                    "options": getattr(question, 'options', []),
                    "metadata": getattr(question, 'metadata', {})
                }

            # Create a fresh ORM instance directly - don't try to convert domain model to ORM
            orm_question = CandlestickQuestionORM(
                question_id=question.id,
                difficulty=question.difficulty.value,
                pattern_type=question.pattern,
                content=json.dumps(content_dict),
                topics=json.dumps(topics)
            )
            
            async with self.async_session() as session:
                async with session.begin():
                    try:
                        # Try to get existing question
                        result = await session.execute(
                            select(CandlestickQuestionORM).where(
                                CandlestickQuestionORM.question_id == orm_question.question_id
                            )
                        )
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            # Update existing question
                            existing.difficulty = orm_question.difficulty
                            existing.pattern_type = orm_question.pattern_type
                            existing.content = orm_question.content
                            existing.topics = orm_question.topics
                            existing.updated_at = datetime.datetime.utcnow()
                            # Use the existing instance
                            orm_to_return = existing
                        else:
                            # Add new question - ensure we're using the ORM model
                            session.add(orm_question)
                            orm_to_return = orm_question
                        
                        # Flush to ensure we have the updated/inserted data
                        await session.flush()
                        
                        # Map back to domain object and return it
                        return self._map_orm_to_domain(orm_to_return)
                    except Exception as db_error:
                        logger.error(f"Database error in candlestick_question repository: {str(db_error)}")
                        raise RepositoryError(f"Database error: {str(db_error)}", original_exception=db_error)
                    
        except RepositoryError:
            # Re-raise RepositoryError if it originated from mapping or database
            raise
        except Exception as e:
            # Log and raise repository error for other exceptions
            logger.error(f"Error saving question {question.id}: {str(e)}", exc_info=True)
            raise RepositoryError(f"Failed to save question {question.id}", original_exception=e)

    async def delete(self, question_id: str) -> bool:
        """ Delete a question by ID from the database. """
        if not question_id:
            raise RepositoryError("Question ID cannot be empty for deletion")
        try:
             async with self.async_session() as session:
                  async with session.begin():
                       result = await session.execute(
                           sql_delete(CandlestickQuestionORM).where(CandlestickQuestionORM.question_id == question_id)
                       )
                       deleted_count = result.rowcount
                  logger.debug(f"Deleted {deleted_count} question(s) with ID {question_id}")
                  return deleted_count > 0
        except Exception as e:
             logger.error(f"Database error deleting question {question_id}: {e}", exc_info=True)
             raise RepositoryError(f"Database error deleting question {question_id}", original_exception=e)


    async def find_by_difficulty(self, difficulty: str, limit: int = 10, offset: int = 0) -> List[Q_Domain]:
        """ Find questions by difficulty level from the database. """
        try:
             async with self.async_session() as session:
                  stmt = select(CandlestickQuestionORM)\
                      .where(CandlestickQuestionORM.difficulty == difficulty)\
                      .order_by(CandlestickQuestionORM.question_id) \
                      .limit(limit)\
                      .offset(offset)
                  result = await session.execute(stmt)
                  orm_questions = result.scalars().all()
                  # Map results, handling potential mapping errors
                  domain_questions = []
                  for q_orm in orm_questions:
                      try:
                           domain_questions.append(self._map_orm_to_domain(q_orm))
                      except RepositoryError as map_error:
                           logger.error(f"Skipping question {q_orm.question_id} in find_by_difficulty due to mapping error: {map_error}")
                  return domain_questions
        except RepositoryError:
              raise # Re-raise mapping errors
        except Exception as e:
             logger.error(f"Error finding questions by difficulty {difficulty}: {e}", exc_info=True)
             raise RepositoryError(f"Database error finding questions by difficulty {difficulty}", original_exception=e)


    async def find_by_topics(
        self,
        topics: List[str],
        limit: int = 10,
        offset: int = 0
    ) -> List[Q_Domain]:
        """
        Find questions by topic(s) using JSONB @> operator for efficiency.
        Assumes 'topics' is a JSON array within the 'content' JSONB column.
        Example: content = {"topics": ["topic1", "topic2"], ...}
        """
        if not topics:
            return []

        try:
            async with self.async_session() as session:
                 # Use JSONB operators to check if the 'topics' array in the content column
                 # contains *any* of the provided topics. Use @> for contains.
                 # Need to cast the Python list to a JSONB array for the query.
                 # Note: @> checks if left JSON contains right JSON.
                 # To check if content['topics'] contains any element from the input topics list,
                 # we can use the `?|` (exists any) operator with a text array.
                 stmt = (
                     select(CandlestickQuestionORM)
                     .where(
                         # Access the 'topics' key within the JSONB 'content' column
                         # and check if it has any elements in common with the input 'topics' list.
                         # Requires casting input list to ARRAY(Text) for the ?| operator.
                         CandlestickQuestionORM.content[ 'topics' ].astext.cast(ARRAY(String)).op('?|')(topics)
                         # Alternative: Check if content['topics'] contains ALL provided topics (@>)
                         # CandlestickQuestionORM.content['topics'].as_jsonb().contains(sql_cast(topics, JSONB))
                     )
                     .order_by(CandlestickQuestionORM.question_id) # Or func.random()
                     .limit(limit)
                     .offset(offset)
                 )
                 result = await session.execute(stmt)
                 orm_questions = result.scalars().all()

                 # Map results, handling potential mapping errors
                 domain_questions = []
                 for q_orm in orm_questions:
                     try:
                         domain_questions.append(self._map_orm_to_domain(q_orm))
                     except RepositoryError as map_error:
                         logger.error(f"Skipping question {q_orm.question_id} in find_by_topics due to mapping error: {map_error}")
                     except Exception as e_map:
                          logger.error(f"Unexpected mapping error for question {q_orm.question_id} in find_by_topics: {e_map}")
                 return domain_questions

        except RepositoryError:
            raise # Re-raise mapping errors
        except Exception as e:
             logger.error(f"Database error finding questions by topics {topics}: {e}", exc_info=True)
             raise RepositoryError(f"Database error finding questions by topics {topics}", original_exception=e)

    async def find_by_criteria(self, criteria: Dict[str, Any], limit: int = 10, offset: int = 0) -> List[Q_Domain]:
        """ Find questions based on multiple criteria (difficulty, topics, exclude_ids). """
        try:
             async with self.async_session() as session:
                  stmt = select(CandlestickQuestionORM)
                  filters = []

                  if 'difficulty' in criteria:
                       filters.append(CandlestickQuestionORM.difficulty == criteria['difficulty'])
                  
                  if 'topics' in criteria and criteria['topics']:
                       # Use JSONB operator (?|) for topic matching
                       topics = criteria['topics']
                       filters.append(
                            CandlestickQuestionORM.content[ 'topics' ].astext.cast(ARRAY(String)).op('?|')(topics)
                       )
                       
                  if 'exclude_ids' in criteria and criteria['exclude_ids']:
                       filters.append(CandlestickQuestionORM.question_id.notin_(criteria['exclude_ids']))

                  if filters:
                       stmt = stmt.where(and_(*filters))

                  stmt = stmt.order_by(CandlestickQuestionORM.question_id).limit(limit).offset(offset)
                  result = await session.execute(stmt)
                  orm_questions = result.scalars().all()
                  
                  domain_questions = []
                  for q_orm in orm_questions:
                      try:
                           domain_questions.append(self._map_orm_to_domain(q_orm))
                      except RepositoryError as map_error:
                           logger.error(f"Skipping question {q_orm.question_id} in find_by_criteria due to mapping error: {map_error}")
                      except Exception as e_map:
                           logger.error(f"Unexpected mapping error for question {q_orm.question_id} in find_by_criteria: {e_map}")
                  return domain_questions
        except RepositoryError:
            raise # Re-raise mapping errors
        except Exception as e:
             logger.error(f"Database error finding questions by criteria {criteria}: {e}", exc_info=True)
             raise RepositoryError(f"Database error finding questions by criteria", original_exception=e)

    async def count_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """ Count questions based on multiple criteria. """
        try:
            async with self.async_session() as session:
                 stmt = select(func.count(distinct(CandlestickQuestionORM.question_id)))
                 filters = []
                 if 'difficulty' in criteria:
                      filters.append(CandlestickQuestionORM.difficulty == criteria['difficulty'])
                 if 'topics' in criteria and criteria['topics']:
                      topics = criteria['topics']
                      filters.append(
                           CandlestickQuestionORM.content[ 'topics' ].astext.cast(ARRAY(String)).op('?|')(topics)
                      )
                 if 'exclude_ids' in criteria and criteria['exclude_ids']:
                      filters.append(CandlestickQuestionORM.question_id.notin_(criteria['exclude_ids']))

                 if filters:
                      stmt = stmt.where(and_(*filters))

                 result = await session.execute(stmt)
                 count = result.scalar_one_or_none() or 0
                 return count
        except Exception as e:
            logger.error(f"Database error counting questions by criteria {criteria}: {e}", exc_info=True)
            raise RepositoryError(f"Database error counting questions by criteria", original_exception=e)


class CandlestickSessionRepository(SessionRepository[S_Domain]):
    """
    Repository implementation for candlestick pattern assessment sessions using SQLAlchemy Async.
    """

    def __init__(self):
        """
        Initialize the repository.
        """
        self._session_factory = None
        logger.info("Initialized CandlestickSessionRepository")

    @property
    def async_session(self):
        """Get the async session factory, creating it if necessary."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                get_engine(),
                class_=AsyncSession,
                expire_on_commit=False
            )
        return self._session_factory

    # --- Base Interface Properties ---
    @property
    def domain_type(self) -> str:
        return "candlestick_session"

    @property
    def table_name(self) -> str:
        return CandlestickSessionORM.__tablename__

    # --- Internal Helper Methods (Mapping between Domain and ORM) ---
    def _map_orm_to_domain(self, orm_session: S_ORM) -> S_Domain:
         """ Maps SQLAlchemy ORM object to Domain object. """
         session_data = {}
         if orm_session.data:
              try:
                  session_data = json.loads(orm_session.data)
              except json.JSONDecodeError:
                   logger.warning(f"Failed to decode JSON data for session {orm_session.session_id}")
                   pass # Continue with empty data

         try:
            # Create domain model instance, converting types as needed
            domain_instance = CandlestickSessionDomain(
                    id=orm_session.session_id,
                    user_id=orm_session.user_id,
                    assessment_type=AssessmentType(orm_session.assessment_type),
                    created_at=orm_session.created_at,
                    completed_at=orm_session.completed_at,
                    status=SessionStatus(orm_session.status), # Use the dedicated status column
                    # Map other fields from session_data
                    questions=session_data.get('questions', []),
                    answers=session_data.get('answers', {}), # Ensure answers structure matches domain
                    current_question_index=int(session_data.get('current_question_index', 0)),
                    settings=session_data.get('settings', {}),
                    metadata=session_data.get('metadata', {}),
                    patterns_identified=session_data.get('patterns_identified', []),
                    streak=int(session_data.get('streak', 0)),
                    average_response_time_ms=float(session_data.get('average_response_time_ms', 0.0))
                    # Add score/max_score if stored in JSON or needs calculation from answers
                    # score=float(session_data.get('score', 0.0)),
                    # max_score=int(session_data.get('max_score', 0))
            )
            # Update derived fields if necessary (e.g., is_completed based on status)
            domain_instance.is_completed = (domain_instance.status == SessionStatus.COMPLETED)
            # Recalculate score if needed, based on answers? Or trust stored value?
            # domain_instance.score = sum(ans.evaluation.score for ans in domain_instance.answers.values() if ans and ans.evaluation)

            return domain_instance
         except (ValueError, KeyError, TypeError) as e:
             raise RepositoryError(f"Data mapping ORM->Domain error for session {orm_session.session_id}: {e}", original_exception=e)
         except Exception as e:
             logger.error(f"Unexpected mapping ORM->Domain Error (Session {orm_session.session_id}): {e}", exc_info=True)
             raise RepositoryError(f"Unexpected error mapping session {orm_session.session_id} from ORM", original_exception=e)

    def _map_domain_to_orm(self, domain_session: S_Domain) -> S_ORM:
         """ Maps Domain object to SQLAlchemy ORM object. """
         # Ensure answers dict keys/values are JSON serializable
         serializable_answers = {}
         if domain_session.answers:
            for qid, answer_obj in domain_session.answers.items():
                if hasattr(answer_obj, 'to_dict'): # If it's an object like UserAnswer
                     try:
                          serializable_answers[qid] = answer_obj.to_dict()
                     except Exception as ser_err:
                          logger.error(f"Failed to serialize answer object for qid {qid} in session {domain_session.id}: {ser_err}")
                          serializable_answers[qid] = {"error": "Serialization failed", "value": str(answer_obj)}
                elif isinstance(answer_obj, dict): # If it's already a dict
                     serializable_answers[qid] = answer_obj
                else: # Fallback for simple values? Review required structure.
                     serializable_answers[qid] = str(answer_obj)

         session_data = {
            'questions': domain_session.questions, # Assuming list of IDs or simple data
            'answers': serializable_answers,
            'current_question_index': domain_session.current_question_index,
            'settings': domain_session.settings,
            'metadata': domain_session.metadata,
            'patterns_identified': domain_session.patterns_identified,
            'streak': domain_session.streak,
            'average_response_time_ms': domain_session.average_response_time_ms
            # Add score/max_score if they should be persisted in JSON
            # 'score': domain_session.score,
            # 'max_score': domain_session.max_score
         }
         try:
            data_json = json.dumps(session_data)
         except TypeError as e:
              raise RepositoryError(f"Failed to serialize session data Domain->ORM for {domain_session.id}", original_exception=e)

         try:
            # Create ORM instance
            orm_instance = CandlestickSessionORM(
                session_id=domain_session.id,
                user_id=domain_session.user_id,
                assessment_type=domain_session.assessment_type.value,
                created_at=domain_session.created_at,
                completed_at=domain_session.completed_at, # This should be updated when status changes to COMPLETED
                status=domain_session.status.value, # Set the dedicated status column
                data=data_json
            )
            return orm_instance
         except Exception as e:
              logger.error(f"ORM instance creation error for session {domain_session.id}: {e}")
              raise RepositoryError(f"Failed to create ORM instance for session {domain_session.id}", original_exception=e)

    # --- Base Interface Method Implementations (Now Async with DB) ---
    async def get_by_id(self, session_id: str) -> Optional[S_Domain]:
        """ Get a session by ID from the database. """
        if not session_id:
            logger.warning("Attempted to get session with empty ID")
            return None # Or raise RepositoryError
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(CandlestickSessionORM).where(CandlestickSessionORM.session_id == session_id)
                )
                orm_session = result.scalar_one_or_none()
                if orm_session:
                    # Mapping errors handled within _map_orm_to_domain
                    return self._map_orm_to_domain(orm_session)
                return None # Not found
        except RepositoryError:
            raise # Re-raise mapping errors
        except Exception as e:
            raise RepositoryError(f"Database error retrieving session {session_id}", original_exception=e)

    async def save(self, domain_session_obj: S_Domain) -> Optional[S_Domain]:
        """ Save a session to the database (insert or update). """
        if not domain_session_obj or not domain_session_obj.id:
             raise RepositoryError("Invalid session object provided for saving")

        # Update completed_at if status is COMPLETED and it's not set
        if domain_session_obj.status == SessionStatus.COMPLETED and not domain_session_obj.completed_at:
            domain_session_obj.completed_at = datetime.utcnow()

        try:
            # Mapping errors handled within _map_domain_to_orm
            orm_session = self._map_domain_to_orm(domain_session_obj)
        except RepositoryError:
             raise # Re-raise mapping errors

        try:
            async with self.async_session() as db_session:
                 async with db_session.begin():
                    # Use merge to handle both insert and update
                    merged_orm = await db_session.merge(orm_session)
                    # Refresh to get any database-generated values
                    await db_session.refresh(merged_orm)
                    # Map back to domain object
                    return self._map_orm_to_domain(merged_orm)
        except Exception as e:
            logger.error(f"Error saving session {domain_session_obj.id}: {e}", exc_info=True)
            # Rollback handled by session.begin()
            raise RepositoryError(f"Database error saving session {domain_session_obj.id}", original_exception=e)

    async def delete(self, session_id: str) -> bool:
        """ Delete a session from the database. """
        if not session_id:
            raise RepositoryError("Empty session ID provided for deletion")
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sql_delete(CandlestickSessionORM).where(CandlestickSessionORM.session_id == session_id)
                    )
                    deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.debug(f"Deleted session {session_id} from database.")
                    return True
                else:
                    logger.warning(f"Attempted to delete non-existent session {session_id}")
                    return False # Not found
        except Exception as e:
             logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
             # Rollback handled automatically
             raise RepositoryError(f"Database error deleting session {session_id}", original_exception=e)

    async def find_by_user_id(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None # Expecting SessionStatus enum value string
    ) -> List[S_Domain]:
        """ Find sessions by user ID and optional status from the database. """
        if not user_id: return []
        try:
            async with self.async_session() as session:
                conditions = [CandlestickSessionORM.user_id == user_id]
                if status:
                    try:
                        # Validate status against enum before querying
                        SessionStatus(status)
                        conditions.append(CandlestickSessionORM.status == status)
                    except ValueError:
                         logger.warning(f"Invalid status value '{status}' provided to find_by_user_id, ignoring status filter.")

                stmt = select(CandlestickSessionORM)\
                           .where(and_(*conditions))\
                           .order_by(CandlestickSessionORM.created_at.desc())\
                           .limit(limit)\
                           .offset(offset)

                result = await session.execute(stmt)
                orm_sessions = result.scalars().all()
                # Map results, handling potential mapping errors
                domain_sessions = []
                for orm_s in orm_sessions:
                    try:
                         domain_sessions.append(self._map_orm_to_domain(orm_s))
                    except RepositoryError as map_error:
                         logger.error(f"Skipping session {orm_s.session_id} in find_by_user_id due to mapping error: {map_error}")
                return domain_sessions
        except RepositoryError:
             raise # Re-raise mapping errors
        except Exception as e:
             logger.error(f"Error finding sessions by user '{user_id}': {e}", exc_info=True)
             raise RepositoryError(f"Database error finding sessions for user '{user_id}'", original_exception=e)

    async def find_by_date_range(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10,
        offset: int = 0
    ) -> List[S_Domain]:
         """ Find sessions for a user within a date range from the database. """
         if not user_id: return []
         try:
            async with self.async_session() as session:
                 stmt = select(CandlestickSessionORM)\
                    .where(CandlestickSessionORM.user_id == user_id)\
                    .where(CandlestickSessionORM.created_at >= start_date)\
                    .where(CandlestickSessionORM.created_at <= end_date)\
                    .order_by(CandlestickSessionORM.created_at.desc())\
                    .limit(limit)\
                    .offset(offset)
                 result = await session.execute(stmt)
                 orm_sessions = result.scalars().all()
                 # Map results, handling potential mapping errors
                 domain_sessions = []
                 for orm_s in orm_sessions:
                     try:
                          domain_sessions.append(self._map_orm_to_domain(orm_s))
                     except RepositoryError as map_error:
                          logger.error(f"Skipping session {orm_s.session_id} in find_by_date_range due to mapping error: {map_error}")
                 return domain_sessions
         except RepositoryError:
              raise # Re-raise mapping errors
         except Exception as e:
             logger.error(f"Error finding sessions by date range for user '{user_id}': {e}", exc_info=True)
             raise RepositoryError(f"Database error finding sessions by date range for user '{user_id}'", original_exception=e)

    async def find_by_criteria(
        self,
        criteria: Dict[str, Any],
        limit: int = 10,
        offset: int = 0
    ) -> List[S_Domain]:
        """ Find sessions matching the specified criteria from the database. """
        try:
            async with self.async_session() as session:
                stmt = select(CandlestickSessionORM)
                conditions = []
                if "user_id" in criteria:
                    conditions.append(CandlestickSessionORM.user_id == criteria["user_id"])
                if "status" in criteria:
                     try:
                         SessionStatus(criteria["status"]) # Validate
                         conditions.append(CandlestickSessionORM.status == criteria["status"])
                     except ValueError:
                         logger.warning(f"Invalid status '{criteria['status']}' in criteria, ignoring.")
                if "assessment_type" in criteria:
                     try:
                         AssessmentType(criteria["assessment_type"]) # Validate
                         conditions.append(CandlestickSessionORM.assessment_type == criteria["assessment_type"])
                     except ValueError:
                         logger.warning(f"Invalid assessment_type '{criteria['assessment_type']}' in criteria, ignoring.")
                if "start_date" in criteria:
                     conditions.append(CandlestickSessionORM.created_at >= criteria["start_date"])
                if "end_date" in criteria:
                     conditions.append(CandlestickSessionORM.created_at <= criteria["end_date"])

                if conditions:
                    stmt = stmt.where(and_(*conditions))

                stmt = stmt.order_by(CandlestickSessionORM.created_at.desc())\
                           .limit(limit)\
                           .offset(offset)
                result = await session.execute(stmt)
                orm_sessions = result.scalars().all()
                # Map results, handling potential mapping errors
                domain_sessions = []
                for orm_s in orm_sessions:
                    try:
                         domain_sessions.append(self._map_orm_to_domain(orm_s))
                    except RepositoryError as map_error:
                         logger.error(f"Skipping session {orm_s.session_id} in find_by_criteria due to mapping error: {map_error}")
                return domain_sessions
        except RepositoryError:
             raise # Re-raise mapping errors
        except Exception as e:
            logger.error(f"Error finding sessions by criteria {criteria}: {e}", exc_info=True)
            raise RepositoryError(f"Database error finding sessions by criteria {criteria}", original_exception=e)

    async def count_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """ Count sessions matching the specified criteria in the database. """
        try:
            async with self.async_session() as session:
                stmt = select(func.count(CandlestickSessionORM.session_id))
                conditions = []
                # Add conditions similar to find_by_criteria, including validation
                if "user_id" in criteria:
                     conditions.append(CandlestickSessionORM.user_id == criteria["user_id"])
                if "status" in criteria:
                    try:
                        SessionStatus(criteria["status"])
                        conditions.append(CandlestickSessionORM.status == criteria["status"])
                    except ValueError:
                        logger.warning(f"Invalid status '{criteria['status']}' in count_by_criteria, ignoring.")
                if "assessment_type" in criteria:
                    try:
                        AssessmentType(criteria["assessment_type"])
                        conditions.append(CandlestickSessionORM.assessment_type == criteria["assessment_type"])
                    except ValueError:
                        logger.warning(f"Invalid assessment_type '{criteria['assessment_type']}' in count_by_criteria, ignoring.")
                if "start_date" in criteria:
                     conditions.append(CandlestickSessionORM.created_at >= criteria["start_date"])
                if "end_date" in criteria:
                     conditions.append(CandlestickSessionORM.created_at <= criteria["end_date"])

                if conditions:
                    stmt = stmt.where(and_(*conditions))

                count_result = await session.execute(stmt)
                return count_result.scalar_one_or_none() or 0
        except Exception as e:
            logger.error(f"Error counting sessions by criteria {criteria}: {e}", exc_info=True)
            raise RepositoryError(f"Database error counting sessions by criteria {criteria}", original_exception=e)

    async def get_latest_session(self, user_id: str) -> Optional[S_Domain]:
        """ Get the most recent session for a user from the database. """
        if not user_id: return None
        try:
            async with self.async_session() as session:
                 stmt = select(CandlestickSessionORM)\
                    .where(CandlestickSessionORM.user_id == user_id)\
                    .order_by(CandlestickSessionORM.created_at.desc())\
                    .limit(1)
                 result = await session.execute(stmt)
                 orm_session = result.scalar_one_or_none()
                 if orm_session:
                    # Mapping errors handled within _map_orm_to_domain
                    return self._map_orm_to_domain(orm_session)
                 return None # Not found
        except RepositoryError:
             raise # Re-raise mapping errors
        except Exception as e:
             logger.error(f"Error getting latest session for user '{user_id}': {e}", exc_info=True)
             raise RepositoryError(f"Database error getting latest session for user '{user_id}'", original_exception=e)

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """ Get aggregated statistics for a user's sessions from the database. """
        logger.debug(f"Getting basic user stats for {user_id}")
        try:
            total_count = await self.count_by_criteria({'user_id': user_id})
            completed_count = await self.count_by_criteria({'user_id': user_id, 'status': SessionStatus.COMPLETED.value})
            # Avg score calculation moved to get_user_performance in the aggregate repo
            avg_score = 0.0 # Placeholder for basic stats
            return {
                 "user_id": user_id,
                 "total_sessions": total_count,
                 "completed_sessions": completed_count,
                 "average_score": avg_score, # Keep placeholder or remove if not calculated here
            }
        except RepositoryError as e:
             # Log and re-raise specific repository errors
             logger.error(f"Repository error getting basic user stats for '{user_id}': {e.original_exception}")
             raise RepositoryError(f"Failed to retrieve basic stats for user '{user_id}'", original_exception=e)
        except Exception as e:
             logger.error(f"Unexpected error getting basic user stats for '{user_id}': {e}", exc_info=True)
             # Raise RepositoryError for unexpected issues
             raise RepositoryError(f"Unexpected error retrieving basic stats for user '{user_id}'", original_exception=e)


# Consolidated Assessment Repository
class CandlestickAssessmentRepositoryImpl(AssessmentRepository):
    """
    Implementation of the aggregate AssessmentRepository interface using Async DB.
    Provides access to both question and session repositories for candlesticks.
    """

    # Use dependency injection
    def __init__(self):
        """ Initialize the aggregate repository. """
        self._question_repo = CandlestickQuestionRepository()
        self._session_repo = CandlestickSessionRepository()
        # Store session_factory for direct use in complex queries if needed
        self.async_session = None
        logger.info("Initialized CandlestickAssessmentRepositoryImpl")

    @property
    def question_repository(self) -> QuestionRepository[Q_Domain]: # Specify generic type
        """ Get the question repository instance. """
        return self._question_repo

    @property
    def session_repository(self) -> SessionRepository[S_Domain]: # Specify generic type
        """ Get the session repository instance. """
        return self._session_repo

    # --- Methods from AssessmentRepository Interface (Now Async) ---

    async def get_questions_for_session(
        self,
        difficulty: Optional[str] = None,
        topics: Optional[List[str]] = None,
        count: int = 10,
        user_id: Optional[str] = None # For excluding seen questions
    ) -> List[Q_Domain]:
        """
        Get suitable questions for a new session, using JSONB for topic filtering.
        """
        if count <= 0:
            return []

        try:
            async with self.async_session() as session:
                # --- 1. Get IDs of questions already attempted by the user ---
                attempted_question_ids = set()
                if user_id:
                    stmt_attempts = (
                        select(distinct(CandlestickAttemptORM.question_id))
                        .join(CandlestickSessionORM, CandlestickAttemptORM.session_id == CandlestickSessionORM.session_id)
                        .where(CandlestickSessionORM.user_id == user_id)
                    )
                    result_attempts = await session.execute(stmt_attempts)
                    attempted_question_ids = set(result_attempts.scalars().all())
                    logger.debug(f"User {user_id} has attempted {len(attempted_question_ids)} questions.")

                # --- 2. Build the main question query with filters ---
                stmt = select(CandlestickQuestionORM)
                filters = []

                # Exclude attempted questions
                if attempted_question_ids:
                    filters.append(CandlestickQuestionORM.question_id.notin_(attempted_question_ids))

                # Filter by difficulty
                if difficulty:
                    filters.append(CandlestickQuestionORM.difficulty == difficulty)

                # Filter by topics using JSONB operator (?|)
                if topics:
                    filters.append(
                        CandlestickQuestionORM.content[ 'topics' ].astext.cast(ARRAY(String)).op('?|')(topics)
                    )

                if filters:
                    stmt = stmt.where(and_(*filters))

                # --- 3. Apply ordering and limit ---
                # Randomize selection using func.random() (ensure DB support)
                stmt = stmt.order_by(func.random()).limit(count)

                # --- 4. Execute query and map results ---
                result = await session.execute(stmt)
                orm_questions = result.scalars().all()

                domain_questions = []
                for orm_q in orm_questions:
                    try:
                        # Use the already updated mapping function
                        domain_questions.append(self._question_repo._map_orm_to_domain(orm_q))
                    except RepositoryError as map_err:
                         logger.warning(f"Skipping question {orm_q.question_id} due to mapping error in get_questions_for_session: {map_err}")
                    except Exception as unexpected_err:
                         logger.warning(f"Skipping question {orm_q.question_id} due to unexpected mapping error in get_questions_for_session: {unexpected_err}")

                if len(domain_questions) < count:
                     logger.warning(f"Could only find {len(domain_questions)} questions matching criteria for user {user_id} (requested {count}).")

                return domain_questions

        except Exception as e:
            logger.error(f"Error getting questions for session (User: {user_id}, Difficulty: {difficulty}, Topics: {topics}): {e}", exc_info=True)
            raise RepositoryError(f"Database error retrieving questions for session", original_exception=e)


    async def get_user_performance(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate overall user performance statistics based on their past attempts.
        """
        if not user_id:
            raise RepositoryError("User ID is required to get performance stats.")

        try:
            async with self.async_session() as session:
                # Query all attempts for the user, joining with Session to ensure they belong to the user
                stmt = (
                    select(
                        CandlestickAttemptORM.is_correct,
                        CandlestickAttemptORM.response_time_ms
                    )
                    .join(CandlestickSessionORM, CandlestickAttemptORM.session_id == CandlestickSessionORM.session_id)
                    .where(CandlestickSessionORM.user_id == user_id)
                )

                result = await session.execute(stmt)
                attempts_data = result.all() # Fetch all results [(is_correct, response_time_ms), ...]

                if not attempts_data:
                    logger.info(f"No attempts found for user {user_id}. Returning default performance.")
                    # Return default/empty stats if no attempts
                    return {
                        "total_attempts": 0,
                        "correct_attempts": 0,
                        "incorrect_attempts": 0,
                        "accuracy": 0.0,
                        "average_response_time_ms": None,
                        "completed_sessions": 0 # Requires separate query or different aggregation
                        # Add more stats as needed
                    }

                total_attempts = len(attempts_data)
                correct_attempts = sum(1 for is_correct, _ in attempts_data if is_correct)
                incorrect_attempts = total_attempts - correct_attempts
                accuracy = (correct_attempts / total_attempts) * 100 if total_attempts > 0 else 0.0

                # Calculate average response time, excluding None values
                valid_response_times = [rt for _, rt in attempts_data if rt is not None and rt >= 0]
                average_response_time_ms = (sum(valid_response_times) / len(valid_response_times)) if valid_response_times else None

                # TODO: Add calculation for completed_sessions, streaks, etc. if needed by querying Session table

                performance_stats = {
                    "total_attempts": total_attempts,
                    "correct_attempts": correct_attempts,
                    "incorrect_attempts": incorrect_attempts,
                    "accuracy": round(accuracy, 2),
                    "average_response_time_ms": average_response_time_ms,
                    # Add more derived stats here
                }
                logger.info(f"Calculated performance stats for user {user_id}")
                return performance_stats

        except Exception as e:
            logger.error(f"Error getting user performance for user {user_id}: {e}", exc_info=True)
            raise RepositoryError(f"Database error retrieving user performance for {user_id}", original_exception=e)


    async def get_topic_performance(self, user_id: str, topic: str) -> Dict[str, Any]:
        """
        Calculate user performance for a specific topic using JSONB filtering.
        """
        if not user_id or not topic:
            raise RepositoryError("User ID and topic are required to get topic performance.")

        try:
            async with self.async_session() as session:
                 # Query attempts for the user, joining with Question and filtering
                 # on the 'topics' array within the Question's JSONB 'content' field.
                 stmt = (
                     select(
                         CandlestickAttemptORM.is_correct,
                         CandlestickAttemptORM.response_time_ms
                     )
                     .join(CandlestickSessionORM, CandlestickAttemptORM.session_id == CandlestickSessionORM.session_id)
                     .join(CandlestickQuestionORM, CandlestickAttemptORM.question_id == CandlestickQuestionORM.question_id)
                     .where(
                         and_(
                             CandlestickSessionORM.user_id == user_id,
                             # Check if the topic exists in the 'topics' JSON array
                             # Use contains operator (@>) with a JSONB array containing the single topic
                             CandlestickQuestionORM.content[ 'topics' ].as_jsonb().contains(sql_cast([topic], JSONB))
                         )
                     )
                 )

                 result = await session.execute(stmt)
                 topic_attempts_data = result.all() # Fetch [(is_correct, response_time_ms)]

                 if not topic_attempts_data:
                    logger.info(f"No attempts found for user {user_id} on topic '{topic}'. Returning default performance.")
                    return {
                        "topic": topic,
                        "total_attempts": 0,
                        "correct_attempts": 0,
                        "accuracy": 0.0,
                        "average_response_time_ms": None,
                    }

                 # Calculation logic remains the same
                 total_attempts = len(topic_attempts_data)
                 correct_attempts = sum(1 for is_correct, _ in topic_attempts_data if is_correct)
                 accuracy = (correct_attempts / total_attempts) * 100 if total_attempts > 0 else 0.0
                 valid_response_times = [rt for _, rt in topic_attempts_data if rt is not None and rt >= 0]
                 average_response_time_ms = (sum(valid_response_times) / len(valid_response_times)) if valid_response_times else None
                 performance_stats = {
                    "topic": topic,
                    "total_attempts": total_attempts,
                    "correct_attempts": correct_attempts,
                    "accuracy": round(accuracy, 2),
                    "average_response_time_ms": average_response_time_ms,
                 }
                 logger.info(f"Calculated performance stats for user {user_id} on topic '{topic}'")
                 return performance_stats

        except Exception as e:
            logger.error(f"Error getting topic performance for user {user_id}, topic {topic}: {e}", exc_info=True)
            raise RepositoryError(f"Database error retrieving topic performance for {user_id} on {topic}", original_exception=e)


    async def get_recommended_topics(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommended topics based on performance, leveraging JSONB queries.
        """
        logger.debug(f"Getting recommended topics for user {user_id}")
        try:
            attempted_topics_list = []
            async with self.async_session() as session:
                 # Efficiently get distinct topics the user has encountered using JSONB functions.
                 # Use jsonb_array_elements_text to flatten the topics arrays from relevant questions.
                 subquery = (
                     select(distinct(CandlestickAttemptORM.question_id)).
                     join(CandlestickSessionORM, CandlestickAttemptORM.session_id == CandlestickSessionORM.session_id).
                     where(CandlestickSessionORM.user_id == user_id).
                     scalar_subquery()
                 )
                 stmt = (
                     select(distinct(func.jsonb_array_elements_text(CandlestickQuestionORM.content[ 'topics' ])))
                     .where(CandlestickQuestionORM.question_id.in_(subquery))
                 )
                 result = await session.execute(stmt)
                 attempted_topics_list = result.scalars().all()

            if not attempted_topics_list:
                logger.info(f"No attempted topics found for user {user_id} to generate recommendations.")
                return []

            # Fetch performance concurrently for each identified topic
            topic_perf = []
            async def safe_get_topic_performance(uid, topic):
                # ... (safe_get_topic_performance remains the same)
                try:
                    return await self.get_topic_performance(uid, topic)
                except RepositoryError as e:
                    logger.warning(f"Failed to get performance for topic '{topic}' for user {uid}: {e}")
                    return None
                except Exception as unexpected_e:
                     logger.error(f"Unexpected error getting performance for topic '{topic}' for user {uid}: {unexpected_e}", exc_info=True)
                     return None

            perf_tasks = [safe_get_topic_performance(user_id, topic) for topic in attempted_topics_list if topic]
            results = await asyncio.gather(*perf_tasks)

            # Filter and sort logic remains the same
            topic_perf = [
                perf for perf in results
                if perf is not None and isinstance(perf, dict) and perf.get("total_attempts", 0) > 0
            ]

            if not topic_perf:
                 logger.warning(f"Could not calculate performance for any attempted topics for user {user_id}. Returning empty recommendations.")
                 return []

            topic_perf.sort(key=lambda x: x.get('accuracy', 100.0))
            return topic_perf[:limit]

        except Exception as e:
             logger.error(f"Error getting recommended topics for user {user_id}: {e}", exc_info=True)
             raise RepositoryError(f"Failed to get recommended topics for user {user_id}", original_exception=e)


    async def get_difficulty_distribution(self, user_id: str) -> Dict[str, float]:
        """ Get distribution of attempts across difficulty levels for a user. """
        logger.debug(f"Getting difficulty distribution for user {user_id}")
        try:
            async with self.async_session() as session:
                # Query to count attempts per difficulty
                stmt = select(
                        CandlestickQuestionORM.difficulty,
                        func.count(CandlestickAttemptORM.attempt_id).label("attempt_count")
                    ).select_from(CandlestickAttemptORM)\
                    .join(CandlestickQuestionORM, CandlestickAttemptORM.question_id == CandlestickQuestionORM.question_id)\
                    .join(CandlestickSessionORM, CandlestickAttemptORM.session_id == CandlestickSessionORM.session_id)\
                    .where(CandlestickSessionORM.user_id == user_id)\
                    .group_by(CandlestickQuestionORM.difficulty)
                
                result = await session.execute(stmt)
                rows = result.all()
                
                total_attempts = sum(row.attempt_count for row in rows)
                distribution = {}
                if total_attempts > 0:
                    for row in rows:
                         difficulty = row.difficulty or "Unknown" # Handle potential NULL difficulty
                         percentage = round((row.attempt_count / total_attempts) * 100, 2)
                         distribution[difficulty] = percentage
                
                return distribution
                
        except Exception as e:
            logger.error(f"Error getting difficulty distribution for user {user_id}: {e}", exc_info=True)
            raise RepositoryError(f"Database error getting difficulty distribution for {user_id}", original_exception=e)