"""
History Manager - SQLAlchemy based persistent storage for analysis history.
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import UserSession, SearchHistory

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, db: Session):
        self.db = db
        
    def save_analysis(self, result: Dict, session_id: str, user_id: Optional[int] = None):
        """Save analysis result to DB with session mapping and user linkage."""
        try:
            # 1. Check if UserSession exists, create or update user linkage
            user_session = self.db.query(UserSession).filter(UserSession.session_id == session_id).first()
            if not user_session:
                user_session = UserSession(session_id=session_id, user_id=user_id)
                self.db.add(user_session)
            elif user_id and not user_session.user_id:
                user_session.user_id = user_id

            # 2. Extract metrics and save SearchHistory
            analysis = result.get('analysis', {})
            risk = analysis.get('infringement', {}).get('risk_level', 'unknown')
            score = analysis.get('similarity', {}).get('score', 0)
            
            timestamp_str = result.get('timestamp', datetime.utcnow().isoformat())
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.utcnow()
                
            history_record = SearchHistory(
                session_id=session_id,
                user_idea=result.get('user_idea', ''),
                result_json=json.dumps(result, ensure_ascii=False),
                risk_level=risk,
                score=score,
                timestamp=timestamp
            )
            
            self.db.add(history_record)
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error saving history: {e}", exc_info=True)
            return False
            
    def load_recent(
        self,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
        keyword: Optional[str] = None,
        sort_by: str = "desc",
    ) -> List[Dict]:
        """
        Load recent analysis history.
        If user_id is provided, loads all history for that user across sessions.
        Otherwise, loads by session_id.
        """
        try:
            query = self.db.query(SearchHistory).join(UserSession)
            if user_id:
                query = query.filter(UserSession.user_id == user_id)
            elif session_id:
                query = query.filter(UserSession.session_id == session_id)
            else:
                return []

            keyword_text = (keyword or "").strip()
            if keyword_text:
                query = query.filter(SearchHistory.user_idea.ilike(f"%{keyword_text}%"))

            if sort_by == "asc":
                query = query.order_by(SearchHistory.timestamp.asc(), SearchHistory.id.asc())
            elif sort_by == "risk_desc":
                query = query.order_by(
                    SearchHistory.score.desc(),
                    SearchHistory.timestamp.desc(),
                    SearchHistory.id.desc(),
                )
            elif sort_by == "risk_asc":
                query = query.order_by(
                    SearchHistory.score.asc(),
                    SearchHistory.timestamp.desc(),
                    SearchHistory.id.desc(),
                )
            else:
                query = query.order_by(SearchHistory.timestamp.desc(), SearchHistory.id.desc())

            recent_histories = query.limit(limit).all()

            parsed_histories: List[Dict] = []
            for record in recent_histories:
                try:
                    payload = json.loads(record.result_json)
                    if not isinstance(payload, dict):
                        payload = {}
                except Exception:
                    payload = {}

                # Legacy 데이터/트리거 혼합 상황에서도 프론트가 안정적으로 읽을 수 있게 보강
                payload.setdefault("user_idea", record.user_idea)
                payload.setdefault(
                    "timestamp",
                    record.timestamp.isoformat() if isinstance(record.timestamp, datetime) else datetime.utcnow().isoformat(),
                )
                payload.setdefault("risk_level", record.risk_level)
                payload.setdefault("score", record.score)

                parsed_histories.append(payload)

            return parsed_histories
        except Exception as e:
            logger.error(f"Failed to load recent history: {e}", exc_info=True)
            return []

    def find_cached_result(self, user_idea: str, session_id: str) -> Optional[Dict]:
        """Find the most recent identical query in history to act as a cache."""
        try:
            cached = (
                self.db.query(SearchHistory)
                .filter(SearchHistory.session_id == session_id, SearchHistory.user_idea == user_idea)
                .order_by(SearchHistory.id.desc())
                .first()
            )
            return json.loads(cached.result_json) if cached else None
        except Exception as e:
            logger.error(f"Failed to check cache history: {e}", exc_info=True)
            return None

    def clear_history(self, session_id: str):
        """Delete history for specific user session."""
        try:
            user_session = self.db.query(UserSession).filter(UserSession.session_id == session_id).first()
            if user_session:
                self.db.delete(user_session)
                self.db.commit()
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to clear history for {session_id}: {e}", exc_info=True)
