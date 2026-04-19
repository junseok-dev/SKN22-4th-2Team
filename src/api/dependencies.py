import logging
from typing import Optional
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from src.patent_agent import PatentAgent
from src.history_manager import HistoryManager
from src.database.connection import get_db
from src.database.models import User
from src.api.services.security import decode_token

logger = logging.getLogger(__name__)

# 싱글턴 인스턴스 및 초기화 상태 트래킹
_patent_agent = None
_init_error = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_patent_agent() -> PatentAgent:
    global _patent_agent, _init_error
    
    # 이미 초기화 성공한 경우 그대로 반환
    if _patent_agent is not None:
        return _patent_agent
        
    # 이전 초기화 과정에서 치명적 에러가 발생했던 경우, 
    # 매 요청마다 무거운 초기화를 반복하지 않고 즉시 에러 리턴
    if _init_error is not None:
        logger.error(f"Analysis engine is in failed state: {_init_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"분석 엔진 초기화 실패 상태입니다. 관리자에게 문의하세요. (Error: {_init_error})"
        )

    logger.info("Initializing PatentAgent instance...")
    try:
        _patent_agent = PatentAgent()
        logger.info("PatentAgent initialized successfully.")
    except Exception as e:
        _init_error = f"{type(e).__name__}: {str(e)}"
        logger.error(
            f"PatentAgent 초기화 실패: {_init_error}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 엔진(PatentAgent) 초기화 중 오류가 발생했습니다: {_init_error}"
        )
    return _patent_agent

def get_history_manager(db: Session = Depends(get_db)) -> HistoryManager:
    """Provides a HistoryManager instance with the active DB session."""
    return HistoryManager(db=db)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Bearer 토큰을 검증하고 현재 로그인 사용자를 반환합니다. 인증 필수 엔드포인트에서 사용."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보가 유효하지 않습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = decode_token(token)
        if payload is None:
            raise credentials_exception
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user

async def get_optional_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Bearer 토큰이 있으면 사용자 반환, 없으면 None 반환합니다. 비로그인 허용 엔드포인트에서 사용."""
    if not token:
        return None
    try:
        payload = decode_token(token)
        if payload is None:
            return None
        user_id: str = payload.get("sub")
        if not user_id:
            return None
        return db.query(User).filter(User.id == int(user_id)).first()
    except Exception:
        return None
