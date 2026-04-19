"""
Database connection and session management.

변경 이력:
  - 2026-03-03: Lazy 초기화 패턴 적용 (Antigravity)
    bootstrap_secrets()가 os.environ에 DATABASE_URL을 주입하기 전에
    connection.py가 임포트되면 SQLite 폴백으로 고정되는 타이밍 버그 수정.
    engine / SessionLocal을 최초 get_db() 호출 시점에 생성하도록 변경.
"""
import logging
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# SQLite 폴백 경로 (로컬 개발 전용)
_DB_PATH = Path(__file__).parent.parent / "data" / "history.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_DEFAULT_DB_URL = f"sqlite:///{_DB_PATH}"

Base = declarative_base()

# ── Lazy 초기화용 내부 상태 ────────────────────────────────────────────────
# engine / SessionLocal은 최초 get_db() 호출 시점에 생성됩니다.
# 이렇게 해야 bootstrap_secrets()가 DATABASE_URL을 주입한 뒤
# 올바른 값으로 engine이 생성됩니다.
_engine = None
_SessionLocal = None


def _get_engine():
    """DATABASE_URL 환경 변수를 읽어 SQLAlchemy engine을 반환합니다 (Lazy)."""
    global _engine, _SessionLocal

    if _engine is not None:
        return _engine

    database_url: str = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)

    # PostgreSQL URL은 psycopg2 드라이버를 명시해야 연결이 안정적입니다.
    # SQLAlchemy가 'postgresql://' 스킴을 자동으로 인식하지만,
    # RDS 연결 시 'postgresql+psycopg2://' 로 명시하면 드라이버 오류를 방지합니다.
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

    logger.info(
        "데이터베이스 엔진 초기화: %s",
        database_url.split("@")[-1] if "@" in database_url else database_url,  # 비밀번호 마스킹
    )

    try:
        _engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,       # 연결 유효성 사전 검사 (RDS 재시작 대응)
            pool_recycle=1800,        # 30분마다 커넥션 재생성 (MySQL/PG idle timeout 대응)
            echo=False,               # SQL 쿼리 로깅 비활성화 (운영 환경)
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        logger.info("데이터베이스 엔진 초기화 완료.")
    except Exception as exc:
        logger.critical("데이터베이스 엔진 생성 실패: %s", exc, exc_info=True)
        raise

    return _engine


def get_engine():
    """외부에서 engine에 접근할 때 사용합니다 (예: Base.metadata.create_all)."""
    return _get_engine()


# 하위 호환성: main.py가 `from src.database.connection import engine`으로 참조하는 경우 대응
# → 해당 코드를 get_engine() 호출로 마이그레이션 권장
class _LazyEngine:
    """engine 속성을 지연 초기화하는 프록시 객체."""

    def __getattr__(self, name: str):
        return getattr(_get_engine(), name)


engine = _LazyEngine()  # type: ignore[assignment]


def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends 전용: 요청당 DB 세션을 제공하고 완료 후 반드시 닫습니다."""
    _get_engine()  # engine / SessionLocal이 초기화됐음을 보장
    assert _SessionLocal is not None, "SessionLocal이 초기화되지 않았습니다."

    db: Session = _SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def verify_db_connection() -> dict:
    """
    데이터베이스 연결 상태를 검증하고 결과를 반환합니다.
    헬스체크 엔드포인트 또는 시작 시 연동 검증에 사용합니다.

    Returns:
        {"ok": True, "url_hint": "...", "db_type": "postgresql" | "sqlite"}
    """
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        url_raw: str = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
        db_type = "sqlite" if url_raw.startswith("sqlite") else "postgresql"
        url_hint = url_raw.split("@")[-1] if "@" in url_raw else url_raw
        logger.info("DB 연결 검증 성공 (%s): %s", db_type, url_hint)
        return {"ok": True, "url_hint": url_hint, "db_type": db_type}
    except Exception as exc:
        logger.error("DB 연결 검증 실패: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


def ensure_schema_compatibility() -> None:
    """
    로컬 SQLite 스키마의 하위 호환성을 보정합니다.

    현재 보정 항목:
      - usersession.user_id 컬럼 누락 시 자동 추가
      - usersession.user_id 인덱스 누락 시 자동 생성
    """
    eng = _get_engine()
    # 운영(PostgreSQL)은 Alembic 마이그레이션 기준으로 관리하고,
    # 로컬 SQLite만 경량 자동 보정을 수행합니다.
    if not str(eng.url).startswith("sqlite"):
        return

    try:
        with eng.begin() as conn:
            rows = conn.execute(text("PRAGMA table_info(usersession)")).fetchall()
            existing_cols = {r[1] for r in rows}  # r[1] == column name

            if "user_id" not in existing_cols:
                logger.warning("usersession.user_id 컬럼이 없어 자동 추가를 수행합니다.")
                conn.execute(text("ALTER TABLE usersession ADD COLUMN user_id INTEGER"))

            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_usersession_user_id ON usersession (user_id)"))
    except Exception as exc:
        logger.error("스키마 호환성 보정 실패: %s", exc, exc_info=True)
        raise
