import os
import uuid
import logging
import socket
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# 애플리케이션 모듈 임포트 전 가장 먼저 시크릿을 로드합니다.
# AWS Secrets Manager가 값을 주입했다면 os.getenv를 통해 조회 가능합니다.
from src.secrets_manager import bootstrap_secrets

# 시크릿 부트스트랩 (AWS Secrets Manager 또는 .env 로드)
bootstrap_secrets()

# ── 핵심 환경 변수 선행 검증 (Fast-Fail) ──────────────────────────────────
# 앱 구동 전 필수 키가 누락되었다면 에러 로그를 남기지만, 컨테이너 헬스체크 통과를 위해 종료(sys.exit)하지는 않습니다.
# PINECONE_ENVIRONMENT: Pinecone v3 Serverless에서는 불필요 (v2 레거시) — 체크에서 제외
critical_env_vars = {
    "OPENAI_API_KEY": "OpenAI API 키가 누락되었습니다.",
    "PINECONE_API_KEY": "Pinecone API 키가 누락되었습니다.",
}

missing_vars = []
for var, msg in critical_env_vars.items():
    if not os.getenv(var):
        logger.critical(f"Missing critical environment variable: {var} ({msg})")
        missing_vars.append(var)

if missing_vars:
    # Issue #33: 필수 환경 변수 누락 시 애플리케이션을 즉시 중단합니다 (Fast-fail).
    # AWS Secrets Manager 또는 ECS Task Definition의 secrets 매핑이 정상적으로
    # 구성되어 있다면 이 지점에 도달하지 않습니다.
    raise ValueError(
        f"애플리케이션 구동 불가: 필수 환경 변수가 누락되었습니다 → {', '.join(missing_vars)}. "
        f"AWS Secrets Manager 또는 .env 설정을 확인하세요."
    )

# 검증 통과 완료 시 config 및 나머지 모듈 로드
from src.config import config

from contextlib import asynccontextmanager
from src.api.v1.router import router as api_v1_router
from src.database.connection import Base, get_engine, verify_db_connection, ensure_schema_compatibility
from src.utils import configure_json_logging
from src.api.middleware import SecurityMiddleware
from src.security import PromptInjectionError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def _check_tcp_connectivity(host: str, port: int = 443, timeout: float = 2.0) -> bool:
    """Lightweight outbound TCP connectivity probe for dependency health checks."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """트래픽 수신 전 의존성 사전 초기화 (Pre-warm)"""
    from src.api.dependencies import get_patent_agent, get_history_manager
    
    logger.info("Checking system readiness & pre-warming dependencies...")
    
    try:
        # PatentAgent 초기화 시 내부적으로 config 검증 및 LLM 연결 테스트가 수행되길 기대합니다.
        agent = get_patent_agent()
        logger.info(f"PatentAgent initialized (Model: {config.agent.analysis_model})")
        
        get_history_manager()
        logger.info("HistoryManager initialized.")
        
        # NLTK 데이터 경로 확인 (Dockerfile ENV와 동기화 확인용)
        import nltk
        logger.info(f"NLTK Data Paths: {nltk.data.path}")
        
        # Database schema initialization
        logger.info("Initializing database schema...")
        from fastapi.concurrency import run_in_threadpool
        # get_engine()을 호출해야 bootstrap_secrets() 이후 주입된 DATABASE_URL로 엔진이 생성됩니다.
        await run_in_threadpool(Base.metadata.create_all, bind=get_engine())
        await run_in_threadpool(ensure_schema_compatibility)
        
        # DB 연결 검증 (RDS PostgreSQL 또는 SQLite 폴백 확인)
        db_status = await run_in_threadpool(verify_db_connection)
        if db_status["ok"]:
            logger.info("DB 연결 검증 완료: type=%s, host=%s", db_status["db_type"], db_status["url_hint"])
        else:
            logger.error("DB 연결 검증 실패: %s", db_status.get("error"))
        logger.info("Database schema initialized.")
        
        logger.info("System health check: PASSED. Ready to receive traffic.")
    except Exception as e:
        logger.critical(f"FATAL: Dependency initialization failed during lifespan: {e}")
        # 초기화 실패 시 컨테이너가 Unhealthy 상태로 남지 않고 일단 켜지게 둡니다 (ALB 200 OK 헬스체크용).
        # 실제 API 요청 시 500 에러+상세메시지로 사용자에게 노출됩니다.
    
    yield
    logger.info("Shutting down FastAPI application...")

def create_app() -> FastAPI:
    # 1. 로깅 초기화
    configure_json_logging(level=logging.INFO)
    logger.info("Starting FastAPI application...")

    # 2. FastAPI 초기화
    app = FastAPI(
        title="쇼특허 (Short-Cut) API 명세서",
        description="AI 기반 특허 선행 기술 조사 시스템 Backend API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # 3. 미들웨어 추가 (순서는 나중에 등록한 것이 먼저 실행됨)
    # CORS 설정: ALLOWED_ORIGINS 환경변수로 허용 도메인을 동적으로 주입합니다.
    # - 개발 환경(APP_ENV != production): ["*"] 허용 (로컬/개발용)
    # - 운영 환경(APP_ENV=production): ALLOWED_ORIGINS 값이 "*"이면 [\"*\"], 아니면 콤마 구분된 도메인 리스트 적용
    _raw_origins = os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    ).strip()
    if _raw_origins == "*":
        allowed_origins = ["*"]
    else:
        allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

    # 프론트엔드가 같은 컨테이너(FastAPI)에서 서빙될 때는 same-origin 요청이므로 CORS가 불필요합니다.
    # 단, API를 별도의 도메인에서 직접 호출하는 경우를 대비해 CORS 미들웨어는 항상 등록합니다.
    _is_production = os.getenv("APP_ENV") == "production"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if _is_production else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 보안 미들웨어 등록 (ASGI 기반)
    app.add_middleware(SecurityMiddleware)

    # 4. 요청 로깅 미들웨어 (모든 요청의 시작/종료 로깅)
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        req_id = uuid.uuid4().hex
        logger.info(f"[RequestStart] {request.method} {request.url.path} (ReqID: {req_id})")
        
        from time import time
        start_time = time()
        
        try:
            response = await call_next(request)
            duration = round(time() - start_time, 3)
            logger.info(f"[RequestEnd] {request.method} {request.url.path} - Status: {response.status_code} (ReqID: {req_id}, Latency: {duration}s)")
            return response
        except Exception as e:
            duration = round(time() - start_time, 3)
            logger.error(f"[RequestError] {request.method} {request.url.path} - Failed (ReqID: {req_id}, Latency: {duration}s): {str(e)}")
            raise e

    # 5. 전역 예외 처리 (Global Exception Handlers)
    @app.exception_handler(PromptInjectionError)
    async def prompt_injection_exception_handler(request: Request, exc: PromptInjectionError):
        req_id = uuid.uuid4().hex
        logger.error(f"[GlobalException] Prompt Injection at {request.url.path} from {request.client.host if request.client else 'Unknown'} (ReqID: {req_id})")
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Forbidden: 악의적인 입력 패턴이 감지되었습니다.", 
                "error_type": "PromptInjectionError",
                "request_id": req_id
            }
        )

    from src.rate_limiter import RateLimitException
    @app.exception_handler(RateLimitException)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
        req_id = uuid.uuid4().hex
        logger.warning(f"[RateLimit] Hit at {request.url.path} (ReqID: {req_id}): {exc.message}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": exc.message,
                "reset_time": exc.reset_time,
                "request_id": req_id
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = uuid.uuid4().hex
        logger.error(f"[GlobalException] Unhandled Error at {request.url.path} (ReqID: {req_id}): {str(exc)}")
        
        # FastAPI 미들웨어가 500 에러 시 가끔 동작하지 않아 브라우저에서 CORS 에러로 덮어씌워지는 현상을 방지하기 위해 명시적 헤더 추가
        origin = request.headers.get("origin")
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        headers = {}
        if origin and (origin in allowed_origins or "*" in allowed_origins or os.getenv("APP_ENV") != "production"):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(exc)}",
                "request_id": req_id,
                "error_type": type(exc).__name__
            },
            headers=headers
        )

    # 5. API Endpoints 라우터 통합
    app.include_router(api_v1_router, prefix="/api/v1")

    # 6. 프론트엔드 서빙 (Vanilla JS vs React dist 자동 감지)
    # React 빌드 산출물(dist)이 있으면 우선 서빙하고, 없으면 기본 frontend 폴더를 서빙합니다.
    # ── 프론트엔드 정적 파일 서빙 설정 ────────────────────────────────────
    # 빌드된 React 에셋(dist)이 있으면 우선 사용, 없으면 소스 폴더(frontend) 사용
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    
    # ── 프론트엔드 정적 파일 서빙 설정 ────────────────────────────────────
    # 빌드된 React 에셋(dist)이 있으면 우선 사용, 없으면 소스 폴더(frontend) 사용
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dist_path = os.path.join(base_dir, "frontend", "dist")
    src_path = os.path.join(base_dir, "frontend")
    
    if os.path.exists(dist_path) and os.path.isdir(dist_path):
        frontend_dir = dist_path
        logger.info(f"Frontend: Using production build from {dist_path}")
    else:
        frontend_dir = src_path
        logger.warning(f"Frontend: Production build not found at {dist_path}. Falling back to source at {src_path}")

    index_file = os.path.join(frontend_dir, "index.html")

    @app.get("/")
    async def serve_index():
        """Root 접속 시 프론트엔드 메인 페이지 반환 (ALB 헬스체크 200 OK)"""
        if os.path.exists(index_file):
            logger.info(f"Serving index.html from {index_file}")
            return FileResponse(
                index_file,
                headers={
                    # SPA 엔트리 파일은 캐시 금지하여 최신 번들 해시를 즉시 반영
                    "Cache-Control": "no-store, no-cache, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        
        logger.error(f"Frontend index.html not found at {index_file}")
        return {
            "status": "error",
            "message": "Frontend index.html not found",
            "path": index_file,
            "cwd": os.getcwd()
        }

    @app.get("/health")
    async def health_check():
        """상태 점검 엔드포인트 (DB 연결 상태 포함)"""
        from fastapi.concurrency import run_in_threadpool
        db_status = await run_in_threadpool(verify_db_connection)
        pinecone_reachable = await run_in_threadpool(_check_tcp_connectivity, "api.pinecone.io", 443, 2.0)
        openai_reachable = await run_in_threadpool(_check_tcp_connectivity, "api.openai.com", 443, 2.0)
        return {
            "status": "ok",
            "message": "Healthy",
            "database": db_status,
            "dependencies": {
                "pinecone_api_tcp_443": pinecone_reachable,
                "openai_api_tcp_443": openai_reachable,
            },
        }

    # API 라우트들을 먼저 등록한 후, 나머지 모든 경로를 정적 파일로 마운트
    # (이미 위에서 라우터들이 등록되었으므로 순서상 안전함)
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app

app = create_app()
