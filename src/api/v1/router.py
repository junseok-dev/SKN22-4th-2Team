from fastapi import APIRouter, Depends, Query, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Any

from src.api.schemas.request import AnalyzeRequest, EmailNotificationRequest
from src.api.schemas.response import HistoryResponse
from src.api.dependencies import get_patent_agent, get_history_manager, get_db
from sqlalchemy.orm import Session
from src.api.services.analyze_service import process_analysis_stream
from src.patent_agent import PatentAgent
from src.history_manager import HistoryManager
import logging

from src.api.v1.auth_router import router as auth_router

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(auth_router)


def _mock_send_email(session_id: str, idea: str, result_data: dict[str, Any]) -> None:
    """SMTP 연동 전 임시 Mock 이메일 발송 작업."""
    risk_level = result_data.get("riskLevel", "unknown")
    risk_score = result_data.get("riskScore", "unknown")
    logger.info(
        "[notify/email] Mock send completed (session_id=%s, idea_len=%d, risk=%s/%s)",
        session_id,
        len(idea),
        risk_level,
        risk_score,
    )

@router.post("/analyze", summary="특허 분석 요청 (SSE 스트리밍 연동)")
async def analyze_patent(
    request: AnalyzeRequest,
    req: Request,
    agent: PatentAgent = Depends(get_patent_agent),
    history: HistoryManager = Depends(get_history_manager),
    db: Session = Depends(get_db)
):
    try:
        user_id = None
        header_session_id = req.headers.get("X-Session-ID")
        request_session_id = (request.session_id or "").strip()
        session_id = (
            request_session_id
            if request_session_id and request_session_id != "anonymous"
            else (header_session_id or "anonymous")
        )

        # Check if streaming is requested
        if request.stream:
            # We use text/event-stream for SSE
            return StreamingResponse(
                process_analysis_stream(
                    request,
                    agent,
                    history,
                    user_id=user_id,
                    session_id=session_id,
                ),
                media_type="text/event-stream"
            )
        else:
            # Full blocking wait
            result = await agent.analyze(
                user_idea=request.user_idea,
                use_hybrid=request.use_hybrid,
                stream=False,
                ipc_filters=request.ipc_filters
            )
            # Save history (session_id = request.session_id, user_id = user_id)
            saved = history.save_analysis(result, session_id=session_id, user_id=user_id)
            if not saved:
                logger.warning("[v51c8a7b] History save failed (session_id=%s)", session_id)
            return result
    except Exception as e:
        logger.error(f"[v51c8a7b] Error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"[v51c8a7b] {str(e)}")


@router.post("/notify/email", summary="분석 결과 이메일 발송 요청")
async def notify_email(
    payload: EmailNotificationRequest,
    req: Request,
    background_tasks: BackgroundTasks,
):
    """
    결과 이메일 발송 API.
    현재는 Mock 동작으로 요청만 수락하고 백그라운드 로그 작업을 수행합니다.
    """
    try:
        session_id = req.headers.get("X-Session-ID", "anonymous")
        background_tasks.add_task(
            _mock_send_email,
            session_id,
            payload.idea,
            payload.resultData,
        )
        return {
            "success": True,
            "message": "이메일 발송 요청이 접수되었습니다.",
        }
    except Exception as e:
        logger.error("[notify/email] Failed to queue email task: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="이메일 발송 요청 처리 중 오류가 발생했습니다.")

@router.get("/history", summary="과거 검색 기록 조회", response_model=HistoryResponse)
async def get_history(
    req: Request,
    limit: int = Query(20, description="최대 조회 개수"),
    keyword: str | None = Query(default=None, description="아이디어 키워드 검색"),
    sort_by: str = Query(default="desc", description="정렬 기준: desc | asc | risk_desc | risk_asc"),
    history: HistoryManager = Depends(get_history_manager),
    db: Session = Depends(get_db)
):
    try:
        user_id = None
        
        # If not logged in, fetch by session_id (X-Session-ID header)
        session_id = None
        if not user_id:
            session_id = req.headers.get("X-Session-ID")
            if not session_id:
                # Fallback to user_id in request body if somehow provided for history
                # But headers are preferred
                return HistoryResponse(user_id="anonymous", history=[])

        normalized_sort = sort_by if sort_by in {"desc", "asc", "risk_desc", "risk_asc"} else "desc"
        safe_limit = max(1, min(limit, 100))
        history_items = history.load_recent(
            user_id=user_id,
            session_id=session_id,
            limit=safe_limit,
            keyword=keyword,
            sort_by=normalized_sort,
        )
        return HistoryResponse(user_id=session_id or "anonymous", history=history_items)
    except Exception as e:
        logger.error(f"[v51c8a7b] Error retrieving history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve history")
