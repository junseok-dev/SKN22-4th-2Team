import json
import logging
from typing import AsyncGenerator, Optional
from src.patent_agent import PatentAgent
from src.history_manager import HistoryManager
from src.api.schemas.request import AnalyzeRequest
from src.api.schemas.response import AnalyzeResponse
from src.security import PromptInjectionError, sanitize_user_input

logger = logging.getLogger(__name__)

async def process_analysis_stream(
    request: AnalyzeRequest,
    agent: PatentAgent,
    history: HistoryManager,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Process analysis and stream results using standard SSE format (event + data).
    Ref: [Issue #51] 스트리밍 UI 버그 수정 - SSE 표준 규격 준수
    """
    effective_session_id = session_id or request.session_id or "anonymous"
    logger.info(f"[AnalysisStart] New request - Session: {effective_session_id}, User: {user_id}, Hybrid: {request.use_hybrid}")
    
    try:
        # 1. Start pipeline: Send initial setup/metadata
        # percent: 5 -> 초기화 단계
        yield "event: progress\n"
        yield f"data: {json.dumps({'percent': 5, 'message': '분석 엔진 초기화 중...'})}\n\n"
        
        # Security sanitization happens inside agent.analyze but let's do initial check
        try:
            sanitized_idea = sanitize_user_input(request.user_idea)
            logger.info(f"[AnalysisStage] Security check passed (Session: {effective_session_id})")
        except PromptInjectionError as e:
            logger.error(f"[Security] Analysis blocked (Session: {effective_session_id}): {e}")
            yield "event: error\n"
            yield f"data: {json.dumps({'detail': str(e), 'error_type': 'SecurityError'})}\n\n"
            return
 
        # 2. Search & initial grading
        # percent: 20 -> 검색 시작
        logger.info(f"[AnalysisStage] Starting patent search (Session: {effective_session_id})")
        yield "event: progress\n"
        yield f"data: {json.dumps({'percent': 20, 'message': '특허 데이터베이스 검색 및 1차 검증 중...'})}\n\n"
        results = await agent.search_with_grading(sanitized_idea, use_hybrid=request.use_hybrid, ipc_filters=request.ipc_filters)
        
        if not results:
            if getattr(agent, "degraded", False):
                detail = (
                    "벡터 DB 연결이 비활성화되어 유사 특허 검색을 수행할 수 없습니다. "
                    "Pinecone SDK/네트워크/API 키/인덱스 상태를 확인하세요."
                )
                logger.error(f"[AnalysisError] Vector DB unavailable (Session: {effective_session_id})")
                yield "event: error\n"
                yield f"data: {json.dumps({'detail': detail, 'error_type': 'VECTOR_DB_UNAVAILABLE'})}\n\n"
            else:
                logger.warning(f"[AnalysisComplete] No patents found (Session: {effective_session_id})")
                yield "event: empty\n"
                yield f"data: {json.dumps({'message': '관련 특허를 찾지 못했습니다.'})}\n\n"
            return
            
        logger.info(f"[AnalysisStage] Found {len(results)} relevant patents (Session: {effective_session_id})")
        
        # Send search results (percent: 50 -> 검색 결과 요약 송신)
        search_results_data = [
            {
                "patent_id": r.publication_number,
                "title": r.title,
                "abstract": r.abstract,
                "claims": r.claims,
                "grading_score": r.grading_score,
                "grading_reason": r.grading_reason,
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "rrf_score": r.rrf_score,
            }
            for r in results
        ]
        
        yield "event: progress\n"
        yield f"data: {json.dumps({'percent': 50, 'message': '검색 결과 집계 완료. 상세 심층 분석을 시작합니다.'})}\n\n"
        
        # 3. Stream Critical Analysis
        # percent: 60-90 -> 생성형 AI 분석 진행 중 (UI에서 부드럽게 증가하도록 구현 권장)
        logger.info(f"[AnalysisStage] Starting AI critical report generation (Session: {effective_session_id})")
        yield "event: progress\n"
        yield f"data: {json.dumps({'percent': 70, 'message': '생성형 AI를 통한 기술 유사도 및 침해 리스크 심위 분석 중...'})}\n\n"
        
        full_analysis_text = ""
        chunk_count = 0
        async for chunk in agent.critical_analysis_stream(sanitized_idea, results):
            # 텍스트 청크 누적 (마지막 구조화 파싱용)
            full_analysis_text += chunk
            chunk_count += 1
            # 생성 중에는 메시지 없이 퍼센트만 유지하거나 미세하게 조정
            if chunk_count % 20 == 0: # 로그 너무 많이 남지 않게 조절
                yield "event: progress\n"
                yield f"data: {json.dumps({'percent': 85, 'message': 'AI 리포트 작성 중...'})}\n\n"
            
        logger.info(f"[AnalysisStage] AI report generation complete. Chunks: {chunk_count} (Session: {effective_session_id})")
            
        # 4. Final structured results (GPT structured parsing replacement)
        # percent: 95 -> 분석 텍스트 구조화 중
        logger.info(f"[AnalysisStage] Parsing stream to structured format (Session: {effective_session_id})")
        yield "event: progress\n"
        yield f"data: {json.dumps({'percent': 95, 'message': '결과 리포트 최종 검토 및 구조화 중...'})}\n\n"
        
        structured_analysis = await agent.parse_streaming_to_structured(
            sanitized_idea, full_analysis_text, results
        )
        
        from datetime import datetime
        final_result = {
            "user_idea": sanitized_idea,
            "search_results": search_results_data,
            "analysis": {
                "similarity": {
                    "score": structured_analysis.similarity.score,
                    "common_elements": structured_analysis.similarity.common_elements,
                    "summary": structured_analysis.similarity.summary,
                    "evidence": structured_analysis.similarity.evidence_patents,
                },
                "infringement": {
                    "risk_level": structured_analysis.infringement.risk_level,
                    "risk_factors": structured_analysis.infringement.risk_factors,
                    "summary": structured_analysis.infringement.summary,
                    "evidence": structured_analysis.infringement.evidence_patents,
                },
                "avoidance": {
                    "strategies": structured_analysis.avoidance.strategies,
                    "alternatives": structured_analysis.avoidance.alternative_technologies,
                    "summary": structured_analysis.avoidance.summary,
                    "evidence": structured_analysis.avoidance.evidence_patents,
                },
                "conclusion": structured_analysis.conclusion,
            },
            "timestamp": datetime.now().isoformat(),
            "search_type": "hybrid" if request.use_hybrid else "dense",
        }

        # 5. Notify completion with full result
        yield "event: complete\n"
        yield f"data: {json.dumps({'percent': 100, 'result': final_result})}\n\n"
        
        # Save to history after successful stream
        logger.info(f"[AnalysisComplete] Success. Saving to history (Session: {effective_session_id})")
        saved = history.save_analysis(final_result, session_id=effective_session_id, user_id=user_id)
        if not saved:
            logger.warning(f"[AnalysisComplete] History save failed (Session: {effective_session_id})")
        
    except Exception as e:
        logger.error(f"[AnalysisError] Failed for session {effective_session_id}: {str(e)}", exc_info=True)
        yield "event: error\n"
        yield f"data: {json.dumps({'detail': f'분석 중 오류 발생: {str(e)}'})}\n\n"
