from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AnalyzeRequest(BaseModel):
    user_idea: str = Field(..., description="사용자의 특허 아이디어", min_length=10)
    use_hybrid: bool = Field(default=True, description="하이브리드 검색 사용 여부")
    stream: bool = Field(default=True, description="스트리밍 응답 여부 (SSE)")
    session_id: Optional[str] = Field(default="anonymous", description="브라우저 세션 ID (JWT 연동 시 히스토리 연결용)")
    ipc_filters: Optional[List[str]] = Field(default=None, description="IPC 분류 필터링")


class EmailNotificationRequest(BaseModel):
    """분석 결과 이메일 발송 요청 페이로드"""
    idea: str = Field(..., description="사용자 입력 아이디어", min_length=1)
    resultData: Dict[str, Any] = Field(default_factory=dict, description="분석 결과 데이터")
