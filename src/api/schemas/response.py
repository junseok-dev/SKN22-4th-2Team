from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SearchResultDTO(BaseModel):
    patent_id: str = Field(..., description="특허 공개/등록 번호")
    title: str = Field(..., description="특허 명칭")
    abstract: str = Field(..., description="특허 초록 요약")
    claims: str = Field(..., description="청구항 내용 일부")
    grading_score: float = Field(default=0.0)
    grading_reason: str = Field(default="")
    dense_score: float = Field(default=0.0)
    sparse_score: float = Field(default=0.0)
    rrf_score: float = Field(default=0.0)

class AnalysisDTO(BaseModel):
    similarity: Dict[str, Any]
    infringement: Dict[str, Any]
    avoidance: Dict[str, Any]
    conclusion: str

class AnalyzeResponse(BaseModel):
    user_idea: str
    search_results: List[SearchResultDTO]
    analysis: Optional[AnalysisDTO] = None
    timestamp: str
    search_type: str
    error: Optional[str] = None
    security_alert: Optional[bool] = None

class HistoryItemResponse(BaseModel):
    id: int
    timestamp: str
    user_idea: str
    risk_level: str
    score: int
    result_json: AnalyzeResponse

class HistoryResponse(BaseModel):
    user_id: str = Field(..., description="사용자 ID (세션 ID에 해당)")
    history: List[Dict[str, Any]] = Field(..., description="검색 이력 목록 (최신순)")
