"""
Short-Cut - Self-RAG Patent Agent with Hybrid Search & Streaming
==========================================================================
Advanced RAG pipeline with HyDE, Hybrid Search (RRF), Streaming, and CoT Analysis.

Features:
1. HyDE (Hypothetical Document Embedding) - Generate virtual claims for better retrieval
2. Hybrid Search - Dense (FAISS) + Sparse (BM25) with RRF fusion
3. LLM Streaming Response - Real-time analysis output
4. Critical CoT Analysis - Detailed similarity/infringement/avoidance analysis

Author: Team 뀨💕
License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from types import SimpleNamespace

from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from pydantic import BaseModel, Field
import httpx
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from openai import APIStatusError  # for other API errors if needed
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from src.security import sanitize_user_input, wrap_user_query, PromptInjectionError

from src.serialization import json_loads, json_dumps

# =============================================================================
# Logging Setup — 구조화 JSON 포맷 적용 (CloudWatch / ELK 연동)
# =============================================================================

from src.utils import configure_json_logging, LogEvent

configure_json_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (Environment Variables)
# =============================================================================

from src.config import config

# Data paths - relative to this file
from pathlib import Path

# 데이터 경로 (이 파일 기준 상대 경로)
DATA_DIR: Path = Path(__file__).resolve().parent / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================


class GradingResult(BaseModel):
    """Structured grading result from GPT."""

    patent_id: str = Field(description="Patent publication number")
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation for the score")


class GradingResponse(BaseModel):
    """Response containing all grading results."""

    results: List[GradingResult] = Field(description="List of grading results")
    average_score: float = Field(description="Average score across all results")
    filter_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="컷오프 필터링 통계 (grade_results에서 자동 설정)",
    )


class QueryRewriteResponse(BaseModel):
    """Optimized search query from GPT."""

    optimized_query: str = Field(description="Improved search query")
    keywords: List[str] = Field(description="Key technical terms to search")
    reasoning: str = Field(description="Why this query should work better")


class SimilarityAnalysis(BaseModel):
    """유사도 평가 section."""

    score: int = Field(description="Technical similarity score 0-100")
    common_elements: List[str] = Field(description="Shared technical elements")
    summary: str = Field(description="Overall similarity assessment")
    evidence_patents: List[str] = Field(
        description="Patent IDs supporting this analysis"
    )


class InfringementAnalysis(BaseModel):
    """침해 리스크 section."""

    risk_level: str = Field(description="high, medium, or low")
    risk_factors: List[str] = Field(description="Specific infringement concerns")
    summary: str = Field(description="Overall risk assessment")
    evidence_patents: List[str] = Field(
        description="Patent IDs supporting this analysis"
    )


class AvoidanceStrategy(BaseModel):
    """회피 전략 section."""

    strategies: List[str] = Field(description="Design-around approaches")
    alternative_technologies: List[str] = Field(
        description="Alternative implementations"
    )
    summary: str = Field(description="Recommended avoidance approach")
    evidence_patents: List[str] = Field(
        description="Patent IDs informing these strategies"
    )


class ComponentComparison(BaseModel):
    """구성요소 대비표 - Element-by-element comparison."""

    idea_components: List[str] = Field(
        description="User idea's key technical components"
    )
    matched_components: List[str] = Field(
        description="Components found in prior patents"
    )
    unmatched_components: List[str] = Field(
        description="Novel components not in prior art"
    )
    risk_components: List[str] = Field(
        description="Components causing infringement risk"
    )


class CriticalAnalysisResponse(BaseModel):
    """Complete critical analysis response."""

    similarity: SimilarityAnalysis
    infringement: InfringementAnalysis
    avoidance: AvoidanceStrategy
    component_comparison: ComponentComparison = Field(
        description="Element comparison table"
    )
    conclusion: str = Field(description="Final recommendation")


# =============================================================================
# Patent Search Result
# =============================================================================


@dataclass
class PatentSearchResult:
    """A single patent search result."""

    publication_number: str
    title: str
    abstract: str
    claims: str
    ipc_codes: List[str]
    similarity_score: float = 0.0  # Vector similarity
    grading_score: float = 0.0  # LLM grading score
    grading_reason: str = ""

    # Hybrid search scores
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    is_prioritized: bool = False  # Flag for patents explicitly mentioned in query


# =============================================================================
# Patent Agent - Main Class
# =============================================================================


class PatentAgent:
    """
    Self-RAG Patent Analysis Agent (v3.0).

    Features:
    - Pinecone Serverless Hybrid Search (Dense + Sparse)
    - OpenAI API for embeddings and LLM
    - Streaming response for real-time analysis

    Implements:
    1. HyDE - Hypothetical Document Embedding
    2. Hybrid Search - Dense + Sparse with RRF
    3. Grading & Rewrite Loop
    4. Critical CoT Analysis with Streaming
    """

    def __init__(self, db_client=None):
        if not config.embedding.api_key:
            raise ValueError("config.embedding.api_key not set. Check .env file.")

        # 전역 타임아웃 설정: 전체 요청 60초, TCP 연결 10초
        # OpenAI 서버 지연 시 이벤트 루프 무한 점유 방지
        self.client = AsyncOpenAI(
            api_key=config.embedding.api_key,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

        # Initialize Vector DB client with hybrid search
        if db_client is not None:
            self.db_client = db_client
        else:
            # Use PineconeClient for v3.0 Migration
            try:
                from src.vector_db import PineconeClient

                self.db_client = PineconeClient()
                self._try_load_local_cache()
                self.degraded = False
            except Exception as e:
                logger.error(
                    f"PineconeClient 초기화 실패: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                # Don't raise here to allow the API to start in degraded mode.
                # This enables the server to serve health checks and other endpoints
                # while disabling vector DB dependent features.
                self.db_client = None
                self.degraded = True
                logger.warning(
                    "Running in degraded mode: Pinecone DB unavailable. Vector search disabled."
                )

    def _try_load_local_cache(self) -> bool:
        """Try to load local metadata cache and BM25 index."""
        loaded = self.db_client.load_local()
        if loaded:
            stats = self.db_client.get_stats()
            logger.info(f"Loaded local cache: {stats.get('bm25_docs', 0)} docs in BM25")
            return True
        else:
            logger.warning("No local cache found. Run pipeline to build BM25 index.")
            return False

    def index_loaded(self) -> bool:
        """Check if DB is ready."""
        return self.db_client is not None

    # =========================================================================
    # 컷오프 필터 통계 헬퍼 (DRY — Issue #18 리팩토링)
    # =========================================================================

    def _compute_filter_stats(
        self,
        results: List[PatentSearchResult],
        threshold: float = config.agent.cutoff_threshold,
    ) -> Dict[str, Any]:
        """컷오프 임계값 기준 필터링 통계를 계산합니다.

        Args:
            results: 그레이딩 완료된 특허 검색 결과 목록
            threshold: 컷오프 임계값 (기본값: 전역 config.agent.cutoff_threshold)

        Returns:
            필터 통계 딕셔너리 (before_filter, after_filter, filtered_out, filter_ratio_pct, threshold)
        """
        total = len(results)
        passed = sum(1 for r in results if r.grading_score >= threshold)
        filtered = total - passed
        ratio = (filtered / total) if total > 0 else 0.0
        return {
            "before_filter": total,
            "after_filter": passed,
            "filtered_out": filtered,
            "filter_ratio_pct": round(ratio * 100, 1),
            "threshold": threshold,
        }

    def _log_filter_stats(
        self,
        stats: Dict[str, Any],
        stage: str,
        *,
        extra_fields: Optional[Dict[str, Any]] = None,
        warn_ratio: float = 0.8,
    ) -> None:
        """컷오프 필터 통계를 로그로 발행합니다.

        필터링 비율이 warn_ratio를 초과하면 WARNING, 아니면 INFO로 기록합니다.
        """
        log_payload: Dict[str, Any] = {
            "event": (
                LogEvent.ANALYSIS_CUTOFF
                if "analysis" in stage
                else LogEvent.CUTOFF_FILTER
            ),
            "stage": stage,
            **stats,
        }
        if extra_fields:
            log_payload.update(extra_fields)

        ratio_pct = stats.get("filter_ratio_pct", 0.0)
        if ratio_pct > warn_ratio * 100:
            log_payload["event"] = LogEvent.HIGH_CUTOFF_WARNING
            logger.warning(
                f"[{stage}] 컷오프 필터링 비율이 임계값({int(warn_ratio * 100)}%)을 초과했습니다. "
                "검색 품질 저하가 의심됩니다.",
                extra=log_payload,
            )
        else:
            logger.info(f"[{stage}] 컷오프 필터링 결과", extra=log_payload)

    # =========================================================================
    # Keyword Extraction for Hybrid Search
    # =========================================================================

    async def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for BM25 search.
        Uses both rule-based extraction and optional LLM enhancement.
        """
        from src.vector_db import KeywordExtractor

        # Rule-based extraction
        keywords = KeywordExtractor.extract(text, max_keywords=15)

        return keywords

    def extract_patent_ids(self, text: str) -> List[str]:
        """
        Extract patent IDs (e.g., CN-119821168-A, KR-102842452-B1) from text.
        """
        # Precise pattern for CC-NUMBER-SUFFIX or CC-NUMBER
        pattern = r"\b([A-Z]{2}[-]?\d{4,}(?:[-][A-Z0-9]+)?)\b"

        matches = re.findall(pattern, text, re.ASCII)
        # Filter and clean
        cleaned = []
        for m in matches:
            if re.search(r"\d{4,}", m):  # Ensure it has enough digits to be a patent ID
                cleaned.append(m.upper())

        return list(set(cleaned))

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, ValueError)
        ),
    )
    async def _fetch_by_ids_safe(self, ids: List[str]) -> List[Any]:
        """Wrapper for ID fetch with retry AND validation."""
        results = await self.db_client.async_fetch_by_ids(ids)

        # Validation: If we requested N IDs, we expect N results (or reasonably close)
        # Note: Pinecone might return fewer if not found, but in our Golden Dataset,
        # we assume all IDs exist. If not found, it's likely a consistency/timeout issue.
        if len(results) < len(ids):
            missing_count = len(ids) - len(results)
            # Create a custom error to trigger retry
            raise ValueError(
                f"Partial retrieval detected. Requested {len(ids)}, got {len(results)}. Missing {missing_count} items."
            )

        return results

    # =========================================================================
    # 1. HyDE - Hypothetical Document Embedding
    # =========================================================================

    async def generate_hypothetical_claim(self, user_idea: str) -> str:
        # Skip HyDE model when degraded
        if self.db_client is None:
            logger.warning("generate_hypothetical_claim skipped in degraded mode")
            return user_idea

        """
        Generate a hypothetical patent claim from user's idea.
        """
        system_prompt = """당신은 20년 경력의 특허 분쟁 대응 전문 변리사입니다. 
당신의 목표는 사용자의 추상적인 아이디어를 바탕으로, 법적/기술적으로 가장 명확하고 구체적인 '독립 청구항(Independent Claim)'의 형태로 가상의 특허를 작성하는 것입니다.

이 가상 청구항은 실제 특허 데이터셋에서 유사한 기술을 찾아내기 위한 검색 쿼리로 사용됩니다.
반드시 사용자가 제공한 아이디어 범위 내에서만 작성하십시오."""

        # 사용자 입력을 <user_query> 태그로 감싸 시스템 프롬프트와 분리 (Prompt Injection 방어)
        user_prompt = f"아이디어:\n{wrap_user_query(user_idea)}\n\n위 아이디어를 바탕으로 한 전문적인 가상 제1항(독립항)을 작성하십시오."

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.hyde_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            hypothetical_claim = response.choices[0].message.content.strip()
            logger.info(
                f"Generated hypothetical claim: {hypothetical_claim[:100]}...",
                extra={"event": LogEvent.HYDE_START},
            )

            return hypothetical_claim
        except Exception:
            logger.exception(
                "HyDE 청구항 생성 실패. 원본 아이디어를 폴백으로 반환합니다."
            )
            return user_idea

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI text-embedding-3-small."""
        try:
            response = await self.client.embeddings.create(
                model=config.agent.embedding_model,
                input=text,
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return a zero vector as fallback to avoid crashing the whole pipeline
            # 1536 is the dimension for text-embedding-3-small
            dim = 1536 if "small" in config.agent.embedding_model else 3072
            return np.zeros(dim, dtype=np.float32)

    async def generate_multi_queries(self, user_idea: str) -> List[str]:
        # Skip LLM query generation in degraded mode
        if self.db_client is None:
            logger.warning("generate_multi_queries skipped in degraded mode")
            return [user_idea]

        """
        Generate multiple search queries for better coverage.
        Returns 3 queries:
        1. Technical reformulation (synonyms)
        2. Claim-style phrasing
        3. Problem-solution keywords
        """
        system_prompt = """당신은 특허 검색 전문가입니다. 사용자의 아이디어를 바탕으로 검색 범위를 넓히기 위해 3가지 다른 관점의 검색 쿼리를 생성하십시오.
JSON 형식으로 응답하십시오:
{
  "queries": [
    "쿼리 1: 전문 용어 및 유의어 중심 (Technical Formulation)",
    "쿼리 2: 청구항 스타일 구문 (Claim-style Phrasing)",
    "쿼리 3: 해결하려는 과제와 솔루션 키워드 (Problem-Solution)"
  ]
}"""

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.hyde_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": wrap_user_query(user_idea)},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            data = json_loads(response.choices[0].message.content)
            queries = data.get("queries", [])
            logger.info(f"Generated {len(queries)} multi-queries")
            return queries[:3]  # Ensure max 3

        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}")
            return [user_idea]  # Fallback to original

    async def hyde_search(
        self,
        user_idea: str,
        top_k: int = config.agent.top_k_results,
        use_hybrid: bool = True,
    ) -> Tuple[str, List[PatentSearchResult]]:
        """
        HyDE-enhanced patent search (Single Query Version).
        """
        # Generate hypothetical claim
        hypothetical_claim = await self.generate_hypothetical_claim(user_idea)

        # Check if index is available
        if not self.index_loaded():
            logger.warning("Index not loaded. Returning empty results.")
            return hypothetical_claim, []

        results = await self._execute_search(
            hypothetical_claim, user_idea, top_k, use_hybrid
        )
        return hypothetical_claim, results

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, httpx.RequestError)
        ),  # Retry on network/API related exceptions
    )
    async def _execute_search(
        self,
        query_text: str,
        context_text: str,
        top_k: int,
        use_hybrid: bool,
        ipc_filters: Optional[List[str]] = None,
    ) -> List[PatentSearchResult]:
        """Internal helper to execute actual search."""
        # Fallback: If vector DB is unavailable (degraded mode), perform a
        # lightweight keyword-based search over a small builtin mock dataset.
        # This enables UI testing and avoids returning empty results for common
        # queries such as 'drone delivery'. In production, ensure `PINECONE_API_KEY`
        # is set so full hybrid search is used.
        if self.db_client is None:
            logger.warning(
                "DB client unavailable: using local keyword fallback search."
            )
            # Very small mock corpus — extend as needed for local testing
            mock_corpus = [
                {
                    "publication_number": "US-20210123456-A1",
                    "title": "Autonomous aerial vehicle for parcel delivery",
                    "abstract": "An unmanned aerial vehicle (UAV) system for delivering goods to customers, including navigation and package release mechanisms.",
                    "claims": "1. An unmanned aerial delivery system comprising ...",
                    "ipc_code": "B64C",
                },
                {
                    "publication_number": "KR-20220098765-B1",
                    "title": "Drone-based food delivery apparatus and method",
                    "abstract": "A drone platform designed to transport and deliver prepared food items to specified locations with temperature control.",
                    "claims": "1. A method for delivering food using an aerial drone ...",
                    "ipc_code": "B64C",
                },
            ]

            q = (query_text + " " + context_text).lower()
            results: List[PatentSearchResult] = []
            for doc in mock_corpus:
                score = 0.0
                # simple keyword matching heuristic
                # match both English and Korean keywords
                if any(
                    k in q
                    for k in [
                        "drone",
                        "uav",
                        "aerial",
                        "delivery",
                        "food",
                        "드론",
                        "음식",
                        "배달",
                        "무인",
                        "무인항공기",
                    ]
                ):
                    score = 0.9
                elif any(
                    k in q
                    for k in [
                        "autonomous",
                        "navigation",
                        "package",
                        "착륙",
                        "온도",
                        "수납",
                    ]
                ):
                    score = 0.6

                if score > 0.0:
                    results.append(
                        PatentSearchResult(
                            publication_number=doc["publication_number"],
                            title=doc["title"],
                            abstract=doc["abstract"],
                            claims=doc["claims"],
                            ipc_codes=[doc.get("ipc_code", "")],
                            similarity_score=score,
                            dense_score=score,
                            sparse_score=score,
                            rrf_score=score,
                        )
                    )

            # sort and trim
            results.sort(key=lambda r: r.rrf_score, reverse=True)
            return results[:top_k]
        # Embed query
        query_embedding = await self.embed_text(query_text)

        # Extract keywords
        keywords = await self.extract_keywords(context_text + " " + query_text)
        keyword_query = " ".join(keywords)

        # Search
        if use_hybrid:
            search_results = await self.db_client.async_hybrid_search(
                query_embedding,
                keyword_query,
                top_k=top_k,
                dense_weight=config.agent.dense_weight,
                sparse_weight=config.agent.sparse_weight,
                ipc_filters=ipc_filters,
            )
        else:
            search_results = await self.db_client.async_search(
                query_embedding,
                top_k=top_k,
                ipc_filters=ipc_filters,
            )

        # Convert objects
        results = []
        for r in search_results:
            results.append(
                PatentSearchResult(
                    publication_number=r.patent_id,
                    title=r.metadata.get("title", ""),
                    abstract=r.metadata.get("abstract", r.content[:500]),
                    claims=r.metadata.get("claims", ""),
                    ipc_codes=(
                        [r.metadata.get("ipc_code", "")]
                        if r.metadata.get("ipc_code")
                        else []
                    ),
                    similarity_score=r.score,
                    dense_score=r.dense_score,
                    sparse_score=r.sparse_score,
                    rrf_score=r.rrf_score,
                )
            )
        return results

    async def search_multi_query(
        self,
        user_idea: str,
        top_k: int = config.agent.top_k_results,
        use_hybrid: bool = True,
        ipc_filters: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[PatentSearchResult]]:
        # If running without a DB client (degraded mode), avoid all LLM calls
        if self.db_client is None:
            logger.warning(
                "search_multi_query in degraded mode: skipping LLM and returning fresh fallback results."
            )
            # directly return single-query and run _execute_search which will use keyword fallback
            results = await self._execute_search(
                user_idea, user_idea, top_k, use_hybrid
            )
            return [user_idea], results

        # 1. Detect specific patent IDs in user idea
        target_ids = self.extract_patent_ids(user_idea)
        target_results = []
        if target_ids:
            logger.info(f"Detected target patents in query: {target_ids}")
            raw_target_results = await self._fetch_by_ids_safe(target_ids)

            # Convert to PatentSearchResult
            for r in raw_target_results:
                target_results.append(
                    PatentSearchResult(
                        publication_number=r.patent_id,
                        title=r.metadata.get("title", ""),
                        abstract=r.metadata.get("abstract", r.content[:500]),
                        claims=r.metadata.get("claims", ""),
                        ipc_codes=(
                            [r.metadata.get("ipc_code", "")]
                            if r.metadata.get("ipc_code")
                            else []
                        ),
                        similarity_score=r.score,
                        dense_score=r.dense_score,
                        sparse_score=r.sparse_score,
                        rrf_score=r.rrf_score,
                        is_prioritized=True,  # Mark as prioritized
                    )
                )
            logger.info(f"Found {len(target_results)} requested patents in DB")

        # 2. Generate queries for broader search
        queries = await self.generate_multi_queries(user_idea)
        if not queries:
            queries = [user_idea]

        logger.info(f"Executing Multi-Query Search with: {queries}")

        # 3. Parallel Execution using asyncio.gather
        tasks = [
            self._execute_search(
                query, user_idea, top_k, use_hybrid, ipc_filters=ipc_filters
            )
            for query in queries
        ]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Deduplication & Fusion
        seen_ids = set()
        merged_results = []

        # Pre-populate with target results so they are definitely included
        for r in target_results:
            if r.publication_number not in seen_ids:
                seen_ids.add(r.publication_number)
                r.is_prioritized = True
                merged_results.append(r)

        # Simple Fusion: Round-Robin or Score-based?
        # Using Score-based here (Flatten and sort by RRF/Sim score)
        all_results = []
        for res in results_list:
            if isinstance(res, Exception):
                logger.error(f"Multi-query task failed: {res}")
            else:
                all_results.extend(res)

        # Sort by score descending before dedup to keep highest scoring instance
        all_results.sort(
            key=lambda x: x.rrf_score if use_hybrid else x.similarity_score,
            reverse=True,
        )

        for r in all_results:
            if r.publication_number not in seen_ids:
                seen_ids.add(r.publication_number)
                merged_results.append(r)
            else:
                # If it's a target patent seen again, ensure the is_prioritized flag is preserved
                # if it was already marked as such in merged_results
                pass

        logger.info(
            f"Multi-Query: {len(all_results)} total -> {len(merged_results)} unique results"
        )
        return (
            queries,
            merged_results[: top_k * 2],
        )  # Return more candidates for grading

    # =========================================================================
    # 2. Grading & Rewrite Loop
    # =========================================================================

    async def grade_results(
        self,
        user_idea: str,
        results: List[PatentSearchResult],
    ) -> GradingResponse:
        """Grade each search result for relevance to user's idea."""
        if not results:
            logger.warning("No results to grade", extra={"event": LogEvent.ERROR})
            return GradingResponse(results=[], average_score=0.0)

        logger.info(
            f"Grading {len(results)} results", extra={"event": LogEvent.GRADING_START}
        )

        results_text = "\n\n".join(
            [
                f"[특허 {i+1}: {r.publication_number}]\n"
                f"제목: {r.title}\n"
                f"초록: {r.abstract[:300]}...\n"
                f"청구항: {r.claims[:300]}..."
                for i, r in enumerate(results)
            ]
        )

        system_prompt = """당신은 20년 경력의 특허 분쟁 대응 전문 변리사입니다. 당신의 목표는 검색된 특허가 사용자의 아이디어와 기술적으로 실질적인 관련이 있는지를 '매우 비판적이고 보수적인' 관점에서 평가하는 것입니다.

평가 지침 (CRITICAL):
1. **사실에 기반한 엄격한 평가 (Strict Grounding)**: 검색된 내용에 명백히 존재하지 않는 내용을 유추하여 관련성을 높게 평가하지 마십시오. 
   - 사용자의 아이디어 특징 중 선행 기술에 명시되지 않은 요소가 있다면 이 부분의 가중치를 높게 두어 차이점으로 인지하십시오.
2. **기술적 실현 가능성 및 논리**: 아이디어가 논리적으로 성립하지 않거나 전혀 다른 성질의 기술이 물리적/비물리적으로 결합 불가한 경우, 단순한 키워드 짜집기인 경우 가차없이 0점에 가까운 점수를 부여하십시오.
3. **기술 분야 및 목적의 일치성**: 아이디어의 '진정한 기술적 과제'와 특허의 '해결하려는 과제'가 일치하는지 최우선으로 검토하십시오.
4. **점수 부여 기준표 (Rubric) (0.0 ~ 1.0)**:
   - 0.8~1.0: 기술적 수단과 목적이 거의 동일하며 선행 특허의 청구항 내에 아이디어 내용이 모두 포함됨 (직접적 침해 리스크)
   - 0.5~0.7: 기술 분야는 같으나 세부 달성 수단(구성요소)에 명확한 차이가 있음 (개량 또는 회피 가능)
   - 0.1~0.4: 단어/키워드만 일부 일치할 뿐 기술적 맥락이나 해결하려는 과제가 완전 상이함 (단순 참고)
   - 0.0: 기술적으로 전혀 무관함 (환각 방지 컷오프)

평가 시 '오이맛 소고기'와 같이 키워드(육종, 소고기, 오이)는 존재하나 기술적 실체가 불분명하거나 논리적 비약이 있는 경우, 유사도가 높게 측정되지 않도록 엄격하게 심사하십시오.
반드시 JSON 형식으로 응답하십시오."""

        user_prompt = f"""[사용자 아이디어]
{wrap_user_query(user_idea)}

[검색된 특허 목록]
{results_text}

각 특허에 대해 다음 JSON 형식으로 평가하십시오:
{{
  "results": [
    {{"patent_id": "특허번호", "score": 0.0-1.0, "reason": "평가 이유"}}
  ],
  "average_score": 전체평균점수
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.grading_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"Grading API call failed: {e}")
            # Fallback for prioritized results
            for result in results:
                if result.is_prioritized:
                    result.grading_score = 1.0
                    result.grading_reason = (
                        "[PRIORITIZED] Grading API failed but ID matched"
                    )
            return GradingResponse(results=[], average_score=0.0)

        try:
            grading_data = json_loads(response.choices[0].message.content)
            grading_response = GradingResponse(**grading_data)

            for grade in grading_response.results:
                for result in results:
                    if result.publication_number == grade.patent_id:
                        # 우선순위 부스트: 명시적으로 요청된 특허는 점수를 1.0으로 강제
                        if result.is_prioritized:
                            result.grading_score = 1.0
                            result.grading_reason = f"[PRIORITIZED] {grade.reason}"
                        else:
                            result.grading_score = grade.score
                            result.grading_reason = grade.reason

            # Failsafe: 우선순위 결과는 LLM이 누락하더라도 항상 부스트 보장
            for result in results:
                if result.is_prioritized:
                    result.grading_score = 1.0
                    if not result.grading_reason:
                        result.grading_reason = (
                            "[PRIORITIZED] Explicitly requested by user"
                        )
                    elif "[PRIORITIZED]" not in result.grading_reason:
                        result.grading_reason = f"[PRIORITIZED] {result.grading_reason}"

            # [Issue #18] 컷오프 필터링 통계 계산 및 로깅 (헬퍼 활용 — DRY)
            filter_stats = self._compute_filter_stats(results)
            grading_response.filter_stats = filter_stats
            self._log_filter_stats(
                filter_stats,
                stage="grade_results",
                extra_fields={
                    "average_grading_score": round(grading_response.average_score, 3)
                },
            )

            return grading_response

        except Exception as e:
            logger.error(f"Failed to parse grading response: {e}")
            # 오류 발생 시에도 우선순위 결과는 복원
            for result in results:
                if result.is_prioritized:
                    result.grading_score = 1.0
                    result.grading_reason = (
                        "[PRIORITIZED] Grading failed but ID matched"
                    )
            return GradingResponse(results=[], average_score=0.0)

    async def rewrite_query(
        self,
        user_idea: str,
        previous_results: List[PatentSearchResult],
    ) -> QueryRewriteResponse:
        """Optimize search query based on poor results."""
        results_summary = "\n".join(
            [
                f"- {r.publication_number}: score={r.grading_score:.2f}, {r.grading_reason}"
                for r in previous_results
            ]
        )

        prompt = f"""검색 결과가 관련성이 낮습니다. 검색 쿼리를 최적화해주세요.

[원래 아이디어]
{wrap_user_query(user_idea)}

[이전 검색 결과 (낮은 점수)]
{results_summary}

JSON 형식으로 응답:
{{
  "optimized_query": "개선된 검색 쿼리",
  "keywords": ["핵심", "기술", "키워드"],
  "reasoning": "개선 이유"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.grading_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Rewrite API call failed: {e}")
            return QueryRewriteResponse(
                optimized_query=user_idea,
                keywords=[],
                reasoning="Rewrite API call failed",
            )

        try:
            data = json_loads(response.choices[0].message.content)
            return QueryRewriteResponse(**data)
        except Exception as e:
            logger.error(f"Failed to parse rewrite response: {e}")
            return QueryRewriteResponse(
                optimized_query=user_idea, keywords=[], reasoning="Failed to optimize"
            )

    async def search_with_grading(
        self,
        user_idea: str,
        use_hybrid: bool = True,
        ipc_filters: Optional[List[str]] = None,
    ) -> List[PatentSearchResult]:
        """Complete search pipeline with grading and optional rewrite."""
        # Initial Search (Multi-Query handles ID prioritization)
        queries, results = await self.search_multi_query(
            user_idea, use_hybrid=use_hybrid, ipc_filters=ipc_filters
        )

        if not results:
            logger.warning("No search results found")
            return []

        # 그레이딩 실행
        # If running in degraded/local fallback mode (no DB client), skip LLM grading
        if self.db_client is None:
            logger.info(
                "DB client missing — applying local fallback grading to avoid LLM calls."
            )
            for r in results:
                # boost fallback results so they are considered relevant for analysis
                r.grading_score = 0.9
                r.grading_reason = "[LOCAL_FALLBACK] heuristic match"
            avg_score = sum(r.grading_score for r in results) / len(results)
            filter_stats = self._compute_filter_stats(results)
            grading = SimpleNamespace(
                average_score=avg_score, filter_stats=filter_stats
            )
            logger.info(
                f"Initial grading (fallback) - Average score: {grading.average_score:.2f}"
            )
        else:
            grading = await self.grade_results(user_idea, results)
            logger.info(f"Initial grading - Average score: {grading.average_score:.2f}")

        # [Issue #18] grade_results()가 반환한 filter_stats 재활용 (중복 연산 제거)
        if grading.filter_stats:
            self._log_filter_stats(
                grading.filter_stats,
                stage="search_with_grading",
                extra_fields={
                    "rewrite_trigger_threshold": config.agent.grading_threshold,
                    "will_rewrite": grading.average_score
                    < config.agent.grading_threshold,
                },
            )

        # Check if rewrite is needed
        if grading.average_score < config.agent.grading_threshold:
            logger.info(
                f"Score below threshold ({config.agent.grading_threshold}), attempting query rewrite..."
            )

            rewrite = await self.rewrite_query(user_idea, results)
            logger.info(f"Rewritten query: {rewrite.optimized_query}")

            _, new_results = await self.search_multi_query(
                rewrite.optimized_query, use_hybrid=use_hybrid, ipc_filters=ipc_filters
            )

            new_grading = await self.grade_results(user_idea, new_results)
            logger.info(
                f"After rewrite - Average score: {new_grading.average_score:.2f}"
            )

            if new_grading.average_score > grading.average_score:
                results = new_results
                grading = new_grading

        results.sort(key=lambda x: x.grading_score, reverse=True)

        return results

    # =========================================================================
    # 3. Critical CoT Analysis - Standard (Non-Streaming)
    # =========================================================================

    async def critical_analysis(
        self,
        user_idea: str,
        results: List[PatentSearchResult],
    ) -> CriticalAnalysisResponse:
        """
        Perform critical Chain-of-Thought analysis (non-streaming).
        """
        if not results:
            return self._empty_analysis()

        # [Issue #18] 분석 진입 전 컷오프 필터 적용 및 로깅 (헬퍼 활용)
        logger.info(
            "Starting critical analysis", extra={"event": LogEvent.ANALYSIS_START}
        )
        relevant_results = [
            r for r in results if r.grading_score >= config.agent.cutoff_threshold
        ][:5]
        filter_stats = self._compute_filter_stats(results)
        # 실제 분석에 사용되는 수 반영 (top-5 제한 포함)
        filter_stats["after_filter"] = len(relevant_results)
        filter_stats["filtered_out"] = filter_stats["before_filter"] - len(
            relevant_results
        )
        self._log_filter_stats(filter_stats, stage="critical_analysis")

        if not relevant_results:
            # 분석 가능한 결과 없음 → 환각 방지를 위해 명시적 메시지 반환
            patents_text = "제공된 검색 결과 중 분석할 가치가 있는(점수 0.3 이상) 관련 특허가 없습니다."
        else:
            patents_text = "\n\n".join(
                [
                    f"=== 특허 {r.publication_number} ===\n"
                    f"제목: {r.title}\n"
                    f"IPC: {', '.join(r.ipc_codes[:3])}\n"
                    f"초록: {r.abstract}\n"
                    f"청구항: {r.claims}\n"
                    f"관련성 점수: {r.grading_score:.2f} ({r.grading_reason})"
                    for r in relevant_results
                ]
            )

        system_prompt, user_prompt = self._build_analysis_prompts(
            user_idea, patents_text
        )

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.analysis_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2500,
            )

            data = json_loads(response.choices[0].message.content)
            return CriticalAnalysisResponse(**data)

        except Exception as e:
            logger.error(f"Analysis failed with {config.agent.analysis_model}: {e}")
            logger.warning(f"Falling back to {config.agent.fallback_model}...")

            try:
                # Fallback implementation
                response = await self.client.chat.completions.create(
                    model=config.agent.fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=2500,
                )

                data = json_loads(response.choices[0].message.content)
                return CriticalAnalysisResponse(**data)
            except Exception as fallback_error:
                logger.error(f"Fallback analysis failed: {fallback_error}")
                return self._empty_analysis()

    # =========================================================================
    # 4. Critical CoT Analysis - Streaming
    # =========================================================================

    async def critical_analysis_stream(
        self,
        user_idea: str,
        results: List[PatentSearchResult],
    ) -> AsyncGenerator[str, None]:
        """
        Perform critical Chain-of-Thought analysis with streaming.

        Yields:
            Tokens as they are generated by the LLM
        """
        if not results:
            yield "분석할 특허가 없습니다."
            return

        # [Issue #18] 스트리밍 분석 진입 전 컷오프 필터 로깅 (헬퍼 활용)
        logger.info(
            "Starting critical analysis stream",
            extra={"event": LogEvent.ANALYSIS_STREAM_START},
        )
        relevant_results = [
            r for r in results if r.grading_score >= config.agent.cutoff_threshold
        ][:5]
        filter_stats = self._compute_filter_stats(results)
        filter_stats["after_filter"] = len(relevant_results)
        filter_stats["filtered_out"] = filter_stats["before_filter"] - len(
            relevant_results
        )
        self._log_filter_stats(filter_stats, stage="critical_analysis_stream")

        if not relevant_results:
            patents_text = "제공된 검색 결과 중 분석할 가치가 있는(점수 0.3 이상) 관련 특허가 없습니다."
        else:
            patents_text = "\n\n".join(
                [
                    f"=== 특허 {r.publication_number} ===\n"
                    f"제목: {r.title}\n"
                    f"IPC: {', '.join(r.ipc_codes[:3])}\n"
                    f"초록: {r.abstract[:500]}\n"
                    f"청구항: {r.claims[:500]}\n"
                    f"관련성 점수: {r.grading_score:.2f}"
                    for r in relevant_results
                ]
            )

        system_prompt = """당신은 20년 경력의 특허 분쟁 대응 전문 변리사입니다. 당신의 목표는 제공된 선행 특허(Context)와 사용자의 아이디어를 '매우 비판적이고 보수적인' 관점에서 대비하여 침해 리스크와 기술적 유사도를 정밀하게 분석하는 것입니다.

분석 원칙 (CRITICAL):
1. **사실에만 기반 (Strict Faithfulness, No Hallucination)**: 
   - 오직 아래에 제공된 참조 특허(Context) 텍스트의 '문언'에만 근거하십시오.
   - **절대 Context에 없는 구성요소나 기술적 효과를 만들어내지 마십시오 (NEVER FABRICATE).**
   - 사용자의 아이디어 특징 중 선행 특허에서 찾을 수 없는 독창적인 부분은 억지로 유사성을 지어내지 말고, "해당 구성요소는 선행 특허에서 조회되지 않음"으로 명시하십시오.

2. **명시적 인용 의무 (Explicit Citation Standard)**:
   - 분석의 모든 근거 문장 끝에는 반드시 `[출처: 특허번호]` 형태로 인용하십시오. 
   - 예시: "A 센서를 이용하여 데이터를 수집하는 특징이 동일합니다. [출처: KR-101234567-B1]"
   - 인용할 특정 특허나 문구가 없다면, 해당 내용을 창작하지 마십시오.

3. **불확실성 인정 (Acknowledge Uncertainty)**:
   - Context에 판단을 내릴 충분한 정보가 없다면 "정보 부족"으로 표기하거나 "관련 내용을 찾을 수 없습니다"라고 명확하게 선언하십시오.

4. **엄격한 구성요소 대비 (All Elements Rule)**: 
   - 판단 시 청구항을 잘게 쪼개어 구성요소를 1:1로 대비하고, 하나라도 아이디어에 결여되어 있거나 선행 특허에 없다면 비침해(또는 신규성 있음) 방향으로 판단하십시오.

**중요**: 마크다운 형식으로 실시간 출력하십시오.

[출력 형식 및 Few-Shot 예시]
## 1. 유사도 평가
- **핵심 기술**: (아이디어 정의)
- **종합 점수**: (0-100점)
- (특허별 비교 코멘트 예시: 본 아이디어의 '지문인식 결제' 요소는 [출처: KR-100000000-B1]의 청구항 1에 명시된 특징과 거의 일치함. 그러나 '홍채 인식' 관련 구성은 제시된 선행 특허 어디에서도 찾아볼 수 없음.)

## 2. 청구항 기반 침해 리스크
※ 각 특허별로 가장 위험한 청구항을 분석합니다.

### [특허번호] 제목
- **위험 청구항**: 제1항
- **구성요소 대비**:
  - [아이디어 구성] vs [청구항 구성] → **불일치** (선행 특허에 해당 내용 없음)
- **리스크**: 🟢 Low (선행 특허에 핵심 요소가 결여되어 있으므로 실시 가능성이 높음)

(다른 특허 반복...)

## 3. 회피 전략
(엄격한 분석 통과 후 제안 가능한 방향)

## 4. 결론
(최종 권고)"""

        user_prompt = f"""[분석 대상: 사용자 아이디어]
{wrap_user_query(user_idea)}

[참조 특허 목록 (선행 기술)]
{patents_text}

위 선행 특허들의 **청구항(Claims)**을 중심으로 아이디어와 정밀 대비 분석을 수행하십시오."""

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.analysis_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=2500,
            )
        except Exception as e:
            logger.error(f"Analysis stream API call failed: {e}")
            yield "분석 서비스를 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해주세요."
            return

        try:
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            logger.exception("스트리밍 중 오류 발생. 스트림을 종료합니다.")
            return

    def _build_analysis_prompts(
        self, user_idea: str, patents_text: str
    ) -> Tuple[str, str]:
        """Build system and user prompts for analysis."""
        system_prompt = """당신은 20년 경력의 특허 분쟁 대응 전문 변리사입니다. 
당신의 목표는 제공된 선행 특허(Context)와 사용자의 아이디어를 대비하여, 신규성이나 진보성이 부정될 수 있는지 혹은 침해 리스크가 있는지를 '매우 비판적이고 보수적인' 관점에서 정밀 분석하는 것입니다.

분석 원칙 (CRITICAL):
1. **사실에만 기반 (Strict Faithfulness, No Hallucination)**: 
   - 오직 아래에 제공된 참조 특허(Context) 텍스트의 '문언'에만 근거하십시오.
   - **절대 Context에 없는 구성요소나 기술적 효과를 만들어내지 마십시오 (NEVER FABRICATE).**
   - 사용자의 아이디어 특징 중 선행 특허에서 찾을 수 없는 독창적인 부분은 "해당 구성요소는 선행 특허에서 조회되지 않음"이라고 명확히 답변하고 억지로 유사성을 끼워맞추지 마십시오.

2. **명시적 인용 의무 (Explicit Citation Standard)**:
   - 분석의 모든 핵심 근거에는 반드시 `[출처: 특허번호]` 형태로 명시적 인용을 다십시오.
   - 예: "이미지 인식 기술을 활용한 자율주행 제어부가 일치합니다. [출처: JP-9876543-B2]"
   - 인용할 특정 특허 내용이 없다면, 관련된 내용을 임의로 창작하지 마십시오.

3. **불확실성 인정 (Acknowledge Uncertainty)**:
   - Context에 판단을 내릴 충분한 정보가 없다면 "정보 부족"으로 표기하거나 "관련 내용을 찾을 수 없습니다"라고 명언하십시오. 반대로 무언가가 있다고 추측하는 것은 금지됩니다.

4. **엄격한 구성요소 대비 (All Elements Rule)**: 
   - 청구항의 각 구성요소를 아이디어의 단위 요소별로 1:1 대비하여, 어느 한 쪽이라도 누락된 구성요소가 있다면 엄격히 '불일치'로 취급하여 비침해/신규성 인정을 도출하십시오.
"""

        user_prompt = f"""[분석 대상: 사용자 아이디어]
{wrap_user_query(user_idea)}

[참조 특허 목록 (선행 기술)]
{patents_text}

위 선행 특허들과 사용자 아이디어를 대비 분석하여 아래 JSON 형식으로 응답하십시오:
{{
  "similarity": {{
    "score": 0-100,
    "common_elements": ["공통 구성요소"],
    "summary": "분석 결과",
    "evidence_patents": ["특허번호"]
  }},
  "infringement": {{
    "risk_level": "high/medium/low",
    "risk_factors": ["위험 요소"],
    "summary": "리스크 평가",
    "evidence_patents": ["특허번호"]
  }},
  "avoidance": {{
    "strategies": ["회피 전략"],
    "alternative_technologies": ["대안 기술"],
    "summary": "회피 권고",
    "evidence_patents": ["특허번호"]
  }},
  "component_comparison": {{
    "idea_components": ["아이디어 구성요소"],
    "matched_components": ["일치 구성요소"],
    "unmatched_components": ["신규 구성요소"],
    "risk_components": ["위험 구성요소"]
  }},
  "conclusion": "최종 권고"
}}"""

        return system_prompt, user_prompt

    def _empty_analysis(self) -> CriticalAnalysisResponse:
        """Return empty analysis when no results."""
        return CriticalAnalysisResponse(
            similarity=SimilarityAnalysis(
                score=0,
                common_elements=[],
                summary="분석할 특허가 없습니다.",
                evidence_patents=[],
            ),
            infringement=InfringementAnalysis(
                risk_level="unknown",
                risk_factors=[],
                summary="분석할 특허가 없습니다.",
                evidence_patents=[],
            ),
            avoidance=AvoidanceStrategy(
                strategies=[],
                alternative_technologies=[],
                summary="분석할 특허가 없습니다.",
                evidence_patents=[],
            ),
            component_comparison=ComponentComparison(
                idea_components=[],
                matched_components=[],
                unmatched_components=[],
                risk_components=[],
            ),
            conclusion="검색 결과가 없어 분석을 수행할 수 없습니다.",
        )

    async def parse_streaming_to_structured(
        self,
        user_idea: str,
        streamed_text: str,
        results: List[PatentSearchResult],
    ) -> CriticalAnalysisResponse:
        """
        스트리밍 분석 결과(마크다운)를 경량 모델(GPT-4o-mini)로 JSON 구조화 파싱.

        기존 critical_analysis()의 GPT-4o 호출을 대체하여 비용을 절감합니다.
        스트리밍 텍스트가 비어있거나 파싱 실패 시 _empty_analysis()로 폴백합니다.

        Args:
            user_idea: 사용자의 원본 아이디어 텍스트
            streamed_text: critical_analysis_stream()에서 생성된 마크다운 분석 텍스트
            results: 검색된 특허 결과 리스트 (컨텍스트 보강용)

        Returns:
            CriticalAnalysisResponse: 구조화된 분석 결과
        """
        if not streamed_text or not streamed_text.strip():
            logger.warning("스트리밍 텍스트가 비어있어 빈 분석 결과를 반환합니다.")
            return self._empty_analysis()

        # 참조 특허 번호 목록 (파싱 모델에게 컨텍스트 제공)
        patent_ids = [r.publication_number for r in results if r.grading_score >= 0.3][
            :5
        ]

        system_prompt = """당신은 특허 분석 보고서를 JSON으로 변환하는 데이터 파서입니다.
아래에 제공되는 마크다운 형식의 특허 분석 보고서를 읽고, 정확히 지정된 JSON 스키마로 변환하십시오.

규칙:
1. 보고서에 명시된 정보만 추출하십시오. 새로운 정보를 추가하지 마십시오.
2. 보고서에 해당 필드의 정보가 없으면 빈 문자열 또는 빈 배열로 채우십시오.
3. evidence_patents 필드에는 보고서에 언급된 특허번호만 포함하십시오.
4. score는 0-100 범위의 정수, risk_level은 'high', 'medium', 'low' 중 하나입니다."""

        user_prompt = f"""[사용자 아이디어]
{wrap_user_query(user_idea)}

[참조 특허 번호]
{', '.join(patent_ids) if patent_ids else 'N/A'}

[마크다운 분석 보고서]
{streamed_text}

위 보고서를 아래 JSON 스키마로 변환하십시오:
{{
  "similarity": {{
    "score": 0-100,
    "common_elements": ["공통 구성요소"],
    "summary": "유사도 평가 요약",
    "evidence_patents": ["특허번호"]
  }},
  "infringement": {{
    "risk_level": "high/medium/low",
    "risk_factors": ["위험 요소"],
    "summary": "침해 리스크 요약",
    "evidence_patents": ["특허번호"]
  }},
  "avoidance": {{
    "strategies": ["회피 전략"],
    "alternative_technologies": ["대안 기술"],
    "summary": "회피 전략 요약",
    "evidence_patents": ["특허번호"]
  }},
  "component_comparison": {{
    "idea_components": ["아이디어 구성요소"],
    "matched_components": ["일치 구성요소"],
    "unmatched_components": ["신규 구성요소"],
    "risk_components": ["위험 구성요소"]
  }},
  "conclusion": "최종 권고"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=config.agent.parsing_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
                timeout=30.0,
            )

            data = json_loads(response.choices[0].message.content)
            logger.info(
                f"스트리밍 결과 JSON 파싱 성공 (모델: {config.agent.parsing_model})"
            )
            return CriticalAnalysisResponse(**data)

        except Exception as e:
            logger.error(f"스트리밍 결과 파싱 실패 ({config.agent.parsing_model}): {e}")
            logger.warning("폴백: 빈 분석 결과를 반환합니다.")
            return self._empty_analysis()

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    async def analyze(
        self,
        user_idea: str,
        use_hybrid: bool = True,
        stream: bool = False,
        ipc_filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Complete Self-RAG pipeline.

        Args:
            user_idea: User's patent idea
            use_hybrid: Use hybrid search (dense + sparse)
            stream: Stream analysis output (not applicable for dict output)
        """
        # [Security] 사용자 입력 샌드박싱 적용 (Issue #17)
        try:
            user_idea = sanitize_user_input(user_idea)
        except PromptInjectionError as e:
            logger.error(f"[Security] Analysis blocked: {e}")
            return {"error": str(e), "security_alert": True}

        logger.info(
            "RAG 파이프라인 시작",
            extra={
                "event": LogEvent.PIPELINE_START,
                "idea_preview": user_idea[:100],
                "use_hybrid": use_hybrid,
            },
        )

        logger.info("Step 1-2: HyDE + Hybrid Search & Grading 시작")
        results = await self.search_with_grading(
            user_idea, use_hybrid=use_hybrid, ipc_filters=ipc_filters
        )

        if not results:
            return {"error": "No relevant patents found"}

        logger.info(
            "검색 완료",
            extra={"event": LogEvent.SEARCH_DONE, "result_count": len(results)},
        )
        for r in results[:3]:
            logger.info(
                "상위 검색 결과",
                extra={
                    "event": LogEvent.TOP_RESULT,
                    "patent_id": r.publication_number,
                    "grading_score": round(r.grading_score, 4),
                    "rrf_score": round(r.rrf_score, 4),
                },
            )

        logger.info("Step 3: Critical CoT Analysis 시작")
        if self.db_client is None:
            logger.warning(
                "Degraded mode: skipping LLM critical analysis; returning placeholder analysis."
            )
            analysis = self._empty_analysis()
        else:
            analysis = await self.critical_analysis(user_idea, results)

        output = {
            "user_idea": user_idea,
            "search_results": [
                {
                    "patent_id": r.publication_number,
                    "title": r.title,
                    "abstract": r.abstract,  # Added for DeepEval Faithfulness
                    "claims": r.claims,  # Added for DeepEval Faithfulness
                    "grading_score": r.grading_score,
                    "grading_reason": r.grading_reason,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                    "rrf_score": r.rrf_score,
                }
                for r in results
            ],
            "analysis": {
                "similarity": {
                    "score": analysis.similarity.score,
                    "common_elements": analysis.similarity.common_elements,
                    "summary": analysis.similarity.summary,
                    "evidence": analysis.similarity.evidence_patents,
                },
                "infringement": {
                    "risk_level": analysis.infringement.risk_level,
                    "risk_factors": analysis.infringement.risk_factors,
                    "summary": analysis.infringement.summary,
                    "evidence": analysis.infringement.evidence_patents,
                },
                "avoidance": {
                    "strategies": analysis.avoidance.strategies,
                    "alternatives": analysis.avoidance.alternative_technologies,
                    "summary": analysis.avoidance.summary,
                    "evidence": analysis.avoidance.evidence_patents,
                },
                "conclusion": analysis.conclusion,
            },
            "timestamp": datetime.now().isoformat(),
            "search_type": "hybrid" if use_hybrid else "dense",
        }

        logger.info(
            "파이프라인 완료",
            extra={
                "event": LogEvent.PIPELINE_COMPLETE,
                "similarity_score": analysis.similarity.score,
                "risk_level": analysis.infringement.risk_level,
                "conclusion_preview": analysis.conclusion[:100],
            },
        )

        return output


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main():
    """Interactive CLI for patent analysis."""
    print("\n" + "=" * 70)
    print("⚡ 쇼특허 (Short-Cut) v3.0 - Self-RAG Patent Agent")
    print("    Hybrid Search + Streaming Edition")
    print("=" * 70)
    print("\n특허 분석을 위한 아이디어를 입력하세요.")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")

    agent = PatentAgent()

    if not agent.index_loaded():
        print("⚠️  Index not found. Please run the pipeline first:")
        print("   python pipeline.py --stage 5\n")

    while True:
        try:
            # input() is blocking, run in executor to keep event loop free
            user_input = (await asyncio.to_thread(input, "\n💡 Your idea: ")).strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("❌ Please enter an idea.")
                continue

            result = await agent.analyze(user_input, use_hybrid=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"analysis_{timestamp}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_dumps(result))

            print(f"\n💾 Result saved to: {output_path}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
