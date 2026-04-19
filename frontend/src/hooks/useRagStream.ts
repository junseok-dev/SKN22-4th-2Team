import { useState, useRef, useCallback } from 'react';
import { RagAnalysisResult, PatentContext } from '../types/rag';
import { getSessionId } from '../utils/session';
import type { ErrorType } from '../components/common/ErrorFallback';

export interface RagErrorInfo {
    title: string;
    message: string;
    errorType?: ErrorType; // ErrorFallback 타입 기반 분기를 위한 에러 종류
}

function toFiniteNumber(value: unknown, fallback = 0): number {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function clampPercent(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, Math.round(value)));
}

function normalizeRiskLevel(raw: unknown): 'Low' | 'Medium' | 'High' {
    const level = String(raw || '').toLowerCase();
    if (level.startsWith('h')) return 'High';
    if (level.startsWith('m')) return 'Medium';
    if (level.startsWith('l')) return 'Low';
    return 'Medium';
}

function normalizeUniqueness(raw: unknown, riskLevel: 'Low' | 'Medium' | 'High'): 'Low' | 'Medium' | 'High' {
    const value = String(raw || '').toLowerCase();
    if (value.startsWith('h')) return 'High';
    if (value.startsWith('m')) return 'Medium';
    if (value.startsWith('l')) return 'Low';

    // 위험도가 높을수록 차별성은 낮다고 가정
    if (riskLevel === 'High') return 'Low';
    if (riskLevel === 'Low') return 'High';
    return 'Medium';
}

function normalizeSimilarity(raw: unknown): number {
    const score = toFiniteNumber(raw, 0);
    const percent = score <= 1 ? score * 100 : score;
    return clampPercent(percent);
}

function normalizeTopPatents(raw: any): PatentContext[] {
    if (Array.isArray(raw?.topPatents)) {
        return raw.topPatents.map((p: any, idx: number) => ({
            id: String(p?.id || p?.patent_id || `PAT-${idx + 1}`),
            similarity: normalizeSimilarity(p?.similarity),
            title: String(p?.title || '제목 정보 없음'),
            summary: String(p?.summary || p?.abstract || p?.grading_reason || '').slice(0, 500),
        }));
    }

    const fromSearch = Array.isArray(raw?.search_results) ? raw.search_results : [];
    return fromSearch.slice(0, 5).map((p: any, idx: number) => ({
        id: String(p?.patent_id || p?.id || p?.publication_number || `PAT-${idx + 1}`),
        similarity: normalizeSimilarity(p?.grading_score ?? p?.rrf_score ?? p?.dense_score ?? p?.score),
        title: String(p?.title || '제목 정보 없음'),
        summary: String(p?.grading_reason || p?.abstract || p?.claims || '').slice(0, 500),
    }));
}

function normalizeRagResult(raw: any): RagAnalysisResult {
    const riskLevel = normalizeRiskLevel(raw?.riskLevel ?? raw?.analysis?.infringement?.risk_level);
    const topPatents = normalizeTopPatents(raw);

    let riskScore = toFiniteNumber(raw?.riskScore, NaN);
    if (!Number.isFinite(riskScore)) {
        riskScore = toFiniteNumber(raw?.analysis?.similarity?.score, NaN);
    }
    if (!Number.isFinite(riskScore)) {
        riskScore = riskLevel === 'High' ? 80 : riskLevel === 'Low' ? 30 : 55;
    }

    const similarCount = Number.isFinite(Number(raw?.similarCount))
        ? Math.max(0, Math.round(Number(raw.similarCount)))
        : (Array.isArray(raw?.search_results) ? raw.search_results.length : topPatents.length);

    return {
        riskLevel,
        riskScore: clampPercent(riskScore),
        similarCount,
        uniqueness: normalizeUniqueness(raw?.uniqueness, riskLevel),
        topPatents,
    };
}

export function useRagStream() {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isSkeletonVisible, setIsSkeletonVisible] = useState(false);
    const [isComplete, setIsComplete] = useState(false);
    const [percent, setPercent] = useState(0);
    const [message, setMessage] = useState('');
    const [resultData, setResultData] = useState<RagAnalysisResult | null>(null);
    const [errorInfo, setErrorInfo] = useState<RagErrorInfo | null>(null);

    // 진행중인 fetch 요청을 취소하기 위한 AbortController
    const abortControllerRef = useRef<AbortController | null>(null);
    const idleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const startAnalysis = useCallback(async (
        userIdea: string,
        ipcFilters: string[] | null = null,
        useHybrid: boolean = true
    ) => {
        setIsAnalyzing(true);
        setIsSkeletonVisible(true);
        setIsComplete(false);
        setPercent(0);
        setMessage('네트워크 상의 특허 DB 연결을 시도합니다...');
        setResultData(null);
        setErrorInfo(null);

        // 이전 요청이 있다면 취소
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        const MAX_TIMEOUT_MS = 300000; // 전체 분석 상한: 5분
        const IDLE_TIMEOUT_MS = 120000; // 무응답 상한: 2분

        const resetIdleTimeout = () => {
            if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
            idleTimeoutRef.current = setTimeout(() => {
                if (abortControllerRef.current) {
                    abortControllerRef.current.abort(new Error('TIMEOUT'));
                }
            }, IDLE_TIMEOUT_MS);
        };

        const timeoutId = setTimeout(() => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort(new Error('TIMEOUT'));
            }
        }, MAX_TIMEOUT_MS);
        resetIdleTimeout();

        try {
            // 백엔드 FastAPI SSE 엔드포인트 호출 (POST)
            // 시니어 리뷰 반영: VITE_API_BASE_URL 환경변수 사용
            // 운영 환경(Docker)에서 같은 origin인 경우 서버 주소 생략(상대경로) 가능하도록 기본값 '' 설정
            const apiUrl = import.meta.env.VITE_API_BASE_URL || '';
            // 백엔드 AnalyzeRequest 스키마 필드명에 맞춰 요청 Body 구성
            const sessionId = getSessionId();
            const reqBody = {
                user_idea: userIdea,
                session_id: sessionId,
                use_hybrid: useHybrid,
                ipc_filters: ipcFilters && ipcFilters.length > 0 ? ipcFilters : null,
                stream: true,
            };
            // API 버전 prefix: /api/v1/analyze (app.js 기준으로 통일)
            const response = await fetch(`${apiUrl}/api/v1/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'X-Session-ID': sessionId, // Issue #24: 세션 식별자 헤더
                },
                body: JSON.stringify(reqBody),
                signal: abortController.signal
            });

            if (!response.ok) {
                // 백엔드의 에러 Response Body (detail, error_type) 파싱 시도
                try {
                    const errorData = await response.json();
                    const errorType = errorData.error_type || 'SERVER_ERROR';
                    const detail = errorData.detail || '백엔드 서버에서 오류를 반환했습니다.';
                    
                    // Error 객체에 추가 정보를 담아서 throw
                    const error = new Error(errorType);
                    (error as any).detail = detail;
                    throw error;
                } catch (e: any) {
                    // JSON 파싱 실패 혹은 이미 throw 된 경우 처리
                    if (e.detail) throw e; 

                    if (response.status === 429) {
                        throw new Error('RATE_LIMIT');
                    } else if (response.status === 413 || response.status === 422) {
                        throw new Error('TOKEN_EXCEEDED');
                    } else if (response.status === 404) {
                        throw new Error('NOT_FOUND');
                    } else {
                        throw new Error('SERVER_ERROR');
                    }
                }
            }

            if (!response.body) {
                throw new Error('NETWORK_ERROR');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                resetIdleTimeout();

                // 청크 디코딩 후 버퍼에 누적
                buffer += decoder.decode(value, { stream: true });

                // SSE 표준: 이벤트 블록은 \n\n으로 구분
                const blocks = buffer.split('\n\n');
                // 마지막 블록은 아직 불완전할 수 있으므로 버퍼에 남김
                buffer = blocks.pop() || '';

                for (const block of blocks) {
                    if (!block.trim()) continue;

                    // 표준 SSE 이벤트 블록에서 event:와 data: 추출
                    const eventMatch = block.match(/event:\s*([^\n]+)/);
                    const dataMatch = block.match(/data:\s*([^\n]+)/);

                    const eventType = eventMatch ? eventMatch[1].trim() : 'message';
                    const dataStr = dataMatch ? dataMatch[1].trim() : '';

                    if (!dataStr) continue;

                    try {
                        const parsed = JSON.parse(dataStr);
                        console.debug(`[useRagStream] SSE Event: ${eventType}`, parsed);

                        if (eventType === 'progress') {
                            // 스켈레톤 숨김 (처음 progress 이벤트부터)
                            setIsSkeletonVisible(false);
                            setPercent(parsed.percent ?? 0);
                            setMessage(parsed.message || '분석 중...');
                            resetIdleTimeout();
                        } else if (eventType === 'complete') {
                            console.info('[useRagStream] Analysis Complete:', parsed.result);
                            setPercent(100);
                            setMessage('분석이 모두 완료되었습니다.');
                            setResultData(normalizeRagResult(parsed.result));
                            setTimeout(() => {
                                setIsAnalyzing(false);
                                setIsComplete(true);
                            }, 1500);
                        } else if (eventType === 'empty') {
                            console.warn('[useRagStream] Received empty event');
                            const err = new Error('NOT_FOUND');
                            (err as any).detail = parsed.message;
                            throw err;
                        } else if (eventType === 'error') {
                            console.error('[useRagStream] Received error event:', parsed.detail);
                            const errorType = parsed.error_type || 'SERVER_ERROR';
                            const err = new Error(errorType);
                            (err as any).detail = parsed.detail || '백엔드 처리 중 오류가 발생했습니다.';
                            throw err;
                        }
                    } catch (e: any) {
                        if (
                            e.message === 'NOT_FOUND' ||
                            e.message === 'NETWORK_ERROR' ||
                            e.message === 'SERVER_ERROR' ||
                            e.message === 'VECTOR_DB_UNAVAILABLE'
                        ) throw e;
                        console.error('[useRagStream] SSE parsing/processing error:', e, 'Raw block:', block);
                    }
                }
            }
            clearTimeout(timeoutId);
            if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
        } catch (error: any) {
            clearTimeout(timeoutId);
            if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);

            const signalReason = abortControllerRef.current?.signal?.reason as any;
            const isTimeoutAbort =
                error?.message === 'TIMEOUT' ||
                error?.cause?.message === 'TIMEOUT' ||
                signalReason?.message === 'TIMEOUT';

            // AbortController.abort() 발생 시
            if (error.name === 'AbortError' || isTimeoutAbort) {
                // DOMException AbortError가 타임아웃 타이머에 의해 트리거된 경우를 명시적으로 체킹하기엔 어렵지만 name 또는 custom error throw 패턴
                if (isTimeoutAbort) {
                    setErrorInfo({
                        title: '분석 시간 초과 (Timeout) ⏱️',
                        message: '분석 시간이 길어지고 있습니다. 잠시 후 다시 시도해 주세요.',
                        errorType: 'TIMEOUT',
                    });
                } else {
                    console.log('Analysis request aborted by user');
                }
            } else {
                console.error('Analysis failed:', error);

                // 에러 종류별 매핑
                if (error.message === 'TOKEN_EXCEEDED') {
                    setErrorInfo({
                        title: '입력 텍스트가 너무 깁니다 🚫',
                        message: '입력하신 특허 아이디어가 백엔드 처리 한도를 초과했습니다.',
                        errorType: 'TOKEN_EXCEEDED',
                    });
                } else if (error.message === 'RATE_LIMIT') {
                    setErrorInfo({
                        title: '오늘의 분석 한도를 소진했습니다 🚫',
                        message: '일일 무료 분석 횟수(10회)를 모두 사용했습니다. 내일 다시 시도해 주세요.',
                        errorType: 'RATE_LIMIT',
                    });
                } else if (error.message === 'NOT_FOUND') {
                    setErrorInfo({
                        title: '유사 특허 결과를 찾지 못했습니다 📭',
                        message: error.detail || '입력하신 내용과 일치하는 선행 특허가 없습니다.',
                        errorType: 'NOT_FOUND',
                    });
                } else if (error.message === 'VECTOR_DB_UNAVAILABLE') {
                    setErrorInfo({
                        title: '특허 DB 연결 오류 🛰️',
                        message: error.detail || '벡터 DB 연결에 실패했습니다. 서버 설정을 확인해 주세요.',
                        errorType: 'SERVER_ERROR',
                    });
                } else if (error.message === 'SERVER_ERROR') {
                    setErrorInfo({
                        title: '서버 내부 오류 발생 🛠️',
                        message: error.detail || '백엔드 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.',
                        errorType: 'SERVER_ERROR',
                    });
                } else {
                    setErrorInfo({
                        title: '네트워크 연결 오류 🔌',
                        message: error.detail || '일시적인 연결 문제가 발생했습니다. 백엔드 서버가 켜져 있는지 확인하고 잠시 후 다시 시도해 주세요.',
                        errorType: 'NETWORK_ERROR',
                    });
                }
            }

            // UI 상태 초기화로 Fallback 이나 기본화면 노출 유도
            setIsAnalyzing(false);
            setIsSkeletonVisible(false);
            setPercent(0);
        } finally {
            if (idleTimeoutRef.current) {
                clearTimeout(idleTimeoutRef.current);
                idleTimeoutRef.current = null;
            }
            abortControllerRef.current = null;
        }
    }, []);

    const cancelAnalysis = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        setIsAnalyzing(false);
        setIsSkeletonVisible(false);
        setIsComplete(false);
        setPercent(0);
        setMessage('');
        setResultData(null);
        setErrorInfo(null);
    }, []);

    return {
        isAnalyzing,
        isSkeletonVisible,
        isComplete,
        percent,
        message,
        resultData,
        errorInfo,
        startAnalysis,
        cancelAnalysis,
        setIsComplete,
        setErrorInfo
    };
}
