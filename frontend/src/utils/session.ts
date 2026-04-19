/**
 * 클라이언트 고유 세션 ID 관리 유틸리티 (Issue #24)
 *
 * - 최초 접속 시 UUID v4를 생성하여 localStorage에 영구 저장
 * - 시크릿 모드 등 localStorage 접근 불가 환경은 sessionStorage로 fallback
 * - 다중 탭: 동일 localStorage를 공유하므로 같은 세션 ID 유지
 * - 모든 API 요청의 X-Session-ID 헤더에 사용
 */

const SESSION_KEY = 'shortcut_session_id';

/**
 * 세션 ID 반환 함수.
 * localStorage에 저장된 ID가 없으면 새 UUID를 생성하여 저장 후 반환합니다.
 */
export function getSessionId(): string {
    const generateUUID = () => {
        try {
            if (typeof crypto !== 'undefined' && crypto.randomUUID) {
                return crypto.randomUUID();
            }
        } catch (e) {
            console.warn('crypto.randomUUID failed, falling back to Math.random', e);
        }
        // Fallback for non-secure contexts or older browsers
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            const r = (Math.random() * 16) | 0;
            const v = c === 'x' ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    };

    try {
        let sessionId = localStorage.getItem(SESSION_KEY);
        if (!sessionId) {
            sessionId = generateUUID();
            localStorage.setItem(SESSION_KEY, sessionId);
        }
        return sessionId;
    } catch {
        try {
            let sessionId = sessionStorage.getItem(SESSION_KEY);
            if (!sessionId) {
                sessionId = generateUUID();
                sessionStorage.setItem(SESSION_KEY, sessionId);
            }
            return sessionId;
        } catch {
            return generateUUID();
        }
    }
}

/**
 * 세션 ID를 강제로 초기화합니다.
 * (로그아웃, 테스트 등의 목적으로 사용)
 */
export function resetSessionId(): void {
    try {
        localStorage.removeItem(SESSION_KEY);
        sessionStorage.removeItem(SESSION_KEY);
    } catch {
        // 접근 불가 환경이면 무시
    }
}
