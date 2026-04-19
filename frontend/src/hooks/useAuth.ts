import { useState, useEffect, useCallback } from 'react';
import { UserResponse } from '../types/auth';

// 백엔드 API 베이스 URL
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// localStorage 키
const TOKEN_KEY = 'shortcut_access_token';
const AUTH_CHANGED_EVENT = 'shortcut-auth-changed';

interface AuthState {
    user: UserResponse | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    error: string | null;
}

interface UseAuthReturn extends AuthState {
    login: (email: string, password: string) => Promise<void>;
    signup: (email: string, password: string, name?: string) => Promise<void>;
    logout: () => void;
    clearError: () => void;
}

export const useAuth = (): UseAuthReturn => {
    const [state, setState] = useState<AuthState>({
        user: null,
        isAuthenticated: false,
        isLoading: true,
        error: null,
    });

    const emitAuthChanged = useCallback(() => {
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
        }
    }, []);

    /** 저장된 JWT로 사용자 정보를 복원합니다 (새로고침 시) */
    const restoreUser = useCallback(async () => {
        const token = localStorage.getItem(TOKEN_KEY);
        if (!token) {
            setState((s: AuthState) => ({ ...s, isLoading: false }));
            return;
        }
        try {
            const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                const user: UserResponse = await res.json();
                setState({ user, isAuthenticated: true, isLoading: false, error: null });
            } else {
                // 토큰 만료 등 → 로컬 삭제
                localStorage.removeItem(TOKEN_KEY);
                setState({ user: null, isAuthenticated: false, isLoading: false, error: null });
            }
        } catch {
            localStorage.removeItem(TOKEN_KEY);
            setState({ user: null, isAuthenticated: false, isLoading: false, error: null });
        }
    }, []);

    // 마운트 시 토큰 복원
    useEffect(() => {
        restoreUser();

        const handleAuthChanged = () => {
            restoreUser();
        };
        const handleStorage = (event: StorageEvent) => {
            if (event.key === TOKEN_KEY) {
                restoreUser();
            }
        };

        window.addEventListener(AUTH_CHANGED_EVENT, handleAuthChanged);
        window.addEventListener('storage', handleStorage);

        return () => {
            window.removeEventListener(AUTH_CHANGED_EVENT, handleAuthChanged);
            window.removeEventListener('storage', handleStorage);
        };
    }, [restoreUser]);

    /** 로그인: 이메일 + 비밀번호 → JWT 저장 → 사용자 정보 세팅 */
    const login = useCallback(async (email: string, password: string) => {
        console.log('[useAuth] Attempting login for:', email);
        setState((s: AuthState) => ({ ...s, isLoading: true, error: null }));
        try {
            const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            // 응답 텍스트를 먼저 읽고 안전하게 JSON 파싱
            const text = await res.text();
            let data: any;
            try {
                data = text ? JSON.parse(text) : {};
            } catch {
                console.error('[useAuth] JSON 파싱 실패, 원본 응답:', text.substring(0, 200));
                throw new Error('서버 응답을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.');
            }

            if (!res.ok) {
                throw new Error(data.detail || '로그인에 실패했습니다.');
            }

            localStorage.setItem(TOKEN_KEY, data.access_token);

            // 토큰으로 사용자 정보 조회
            const meRes = await fetch(`${API_BASE}/api/v1/auth/me`, {
                headers: { Authorization: `Bearer ${data.access_token}` },
            });
            if (!meRes.ok) throw new Error('사용자 정보를 가져올 수 없습니다.');
            const user: UserResponse = await meRes.json();

            console.info('[useAuth] Login success:', user.email);
            setState({ user, isAuthenticated: true, isLoading: false, error: null });
            emitAuthChanged();
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : '로그인 중 오류가 발생했습니다.';
            console.error('[useAuth] Login failed:', e);
            setState((s: AuthState) => ({ ...s, isLoading: false, error: msg }));
            throw e; // 폼 레벨에서 처리 가능하도록 re-throw
        }
    }, [emitAuthChanged]);

    /** 회원가입: 이메일 + 비밀번호 + 이름 → 자동 로그인 */
    const signup = useCallback(async (email: string, password: string, name?: string) => {
        console.log('[useAuth] Starting signup for:', email);
        setState((s: AuthState) => ({ ...s, isLoading: true, error: null }));
        try {
            const res = await fetch(`${API_BASE}/api/v1/auth/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, name }),
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || '회원가입에 실패했습니다.');
            }
            // 회원가입 성공 → 자동 로그인
            console.info('[useAuth] Signup success, attempting auto-login.');
            await login(email, password);
        } catch (e: unknown) {
            // login() 실패 시 이미 state 업데이트됨
            if (!(e instanceof Error && e.message.includes('로그인'))) {
                const msg = e instanceof Error ? e.message : '회원가입 중 오류가 발생했습니다.';
                console.error('[useAuth] Signup failed:', e);
                setState((s: AuthState) => ({ ...s, isLoading: false, error: msg }));
            }
            throw e;
        }
    }, [login]);

    /** 로그아웃: 토큰 삭제 + 상태 초기화 */
    const logout = useCallback(() => {
        localStorage.removeItem(TOKEN_KEY);
        setState({ user: null, isAuthenticated: false, isLoading: false, error: null });
        emitAuthChanged();
    }, [emitAuthChanged]);

    const clearError = useCallback(() => {
        setState((s: AuthState) => ({ ...s, error: null }));
    }, []);

    return {
        ...state,
        login,
        signup,
        logout,
        clearError,
    };
};
