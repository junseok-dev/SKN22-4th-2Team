import { ReactNode } from 'react';
import { SignupForm } from './SignupForm';
import { LoginForm } from './LoginForm';
import { User } from '../../types/auth';

interface AuthGuardProps {
    children: ReactNode;
    isGuest: boolean;
    setIsGuest: (v: boolean) => void;
    authView: 'login' | 'signup';
    setAuthView: (v: 'login' | 'signup') => void;
    user: User;
    onAuthSuccess?: (message: string) => void;
}


/**
 * 전역 인증 가드 컴포넌트
 * - 인증되지 않은 상태에서 분석을 시도할 때 로그인/회원가입 모달을 표시합니다.
 * - 배경(메인 콘텐츠)은 항상 렌더링되므로, 첫 접속 시 메인 화면이 보입니다.
 */
export const AuthGuard = (props: AuthGuardProps) => {
    const {
        children,
        isGuest,
        setIsGuest,
        authView,
        setAuthView,
        user,
        onAuthSuccess
    } = props;

    // 로그인/회원가입 성공 시 모달 닫기
    const handleSuccess = (message: string) => {
        setIsGuest(true);
        onAuthSuccess?.(message);
    };

    return (
        <div className="relative min-h-screen">
            {/* 메인 서비스 콘텐츠 (항상 렌더링하여 배경으로 유지) */}
            {children}

            {/* 인증 모달 오버레이 (비로그인 상태이며 게스트 모드가 아닐 때만 표시) */}
            {!user && isGuest === false && (
                <div
                    className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm"
                    aria-modal="true"
                    role="dialog"
                    onClick={e => {
                        // 배경 클릭 시 모달 닫기 (게스트 모드로 전환)
                        if (e.target === e.currentTarget) setIsGuest(true);
                    }}
                >
                    <div className="w-full max-w-md p-4">
                        {authView === 'signup' ? (
                            <SignupForm
                                onSwitchToLogin={() => setAuthView('login')}
                                onSuccess={handleSuccess}
                            />
                        ) : (
                            <LoginForm
                                onSwitchToSignup={() => setAuthView('signup')}
                                onSuccess={handleSuccess}
                            />
                        )}
                        <div className="mt-3 text-center">
                            <button
                                onClick={() => setIsGuest(true)}
                                className="text-sm text-gray-300 hover:text-white underline transition-colors"
                            >
                                지금은 둘러보기만 할게요 →
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
