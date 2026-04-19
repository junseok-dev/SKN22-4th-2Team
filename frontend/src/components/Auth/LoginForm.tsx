import { useState, FormEvent } from 'react';
import { useAuth } from '../../hooks/useAuth';

interface LoginFormProps {
    onSwitchToSignup: () => void;
    onSuccess?: (message: string) => void;
}

/**
 * 로그인 폼 컴포넌트
 * - 이메일/비밀번호 입력 → POST /api/v1/auth/login 호출
 * - 성공 시 모달 닫힘 (onSuccess 콜백)
 */
export const LoginForm = ({ onSwitchToSignup, onSuccess }: LoginFormProps) => {
    const { login, isLoading } = useAuth();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [errorMsg, setErrorMsg] = useState('');

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setErrorMsg('');
        if (!email || !password) {
            setErrorMsg('이메일과 비밀번호를 입력해주세요.');
            return;
        }
        try {
            await login(email, password);
            onSuccess?.('로그인 되셨습니다.');
        } catch (err: unknown) {
            setErrorMsg(err instanceof Error ? err.message : '로그인에 실패했습니다.');
        }
    };

    return (
        <div className="bg-white rounded-2xl shadow-2xl p-8 w-full">
            {/* 헤더 */}
            <div className="mb-6">
                <h2 className="text-2xl font-extrabold text-gray-900">로그인</h2>
                <p className="text-sm text-gray-500 mt-1">특허 분석 기능을 이용하려면 로그인하세요.</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                {/* 이메일 */}
                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-1">이메일</label>
                    <input
                        type="email"
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        placeholder="your@email.com"
                        autoComplete="username"
                        className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-gray-800 text-sm"
                        required
                        disabled={isLoading}
                    />
                </div>

                {/* 비밀번호 */}
                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-1">비밀번호</label>
                    <input
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        placeholder="••••••"
                        autoComplete="current-password"
                        className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-gray-800 text-sm"
                        required
                        disabled={isLoading}
                    />
                </div>

                {/* 에러 메시지 */}
                {errorMsg && (
                    <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                        {errorMsg}
                    </div>
                )}

                {/* 로그인 버튼 */}
                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition-all shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                    {isLoading ? '로그인 중...' : '로그인'}
                </button>
            </form>

            {/* 회원가입 전환 */}
            <p className="text-center text-sm text-gray-500 mt-5">
                계정이 없으신가요?{' '}
                <button
                    onClick={onSwitchToSignup}
                    className="text-blue-600 font-semibold hover:underline"
                >
                    회원가입
                </button>
            </p>
        </div>
    );
};
