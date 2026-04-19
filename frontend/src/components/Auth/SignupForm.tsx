import { useState, FormEvent } from 'react';
import { useAuth } from '../../hooks/useAuth';

interface SignupFormProps {
    onSwitchToLogin: () => void;
    onSuccess?: (message: string) => void;
}

/**
 * 회원가입 폼 컴포넌트
 * - 이름 / 이메일 / 비밀번호 / 비밀번호 확인 입력
 * - POST /api/v1/auth/signup 호출 → 성공 시 자동 로그인
 */
export const SignupForm = ({ onSwitchToLogin, onSuccess }: SignupFormProps) => {
    const { signup, isLoading } = useAuth();
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirm, setConfirm] = useState('');
    const [errorMsg, setErrorMsg] = useState('');

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setErrorMsg('');

        if (!email || !password) {
            setErrorMsg('이메일과 비밀번호를 입력해주세요.');
            return;
        }
        if (password.length < 6) {
            setErrorMsg('비밀번호는 최소 6자 이상이어야 합니다.');
            return;
        }
        if (password !== confirm) {
            setErrorMsg('비밀번호가 일치하지 않습니다.');
            return;
        }

        try {
            await signup(email, password, name || undefined);
            onSuccess?.('회원가입 완료 되었습니다.');
        } catch (err: unknown) {
            setErrorMsg(err instanceof Error ? err.message : '회원가입에 실패했습니다.');
        }
    };

    return (
        <div className="bg-white rounded-2xl shadow-2xl p-8 w-full">
            {/* 헤더 */}
            <div className="mb-6">
                <h2 className="text-2xl font-extrabold text-gray-900">회원가입</h2>
                <p className="text-sm text-gray-500 mt-1">무료로 가입하고 특허 분석 기능을 이용하세요.</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                {/* 이름 (선택) */}
                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-1">
                        이름 <span className="text-gray-400 font-normal">(선택)</span>
                    </label>
                    <input
                        type="text"
                        value={name}
                        onChange={e => setName(e.target.value)}
                        placeholder="홍길동"
                        autoComplete="name"
                        className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-gray-800 text-sm"
                        disabled={isLoading}
                    />
                </div>

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
                    <label className="block text-sm font-semibold text-gray-700 mb-1">비밀번호 (최소 6자)</label>
                    <input
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        placeholder="••••••"
                        autoComplete="new-password"
                        className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-gray-800 text-sm"
                        required
                        disabled={isLoading}
                    />
                </div>

                {/* 비밀번호 확인 */}
                <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-1">비밀번호 확인</label>
                    <input
                        type="password"
                        value={confirm}
                        onChange={e => setConfirm(e.target.value)}
                        placeholder="••••••"
                        autoComplete="new-password"
                        className={`w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition text-gray-800 text-sm ${confirm && password !== confirm
                                ? 'border-red-400 focus:border-red-400 bg-red-50'
                                : 'border-gray-300 focus:border-blue-500'
                            }`}
                        required
                        disabled={isLoading}
                    />
                    {confirm && password !== confirm && (
                        <p className="text-xs text-red-500 mt-1">비밀번호가 일치하지 않습니다.</p>
                    )}
                </div>

                {/* 에러 메시지 */}
                {errorMsg && (
                    <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                        {errorMsg}
                    </div>
                )}

                {/* 가입 버튼 */}
                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition-all shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                    {isLoading ? '처리 중...' : '무료로 시작하기'}
                </button>
            </form>

            {/* 로그인 전환 */}
            <p className="text-center text-sm text-gray-500 mt-5">
                이미 계정이 있으신가요?{' '}
                <button
                    onClick={onSwitchToLogin}
                    className="text-blue-600 font-semibold hover:underline"
                >
                    로그인
                </button>
            </p>
        </div>
    );
};
