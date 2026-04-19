import { useState, useEffect, useRef } from 'react';
import { TimeoutToast } from './components/Loading/TimeoutToast';
import { IdeaInput } from './components/Form/IdeaInput';
import { IpcFilterSelector } from './components/Form/IpcFilterSelector';
import { ResultView } from './components/Result/ResultView';
import { ErrorFallback } from './components/common/ErrorFallback';
import { HistorySidebar } from './components/History/HistorySidebar';
import { useRagStream } from './hooks/useRagStream';
import { useAuth } from './hooks/useAuth';
import { AuthGuard } from './components/Auth/AuthGuard';

function App() {
    const { user, logout, isLoading: authLoading } = useAuth();
    const [authToast, setAuthToast] = useState('');
    const [successToast, setSuccessToast] = useState('');
    const authToastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const successToastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

    /** 비로그인 시 상단 배너 토스트 표시 (alert 대체) */
    const showAuthToast = (msg: string) => {
        setAuthToast(msg);
        if (authToastTimer.current) clearTimeout(authToastTimer.current);
        authToastTimer.current = setTimeout(() => setAuthToast(''), 3000);
    };

    /** 로그인/회원가입 성공 시 상단 배너 토스트 표시 */
    const showSuccessToast = (msg: string) => {
        setSuccessToast(msg);
        if (successToastTimer.current) clearTimeout(successToastTimer.current);
        successToastTimer.current = setTimeout(() => setSuccessToast(''), 2500);
    };

    useEffect(() => {
        return () => {
            if (authToastTimer.current) clearTimeout(authToastTimer.current);
            if (successToastTimer.current) clearTimeout(successToastTimer.current);
        };
    }, []);
    const [isGuest, setIsGuest] = useState(true);
    const [authView, setAuthView] = useState<'login' | 'signup'>('login');

    const [idea, setIdea] = useState('');
    /** IPC 카테고리 필터: IpcFilterSelector에서 선택한 코드 배열 */
    const [ipcFilters, setIpcFilters] = useState<string[]>([]);
    /** 히스토리 원클릭 시 IdeaInput에 주입할 값 */
    const [historyIdea, setHistoryIdea] = useState<string | undefined>(undefined);
    /** 히스토리 자동 갱신 카운터: isComplete 전환마다 증가 */
    const [refreshCount, setRefreshCount] = useState(0);

    // RAG 분석 상태 관리 훅
    const {
        isAnalyzing,
        isComplete,
        resultData,
        errorInfo,
        startAnalysis,
        setIsComplete,
        setErrorInfo
    } = useRagStream();

    // 폼 제출: 아이디어 텍스트를 상태에 저장하고 RAG 분석 시작
    const handleSubmitIdea = (inputIdea: string) => {
        if (!user) {
            showAuthToast('특허 분석 기능을 이용하려면 로그인이 필요합니다.');
            setIsGuest(false);
            setAuthView('login');
            return;
        }
        setIdea(inputIdea);
        startAnalysis(inputIdea, ipcFilters.length > 0 ? ipcFilters : null, true);
    };

    // 히스토리 원클릭 → IdeaInput 값 동기화 + startAnalysis() 직접 호출
    const handleSelectHistory = (selectedIdea: string) => {
        if (!user) {
            showAuthToast('기록을 확인하려면 로그인이 필요합니다.');
            setIsGuest(false);
            setAuthView('login');
            return;
        }
        setIdea(selectedIdea);
        setHistoryIdea(selectedIdea);
        startAnalysis(selectedIdea, ipcFilters.length > 0 ? ipcFilters : null, true);
    };

    const handleReset = () => {
        setIdea('');
        setHistoryIdea('');
        setIsComplete(false);
    };

    // 분석 완료 시 refreshCount 증가 → HistorySidebar 자동 갱신
    useEffect(() => {
        if (isComplete) {
            setRefreshCount((c: number) => c + 1);
        }
    }, [isComplete]);

    // 인증 초기화 중에는 헤더만 표시 (레이아웃 깜박임 방지)
    if (authLoading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-gray-400 text-sm">불러오는 중...</div>
            </div>
        );
    }

    return (
        <AuthGuard
            isGuest={isGuest}
            setIsGuest={setIsGuest}
            authView={authView}
            setAuthView={setAuthView}
            user={user}
            onAuthSuccess={showSuccessToast}
        >
            <div className="min-h-screen bg-gray-50 relative">

                {/* 비로그인 시 상단 토스트 배너 (alert 대체) */}
                {authToast && (
                    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[10000] bg-orange-500 text-white text-sm font-semibold px-6 py-3 rounded-full shadow-lg animate-bounce">
                        🔒 {authToast}
                    </div>
                )}
                {successToast && (
                    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[10001] bg-emerald-600 text-white text-sm font-semibold px-6 py-3 rounded-full shadow-lg">
                        ✅ {successToast}
                    </div>
                )}

                <header className="bg-white border-b border-gray-100 px-8 py-5 shadow-sm flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-extrabold text-blue-900">💡 쇼특허 (Short-Cut) AI.</h1>
                        <p className="text-gray-500 text-sm font-medium mt-0.5">아이디어만 입력하면 AI가 실시간으로 특허 침해 여부를 분석해 드립니다.</p>
                    </div>

                    <div className="flex items-center gap-4">
                        {user ? (
                            <div className="flex items-center gap-3">
                                <span className="text-sm font-semibold text-gray-700">{user.email}님</span>
                                <button
                                    onClick={logout}
                                    className="text-xs font-bold text-gray-500 hover:text-red-600 transition-colors px-3 py-1.5 border border-gray-200 rounded-md bg-white hover:bg-red-50"
                                >
                                    로그아웃
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => { setAuthView('login'); setIsGuest(false); }}
                                    className="text-xs font-bold text-gray-700 hover:text-blue-600 px-3 py-1.5 transition-colors"
                                >
                                    로그인
                                </button>
                                <button
                                    onClick={() => { setAuthView('signup'); setIsGuest(false); }}
                                    className="text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 px-4 py-1.5 rounded-full shadow-sm transition-all"
                                >
                                    회원가입
                                </button>
                            </div>
                        )}
                    </div>
                </header>

                <TimeoutToast isAnalyzing={isAnalyzing} timeoutMs={30000} />

                <main className="flex flex-col md:flex-row gap-6 p-6 max-w-7xl mx-auto">
                    <HistorySidebar
                        onSelectIdea={handleSelectHistory}
                        isAnalyzing={isAnalyzing}
                        refreshTrigger={refreshCount}
                    />

                    <section className="flex-1 flex flex-col gap-6">
                        {errorInfo ? (
                            <ErrorFallback
                                title={errorInfo.title}
                                message={errorInfo.message}
                                errorType={errorInfo.errorType}
                                onRetry={() => {
                                    handleReset();
                                    setErrorInfo(null);
                                }}
                            />
                        ) : isComplete && resultData ? (
                            <ResultView
                                idea={idea}
                                resultData={resultData}
                                onReset={handleReset}
                            />
                        ) : (
                            <div className="w-full">
                                <IpcFilterSelector
                                    selectedFilters={ipcFilters}
                                    onChange={setIpcFilters}
                                    disabled={isAnalyzing}
                                />
                                <IdeaInput
                                    onSubmit={handleSubmitIdea}
                                    disabled={isAnalyzing}
                                    initialValue={historyIdea}
                                />
                            </div>
                        )}
                    </section>
                </main>
            </div>
        </AuthGuard>
    );
}

export default App;
