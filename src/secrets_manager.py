"""
쇼특허(Short-Cut) – Secrets Manager 유틸리티
=============================================
프로덕션에서는 AWS Secrets Manager를 통해 시크릿을 로드합니다.
개발 환경(APP_ENV=development/dev/local)에서는 `.env` fallback을 허용합니다.

우선순위:
  1. ECS Task Definition secrets 필드로 이미 주입된 환경 변수 (감지 시 SM 호출 skip)
  2. 개발 모드: .env fallback
  3. AWS Secrets Manager → os.environ 주입
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Local .env Fallback (Development Only)
# =============================================================================

def _load_from_dotenv(dotenv_path: Optional[str] = None) -> Dict[str, str]:
    """
    로컬 .env 파일에서 값을 읽어 딕셔너리로 반환합니다.

    운영 환경 의존성을 줄이기 위해 개발 모드에서만 사용됩니다.
    """
    path = dotenv_path or os.getenv("DOTENV_PATH") or ".env"
    if not os.path.exists(path):
        return {}

    try:
        from dotenv import dotenv_values
    except ImportError:
        logger.warning("python-dotenv 미설치로 .env fallback을 사용할 수 없습니다.")
        return {}

    values = dotenv_values(path)
    parsed: Dict[str, str] = {}
    for key, value in values.items():
        if isinstance(value, str) and value.strip():
            parsed[key] = value.strip()
    if parsed:
        logger.info(".env fallback 로드 성공: %s (%d개 키)", path, len(parsed))
    return parsed


# =============================================================================
# AWS Secrets Manager 로더
# =============================================================================

def _load_from_secrets_manager(
    secret_name: str,
    region: str,
) -> Dict[str, str]:
    """
    AWS Secrets Manager에서 단일 시크릿을 읽어 키-값 딕셔너리로 반환합니다.

    Args:
        secret_name: Secrets Manager 시크릿 이름 (예: short-cut/prod/app)
        region:      AWS 리전 (예: us-east-1)

    Returns:
        시크릿 키-값 딕셔너리

    Raises:
        ImportError:  boto3가 설치되지 않은 경우
        ClientError:  Secrets Manager 접근 실패 시
        ValueError:   시크릿 값이 JSON 형식이 아닌 경우
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise ImportError(
            "boto3가 설치되어 있지 않습니다. `pip install boto3` 후 다시 시도하세요."
        ) from exc

    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        logger.info("Secrets Manager에서 시크릿 로드 성공: %s", secret_name)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error(
            "Secrets Manager 접근 실패 (secret=%s, code=%s): %s",
            secret_name,
            error_code,
            exc,
        )
        raise

    secret_string: Optional[str] = response.get("SecretString")
    if not secret_string:
        raise ValueError(
            f"시크릿 '{secret_name}'의 값이 비어 있거나 BinarySecret 형식입니다. "
            "JSON 문자열 형식의 SecretString만 지원합니다."
        )

    try:
        return json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"시크릿 '{secret_name}'의 값이 유효한 JSON이 아닙니다: {exc}"
        ) from exc


def _inject_secrets_to_env(secrets: Dict[str, str]) -> None:
    """
    시크릿 딕셔너리를 os.environ에 주입합니다.
    Secrets Manager 값이 기존 환경 변수보다 우선합니다.
    단, DATABASE_URL은 이미 환경에 설정된 경우 덮어쓰지 않습니다.
    (로컬 개발 시 SQLite 등 로컬 DB를 우선 사용하기 위함)

    Args:
        secrets: 주입할 키-값 딕셔너리
    """
    # DATABASE_URL은 이미 환경에 존재하면 로컬 설정을 우선합니다.
    # (로컬 SQLite ↔ 프로덕션 RDS 혼용 방지)
    LOCAL_PRIORITY_KEYS = {"DATABASE_URL"}

    for key, value in secrets.items():
        if not isinstance(value, str):
            # 숫자·불리언 등 비문자열 값을 문자열로 변환
            value = str(value)

        existing = os.environ.get(key)

        if existing and key in LOCAL_PRIORITY_KEYS:
            logger.info(
                "환경 변수 유지 (로컬 우선): %s → 기존값 사용 (SM 값 무시)", key
            )
            continue

        old = os.environ.get(key)
        os.environ[key] = value
        if old and old != value:
            logger.debug("환경 변수 덮어쓰기: %s (기존값 존재)", key)
        else:
            logger.debug("환경 변수 주입: %s", key)

    logger.info("%d개 시크릿 환경 변수 주입 완료", len(secrets))


# =============================================================================
# GCP 자격증명 처리
# =============================================================================

def _handle_gcp_credentials() -> None:
    """
    GOOGLE_APPLICATION_CREDENTIALS_JSON 환경 변수가 설정된 경우,
    JSON 내용을 임시 파일로 저장하고 GOOGLE_APPLICATION_CREDENTIALS를 설정합니다.

    GCP SDK는 GOOGLE_APPLICATION_CREDENTIALS 경로를 참조하므로
    JSON 문자열을 직접 읽지 못하는 SDK를 위한 호환 처리입니다.
    """
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        return

    import tempfile
    import atexit

    try:
        # 임시 파일에 GCP 자격증명 JSON을 기록합니다.
        # dir='/tmp' 명시: ECS 컨테이너에서 기본 tempdir이 /home/appuser로
        # 설정되어 Permission denied 발생하는 것을 방지합니다.
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="gcp_creds_",
            delete=False,
            encoding="utf-8",
            dir="/tmp",
        )
        tmp.write(credentials_json)
        tmp.flush()
        tmp.close()
        os.chmod(tmp.name, 0o600)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        logger.info(
            "GCP 자격증명 임시 파일 생성 및 GOOGLE_APPLICATION_CREDENTIALS 설정 완료: %s",
            tmp.name,
        )

        # 프로세스 종료 시 임시 파일 삭제
        def _cleanup_tmp_cred(path: str) -> None:
            try:
                os.remove(path)
                logger.debug("GCP 자격증명 임시 파일 삭제 완료: %s", path)
            except OSError:
                pass

        atexit.register(_cleanup_tmp_cred, tmp.name)

    except OSError as exc:
        logger.error("GCP 자격증명 임시 파일 생성 실패: %s", exc)
        raise


# =============================================================================
# 부트스트랩 진입점
# =============================================================================

def bootstrap_secrets(
    secret_name: str = "short-cut/prod/app",
    aws_region: Optional[str] = None,
) -> None:
    """
    AWS Secrets Manager에서 시크릿을 로드하여 os.environ에 주입합니다.
    .env 파일은 사용하지 않습니다.

    Skip 조건: 필수 키(OPENAI_API_KEY, PINECONE_API_KEY)가 이미 환경에 존재하는 경우
    (ECS Task Definition natives secrets 필드로 주입된 경우 해당).

    Args:
        secret_name: Secrets Manager 시크릿 이름. SECRET_NAME 환경변수로 재정의 가능.
        aws_region:  AWS 리전. AWS_REGION 또는 AWS_DEFAULT_REGION 환경변수로 재정의 가능.
    """
    # 환경 변수로 시크릿 이름·리전 재정의 허용 (컨테이너 실행 시 유연성 확보)
    secret_name = os.getenv("SECRET_NAME", secret_name)
    region = aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-northeast-2"

    app_env = (os.getenv("APP_ENV") or "development").lower()
    is_dev_mode = app_env in {"development", "dev", "local"}

    logger.info(
        "시크릿 부트스트랩 시작 (Secret=%s, Region=%s, APP_ENV=%s)",
        secret_name,
        region,
        app_env,
    )

    # 필수 키가 이미 모두 주입돼 있다면 Secrets Manager 호출 생략
    # (ECS Task Definition의 secrets 필드 또는 수동 환경변수 주입 케이스)
    required_keys = ["OPENAI_API_KEY", "PINECONE_API_KEY", "JWT_SECRET_KEY"]
    all_keys_present = all(os.getenv(k) for k in required_keys)

    # 개발 모드에서는 .env 값을 먼저 보강하여 로컬 실행 안정성을 높입니다.
    if not all_keys_present and is_dev_mode:
        dotenv_secrets = _load_from_dotenv()
        if dotenv_secrets:
            _inject_secrets_to_env(dotenv_secrets)
            all_keys_present = all(os.getenv(k) for k in required_keys)

    if all_keys_present:
        logger.info(
            "AWS Secrets Manager 호출 생략: 필수 키가 이미 환경에 존재합니다. "
            "(ECS native secrets injection / .env / 수동 설정)"
        )
    else:
        try:
            secrets = _load_from_secrets_manager(secret_name, region)
            _inject_secrets_to_env(secrets)
        except Exception as exc:
            # 개발 모드에서는 .env fallback이 가능하면 계속 진행합니다.
            if is_dev_mode:
                logger.warning(
                    "AWS Secrets Manager 로드 실패(개발 모드): %s. .env/환경변수 기반으로 계속 진행합니다.",
                    exc,
                )
            else:
                # 운영 모드에서는 시크릿 로드 실패를 치명적으로 처리합니다.
                logger.critical(
                    "AWS Secrets Manager 로드 실패: %s. "
                    "AWS 자격증명 및 SECRET_NAME/AWS_REGION 환경변수를 확인하세요.",
                    exc,
                )
                raise

    # GCP 자격증명 처리 (GOOGLE_APPLICATION_CREDENTIALS_JSON이 설정된 경우)
    _handle_gcp_credentials()

    logger.info("시크릿 부트스트랩 완료")
