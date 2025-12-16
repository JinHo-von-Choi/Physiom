"""
<summary>FastAPI 기반 얼굴 토큰 생성 HTTP 서버</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>키오스크 환경에서 Unity 등 외부 프로세스와 연동하기 위한 엔드포인트 제공.</remarks>
"""

import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import generate_token_from_camera, generate_token_from_directory
from src.config import AppConfig, default_config
from src.exceptions import (
    CameraError,
    ModelError,
    FaceDetectionError,
    NoValidEmbeddingError,
    ImageError,
    ConfigurationError,
    TimeoutError as FaceTimeoutError
)


logger = logging.getLogger("api-http")
logger.setLevel(logging.INFO)


class CameraRequest(BaseModel):
    """
    <summary>POST /token/camera 요청 스키마</summary>
    <remarks>
    Pydantic 모델로 자동 검증 및 OpenAPI 문서 생성.
    maxFrames는 1~20 범위로 제한하여 과도한 리소스 사용 방지.
    </remarks>
    """
    maxFrames: int = Field(default=default_config.max_frames, ge=1, le=20, description="캡처할 최대 프레임 수")


class DirectoryRequest(BaseModel):
    """
    <summary>POST /token/directory 요청 스키마</summary>
    <remarks>
    Pydantic 모델로 자동 검증 및 OpenAPI 문서 생성.
    dirPath는 필수이며, maxImages는 1~20 범위로 제한.
    </remarks>
    """
    dirPath: str = Field(default="", description="이미지 디렉토리 절대/상대 경로")
    maxImages: int = Field(default=default_config.max_images, ge=1, le=20, description="로드할 최대 이미지 수")


def create_app(config: AppConfig = default_config) -> FastAPI:
    """
    <summary>FastAPI 애플리케이션을 생성한다.</summary>
    <param name="config">전역 설정 객체 (HTTP 서버 설정 제외한 나머지 설정 사용)</param>
    <returns>FastAPI 애플리케이션 인스턴스 (엔드포인트 2개 포함)</returns>
    <remarks>
    예외는 success=false 응답으로 변환하여 클라이언트가 명시적으로 처리한다.
    
    제공 엔드포인트:
        - POST /token/camera: 카메라 캡처 → 임베딩 생성
        - POST /token/directory: 디렉토리 이미지 → 임베딩 생성
        - GET /docs: Swagger UI (자동 생성)
        - GET /redoc: ReDoc (자동 생성)
    
    응답 형식:
        성공: {"success": true, "embedding": [512 floats], "error": null}
        실패: {"success": false, "embedding": null, "error": "오류 메시지"}
    
    예외 처리:
        - 에러 타입별로 적절한 HTTP 상태 코드 반환
        - 503: 카메라 오류 / 422: 얼굴 검출 실패 / 408: 타임아웃
        - 404: 파일 미존재 / 403: 권한 부족 / 500: 모델 오류
        - logger로 모든 예외 기록
    
    스레드 안전성:
        - ThreadSafeEncoderManager 사용으로 다중 워커/스레드 환경 안전
        - 각 요청은 독립적인 FaceEncoder 인스턴스 사용
        - gunicorn -w N / uvicorn --workers N 안전하게 지원
        - async/await 비동기 환경에서도 안전 (contextvars 사용)
    
    CORS 미설정:
        - 현재 CORS 미들웨어 미포함
        - 프로덕션 환경에서 필요 시 추가:
          from fastapi.middleware.cors import CORSMiddleware
          app.add_middleware(CORSMiddleware, allow_origins=["*"])
    
    인증 미설정:
        - API Key, Bearer Token 등 인증 미포함
        - 프로덕션 환경에서는 인증 미들웨어 추가 권장
    
    사용 예시:
        # 개발 환경
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=23535)
        
        # 프로덕션 환경 (gunicorn + uvicorn worker)
        # gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api_http:app
    </remarks>
    """
    app: FastAPI = FastAPI(title="Face Token Service", version="1.0.0")

    @app.post("/token/camera")
    async def token_from_camera(request: CameraRequest) -> Dict[str, Any]:
        """
        <summary>카메라로부터 임베딩 생성</summary>
        <remarks>
        에러 타입에 따라 적절한 HTTP 상태 코드와 에러 메시지를 반환한다.
        </remarks>
        """
        try:
            embedding = generate_token_from_camera(max_frames=request.maxFrames, config=config)
            return {"success": True, "embedding": embedding, "error": None}
        
        except CameraError as exc:
            logger.error("카메라 오류: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "camera_error"}
            )
        
        except (FaceDetectionError, NoValidEmbeddingError) as exc:
            logger.warning("얼굴 검출 실패: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "face_detection_error"}
            )
        
        except FaceTimeoutError as exc:
            logger.warning("타임아웃: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "timeout_error"}
            )
        
        except ModelError as exc:
            logger.error("모델 오류: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "model_error"}
            )
        
        except Exception as exc:
            logger.exception("예기치 않은 오류")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "unknown_error"}
            )

    @app.post("/token/directory")
    async def token_from_directory(request: DirectoryRequest) -> Dict[str, Any]:
        """
        <summary>디렉토리로부터 임베딩 생성</summary>
        <remarks>
        에러 타입에 따라 적절한 HTTP 상태 코드와 에러 메시지를 반환한다.
        </remarks>
        """
        try:
            embedding = generate_token_from_directory(dir_path=request.dirPath, max_images=request.maxImages, config=config)
            return {"success": True, "embedding": embedding, "error": None}
        
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning("디렉토리 경로 오류: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "path_error"}
            )
        
        except PermissionError as exc:
            logger.error("권한 오류: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "permission_error"}
            )
        
        except ImageError as exc:
            logger.warning("이미지 로딩 오류: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "image_error"}
            )
        
        except (FaceDetectionError, NoValidEmbeddingError) as exc:
            logger.warning("얼굴 검출 실패: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "face_detection_error"}
            )
        
        except ModelError as exc:
            logger.error("모델 오류: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "model_error"}
            )
        
        except Exception as exc:
            logger.exception("예기치 않은 오류")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"success": False, "embedding": None, "error": str(exc), "error_type": "unknown_error"}
            )

    return app


def run_server(host: str = default_config.server_host, port: int = default_config.server_port, config: AppConfig = default_config) -> None:
    """
    <summary>uvicorn을 통해 FastAPI 서버를 실행</summary>
    <param name="host">리스닝 호스트 (0.0.0.0=모든 인터페이스, 127.0.0.1=로컬 전용)</param>
    <param name="port">리스닝 포트 (기본 23535)</param>
    <param name="config">전역 설정 객체 (HTTP 서버 설정 제외)</param>
    <remarks>
    - 개발 환경에서는 단독 실행, 운영 환경에서는 프로세스 매니저/서비스 래퍼와 함께 구동 권장.
    - 비동기 루프 내에서 재호출하지 않도록 단일 엔트리에서만 사용한다.
    
    프로덕션 배포:
        # systemd (Linux)
        [Unit]
        Description=Face Embedding Service
        After=network.target
        
        [Service]
        Type=simple
        User=faceservice
        WorkingDirectory=/opt/face
        ExecStart=/opt/face/.venv/bin/python main.py server --host 0.0.0.0 --port 23535
        Restart=always
        
        [Install]
        WantedBy=multi-user.target
        
        # Windows Service (NSSM 사용)
        nssm install FaceService "C:\face\.venv\Scripts\python.exe" "C:\face\main.py server --host 0.0.0.0 --port 23535"
        nssm set FaceService AppDirectory "C:\face"
        nssm start FaceService
    
    다중 워커:
        # gunicorn + uvicorn worker (Linux)
        gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
                 -b 0.0.0.0:23535 \
                 --access-logfile - \
                 --error-logfile - \
                 src.api_http:create_app()
    
    HTTPS 설정:
        # Nginx 리버스 프록시 권장
        # uvicorn 직접 TLS:
        uvicorn.run(app, host=host, port=port, 
                    ssl_keyfile="key.pem", ssl_certfile="cert.pem")
    
    성능 튜닝:
        - workers 수: CPU 코어 수 * 2 + 1
        - 메모리: 워커당 ~3GB (모델 로딩)
        - 타임아웃: 기본 30초 (조정 가능)
    
    모니터링:
        - 헬스체크 엔드포인트 추가 권장: GET /health
        - Prometheus 메트릭 엔드포인트: GET /metrics
    
    사용 예시:
        # 로컬 개발
        run_server(host="127.0.0.1", port=23535)
        
        # 외부 접근 허용
        run_server(host="0.0.0.0", port=23535)
        
        # 커스텀 설정
        config = AppConfig(use_gpu=False)
        run_server(host="0.0.0.0", port=8080, config=config)
    </remarks>
    """
    app: FastAPI = create_app(config=config)
    uvicorn.run(app, host=host, port=port)

