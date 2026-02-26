"""
<module_description>
    FastAPI 기반 안면 인식 마이크로서비스 (HTTP API) 엔트리 포인트.
</module_description>
<technical_overview>
    본 모듈은 안면 토큰 생성 서비스를 원격에서 이용할 수 있도록 HTTP 인터페이스를 제공합니다.
    Uvicorn(ASGI) 엔진과 FastAPI 프레임워크를 기반으로 하며, 모델 싱글톤(Singleton) 관리 및 
    비동기(Asynchronous) 이벤트 루프와 동기적(Synchronous) 모델 추론 사이의 최적화를 보장합니다.
</technical_overview>
<architectural_design>
    1. 서비스 생명주기 관리: Lifespan 이벤트를 통해 서버 기동 시 대규모 신경망 모델(2GB+)을 
       메모리에 한 번만 적재(Singleton)하고, 서버 종료 시 자원을 명확히 해제함.
    2. 동시성 제어: run_in_threadpool을 적용하여 CPU/GPU 집중 연산인 모델 추론 시 
       비동기 이벤트 루프가 차단(Block)되는 것을 방지함으로써 서비스 가용성을 확보함.
    3. 의존성 주입(DI): FastAPI의 Depends 기능을 활용하여, 각 요청에 안정적으로 모델 
       인스턴스를 공급함.
</architectural_design>
<author>최진호 (Senior Principal Software Engineer / Computer Science PhD)</author>
<version>2.2.0</version>
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.config import AppSettings, default_config
from src.face_encoder import FaceEncoder

# 시스템 로깅 설정: 서비스 시작, 종료 및 예외 발생 시점 기록을 통한 추적성 확보
logger = logging.getLogger("facer-api")
logger.setLevel(logging.INFO)


class State:
    """
    <summary>애플리케이션 전역 상태 컨테이너</summary>
    <remarks>
    무거운 리소스를 상주시키기 위한 싱글톤 저장소입니다. 
    서버의 구동 시간 동안 유효한 모델 인스턴스를 유지합니다.
    </remarks>
    """
    encoder: Optional[FaceEncoder] = None


# [싱글톤 인스턴스 저장소] 전역 상태를 관리하는 객체
state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    <summary>서버 생명주기(Lifespan) 관리 전략</summary>
    <remarks>
    서버의 탄생과 소멸을 관리하는 전략적 이벤트 핸들러입니다.
    - Startup: 신경망 모델의 가중치 로딩 및 하드웨어 가속기(Execution Provider) 초기화.
    - Running: 비즈니스 로직(추론) 처리.
    - Shutdown: GPU 메모리 할당 해제 및 파일 시스템 스트림 정리.
    </remarks>
    """
    logger.info(">>> [API Gateway] Startup: Loading neural network models into memory...")
    try:
        # Pydantic Settings 기반 전역 설정을 주입받아 인코더 초기화
        state.encoder = FaceEncoder(default_config)
        logger.info(">>> [API Gateway] Success: AI Engine is ready for inference.")

        # ONNX JIT 사전 컴파일: 첫 실제 요청의 레이턴시 스파이크 제거
        logger.info(">>> [API Gateway] Running warm-up inference...")
        _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        state.encoder.detect_faces(_dummy)
        logger.info(">>> [API Gateway] Warm-up complete.")

    except Exception as exc:
        logger.critical(f">>> [API Gateway] Failure: AI Engine initialization failed - {exc}")
        # 초기화 실패 시 시스템 무결성을 위해 서버 기동 중단 (Safe Exit)
        raise RuntimeError(f"Engine initialization error: {exc}")
    
    yield # 요청 대기 상태 (Service Active)
    
    logger.info(">>> [API Gateway] Shutdown: Releasing system resources...")
    state.encoder = None


# FastAPI 프레임워크 인스턴스 정의
app = FastAPI(
    title="Facer API v2.2",
    description="InsightFace(ArcFace) 기반 고성능 안면 토큰 생성 마이크로서비스",
    version="2.2.0",
    lifespan=lifespan
)


class EmbeddingResponse(BaseModel):
    """
    <summary>임베딩 생성 결과 응답 데이터 스키마 (Pydantic 모델)</summary>
    <param name="embedding">512차원 특징 벡터 리스트 (L2 Normalized)</param>
    <param name="error">추론 실패 시 반환되는 오류 코드 (face_not_found, etc.)</param>
    """
    embedding: List[float]
    error: Optional[str] = None


def get_encoder() -> FaceEncoder:
    """
    <summary>의존성 주입(Dependency Injection) 함수</summary>
    <remarks>
    싱글톤으로 관리되는 모델 인스턴스를 요청 핸들러에 안전하게 공급하는 역할을 수행합니다.
    </remarks>
    """
    if state.encoder is None:
        raise HTTPException(status_code=503, detail="AI Engine is not ready.")
    return state.encoder


@app.post("/token/generate", response_model=EmbeddingResponse)
async def generate_token(
    file: UploadFile = File(...),
    encoder: FaceEncoder = Depends(get_encoder)
):
    """
    <summary>이미지 데이터 -> 특징 벡터 변환 엔드포인트 (Inference Pipeline)</summary>
    <param name="file">업로드된 소스 이미지 파일</param>
    <param name="encoder">주입받은 추론 엔진 인스턴스</param>
    <remarks>
    [Concurrency Strategy: run_in_threadpool]
    신경망 연산은 CPU/GPU 집중적인 작업으로 Python의 GIL 및 비동기 이벤트 루프 환경에서 
    이벤트 루프의 응답성을 저해할 수 있습니다. 
    따라서 Starlette의 threadpool을 활용하여 이 연산을 비동기 루프에서 격리(Isolation)함으로써, 
    서버의 전역적 가용성을 유지하며 다중 요청을 처리합니다.
    </remarks>
    """
    # 1. 스트림 데이터 로딩
    contents = await file.read()
    
    # 2. 이미지 디코딩: 바이너리 데이터를 OpenCV 표준 포맷(NumPy)으로 변환
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        logger.warning(f"Invalid image upload: {file.filename}")
        raise HTTPException(status_code=400, detail="Image data decoding failed. Invalid file format.")

    # 3. 모델 추론 실행: 별도 스레드에서 격리하여 동시성 확보
    try:
        embedding, error, _quality = await run_in_threadpool(encoder.extract_embedding, image)
    except Exception as exc:
        logger.error(f"Internal Inference Error: {exc}")
        raise HTTPException(status_code=500, detail="Inference processing exception occurred.")

    # 4. 결과 판별 및 응답 반환
    if error:
        logger.info(f"Token Generation Denied: {error}")
        return EmbeddingResponse(embedding=[], error=error)

    # 5. 임베딩 벡터 직렬화(JSON) 및 최종 반환
    return EmbeddingResponse(embedding=embedding.tolist(), error=None)


@app.get("/health")
async def health_check():
    """
    <summary>서비스 가용성 및 엔진 상태 확인 엔드포인트 (Heartbeat)</summary>
    <remarks>
    부하 분산기(L4/L7)나 오케스트레이터(K8s)의 상태 확인 요청에 대응합니다.
    </remarks>
    """
    is_ready = state.encoder is not None
    return {
        "status": "healthy" if is_ready else "starting",
        "model": default_config.model_name,
        "acceleration": default_config.use_gpu,
        "det_size": default_config.det_size,
    }


def run_server(host: str, port: int, config: AppSettings = default_config):
    """
    <summary>Uvicorn ASGI 서버 런처</summary>
    """
    logger.info(f">>> Service Listening: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
