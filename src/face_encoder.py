"""
<summary>InsightFace 기반 얼굴 검출 및 임베딩 추출 모듈</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>GPU 우선으로 동작하며 CPU로 자동 폴백한다.</remarks>
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.config import AppConfig, default_config
from src.exceptions import (
    ModelLoadError,
    ModelInferenceError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    InvalidImageFormatError
)


logger = logging.getLogger("face-encoder")
logger.setLevel(logging.INFO)


class FaceEncoder:
    """
    <summary>얼굴 검출과 임베딩 생성을 담당하는 클래스</summary>
    <remarks>
    InsightFace(ArcFace) 모델을 활용하여 일관된 임베딩을 생성한다.
    
    책임:
        - 이미지에서 얼굴 검출 (det_10g SCRFD 모델)
        - 검출된 얼굴로부터 512차원 정규화 임베딩 추출 (w600k_r50 ArcFace 모델)
        - GPU/CPU 프로바이더 자동 선택 및 폴백
    
    성능 특성:
        - GPU(RTX 3060): 단일 이미지 ~50ms
        - CPU(i7): 단일 이미지 ~200ms
        - 모델 로딩 초기 비용: ~2초, 메모리 ~2GB
    
    스레드 안전성: InsightFace 내부 onnxruntime 세션은 스레드 안전하지 않으므로,
                   다중 스레드 환경에서는 스레드마다 별도 FaceEncoder 인스턴스 생성 권장.
    
    사용 예시:
        encoder = FaceEncoder(AppConfig(use_gpu=True))
        image = cv2.imread("face.jpg")
        embedding, error = encoder.extract_embedding(image)
        if error is None:
            print(f"임베딩 차원: {embedding.shape}")
    </remarks>
    """

    def __init__(self, config: AppConfig = default_config):
        """
        <summary>모델 초기화 및 런타임 프로바이더 설정</summary>
        <param name="config">전역 설정 객체, 미지정 시 default_config 사용</param>
        <remarks>
        GPU 사용이 불가할 경우 CPU로 자동 폴백한다.
        초기화 시 모델 다운로드(~500MB)가 발생할 수 있으므로 인터넷 연결 필요.
        
        예외:
            - onnxruntime 미설치: ImportError
            - 모델 다운로드 실패: RuntimeError
            - 메모리 부족: MemoryError
        
        부작용:
            - ~/.insightface/models/ 디렉토리에 모델 캐시 생성
            - GPU 메모리 할당 (~1.5GB)
        </remarks>
        """
        self.config: AppConfig    = config
        self.providers: List[str] = self._select_providers()
        self.app: FaceAnalysis    = self._load_model()

    def _select_providers(self) -> List[str]:
        """
        <summary>GPU 사용 가능 여부에 따른 onnxruntime 프로바이더 결정</summary>
        <returns>우선순위가 적용된 프로바이더 리스트</returns>
        <remarks>
        - use_directml 활성화 시 DmlExecutionProvider를 우선 적용한다.
        - use_gpu 활성화 시 CUDAExecutionProvider를 우선 적용하고, 실패 시 CPU로 폴백한다.
        - CPUExecutionProvider는 항상 마지막 안전망으로 포함해 가용성을 보장한다.
        </remarks>
        """
        if self.config.use_directml:
            providers: List[str] = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif self.config.use_gpu:
            providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers: List[str] = ["CPUExecutionProvider"]
        return providers

    def _load_model(self) -> FaceAnalysis:
        """
        <summary>InsightFace 모델을 로드하여 FaceAnalysis 인스턴스를 생성</summary>
        <returns>초기화된 FaceAnalysis 객체</returns>
        <remarks>
        - det_size를 고정(640x640)하여 검출 성능의 일관성을 유지한다.
        - GPU/DirectML 불가 시 CPU로 자동 폴백한다.
        - 모델 캐시는 InsightFace 기본 경로(~/.insightface)에 저장된다.
        </remarks>
        """
        try:
            app: FaceAnalysis = FaceAnalysis(name=self.config.model_name, providers=self.providers)
            ctx_id: int       = 0 if self.config.use_gpu else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("모델 로딩 성공: %s, providers: %s", self.config.model_name, self.providers)
            return app
        except ImportError as exc:
            raise ModelLoadError(
                f"onnxruntime 패키지가 설치되지 않았습니다. "
                f"'pip install onnxruntime-gpu' 또는 'pip install onnxruntime' 실행 필요."
            ) from exc
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"모델 '{self.config.model_name}'을 찾을 수 없습니다. "
                f"인터넷 연결을 확인하거나 모델명을 검증하세요."
            ) from exc
        except MemoryError as exc:
            raise ModelLoadError(
                f"메모리 부족으로 모델 로딩 실패. "
                f"최소 8GB RAM 권장, 현재 사용 가능한 메모리를 확인하세요."
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"모델 로딩 중 예기치 않은 오류 발생: {type(exc).__name__}: {exc}"
            ) from exc

    def detect_faces(self, image: np.ndarray) -> List:
        """
        <summary>이미지에서 얼굴을 검출한다.</summary>
        <param name="image">BGR 포맷의 OpenCV 이미지 (HxWx3, uint8)</param>
        <returns>검출된 Face 객체 리스트 (각 객체는 bbox, kps, normed_embedding 포함)</returns>
        <remarks>
        - 다중 얼굴도 모두 반환하며, 후속 단계에서 필터링한다.
        - BGR 입력을 가정하므로 RGB 이미지 사용 시 cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 변환 필요.
        - 검출 신뢰도는 config.detection_threshold로 조정 가능 (기본 0.6).
        
        전제 조건:
            - image는 유효한 numpy.ndarray (shape: HxWx3, dtype: uint8)
            - 이미지 최소 크기: 32x32 픽셀 권장
        
        후행 조건:
            - 반환된 Face 객체는 InsightFace Face 클래스 인스턴스
            - 빈 리스트 반환 시 얼굴 미검출을 의미
        
        성능:
            - GPU: 640x640 이미지 기준 ~20ms
            - CPU: 640x640 이미지 기준 ~100ms
        </remarks>
        """
        if not isinstance(image, np.ndarray):
            raise InvalidImageFormatError(f"이미지는 numpy.ndarray여야 하지만 {type(image)} 타입입니다.")
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImageFormatError(
                f"이미지는 3채널(BGR)이어야 하지만 shape={image.shape}입니다."
            )
        
        if image.dtype != np.uint8:
            raise InvalidImageFormatError(
                f"이미지 dtype은 uint8이어야 하지만 {image.dtype}입니다."
            )
        
        try:
            faces: List = self.app.get(image)
            return faces
        except Exception as exc:
            raise ModelInferenceError(
                f"얼굴 검출 중 모델 추론 오류 발생: {type(exc).__name__}: {exc}"
            ) from exc

    def extract_embedding(self, image: np.ndarray, raise_on_error: bool = False) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        <summary>단일 이미지에서 얼굴 임베딩을 추출한다.</summary>
        <param name="image">BGR 포맷의 OpenCV 이미지 (HxWx3, uint8)</param>
        <param name="raise_on_error">True시 오류 발생 시 예외 발생, False시 (None, error_code) 반환</param>
        <returns>
        성공 시: (np.ndarray[512], None) - 정규화된 임베딩 벡터와 None
        실패 시: (None, error_code) - None과 오류 코드 문자열 (raise_on_error=False일 때)
        </returns>
        <remarks>
        - 얼굴이 0개 또는 2개 이상이면 None과 오류 코드를 반환한다.
        - 임베딩 추출 실패 시 "embedding_failed"를 반환하여 호출 측이 로깅/필터링할 수 있게 한다.
        - raise_on_error=True 설정 시 명확한 예외 발생으로 에러 처리를 강제할 수 있다.
        
        오류 코드:
            - "face_not_found": 얼굴 미검출
            - "multiple_faces": 2개 이상 얼굴 검출
            - "embedding_failed": 임베딩 추출 실패 (드물게 발생)
        
        전제 조건:
            - image는 유효한 numpy.ndarray (shape: HxWx3, dtype: uint8, BGR 포맷)
        
        후행 조건:
            - 성공 시 반환된 임베딩은 L2 정규화됨 (np.linalg.norm == 1.0)
            - 임베딩 shape: (512,), dtype: float32
        
        사용 예시:
            # 방법 1: 오류 코드 반환 (기본)
            embedding, error = encoder.extract_embedding(image)
            if error is None:
                similarity = np.dot(embedding, other_embedding)
            else:
                logger.warning(f"추출 실패: {error}")
            
            # 방법 2: 예외 발생
            try:
                embedding, _ = encoder.extract_embedding(image, raise_on_error=True)
            except NoFaceDetectedError:
                print("얼굴을 찾을 수 없습니다.")
        
        성능:
            - GPU: 단일 이미지 ~50ms (검출 + 임베딩)
            - CPU: 단일 이미지 ~200ms
        </remarks>
        """
        try:
            faces: List = self.detect_faces(image)
        except (InvalidImageFormatError, ModelInferenceError):
            if raise_on_error:
                raise
            return None, "detection_error"
        
        if len(faces) == 0:
            if raise_on_error:
                raise NoFaceDetectedError("이미지에서 얼굴을 검출하지 못했습니다.")
            return None, "face_not_found"
        
        if len(faces) > 1:
            if raise_on_error:
                raise MultipleFacesDetectedError(
                    f"이미지에서 {len(faces)}개의 얼굴이 검출되었습니다. 1개만 허용됩니다."
                )
            return None, "multiple_faces"

        face               = faces[0]
        embedding: np.ndarray = face.normed_embedding
        
        if embedding is None or len(embedding) == 0:
            if raise_on_error:
                raise ModelInferenceError("임베딩 추출에 실패했습니다.")
            return None, "embedding_failed"
        
        return embedding, None


def extract_face_embeddings(images: List[np.ndarray], encoder: Optional[FaceEncoder] = None) -> List[np.ndarray]:
    """
    <summary>여러 이미지에서 얼굴 임베딩을 추출하여 리스트로 반환한다.</summary>
    <param name="images">BGR 포맷 이미지 리스트 (각 이미지는 HxWx3, uint8)</param>
    <param name="encoder">FaceEncoder 인스턴스, 미지정 시 default_config로 새로 생성</param>
    <returns>유효한 임베딩 벡터 리스트 (각 임베딩은 shape: (512,), dtype: float32)</returns>
    <remarks>
    0개 또는 2개 이상 얼굴이 있는 이미지는 스킵하며 경고 로그만 기록한다.
    
    동작 특성:
        - 얼굴이 정확히 1개 검출된 이미지만 처리
        - 실패한 이미지는 결과에서 제외되고, 빈 리스트 반환 가능
        - encoder 파라미터로 모델 재사용 가능 (성능 향상)
    
    전제 조건:
        - images는 비어있지 않은 리스트 (빈 리스트 시 빈 결과 반환)
        - 각 이미지는 유효한 numpy.ndarray (BGR, HxWx3, uint8)
    
    후행 조건:
        - 반환된 임베딩은 모두 L2 정규화됨
        - len(embeddings) <= len(images) (필터링으로 인한 감소)
    
    사용 예시:
        # 방법 1: 인코더 자동 생성 (매번 모델 로딩, 느림)
        embeddings = extract_face_embeddings(images)
        
        # 방법 2: 인코더 재사용 (권장)
        encoder = FaceEncoder(config)
        embeddings = extract_face_embeddings(images, encoder=encoder)
        
        # 평균화
        if len(embeddings) > 0:
            mean_embedding = np.mean(embeddings, axis=0)
    
    성능:
        - GPU: 5장 이미지 기준 ~250ms (모델 재사용 시)
        - CPU: 5장 이미지 기준 ~1000ms
    
    로깅:
        - WARNING: 각 실패한 이미지마다 "이미지 N 처리 스킵: error_code" 출력
    </remarks>
    """
    active_encoder: FaceEncoder   = encoder or FaceEncoder()
    embeddings: List[np.ndarray]  = []

    for index, image in enumerate(images):
        embedding, error = active_encoder.extract_embedding(image)
        if error is not None:
            logger.warning("이미지 %d 처리 스킵: %s", index, error)
            continue
        embeddings.append(embedding)
    return embeddings

