"""
<module_description>
    InsightFace(ArcFace) 기반 고성능 안면 검출 및 512차원 특징 벡터(Embedding) 추출 엔진.
</module_description>
<technical_overview>
    본 엔진은 현대적인 안면 인식 파이프라인인 SCRFD(Detection)와 ArcFace(Recognition)를 통합하여 구현되었습니다.
    입력 이미지로부터 안면 영역을 정밀하게 추출하고, 정렬(Alignment) 과정을 거쳐 고유한 특징을 512차원의 
    L2 정규화된 유클리드 공간 벡터로 사상(Mapping)합니다. 이를 통해 코사인 유사도 연산만으로 
    객체 간의 동일성 여부를 고속으로 판별할 수 있는 기반을 제공합니다.
</technical_overview>
<architectural_design>
    1. 하드웨어 가속 전략: CUDA(NVIDIA), DirectML(AMD/Intel) 프로바이더를 우선 탐색하며, 가용 불가 시 
       CPU 런타임으로 자동 전환되는 폴백(Fallback) 메커니즘을 내장함.
    2. 전처리 및 품질 보증: Laplacian 기반 선명도 검사(FIQA)를 수행하여 저품질 데이터에 의한 
       오인식(False Acceptance) 가능성을 사전에 차단함.
    3. 지능형 객체 선정: 다중 안면 검출 시 바운딩 박스의 면적을 기준으로 지배적 객체(Dominant Subject)를 
       자동 선정하여 단일 토큰 생성의 일관성을 확보함.
</architectural_design>
<author>최진호 (Senior Principal Software Engineer / Computer Science PhD)</author>
<version>2.2.0</version>
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.config import AppSettings, default_config
from src.exceptions import (
    ModelLoadError,
    ModelInferenceError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    InvalidImageFormatError
)

# 표준 로깅 인터페이스: 시스템 가시성(Observability) 확보를 위해 표준 라이브러리 활용
logger = logging.getLogger("face-encoder")
logger.setLevel(logging.INFO)


class FaceEncoder:
    """
    <summary>안면 인식 특징 추출 프로세스를 캡슐화한 핵심 도메인 클래스</summary>
    <remarks>
    본 클래스는 신경망 모델의 생명주기와 추론 로직을 관리합니다. 
    모델 가중치는 메모리에 상주하며, 요청 시 원자적(Atomic) 추론 작업을 수행합니다.
    </remarks>
    """

    def __init__(self, config: AppSettings = default_config):
        """
        <summary>클래스 생성자: 추론 엔진 초기화</summary>
        <param name="config">애플리케이션 전역 설정 객체 (의존성 주입 패턴 적용)</param>
        """
        self.config: AppSettings   = config
        self.providers: List[str]  = self._select_providers()
        self.app: FaceAnalysis     = self._load_model()

    def _select_providers(self) -> List[str]:
        """
        <summary>가용 하드웨어 가속기(Execution Providers) 판별 및 우선순위 설정</summary>
        <remarks>
        ONNX Runtime의 Execution Provider 전략을 기반으로 하드웨어 가속 우선순위를 결정합니다.
        가장 낮은 단계인 CPUExecutionProvider는 가용성 보장을 위해 항상 리스트의 말단에 배치됩니다.
        </remarks>
        """
        providers = []
        if self.config.use_directml:
            providers.append("DmlExecutionProvider")
        if self.config.use_gpu:
            providers.append("CUDAExecutionProvider")
        
        providers.append("CPUExecutionProvider")
        return providers

    def _load_model(self) -> FaceAnalysis:
        """
        <summary>신경망 모델 로딩 및 연산 그래프 최적화</summary>
        <returns>검출 및 인식을 위한 FaceAnalysis 통합 객체</returns>
        <exception cref="ModelLoadError">런타임 패키지 부재 또는 리소스 부족 시 발생</exception>
        """
        try:
            app = FaceAnalysis(
                name=self.config.model_name,
                providers=self.providers,
                allowed_modules=self.config.allowed_modules
            )
            ctx_id = 0 if self.config.use_gpu else -1
            app.prepare(ctx_id=ctx_id, det_size=(self.config.det_size, self.config.det_size))

            # pose 추출 활성화 (1k3d68 모델이 로딩된 경우에만)
            if self.config.require_pose and 'landmark_3d_68' in app.models:
                app.models['landmark_3d_68'].require_pose = True

            logger.info(
                f"Engine Initialized: {self.config.model_name} "
                f"[Provider: {self.providers[0]}] "
                f"[det_size: {self.config.det_size}] "
                f"[modules: {self.config.allowed_modules}]"
            )
            return app
        except Exception as exc:
            logger.critical(f"Failed to load AI models: {exc}")
            raise ModelLoadError(f"추론 엔진 기동 실패: {exc}")

    def _is_pose_acceptable(self, face) -> bool:
        """pitch/yaw 기반 정면 여부 판정 (require_pose=True 시 유효)."""
        pose = face.get('pose') if hasattr(face, 'get') else getattr(face, 'pose', None)
        if pose is None:
            return True  # pose 데이터 없으면 통과
        pitch, yaw, _roll = pose
        return abs(yaw) <= self.config.pose_yaw_limit and abs(pitch) <= self.config.pose_pitch_limit

    def compute_quality_score(self, face, image: np.ndarray) -> float:
        """
        <summary>프레임 품질 점수 계산 (0.0 ~ 1.0)</summary>
        det_score(0.6) + laplacian_sharpness(0.4) 가중 합산.
        """
        det_score = float(getattr(face, 'det_score', 0.0) or 0.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_norm = min(sharpness / 200.0, 1.0)
        return 0.6 * det_score + 0.4 * sharpness_norm

    def detect_faces(self, image: np.ndarray) -> List:
        """
        <summary>이미지 데이터 분석 및 안면 영역 검출</summary>
        <param name="image">BGR 형식의 다차원 배열 데이터 (HxWx3, uint8)</param>
        <returns>검출된 객체(InsightFace Face) 리스트</returns>
        <remarks>
        추론 전 품질 검사(Sanity Check) 단계:
        1. 자료형 검증: 데이터 구조의 무결성을 확인하여 런타임 에러 방지.
        2. 이미지 선명도 분석(FIQA): Laplacian Variance 알고리즘을 사용하여 
           임계값 미만의 블러(Blur) 이미지를 사전에 필터링함으로써 인식 정합성을 보장함.
        </remarks>
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise InvalidImageFormatError("입력 데이터가 유효한 이미지 다차원 배열 형식이 아닙니다.")
        
        try:
            # Face Image Quality Assessment (FIQA): 선명도 기반 사전 필터링
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 라플라시안 분산은 이미지의 고주파 성분 밀도를 측정함 (낮을수록 블러링이 심함)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if score < 50: 
                logger.warning(f"Low image quality detected (Sharpness: {score:.2f}). Inference aborted.")
                return []

            # 객체 검출 추론 실행
            faces = self.app.get(image)
            return faces
        except Exception as exc:
            raise ModelInferenceError(f"Inference processing error: {exc}")

    def extract_embedding(self, image: np.ndarray, raise_on_error: bool = False) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        <summary>단일 객체에 대한 안면 임베딩 벡터 추출</summary>
        <param name="image">입력 소스 이미지 데이터</param>
        <param name="raise_on_error">예외 전파 여부 제어 플래그</param>
        <returns>(embedding, error_code, quality_score) 튜플. quality_score: 0.0 ~ 1.0</returns>
        """
        try:
            faces = self.detect_faces(image)
        except Exception as exc:
            if raise_on_error: raise
            return None, "detection_error", 0.0

        if not faces:
            if raise_on_error: raise NoFaceDetectedError("객체를 검출하지 못했습니다.")
            return None, "face_not_found", 0.0

        if len(faces) > 1:
            logger.info(f"Multiple subjects ({len(faces)}) detected. Selecting the dominant subject.")
            faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)

        target_face = faces[0]

        # Pose 필터링
        if not self._is_pose_acceptable(target_face):
            if raise_on_error: raise ModelInferenceError("포즈가 허용 범위를 벗어났습니다.")
            return None, "pose_out_of_range", 0.0

        quality = self.compute_quality_score(target_face, image)

        if quality < self.config.min_quality_score:
            if raise_on_error: raise ModelInferenceError(f"품질 점수 미달: {quality:.3f}")
            return None, "quality_too_low", quality

        embedding = target_face.normed_embedding

        if embedding is None:
            if raise_on_error: raise ModelInferenceError("특징 벡터 추출 실패")
            return None, "embedding_failed", 0.0

        return embedding, None, quality

    def batch_extract_embeddings(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """
        <summary>다량 이미지 데이터에 대한 배치(Batch) 추론 처리</summary>
        <returns>(embeddings, quality_scores) 튜플</returns>
        """
        valid_embs: List[np.ndarray] = []
        valid_scores: List[float]    = []
        for i, img in enumerate(images):
            emb, err, score = self.extract_embedding(img)
            if err is None:
                valid_embs.append(emb)
                valid_scores.append(score)
            else:
                logger.debug(f"Skipping index {i} due to: {err}")
        return valid_embs, valid_scores


def extract_face_embeddings(images: List[np.ndarray], encoder: Optional[FaceEncoder] = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    <summary>함수형 인터페이스를 통한 특징 벡터 대량 추출 유틸리티</summary>
    <returns>(embeddings, quality_scores) 튜플</returns>
    """
    active_encoder = encoder or FaceEncoder()
    return active_encoder.batch_extract_embeddings(images)
