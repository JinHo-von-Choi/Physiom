"""
<summary>얼굴 임베딩 기반 토큰 생성 서비스의 퍼사드 모듈</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>
카메라와 디렉토리 입력을 추상화한 고수준 API를 제공한다.
스레드 안전성을 위해 threading.local을 사용한 스레드별 인코더 관리.
</remarks>
"""

import contextvars
import threading
from typing import List, Optional

import numpy as np

from src.camera_capture import capture_from_camera
from src.config import AppConfig, default_config
from src.face_encoder import FaceEncoder, extract_face_embeddings
from src.image_loader import load_images_from_directory
from src.token_generator import generate_token_from_embeddings
from src.exceptions import NoValidEmbeddingError


_context_encoder: contextvars.ContextVar[Optional[FaceEncoder]] = contextvars.ContextVar('encoder', default=None)


class ThreadSafeEncoderManager:
    """
    <summary>스레드/비동기 안전한 FaceEncoder 인스턴스 관리 클래스</summary>
    <remarks>
    threading.local과 contextvars를 함께 사용하여 동기/비동기 환경 모두 지원한다.
    InsightFace의 onnxruntime 세션은 스레드 안전하지 않으므로 이 매니저를 통해 안전성을 확보한다.
    
    동작 원리:
        - 비동기 환경(FastAPI async def): contextvars 사용
        - 동기 환경(threading): threading.local 사용
        - 각 실행 컨텍스트별로 독립적인 FaceEncoder 인스턴스 유지
    
    스레드 안전성: 
        - threading.local: 스레드 안전성 보장
        - contextvars: 비동기 태스크 안전성 보장
    
    메모리: 
        - 동시 실행 컨텍스트당 ~2GB (모델 메모리)
        - 워커/스레드 수 제한 권장 (gunicorn -w 4 등)
    
    사용 예시:
        # 일반 Python 코드
        encoder = encoder_manager.get_encoder(config)
        
        # FastAPI 엔드포인트 (자동)
        @app.post("/token/camera")
        async def handler():
            # 내부적으로 encoder_manager.get_encoder() 사용
            embedding = generate_token_from_camera()
    </remarks>
    """
    
    def __init__(self):
        """
        <summary>동기/비동기 스토리지 초기화</summary>
        """
        self._local = threading.local()
    
    def get_encoder(self, config: AppConfig = default_config) -> FaceEncoder:
        """
        <summary>현재 실행 컨텍스트의 FaceEncoder 인스턴스를 반환</summary>
        <param name="config">전역 설정 객체</param>
        <returns>현재 컨텍스트 전용 FaceEncoder 인스턴스</returns>
        <remarks>
        비동기 환경에서는 contextvars 우선 사용, 동기 환경에서는 threading.local 사용.
        최초 호출 시 새 인코더를 생성하고, 이후 호출에서는 캐시된 인코더를 반환한다.
        
        성능:
            - 최초 호출: ~2초 (모델 로딩)
            - 이후 호출: ~0.01ms (캐시)
        </remarks>
        """
        context_encoder = _context_encoder.get()
        if context_encoder is not None:
            return context_encoder
        
        if not hasattr(self._local, 'encoder') or self._local.encoder is None:
            encoder = FaceEncoder(config)
            self._local.encoder = encoder
            self._local.config_id = id(config)
            
            try:
                _context_encoder.set(encoder)
            except Exception:
                pass
            
            return encoder
        
        return self._local.encoder
    
    def clear_encoder(self) -> None:
        """
        <summary>현재 컨텍스트의 인코더를 명시적으로 정리</summary>
        <remarks>
        메모리 최적화나 설정 변경 시 사용. 일반적으로는 자동 정리되므로 불필요.
        </remarks>
        """
        try:
            _context_encoder.set(None)
        except Exception:
            pass
        
        if hasattr(self._local, 'encoder'):
            self._local.encoder = None
            self._local.config_id = None


encoder_manager: ThreadSafeEncoderManager = ThreadSafeEncoderManager()


def generate_token_from_camera(max_frames: int = default_config.max_frames, config: AppConfig = default_config) -> list[float]:
    """
    <summary>카메라 입력으로부터 대표 임베딩을 생성하는 통합 함수</summary>
    <param name="max_frames">캡처할 최대 프레임 수 (1~20 권장)</param>
    <param name="config">전역 설정 객체 (camera_index, use_gpu 등 포함)</param>
    <returns>대표 임베딩 벡터를 리스트(float)로 반환 (길이: 512)</returns>
    <remarks>
    - 캡처→검출→임베딩→평균을 한 번에 수행한다.
    - 얼굴이 전혀 검출되지 않으면 ValueError를 발생시켜 호출 측이 흐름을 제어하도록 한다.
    - 반환값은 JSON 직렬화가 용이하도록 리스트 형태로 노출한다.
    
    처리 흐름:
        1. capture_from_camera: 카메라에서 max_frames 장 캡처 (얼굴 1개 필터)
        2. extract_face_embeddings: 각 프레임에서 임베딩 추출
        3. generate_token_from_embeddings: 임베딩 평균화
        4. tolist(): numpy → Python list 변환
    
    전제 조건:
        - 카메라가 연결되어 있고 액세스 가능
        - shared_encoder가 초기화됨 (모듈 로드 시 자동)
    
    후행 조건:
        - 카메라 리소스 해제 (capture_from_camera 내부에서 보장)
        - 반환된 리스트는 512개 float 값
    
    예외:
        - RuntimeError: 카메라 열기 실패
        - ValueError: 얼굴 미검출로 임베딩 생성 불가
        - 기타 InsightFace/OpenCV 예외
    
    사용 예시:
        # 기본 사용
        embedding = generate_token_from_camera(max_frames=5)
        print(f"임베딩 길이: {len(embedding)}")  # 512
        
        # 커스텀 설정
        config = AppConfig(camera_index=1, use_gpu=False)
        embedding = generate_token_from_camera(max_frames=3, config=config)
        
        # 유사도 비교
        import numpy as np
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    성능:
        - GPU: 5 프레임 기준 ~300ms (캡처 포함)
        - CPU: 5 프레임 기준 ~1200ms
    
    스레드 안전성:
        - ThreadLocalEncoderManager 사용으로 멀티스레드 환경에서 완전히 안전
        - 각 스레드는 독립적인 FaceEncoder 인스턴스 사용
        - FastAPI의 다중 워커/스레드 환경에서도 안전하게 동작
    </remarks>
    """
    encoder: FaceEncoder         = encoder_manager.get_encoder(config)
    frames: List[np.ndarray]     = capture_from_camera(max_frames=max_frames, config=config, encoder=encoder)
    embeddings: List[np.ndarray] = extract_face_embeddings(frames, encoder=encoder)
    
    if len(embeddings) == 0:
        raise NoValidEmbeddingError(
            f"캡처한 {len(frames)}개 프레임에서 얼굴 임베딩을 추출하지 못했습니다. "
            f"정면 얼굴로 재촬영하거나 조명을 개선하세요."
        )
    
    vector: np.ndarray = generate_token_from_embeddings(embeddings, config=config)
    return vector.tolist()


def generate_token_from_directory(dir_path: str, max_images: int = default_config.max_images, config: AppConfig = default_config) -> list[float]:
    """
    <summary>디렉토리 내 이미지로부터 대표 임베딩을 생성하는 통합 함수</summary>
    <param name="dir_path">이미지 디렉토리 절대/상대 경로</param>
    <param name="max_images">읽을 최대 이미지 수 (1~20 권장)</param>
    <param name="config">전역 설정 객체</param>
    <returns>대표 임베딩 벡터를 리스트(float)로 반환 (길이: 512)</returns>
    <remarks>
    - 로드→검출→임베딩→평균을 수행한다.
    - 얼굴이 검출되지 않으면 ValueError를 발생시켜 호출 측이 명확히 실패를 처리하도록 한다.
    - 파일 확장자 필터링은 image_loader에서 수행된다 (jpg/jpeg/png만).
    
    처리 흐름:
        1. load_images_from_directory: 디렉토리에서 max_images 장 로딩
        2. extract_face_embeddings: 각 이미지에서 임베딩 추출 (얼굴 1개 필터)
        3. generate_token_from_embeddings: 임베딩 평균화
        4. tolist(): numpy → Python list 변환
    
    전제 조건:
        - dir_path는 유효한 디렉토리 경로
        - 디렉토리 내에 jpg/jpeg/png 이미지 존재
        - 읽기 권한 필요
    
    후행 조건:
        - 반환된 리스트는 512개 float 값
        - 원본 이미지 파일은 변경되지 않음
    
    예외:
        - FileNotFoundError: 디렉토리 미존재
        - ValueError: 얼굴 미검출로 임베딩 생성 불가
        - PermissionError: 읽기 권한 부족
    
    사용 예시:
        # 기본 사용
        embedding = generate_token_from_directory("C:/faces/user1")
        
        # 최대 10장 사용
        embedding = generate_token_from_directory("./faces", max_images=10)
        
        # 등록(Enroll) 시나리오
        enroll_embedding = generate_token_from_directory("C:/enroll/user123", max_images=5)
        # DB에 저장: save_to_db(user_id=123, embedding=enroll_embedding)
        
        # 검증(Verify) 시나리오
        verify_embedding = generate_token_from_directory("C:/verify/temp", max_images=3)
        similarity = cosine_similarity(enroll_embedding, verify_embedding)
        if similarity >= 0.40:
            print("동일인 확인")
    
    성능:
        - GPU: 5 이미지 기준 ~300ms (로딩 + 추론)
        - CPU: 5 이미지 기준 ~1200ms
        - SSD 권장 (HDD는 로딩 지연 발생)
    
    보안 고려사항:
        - 디렉토리 경로 검증 필요 (경로 순회 공격 방지)
        - 프로덕션 환경에서는 허용 경로 화이트리스트 사용 권장
    
    스레드 안전성:
        - ThreadLocalEncoderManager 사용으로 멀티스레드 환경에서 완전히 안전
    </remarks>
    """
    encoder: FaceEncoder         = encoder_manager.get_encoder(config)
    images: List[np.ndarray]     = load_images_from_directory(dir_path=dir_path, max_images=max_images, config=config)
    embeddings: List[np.ndarray] = extract_face_embeddings(images, encoder=encoder)
    
    if len(embeddings) == 0:
        raise NoValidEmbeddingError(
            f"디렉토리의 {len(images)}개 이미지에서 얼굴 임베딩을 추출하지 못했습니다. "
            f"정면 얼굴 사진을 사용하거나 단일 얼굴만 포함된 이미지를 제공하세요."
        )
    
    vector: np.ndarray = generate_token_from_embeddings(embeddings, config=config)
    return vector.tolist()

