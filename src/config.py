"""
<summary>전역 설정값과 상수 정의 모듈</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>GPU 가속 사용 여부 및 카메라 기본 인덱스를 정의한다.</remarks>
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """
    <summary>애플리케이션 전역 설정 데이터 클래스</summary>
    <remarks>
    불변 설정 객체로 설계되어 의존성 주입이 용이하다.
    프로덕션 환경에서는 환경변수 또는 설정 파일로부터 값을 주입하여 인스턴스를 생성한다.
    
    예시:
        config = AppConfig(camera_index=1, use_gpu=False)
        encoder = FaceEncoder(config)
    
    스레드 안전성: dataclass는 불변이 아니므로 다중 스레드 환경에서는 읽기 전용으로만 사용해야 한다.
    </remarks>
    """

    camera_index: int          = 0           # 카메라 디바이스 인덱스 (0=첫 번째, 1=두 번째, ...)
    max_frames: int            = 5           # 카메라 캡처 최대 프레임 수 (1~20 권장)
    max_images: int            = 5           # 디렉토리 읽기 최대 이미지 수 (1~20 권장)
    use_gpu: bool              = True        # NVIDIA CUDA GPU 가속 활성화 여부
    use_directml: bool         = True        # AMD DirectML 가속 활성화 여부 (실험적)
    model_name: str            = "buffalo_l" # InsightFace 모델명 (buffalo_s/m/l)
    detection_threshold: float = 0.6         # 얼굴 검출 신뢰도 임계값 (0.0~1.0)
    hash_precision: int        = 3           # 임베딩 양자화 소수점 자리수 (사용 안 함, 하위 호환)
    hash_method: str           = "sha256"    # 해시 알고리즘 (사용 안 함, 하위 호환)
    server_host: str           = "127.0.0.1" # FastAPI 서버 리스닝 호스트
    server_port: int           = 23535       # FastAPI 서버 리스닝 포트
    temp_dir: Path             = Path("./tmp") # 임시 파일 저장 경로 (현재 미사용)


default_config: AppConfig = AppConfig()

