"""
<module_description>
    Pydantic Settings 기반의 전역 설정 관리 모듈.
</module_description>
<technical_overview>
    본 모듈은 '12-Factor App' 설계 원칙을 준수하여 설정(Configuration)을 소스 코드와 분리합니다. 
    환경 변수(Environment Variables) 및 .env 파일로부터 설정을 주입받으며, 
    Pydantic V2의 엄격한 타입 검증 기능을 활용하여 시스템의 안정성을 확보합니다.
</technical_overview>
<architectural_design>
    1. 환경 변수 주입: FACER_ 접두사를 가진 환경 변수를 최우선으로 인식하여 자동 로딩함.
    2. 불변성(Immutability): frozen=True 설정을 통해 런타임 중에 전역 설정이 
       오염되는 것을 원천적으로 차단하여 시스템의 예측 가능성을 높임.
    3. 보호된 네임스페이스 관리: Pydantic 내부 엔진 속성과의 이름 충돌을 방지하기 위한 
       protected_namespaces 설정을 포함함.
</architectural_design>
<author>최진호 (Senior Principal Software Engineer / Computer Science PhD)</author>
<version>2.2.0</version>
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):
    """
    <summary>애플리케이션 전역 설정(Global Configuration) 컨테이너</summary>
    <remarks>
    본 클래스는 시스템의 하드웨어 가속 방식, 모델 사양, 서버 인프라 정보를 중앙 집중적으로 관리합니다.
    데이터 무결성 확보를 위해 불변 객체(frozen)로 설계되었습니다.
    </remarks>
    """

    # --- 하드웨어 인터페이스 및 카메라 설정 ---
    camera_index: int = Field(
        default=0, 
        description="카메라 디바이스 인덱스 (0=Primary, 1=Secondary...)"
    )
    max_frames: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="평균화된 대표 임베딩 생성을 위한 최대 캡처 프레임 수"
    )
    max_images: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="디렉토리 기반 로딩 시 분석할 최대 이미지 개수"
    )
    
    # --- 추론 가속기 및 AI 모델 전략 ---
    use_gpu: bool = Field(
        default=True, 
        description="NVIDIA CUDA GPU 가속 활성화 전략"
    )
    use_directml: bool = Field(
        default=True, 
        description="AMD/Intel GPU용 DirectML 가속 활성화 전략 (실험적)"
    )
    model_name: str = Field(
        default="buffalo_l", 
        description="InsightFace 신경망 모델 명칭 (buffalo_s/m/l)"
    )
    detection_threshold: float = Field(
        default=0.6, 
        ge=0.0, 
        le=1.0, 
        description="안면 검출 신뢰도 하한선 (0.6 이상 시 유효한 안면으로 인식)"
    )
    
    # --- 서버 네트워크 및 인프라스트럭처 설정 ---
    server_host: str = Field(
        default="127.0.0.1", 
        description="FastAPI 서버 리스닝 호스트 주소"
    )
    server_port: int = Field(
        default=23535, 
        description="FastAPI 서버 리스닝 포트 번호"
    )
    temp_dir: Path = Field(
        default=Path("./tmp"), 
        description="임시 데이터 및 캐시 디렉토리 경로"
    )

    # --- 인증 판정 임계값 (Similarity Thresholds) ---
    cosine_threshold_high: float = Field(
        default=0.40, 
        description="동일인 판정 상단 임계값 (상한선 초과 시 신원 확인 긍정)"
    )
    cosine_threshold_low: float = Field(
        default=0.30,
        description="동일인 판정 하단 임계값 (하한선 미달 시 신원 확인 부정)"
    )

    # --- 모듈 선택 및 추론 전략 ---
    allowed_modules: List[str] = Field(
        default=["detection", "landmark_3d_68", "recognition"],
        description="로딩할 InsightFace 모듈 목록 (genderage 제외로 추론 속도 향상)"
    )
    require_pose: bool = Field(
        default=True,
        description="3D 랜드마크 기반 pitch/yaw/roll 추출 활성화"
    )
    det_size: int = Field(
        default=320,
        ge=160,
        le=640,
        description="얼굴 검출 입력 해상도 (근거리 키오스크: 320, 원거리: 640)"
    )

    # --- 정합성 품질 관리 ---
    min_valid_frames: int = Field(
        default=3,
        ge=1,
        description="대표 임베딩 생성 최소 유효 프레임 수"
    )
    min_quality_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="프레임 수용 최소 품질 점수 (det_score + sharpness 가중 합산)"
    )
    outlier_cosine_threshold: float = Field(
        default=0.45,
        description="아웃라이어 임베딩 제거 임계값 (임시 mean과의 cosine 최소값)"
    )
    pose_yaw_limit: float = Field(
        default=25.0,
        ge=0.0,
        le=90.0,
        description="허용 yaw(좌우 회전) 각도 한계 (도)"
    )
    pose_pitch_limit: float = Field(
        default=20.0,
        ge=0.0,
        le=90.0,
        description="허용 pitch(상하 기울기) 각도 한계 (도)"
    )

    # --- 카메라 파이프라인 ---
    camera_warmup_frames: int = Field(
        default=5,
        ge=0,
        description="카메라 open 직후 버릴 초기 프레임 수 (노출 안정화)"
    )
    frame_interval_ms: float = Field(
        default=200.0,
        ge=0.0,
        description="유효 프레임 간 최소 시간 간격(ms) — 유사 프레임 중복 방지"
    )

    # --- Pydantic Settings 엔진 내부 설정 ---
    model_config = SettingsConfigDict(
        # 우선순위 1: .env 파일 데이터 로딩
        env_file=".env",
        
        # 우선순위 2: FACER_ 접두사를 가진 환경 변수 탐색
        # 예시: FACER_SERVER_PORT=8080 주입 시 server_port 속성이 8080으로 자동 갱신됨
        env_prefix="FACER_",
        
        # 파일 인코딩 표준 설정
        env_file_encoding="utf-8",
        
        # 설정 클래스에 정의되지 않은 환경 변수는 조용히 무시하여 유연성 확보
        extra="ignore",
        
        # 불변성(Immutability) 강제: 런타임 오염 방지 및 예측 가능성 증대
        frozen=True,
        
        # Pydantic 2.x 내부 보호 네임스페이스 충돌 회피 전략
        protected_namespaces=('settings_',)
    )


# [하위 호환성 레이어] 레거시 코드와의 인터페이스 유지를 위한 별칭 제공
AppConfig = AppSettings

# [전역 싱글톤 객체] 시스템 전역에서 참조하는 기본 설정 인스턴스
default_config: AppSettings = AppSettings()
