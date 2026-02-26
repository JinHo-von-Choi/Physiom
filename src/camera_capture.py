"""
<summary>카메라로부터 얼굴 프레임 캡처 유틸리티</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>GPU 장착 키오스크 환경에서 기본 카메라를 사용한다.</remarks>
"""

import logging
import time
from typing import List, Optional

import cv2
import numpy as np

from src.config import AppConfig, default_config
from src.face_encoder import FaceEncoder
from src.exceptions import CameraOpenError, CameraFrameError, TimeoutError as FaceTimeoutError


logger = logging.getLogger("camera-capture")
logger.setLevel(logging.INFO)


def capture_from_camera(max_frames: int = default_config.max_frames, config: AppConfig = default_config, encoder: Optional[FaceEncoder] = None) -> List[np.ndarray]:
    """
    <summary>카메라로부터 얼굴이 포함된 프레임을 최대 max_frames 만큼 캡처</summary>
    <param name="max_frames">캡처할 최대 프레임 수 (1~20 권장)</param>
    <param name="config">전역 설정 객체 (camera_index 포함)</param>
    <param name="encoder">얼굴 검출용 인코더 (미지정 시 자동 생성, 성능상 재사용 권장)</param>
    <returns>얼굴이 검출된 프레임 리스트 (각 프레임은 BGR HxWx3 uint8 ndarray)</returns>
    <remarks>
    - Enter 키(ASCII 13) 또는 10초 타임아웃으로 캡처를 중단한다.
    - 얼굴이 정확히 1개 검출된 프레임만 반환하며, 추후 임베딩 추출 정확도를 높인다.
    - 카메라 열기 실패 시 즉시 RuntimeError를 발생시켜 상위 호출자에게 오류를 알린다.
    
    중단 조건:
        1. max_frames 도달
        2. 10초 타임아웃
        3. Enter 키 입력 (cv2.waitKey 사용)
    
    전제 조건:
        - config.camera_index는 유효한 카메라 인덱스 (0~N)
        - 카메라가 다른 프로세스에 의해 점유되지 않음
        - Windows 카메라 프라이버시 설정에서 액세스 허용됨
    
    후행 조건:
        - 카메라 리소스 해제 (finally 블록에서 보장)
        - OpenCV 윈도우 모두 닫힘
        - 반환된 프레임은 모두 얼굴이 1개만 검출됨
    
    예외:
        - RuntimeError: 카메라 열기 실패
        - 기타 cv2 관련 예외: 프레임 읽기 오류 (로그 후 재시도)
    
    사용 예시:
        # 기본 사용
        frames = capture_from_camera(max_frames=5)
        
        # 인코더 재사용 (권장)
        encoder = FaceEncoder(config)
        frames = capture_from_camera(max_frames=5, encoder=encoder)
        
        # 커스텀 카메라 인덱스
        config = AppConfig(camera_index=1)
        frames = capture_from_camera(config=config)
    
    성능:
        - 실시간 처리 (~30fps 카메라 기준)
        - GPU: 얼굴 검출 ~20ms/frame
        - CPU: 얼굴 검출 ~100ms/frame
    
    로깅:
        - WARNING: 프레임 캡처 실패 시
        - INFO: 타임아웃 또는 사용자 중단 시
    
    부작용:
        - 카메라 LED 켜짐
        - OpenCV 윈도우 생성 가능 (GUI 환경)
    </remarks>
    """
    active_encoder: FaceEncoder  = encoder or FaceEncoder(config)
    captured: List[np.ndarray]   = []
    timeout_seconds: float       = 10.0
    max_frame_failures: int      = 10
    frame_failure_count: int     = 0
    start_time: float            = time.time()

    camera: cv2.VideoCapture = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        raise CameraOpenError(
            f"카메라(인덱스: {config.camera_index})를 열 수 없습니다. "
            f"카메라 연결, 드라이버, 프라이버시 설정을 확인하세요."
        )

    # 카메라 초기 프레임 버리기: 자동 노출/화이트밸런스 안정화
    for _ in range(config.camera_warmup_frames):
        camera.read()

    try:
        last_capture_time: float = 0.0  # 마지막 유효 프레임 캡처 시각 (ms)

        while len(captured) < max_frames:
            ret, frame = camera.read()
            if not ret:
                frame_failure_count += 1
                logger.warning(
                    "프레임 캡처 실패 (%d/%d), 재시도합니다.",
                    frame_failure_count,
                    max_frame_failures
                )
                
                if frame_failure_count >= max_frame_failures:
                    raise CameraFrameError(
                        f"프레임 캡처가 {max_frame_failures}회 연속 실패했습니다. "
                        f"카메라 연결 상태를 확인하세요."
                    )
                
                time.sleep(0.1)
                continue
            
            frame_failure_count = 0

            # 프레임 간격 강제: 유사 프레임 중복 방지
            current_ms = time.time() * 1000
            if (current_ms - last_capture_time) < config.frame_interval_ms:
                continue

            try:
                faces = active_encoder.detect_faces(frame)
                if len(faces) == 1:
                    captured.append(frame)
                    last_capture_time = current_ms
                    logger.debug("유효한 얼굴 프레임 캡처 (%d/%d)", len(captured), max_frames)
            except Exception as exc:
                logger.warning("얼굴 검출 중 오류 발생, 프레임 스킵: %s", exc)
                continue

            elapsed: float = time.time() - start_time
            if elapsed >= timeout_seconds:
                if len(captured) == 0:
                    raise FaceTimeoutError(
                        f"캡처 타임아웃({timeout_seconds}초) 내에 유효한 얼굴 프레임을 찾지 못했습니다."
                    )
                logger.info("캡처 타임아웃 도달로 중단합니다. (캡처된 프레임: %d)", len(captured))
                break

            if cv2.waitKey(1) & 0xFF == 13:
                logger.info("사용자 입력으로 캡처를 종료합니다. (캡처된 프레임: %d)", len(captured))
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return captured

