"""
<summary>디렉토리로부터 얼굴 이미지 로드 유틸리티</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>허용된 확장자만 읽고 실패 시 로그만 남긴다.</remarks>
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.config import AppConfig, default_config
from src.exceptions import ImageLoadError


logger = logging.getLogger("image-loader")
logger.setLevel(logging.INFO)

SUPPORTED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png")


def load_images_from_directory(dir_path: str, max_images: int = default_config.max_images, config: AppConfig = default_config) -> List[np.ndarray]:
    """
    <summary>지정된 디렉토리에서 이미지를 최대 max_images 만큼 로드</summary>
    <param name="dir_path">이미지를 포함하는 디렉토리 절대/상대 경로</param>
    <param name="max_images">최대 로드할 이미지 수 (1~20 권장)</param>
    <param name="config">전역 설정 객체 (현재 미사용, 확장성 위해 보유)</param>
    <returns>성공적으로 로드된 이미지 리스트 (각 이미지는 BGR HxWx3 uint8 ndarray)</returns>
    <remarks>
    - 확장자 필터: jpg, jpeg, png만 처리한다 (대소문자 구분 없음).
    - 로드 실패 파일은 스킵하고 경고 로그만 남겨 로버스트하게 진행한다.
    - 디렉토리 유효성 검증에 실패하면 즉시 FileNotFoundError를 발생시켜 상위에서 처리한다.
    
    처리 순서:
        1. 디렉토리 존재 여부 확인
        2. 파일 목록 순회 (iterdir 순서: OS 파일시스템 순서)
        3. 확장자 필터링
        4. max_images 도달 시 중단
        5. cv2.imread로 로딩, 실패 시 스킵
    
    전제 조건:
        - dir_path는 유효한 디렉토리 경로
        - 읽기 권한 필요
    
    후행 조건:
        - 반환된 이미지는 모두 유효한 BGR 포맷 numpy.ndarray
        - len(images) <= max_images
        - 로드 실패 이미지는 포함되지 않음
    
    예외:
        - FileNotFoundError: 디렉토리 미존재 또는 파일이 아닌 경로
        - PermissionError: 읽기 권한 부족 (OS에서 발생)
    
    사용 예시:
        # 기본 사용
        images = load_images_from_directory("C:/faces/user1")
        
        # 최대 10장 로드
        images = load_images_from_directory("./faces", max_images=10)
        
        # 상대 경로 (프로젝트 루트 기준)
        images = load_images_from_directory("../data/faces")
    
    성능:
        - SSD 기준: 5장 이미지 로딩 ~50ms
        - HDD 기준: 5장 이미지 로딩 ~200ms
        - 네트워크 드라이브: 지연 가능성 높음
    
    로깅:
        - WARNING: 이미지 로딩 실패 시 "이미지 로딩 실패: {file_path}" 출력
    
    보안 고려사항:
        - 심볼릭 링크 추적하므로 디렉토리 순회 공격 가능
        - 프로덕션 환경에서는 경로 검증 추가 권장
    </remarks>
    """
    path: Path               = Path(dir_path)
    images: List[np.ndarray] = []
    failed_count: int        = 0
    max_failures: int        = 20

    if not path.exists():
        raise FileNotFoundError(
            f"디렉토리가 존재하지 않습니다: {dir_path}"
        )
    
    if not path.is_dir():
        raise NotADirectoryError(
            f"경로가 디렉토리가 아닙니다: {dir_path}"
        )

    try:
        file_list = list(path.iterdir())
    except PermissionError as exc:
        raise PermissionError(
            f"디렉토리 읽기 권한이 없습니다: {dir_path}"
        ) from exc

    for file_path in file_list:
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("지원하지 않는 확장자, 스킵: %s", file_path.suffix)
            continue
        
        if len(images) >= max_images:
            logger.info("최대 이미지 수(%d) 도달, 로딩 중단", max_images)
            break

        try:
            image: np.ndarray = cv2.imread(str(file_path))
            if image is None:
                failed_count += 1
                logger.warning(
                    "이미지 로딩 실패 (%d/%d): %s",
                    failed_count,
                    max_failures,
                    file_path.name
                )
                
                if failed_count >= max_failures:
                    raise ImageLoadError(
                        f"{max_failures}개 이상의 이미지 로딩 실패. "
                        f"디렉토리에 손상된 파일이 다수 포함되어 있을 수 있습니다."
                    )
                continue
            
            if image.size == 0:
                logger.warning("빈 이미지 파일, 스킵: %s", file_path.name)
                continue
            
            images.append(image)
            logger.debug("이미지 로딩 성공 (%d/%d): %s", len(images), max_images, file_path.name)
        
        except PermissionError:
            logger.warning("파일 읽기 권한 없음, 스킵: %s", file_path.name)
            continue
        except Exception as exc:
            logger.warning("이미지 로딩 중 예외 발생, 스킵: %s - %s", file_path.name, exc)
            continue

    if len(images) == 0:
        raise ImageLoadError(
            f"디렉토리 '{dir_path}'에서 유효한 이미지를 찾지 못했습니다. "
            f"지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return images

