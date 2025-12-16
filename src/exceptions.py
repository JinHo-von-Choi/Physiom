"""
<summary>프로젝트 전용 커스텀 예외 클래스 정의 모듈</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>
명확한 오류 원인 전달과 세밀한 예외 처리를 위해 도메인별 예외를 정의한다.
모든 커스텀 예외는 FaceServiceError를 상속하여 일괄 처리 가능하도록 설계했다.
</remarks>
"""


class FaceServiceError(Exception):
    """
    <summary>얼굴 인식 서비스의 최상위 예외 클래스</summary>
    <remarks>
    모든 커스텀 예외의 베이스 클래스로, 서비스 전체 예외를 한 번에 포착 가능하다.
    
    사용 예시:
        try:
            result = some_face_operation()
        except FaceServiceError as e:
            logger.error(f"서비스 오류: {e}")
    </remarks>
    """


class CameraError(FaceServiceError):
    """
    <summary>카메라 관련 오류</summary>
    <remarks>
    카메라 열기 실패, 프레임 읽기 실패, 권한 부족 등 카메라 작업 중 발생하는 모든 오류.
    
    발생 상황:
        - 카메라 디바이스 미존재
        - 다른 프로세스가 카메라 점유 중
        - Windows 카메라 프라이버시 설정으로 액세스 차단
        - 카메라 드라이버 오류
    </remarks>
    """


class CameraOpenError(CameraError):
    """
    <summary>카메라 열기 실패</summary>
    <remarks>
    cv2.VideoCapture.isOpened()가 False를 반환할 때 발생.
    
    원인:
        - 잘못된 camera_index
        - 카메라 미연결
        - 드라이버 미설치
    </remarks>
    """


class CameraFrameError(CameraError):
    """
    <summary>카메라 프레임 읽기 실패</summary>
    <remarks>
    cv2.VideoCapture.read()가 ret=False를 반환할 때 발생.
    일시적 오류일 수 있으므로 재시도 로직 권장.
    
    원인:
        - 카메라 연결 불안정
        - 버퍼 오버플로우
        - 일시적 하드웨어 오류
    </remarks>
    """


class ModelError(FaceServiceError):
    """
    <summary>얼굴 인식 모델 관련 오류</summary>
    <remarks>
    InsightFace 모델 로딩, 초기화, 추론 중 발생하는 오류.
    </remarks>
    """


class ModelLoadError(ModelError):
    """
    <summary>모델 로딩 실패</summary>
    <remarks>
    InsightFace 모델 다운로드 또는 로딩 중 오류 발생.
    
    원인:
        - 인터넷 연결 끊김 (모델 다운로드 실패)
        - 디스크 공간 부족
        - 손상된 모델 파일
        - onnxruntime 미설치
    </remarks>
    """


class ModelInferenceError(ModelError):
    """
    <summary>모델 추론(inference) 실패</summary>
    <remarks>
    얼굴 검출 또는 임베딩 추출 중 오류 발생.
    
    원인:
        - GPU 메모리 부족
        - 잘못된 입력 형식
        - 모델 내부 오류
    </remarks>
    """


class FaceDetectionError(FaceServiceError):
    """
    <summary>얼굴 검출 관련 오류</summary>
    <remarks>
    얼굴 미검출, 다중 얼굴 검출 등 얼굴 검출 단계의 논리적 오류.
    </remarks>
    """


class NoFaceDetectedError(FaceDetectionError):
    """
    <summary>얼굴 미검출</summary>
    <remarks>
    이미지에서 얼굴이 하나도 검출되지 않음.
    
    원인:
        - 얼굴이 없는 이미지
        - 극단적 각도/조명
        - 이미지 품질 불량
    
    사용자 액션:
        - 정면 얼굴 촬영 요청
        - 조명 개선
        - 카메라 초점 확인
    </remarks>
    """


class MultipleFacesDetectedError(FaceDetectionError):
    """
    <summary>다중 얼굴 검출</summary>
    <remarks>
    이미지에서 2개 이상의 얼굴이 검출됨.
    단일 얼굴만 처리하는 정책에 위배.
    
    사용자 액션:
        - 1명만 촬영 요청
        - 배경 인물 제거
    </remarks>
    """


class EmbeddingError(FaceServiceError):
    """
    <summary>임베딩 생성 관련 오류</summary>
    <remarks>
    임베딩 추출, 평균화, 변환 중 발생하는 오류.
    </remarks>
    """


class NoValidEmbeddingError(EmbeddingError):
    """
    <summary>유효한 임베딩 없음</summary>
    <remarks>
    여러 이미지를 처리했으나 단 하나의 유효한 임베딩도 생성하지 못함.
    
    원인:
        - 모든 이미지에서 얼굴 미검출
        - 모든 이미지가 다중 얼굴
        - 임베딩 추출 모두 실패
    </remarks>
    """


class ImageError(FaceServiceError):
    """
    <summary>이미지 파일 처리 오류</summary>
    <remarks>
    이미지 로딩, 디코딩, 형식 변환 중 발생하는 오류.
    </remarks>
    """


class ImageLoadError(ImageError):
    """
    <summary>이미지 로딩 실패</summary>
    <remarks>
    cv2.imread가 None을 반환하거나 파일 읽기 실패.
    
    원인:
        - 손상된 이미지 파일
        - 지원하지 않는 포맷
        - 파일 권한 부족
        - 파일 미존재
    </remarks>
    """


class InvalidImageFormatError(ImageError):
    """
    <summary>잘못된 이미지 형식</summary>
    <remarks>
    이미지 shape, dtype, 채널 수 등이 요구사항에 맞지 않음.
    
    요구사항:
        - shape: (H, W, 3) - 3채널 컬러 이미지
        - dtype: uint8
        - format: BGR (OpenCV 표준)
    </remarks>
    """


class ConfigurationError(FaceServiceError):
    """
    <summary>설정 관련 오류</summary>
    <remarks>
    잘못된 설정값, 필수 설정 누락 등.
    
    예시:
        - camera_index가 음수
        - max_frames가 0
        - 존재하지 않는 model_name
    </remarks>
    """


class TimeoutError(FaceServiceError):
    """
    <summary>작업 타임아웃</summary>
    <remarks>
    카메라 캡처, 모델 로딩 등의 작업이 제한 시간을 초과함.
    
    발생 상황:
        - 카메라 캡처 10초 타임아웃
        - 모델 다운로드 지연
        - GPU 응답 없음
    </remarks>
    """

