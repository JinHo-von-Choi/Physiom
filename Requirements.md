# Face Token Service Requirements

- Python 3.11 이상
- GPU가 장착된 Windows 10/11, NVIDIA CUDA 11.x 드라이버 권장
- 필수 라이브러리
  - insightface (ArcFace 기반, onnxruntime-gpu 포함)
  - opencv-python
  - numpy
  - fastapi
  - uvicorn[standard]
  - pydantic
- 포트
  - HTTP 서버 기본 포트: 23535 (config로 변경 가능)
- 보안/프라이버시
  - 카메라로 캡처한 이미지는 디스크에 저장하지 않음
  - 임베딩 및 토큰은 메모리에서만 유지
- 입력 제약
  - 디렉토리 이미지 확장자: .jpg, .jpeg, .png
  - 얼굴이 1개인 이미지/프레임만 사용
- 실행 예시
  - python main.py camera --max-frames 5
  - python main.py directory --path "C:/faces/user1" --max-images 5
  - python main.py server --host 0.0.0.0 --port 23535

