# Face Token Service Requirements

## 런타임 환경

- Python 3.11 이상 (3.9~3.12 호환)
- GPU가 장착된 Windows 10/11 또는 Linux, NVIDIA CUDA 드라이버 권장

## 의존성 라이브러리

| 패키지 | 최소 버전 | 용도 |
|---|---|---|
| insightface | 0.7.3 | ArcFace 기반 얼굴 검출·인식 엔진 |
| onnxruntime | 1.16.0 | ONNX 추론 런타임 (CPU 기본) |
| opencv-python | 4.8.0 | 이미지 처리, Laplacian 선명도 분석 |
| numpy | 1.24.0 | 임베딩 벡터 연산 |
| fastapi | 0.104.0 | 비동기 HTTP API 프레임워크 |
| uvicorn[standard] | 0.24.0 | ASGI 서버 |
| pydantic | 2.5.0 | 데이터 검증 |
| pydantic-settings | 2.0.0 | 환경 변수 / .env 파일 기반 설정 관리 |
| python-dotenv | 1.0.0 | .env 파일 로딩 지원 |

GPU 가속이 필요한 경우 `onnxruntime` 대신:
- NVIDIA: `pip install onnxruntime-gpu>=1.16.0`
- AMD/Intel: `pip install onnxruntime-directml>=1.16.0`

## 포트

- HTTP 서버 기본 포트: `23535` (환경 변수 `FACER_SERVER_PORT`로 변경 가능)

## 보안 / 프라이버시

- 카메라로 캡처한 이미지는 디스크에 저장하지 않음
- 임베딩 및 토큰은 메모리에서만 유지
- API 키, 비밀번호 등 민감 정보는 환경 변수 또는 `.env` 파일로 관리

## 입력 제약

- 디렉토리 이미지 확장자: `.jpg`, `.jpeg`, `.png`
- 얼굴이 1개인 이미지/프레임만 사용 (다중 얼굴 시 가장 큰 얼굴 자동 선정)
- 최소 유효 프레임: 3개 이상 (`FACER_MIN_VALID_FRAMES` 조정 가능)
- 허용 포즈 범위: |yaw| ≤ 25°, |pitch| ≤ 20° (`FACER_POSE_YAW_LIMIT/PITCH_LIMIT` 조정 가능)

## 실행 예시

```bash
# 카메라 입력
python main.py camera --max-frames 5

# 디렉토리 입력
python main.py directory --path "C:/faces/user1" --max-images 5

# HTTP API 서버
python main.py server --host 0.0.0.0 --port 23535
```
