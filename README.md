# Face Embedding Service - 엔터프라이즈 얼굴 인증 토큰 생성 모듈

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Pydantic V2](https://img.shields.io/badge/Pydantic-V2-e92063.svg)](https://docs.pydantic.dev/)

## 목차
1. [개요 및 목적](#개요-및-목적)
2. [주요 업데이트 (v2.3.0)](#주요-업데이트-v230)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [기술 스택](#기술-스택)
5. [설치 및 실행](#설치-및-실행)
6. [설정 레퍼런스](#설정-레퍼런스)
7. [핵심 기능](#핵심-기능)
8. [사용 방법](#사용-방법)
9. [테스트 및 검증](#테스트-및-검증)
10. [배포 (Docker)](#배포-docker)
11. [FAQ 및 보안](#faq-및-보안)

---

## 개요 및 목적

본 프로젝트는 엔터프라이즈 환경에서 신뢰할 수 있는 얼굴 임베딩(Embedding)을 생성하고 관리하기 위한 고성능 마이크로서비스입니다. InsightFace(ArcFace)를 기반으로 하며, 근거리 키오스크 환경에서의 정합성 극대화와 추론 성능 최적화에 초점을 맞춥니다.

---

## 주요 업데이트 (v2.3.0)

v2.0.0 대비 추가된 핵심 개선 사항입니다.

- **[P0 버그픽스] L2 재정규화**: `generate_token_from_embeddings`가 평균 벡터를 L2 재정규화하지 않던 문제를 수정. 코사인 유사도 연산의 수학적 정확성 확보.
- **[P1] 추론 모듈 최적화**: `allowed_modules` 설정으로 불필요한 `genderage` 모듈 로딩 제거. 첫 추론 레이턴시를 유발하던 ONNX JIT 컴파일을 서버 기동 시 warm-up으로 선제 처리.
- **[P1] 검출 해상도 최적화**: 검출 입력 해상도를 640×640에서 320×320으로 조정 (`det_size`). 근거리 키오스크 환경에서 속도/정합성 균형 달성.
- **[P2] Pose 필터링**: `landmark_3d_68` 모델의 3D 포즈 데이터(pitch/yaw/roll)를 활용하여 과도한 측면 각도의 얼굴 프레임을 자동 배제.
- **[P2] Outlier 제거**: 임베딩 집계 전 임시 평균과의 코사인 유사도가 임계값 미만인 outlier 벡터를 제거하여 대표 임베딩 품질 향상.
- **[P2] 최소 유효 프레임 검증**: `min_valid_frames` 미달 시 `NoValidEmbeddingError` 조기 발생.
- **[P3] Quality-weighted 집계**: 검출 점수와 Laplacian 선명도를 결합한 2인자 품질 점수로 임베딩 가중 평균 수행.
- **[P3] 카메라 파이프라인 개선**: 카메라 개방 직후 자동 노출 안정화를 위한 warmup 프레임 스킵 및 유사 프레임 중복 방지를 위한 프레임 간격 제어.

---

## 시스템 아키텍처

### 데이터 흐름

```
카메라/이미지 입력
       │
       ▼
camera_capture.py        ← warmup 스킵, 프레임 간격 제어
       │
       ▼
face_encoder.py          ← allowed_modules, det_size 320
  ├── detect_faces()     ← Laplacian blur 체크
  ├── _is_pose_acceptable() ← yaw/pitch 범위 검사
  └── compute_quality_score() ← det_score + sharpness
       │
       ▼
src/__init__.py
  ├── _reject_outliers() ← cosine 기반 아웃라이어 제거
  └── min_valid_frames 검증
       │
       ▼
token_generator.py       ← quality-weighted mean + L2 재정규화
       │
       ▼
  대표 임베딩 (512-dim, L2 norm=1.0)
```

### 디렉토리 구조

```
facer/
├── main.py                    # CLI 엔트리포인트
├── compare_embeddings.py      # 임베딩 비교 유틸리티
├── Dockerfile                 # 컨테이너 배포 설정
├── requirements.txt           # 의존성 목록
├── src/
│   ├── config.py             # Pydantic Settings — 20개 설정 필드
│   ├── face_encoder.py       # FaceEncoder — pose 필터, quality score
│   ├── token_generator.py    # L2 재정규화 + weighted mean
│   ├── api_http.py           # FastAPI — lifespan warm-up
│   ├── camera_capture.py     # 카메라 — warmup 스킵, 프레임 간격
│   ├── image_loader.py       # 디렉토리 이미지 로더
│   ├── exceptions.py         # 도메인 예외 계층
│   └── cli.py                # CLI 핸들러
└── tests/
    ├── test_config.py         # 설정 필드 검증
    ├── test_token_generator.py # L2 정규화, 가중 평균
    ├── test_integration.py    # outlier 제거, min_valid_frames
    ├── test_api.py            # HTTP 엔드포인트
    └── test_robustness.py     # pose 필터, blur 체크 (모델 필요)
```

---

## 기술 스택

- **AI Engine**: InsightFace (ArcFace buffalo_l)
- **Runtime**: ONNX Runtime (GPU/CPU Auto-fallback)
- **Web Framework**: FastAPI (Asynchronous, Type-safe)
- **Validation & Settings**: Pydantic V2 / Pydantic Settings
- **Image Processing**: OpenCV / NumPy

---

## 설치 및 실행

### 1. 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU 가속이 필요한 경우:
```bash
pip install onnxruntime-gpu>=1.16.0
```

### 2. 실행

```bash
# CLI — 카메라
python main.py camera --max-frames 5

# CLI — 디렉토리
python main.py directory --path "./faces/user1" --max-images 5

# HTTP API 서버
python main.py server --host 0.0.0.0 --port 23535
```

---

## 설정 레퍼런스

`src/config.py`의 `AppSettings`는 `FACER_` 접두사를 가진 환경 변수 또는 `.env` 파일로부터 자동 주입됩니다.

### 하드웨어 / 카메라

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `FACER_USE_GPU` | `True` | CUDA GPU 가속 |
| `FACER_USE_DIRECTML` | `True` | AMD/Intel GPU (실험적) |
| `FACER_CAMERA_INDEX` | `0` | 카메라 디바이스 인덱스 |
| `FACER_MAX_FRAMES` | `5` | 카메라 최대 캡처 프레임 수 |
| `FACER_MAX_IMAGES` | `5` | 디렉토리 최대 이미지 수 |

### AI 모델

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `FACER_MODEL_NAME` | `buffalo_l` | InsightFace 모델 (s/m/l) |
| `FACER_ALLOWED_MODULES` | `["detection","landmark_3d_68","recognition"]` | 로딩 모듈 (genderage 제외로 속도 향상) |
| `FACER_REQUIRE_POSE` | `True` | 3D 랜드마크 포즈 추출 활성화 |
| `FACER_DET_SIZE` | `320` | 검출 입력 해상도 (근거리: 320, 원거리: 640) |
| `FACER_DETECTION_THRESHOLD` | `0.6` | 얼굴 검출 신뢰도 하한선 |

### 정합성 품질

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `FACER_MIN_VALID_FRAMES` | `3` | 대표 임베딩 생성 최소 유효 프레임 수 |
| `FACER_MIN_QUALITY_SCORE` | `0.4` | 프레임 수용 최소 품질 점수 |
| `FACER_OUTLIER_COSINE_THRESHOLD` | `0.45` | 아웃라이어 제거 임계값 |
| `FACER_POSE_YAW_LIMIT` | `25.0` | 허용 yaw(좌우) 각도 한계 (도) |
| `FACER_POSE_PITCH_LIMIT` | `20.0` | 허용 pitch(상하) 각도 한계 (도) |

### 카메라 파이프라인

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `FACER_CAMERA_WARMUP_FRAMES` | `5` | 노출 안정화를 위해 버릴 초기 프레임 수 |
| `FACER_FRAME_INTERVAL_MS` | `200.0` | 유효 프레임 간 최소 간격 (ms) |

### 인증 임계값

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `FACER_COSINE_THRESHOLD_HIGH` | `0.40` | 동일인 판정 상단 임계값 |
| `FACER_COSINE_THRESHOLD_LOW` | `0.30` | 동일인 판정 하단 임계값 |

---

## 핵심 기능

### 1. L2 재정규화된 대표 임베딩

여러 프레임의 임베딩을 평균화할 때 결과 벡터가 단위 벡터(norm=1)를 유지하도록 L2 재정규화를 적용합니다. 코사인 유사도 = 내적이 성립하여 비교 연산이 수학적으로 정확합니다.

```python
# 품질 점수 기반 가중 평균 + L2 재정규화
result = generate_token_from_embeddings(embeddings, weights=quality_scores)
# np.linalg.norm(result) == 1.0 보장
```

### 2. Pose 필터링

`landmark_3d_68` 모델의 포즈 데이터로 측면 회전이 심한 프레임을 자동 배제합니다.

```
허용 범위: |yaw| ≤ 25°, |pitch| ≤ 20° (환경 변수로 조정 가능)
```

### 3. Quality-weighted 집계

```
quality_score = 0.6 × det_score + 0.4 × min(laplacian_var / 200, 1.0)
```

검출 신뢰도와 이미지 선명도를 결합한 품질 점수로 임베딩 가중 평균을 수행합니다.

### 4. Outlier 제거

임시 평균과의 코사인 유사도가 `outlier_cosine_threshold`(기본 0.45) 미만인 임베딩을 집계에서 제외합니다. 역광, 측면, 비정상 자세 프레임의 영향을 차단합니다.

### 5. ONNX JIT Warm-up

서버 기동 시(`lifespan`) 더미 이미지로 추론을 한 번 실행하여 ONNX 런타임의 JIT 컴파일을 선제 완료합니다. 첫 실제 요청에서 발생하는 레이턴시 스파이크가 제거됩니다.

### 6. 카메라 파이프라인

- **Warmup 스킵**: 카메라 개방 직후 자동 노출·화이트밸런스 수렴 전 초기 프레임(`camera_warmup_frames`)을 버립니다.
- **프레임 간격 제어**: `frame_interval_ms` 미만 간격으로 들어오는 유사 프레임을 건너뛰어 다양성 있는 샘플을 수집합니다.

---

## 사용 방법

### HTTP API

```bash
# 토큰(임베딩) 생성
curl -X POST http://localhost:23535/token/generate \
     -F "file=@face.jpg"

# 서비스 상태 확인
curl http://localhost:23535/health
# {"status":"healthy","model":"buffalo_l","acceleration":true,"det_size":320}
```

### Python SDK

```python
from src import generate_token_from_directory, generate_token_from_camera
import numpy as np

# 디렉토리 기반 등록
enroll_vec = generate_token_from_directory("./faces/user1", max_images=5)

# 카메라 기반 인증
verify_vec = generate_token_from_camera(max_frames=5)

# 동일인 판정 (L2 정규화 보장으로 cosine = dot product)
similarity = float(np.dot(enroll_vec, verify_vec))
print("동일인" if similarity >= 0.40 else "다른 사람")
```

### 임베딩 비교 유틸리티

```bash
python compare_embeddings.py user_a.npy user_b.npy
```

---

## 테스트 및 검증

```bash
# 전체 테스트 (모델 불필요)
python -m pytest tests/ -v --ignore=tests/test_robustness.py

# 모델 포함 전체 테스트 (InsightFace 모델 설치 필요)
python -m pytest tests/ -v

# 개별 실행
python -m pytest tests/test_token_generator.py -v   # L2 정규화, 가중 평균
python -m pytest tests/test_integration.py -v       # outlier 제거, min_valid_frames
python -m pytest tests/test_api.py -v               # HTTP 엔드포인트
python -m pytest tests/test_config.py -v            # 설정 필드
```

### 빠른 검증

```bash
python -c "from src.config import default_config; print('det_size:', default_config.det_size)"
# det_size: 320

python -c "
from src.token_generator import generate_token_from_embeddings
import numpy as np
v = np.random.rand(512).astype(np.float32); v /= np.linalg.norm(v)
r = generate_token_from_embeddings([v, v])
print('norm:', round(float(np.linalg.norm(r)), 8))
"
# norm: 1.0
```

---

## 배포 (Docker)

```bash
# 이미지 빌드
docker build -t facer-api .

# GPU 지원 컨테이너 실행
docker run -d -p 23535:23535 --gpus all --name facer facer-api

# CPU 전용
docker run -d -p 23535:23535 -e FACER_USE_GPU=false --name facer facer-api
```

---

## FAQ 및 보안

**Q: GPU가 없어도 작동하나요?**
A: 시스템이 자동으로 CPU 모드로 전환합니다. `FACER_USE_GPU=false` 명시를 권장합니다.

**Q: `min_valid_frames=3`인데 프레임이 2개밖에 안 나와요.**
A: 조명 개선, 정면 응시 유도, `FACER_MIN_QUALITY_SCORE` 또는 `FACER_POSE_YAW_LIMIT` 값을 완화하면 유효 프레임 수가 증가합니다.

**Q: 카메라 첫 프레임이 어둡게 나와요.**
A: `FACER_CAMERA_WARMUP_FRAMES`(기본 5)를 늘리면 자동 노출이 안정화된 이후 캡처를 시작합니다.

**Q: 개인정보 보안은?**
A: 카메라로 캡처한 원본 이미지는 디스크에 저장되지 않습니다. 임베딩은 메모리에서만 처리됩니다.

---

**Author**: 최진호
**Update Date**: 2026-02-27
**Version**: 2.3.0
**Status**: Production Ready
