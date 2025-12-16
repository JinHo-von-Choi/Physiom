# Face Embedding Service - 엔터프라이즈 얼굴 인증 토큰 생성 모듈

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/JinHo-von-Choi/face/workflows/CI/badge.svg)](https://github.com/JinHo-von-Choi/face/actions)

## 목차
1. [개요 및 목적](#개요-및-목적)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [기술 스택 및 선택 근거](#기술-스택-및-선택-근거)
4. [시스템 요구사항](#시스템-요구사항)
5. [설치 가이드](#설치-가이드)
6. [설정 및 커스터마이징](#설정-및-커스터마이징)
7. [사용 방법](#사용-방법)
8. [내부 동작 원리](#내부-동작-원리)
9. [임베딩 비교 및 로그인 검증](#임베딩-비교-및-로그인-검증)
10. [보안 및 프라이버시](#보안-및-프라이버시)
11. [에러 처리 및 트러블슈팅](#에러-처리-및-트러블슈팅)
12. [성능 최적화](#성능-최적화)
13. [FAQ](#faq)

---

## 개요 및 목적

본 프로젝트는 키오스크, POS, 출입 통제 시스템 등 얼굴 기반 사용자 인증이 필요한 엔터프라이즈 환경에서 사용할 수 있는 얼굴 임베딩(Embedding) 생성 전용 마이크로서비스입니다.

### 핵심 기능
- 카메라 또는 디렉토리 이미지에서 얼굴을 검출하고 ArcFace 512차원 정규화 임베딩 벡터를 생성
- 동일인의 여러 이미지를 평균화하여 통계적으로 일관된 대표 임베딩 반환
- 인증/로그인 로직은 포함하지 않으며, 순수하게 "얼굴 → 임베딩" 변환만 담당
- FastAPI 기반 HTTP 서버, Python 라이브러리 API, CLI 등 다양한 연동 방식 지원

### 유스케이스
- 사용자 등록(Enroll): 신규 사용자의 얼굴 사진 여러 장을 입력받아 대표 임베딩을 생성하고 데이터베이스에 저장
- 로그인(Verify): 카메라로 캡처한 얼굴 임베딩과 DB 내 등록된 임베딩을 코사인 유사도로 비교하여 본인 확인
- 중복 가입 방지: 신규 등록 시 기존 임베딩과 유사도를 비교해 이미 등록된 사용자인지 검증

### 철학
- Separation of Concerns: 얼굴 임베딩 생성과 인증 로직을 분리하여 각각 독립적으로 확장/테스트 가능
- Privacy by Design: 원본 이미지는 메모리에서만 처리, 임베딩은 역추적이 매우 어려운 고차원 벡터이므로 프라이버시 보호
- Production-Ready: 예외 처리, 로깅, 설정 주입, GPU/CPU 자동 폴백 등 프로덕션 환경 고려 설계

---

## 시스템 아키텍처

### 디렉토리 구조
```
face/
├── main.py                    # CLI 엔트리포인트
├── compare_embeddings.py      # 임베딩 유사도 비교 유틸리티
├── README.md
├── Requirements.md
├── src/
│   ├── __init__.py           # 고수준 API (generate_token_from_camera/directory)
│   ├── config.py             # 전역 설정 (AppConfig dataclass)
│   ├── face_encoder.py       # InsightFace 모델 로딩, 얼굴 검출/임베딩 추출
│   ├── camera_capture.py     # 카메라 프레임 캡처 (얼굴 1개 필터링)
│   ├── image_loader.py       # 디렉토리 이미지 로딩 (jpg/jpeg/png)
│   ├── token_generator.py    # 임베딩 평균화 및 대표 임베딩 생성
│   ├── api_http.py           # FastAPI 서버 및 엔드포인트 정의
│   └── cli.py                # argparse 기반 CLI 파서
└── .venv/                     # Python 가상환경 (권장)
```

### 모듈 간 의존성 흐름
```
main.py
  └── src.cli.main()
       ├── camera → src.__init__.generate_token_from_camera()
       ├── directory → src.__init__.generate_token_from_directory()
       └── server → src.api_http.run_server()

src.__init__.generate_token_from_camera()
  ├── src.camera_capture.capture_from_camera()
  │    └── src.face_encoder.FaceEncoder.detect_faces()
  ├── src.face_encoder.extract_face_embeddings()
  │    └── src.face_encoder.FaceEncoder.extract_embedding()
  └── src.token_generator.generate_embedding_from_embeddings()

src.__init__.generate_token_from_directory()
  ├── src.image_loader.load_images_from_directory()
  ├── src.face_encoder.extract_face_embeddings()
  └── src.token_generator.generate_embedding_from_embeddings()

src.api_http.create_app()
  ├── POST /token/camera → generate_token_from_camera()
  └── POST /token/directory → generate_token_from_directory()
```

### 데이터 플로우
1. 입력: 카메라 프레임 또는 디렉토리 이미지 파일
2. 전처리: OpenCV로 BGR 포맷 로딩, 얼굴 검출 (InsightFace det_10g 모델)
3. 필터링: 얼굴이 정확히 1개 검출된 이미지만 유지 (0개/다중 얼굴은 스킵)
4. 임베딩 추출: ArcFace w600k_r50 모델로 512차원 정규화 벡터 생성
5. 평균화: 여러 이미지의 임베딩을 축별 평균 (np.mean(axis=0))
6. 출력: 512 float 리스트 (JSON) 또는 np.ndarray

---

## 기술 스택 및 선택 근거

### 핵심 라이브러리
- **InsightFace (ArcFace)**: 
  - SOTA 얼굴 인식 정확도 (LFW 99.8%+), 조명·각도·표정 변화에 강건
  - 정규화된 임베딩 제공으로 코사인 유사도 = 내적 계산 가능
  - ONNX 런타임으로 GPU 가속 지원 (CUDA/DirectML)
  - buffalo_l 모델: 속도와 정확도의 균형

- **onnxruntime-gpu / onnxruntime-directml**:
  - GPU 가속 추론 (CUDA/DirectML), CPU 폴백 자동 지원
  - 크로스 플랫폼 (Windows/Linux), 추가 CUDA 설치 없이 pip로 배포 가능

- **OpenCV (cv2)**:
  - 카메라 캡처 (VideoCapture), 이미지 로딩/전처리
  - 경량, 안정적, 산업 표준

- **FastAPI**:
  - 고성능 비동기 웹 프레임워크 (Starlette + Pydantic)
  - 자동 OpenAPI 문서 생성 (/docs)
  - 타입 안전성, 빠른 개발 속도

- **Uvicorn**:
  - ASGI 서버, 프로덕션 배포 시 gunicorn과 함께 사용 가능

### 대안 라이브러리 비교
| 라이브러리         | 장점                          | 단점                          | 선택 이유               |
|--------------------|-------------------------------|-------------------------------|-------------------------|
| InsightFace        | SOTA 정확도, GPU 가속         | ONNX 의존성                   | **선택** (정확도 우선)  |
| face_recognition   | 간단한 API                    | CPU 전용, 느림, dlib 의존성   | 제외 (성능 부족)        |
| DeepFace           | 다양한 모델 래퍼              | 무겁고 느림, 엔터프라이즈 부적합 | 제외 (오버헤드)         |

---

## 시스템 요구사항

### 하드웨어
- **CPU**: Intel Core i5 이상 또는 AMD Ryzen 5 이상 (AVX2 지원 권장)
- **RAM**: 최소 8GB (16GB 권장, 모델 로딩 시 ~2GB 사용)
- **GPU** (선택):
  - NVIDIA: GTX 1060 6GB 이상 (CUDA 11.x/12.x 지원), RTX 시리즈 권장
  - AMD: RX 5700 이상 (DirectML 지원, 호환성 제한 있음)
- **저장공간**: 최소 2GB (모델 다운로드 ~500MB)

### 소프트웨어
- **OS**: Windows 10 (21H2 이상) 또는 Windows 11
- **Python**: 3.11 (3.9~3.12 호환 가능하나 3.11 권장)
- **드라이버**:
  - NVIDIA GPU: 최신 Game Ready/Studio Driver (516.xx 이상)
  - AMD GPU: 최신 Adrenalin Driver (22.xx 이상) + DirectML 지원

### 네트워크
- 인터넷 연결 (모델 자동 다운로드 시 필요, 최초 1회)
- 방화벽: 포트 23535 (HTTP 서버 모드 사용 시)

---

## 설치 가이드

### 1. Python 환경 준비
```powershell
# Python 3.11 설치 확인
python --version  # Python 3.11.x

# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

# pip 업그레이드
python -m pip install --upgrade pip
```

### 2. 의존성 설치

#### NVIDIA GPU 사용 (권장)
```powershell
pip install insightface onnxruntime-gpu opencv-python fastapi uvicorn[standard] pydantic numpy
```

#### AMD GPU (DirectML, 실험적)
```powershell
pip install insightface onnxruntime-directml opencv-python fastapi uvicorn[standard] pydantic numpy
```
- 주의: InsightFace 공식 지원 없음, DmlExecutionProvider 호환성 불확실
- 실패 시 CPU 모드로 폴백됨

#### CPU 전용
```powershell
pip install insightface onnxruntime opencv-python fastapi uvicorn[standard] pydantic numpy
```

### 3. 모델 다운로드 (자동)
- 최초 실행 시 InsightFace가 `~/.insightface/models/buffalo_l/` 경로에 모델 자동 다운로드 (~500MB)
- 수동 다운로드: [InsightFace 모델 저장소](https://github.com/deepinsight/insightface/tree/master/model_zoo)

### 4. 설치 검증
```powershell
python -c "from src.face_encoder import FaceEncoder; print('OK')"
```

---

## 설정 및 커스터마이징

### AppConfig 파라미터 (`src/config.py`)
```python
@dataclass
class AppConfig:
    camera_index: int          = 0        # 기본 카메라 인덱스 (0=첫 번째)
    max_frames: int            = 5        # 카메라 캡처 최대 프레임 수
    max_images: int            = 5        # 디렉토리 읽기 최대 이미지 수
    use_gpu: bool              = True     # GPU 가속 활성화 (CUDA)
    use_directml: bool         = False    # DirectML 사용 (AMD GPU)
    model_name: str            = "buffalo_l"  # InsightFace 모델명
    detection_threshold: float = 0.6      # 얼굴 검출 신뢰도 임계값
    server_host: str           = "127.0.0.1"
    server_port: int           = 23535
```

### 설정 오버라이드
#### 방법 1: `src/config.py` 직접 수정
```python
default_config = AppConfig(
    camera_index=1,      # 두 번째 카메라 사용
    max_frames=10,       # 더 많은 프레임 캡처
    use_gpu=False        # CPU 모드 강제
)
```

#### 방법 2: 런타임 주입
```python
from src.config import AppConfig
from src import generate_token_from_camera

custom_config = AppConfig(camera_index=2, max_frames=3)
embedding = generate_token_from_camera(max_frames=3, config=custom_config)
```

### 모델 변경
- `model_name` 변경 가능 모델: `buffalo_l`, `buffalo_m`, `buffalo_s` (작을수록 빠르지만 정확도 하락)
- 예: `model_name="buffalo_s"` (속도 우선)

### GPU 설정
- **NVIDIA CUDA**: `use_gpu=True`, `use_directml=False` (기본)
- **AMD DirectML**: `use_gpu=False`, `use_directml=True`
- **CPU 전용**: `use_gpu=False`, `use_directml=False`

---

## 사용 방법

### 1. CLI (명령행 인터페이스)

#### 카메라에서 임베딩 생성
```powershell
python main.py camera --max-frames 5
```
**출력 예시**:
```
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
find model: C:\Users\user/.insightface\models\buffalo_l\det_10g.onnx
set det-size: (640, 640)
[0.0149, -0.0146, 0.0526, ..., -0.0224]  # 512 floats
```

#### 디렉토리 이미지에서 임베딩 생성
```powershell
python main.py directory --path "C:\faces\user1" --max-images 5
```
- 지원 확장자: `.jpg`, `.jpeg`, `.png`
- 얼굴이 1개만 검출된 이미지만 사용

#### HTTP 서버 실행
```powershell
python main.py server --host 0.0.0.0 --port 23535
```
- 외부 접근: `--host 0.0.0.0`
- 로컬 전용: `--host 127.0.0.1` (기본)

### 2. Python API

#### 기본 사용
```python
from src import generate_token_from_camera, generate_token_from_directory

# 카메라 캡처
embedding_cam = generate_token_from_camera(max_frames=5)
print(f"Camera embedding: {len(embedding_cam)} dims")  # 512 dims

# 디렉토리 이미지
embedding_dir = generate_token_from_directory(
    dir_path="C:/faces/user1",
    max_images=5
)
print(f"Directory embedding: {embedding_dir[:5]}")  # 처음 5개 값 출력
```

#### 커스텀 설정 사용
```python
from src.config import AppConfig
from src import generate_token_from_directory

config = AppConfig(max_images=10, use_gpu=False)
embedding = generate_token_from_directory(
    dir_path="C:/faces/user1",
    max_images=10,
    config=config
)
```

#### 저수준 API 사용 (고급)
```python
from src.face_encoder import FaceEncoder, extract_face_embeddings
from src.image_loader import load_images_from_directory
from src.token_generator import generate_embedding_from_embeddings
from src.config import AppConfig

config = AppConfig()
encoder = FaceEncoder(config)

# 이미지 로딩
images = load_images_from_directory("C:/faces/user1", max_images=5, config=config)

# 임베딩 추출
embeddings = extract_face_embeddings(images, encoder=encoder)

# 대표 임베딩 생성
final_embedding = generate_embedding_from_embeddings(embeddings)
print(f"Shape: {final_embedding.shape}")  # (512,)
```

### 3. HTTP API

#### 서버 실행
```powershell
python main.py server --host 0.0.0.0 --port 23535
```

#### 엔드포인트

##### POST /token/camera
**요청**:
```bash
curl -X POST http://127.0.0.1:23535/token/camera \
     -H "Content-Type: application/json" \
     -d "{\"maxFrames\":5}"
```

**응답** (성공):
```json
{
  "success": true,
  "embedding": [0.0149, -0.0146, 0.0526, ..., -0.0224],
  "error": null
}
```

**응답** (실패):
```json
{
  "success": false,
  "embedding": null,
  "error": "카메라를 열 수 없습니다."
}
```

##### POST /token/directory
**요청**:
```bash
curl -X POST http://127.0.0.1:23535/token/directory \
     -H "Content-Type: application/json" \
     -d "{\"dirPath\":\"C:/faces/user1\",\"maxImages\":5}"
```

**응답**: `/token/camera`와 동일 형식

#### OpenAPI 문서
- Swagger UI: `http://127.0.0.1:23535/docs`
- ReDoc: `http://127.0.0.1:23535/redoc`

### 4. Unity(C#) 연동 예시

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using System.Text;

[System.Serializable]
public class EmbeddingRequest
{
    public int maxFrames;
}

[System.Serializable]
public class EmbeddingResponse
{
    public bool success;
    public List<float> embedding;
    public string error;
}

public class FaceAuthClient : MonoBehaviour
{
    private const string API_URL = "http://127.0.0.1:23535/token/camera";

    public IEnumerator GetEmbedding(System.Action<List<float>> onSuccess, System.Action<string> onError)
    {
        var request = new EmbeddingRequest { maxFrames = 5 };
        string json = JsonUtility.ToJson(request);

        using (var www = new UnityWebRequest(API_URL, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<EmbeddingResponse>(www.downloadHandler.text);
                if (response.success)
                {
                    onSuccess?.Invoke(response.embedding);
                }
                else
                {
                    onError?.Invoke(response.error);
                }
            }
            else
            {
                onError?.Invoke(www.error);
            }
        }
    }
}
```

---

## 내부 동작 원리

### 1. 얼굴 검출 파이프라인
```
입력 이미지 (BGR, HxWx3)
  ↓
InsightFace det_10g 모델 (SCRFD 기반)
  ↓
검출 결과: [face1, face2, ...] (bounding box, landmarks, score)
  ↓
필터링: 얼굴이 정확히 1개인 경우만 통과
  ↓
검출된 얼굴 ROI
```

### 2. 임베딩 추출 파이프라인
```
얼굴 ROI
  ↓
정렬 (5-point landmarks 기반 affine transform)
  ↓
전처리 (112x112 리사이징, 정규화)
  ↓
ArcFace w600k_r50 모델 (ResNet-50 백본)
  ↓
512차원 임베딩 벡터
  ↓
L2 정규화 (norm = 1.0)
  ↓
정규화된 임베딩 출력
```

### 3. 다중 이미지 평균화
```
임베딩 리스트: [emb1, emb2, emb3, ...]
  ↓
np.stack(axis=0) → shape: (N, 512)
  ↓
np.mean(axis=0) → shape: (512,)
  ↓
대표 임베딩 반환
```

### 4. 왜 평균화가 필요한가?
- 단일 이미지는 조명, 각도, 표정 등 노이즈 포함
- 여러 이미지의 평균은 노이즈를 상쇄하고 본질적 특징만 강화
- 실험적으로 3~5장 평균 시 단일 이미지 대비 인식 정확도 2~5% 향상

---

## 임베딩 비교 및 로그인 검증

### 코사인 유사도 계산
ArcFace 임베딩은 L2 정규화되어 있으므로:
```python
cosine_similarity = np.dot(embedding1, embedding2)
# 또는
cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
```

### 임계값 설정 가이드
| 유사도 범위 | 판정               | 권장 액션                     |
|-------------|--------------------|-------------------------------|
| 0.50+       | 동일인 (매우 높음) | 로그인 허용                   |
| 0.40~0.50   | 동일인 (높음)      | 로그인 허용                   |
| 0.30~0.40   | 경계 구간          | 재캡처 요청 또는 추가 인증    |
| 0.20~0.30   | 타인 가능성 높음   | 로그인 거부                   |
| 0.20 미만   | 타인 (매우 낮음)   | 로그인 거부                   |

- **보수적 설정**: threshold = 0.45 (False Positive 최소화)
- **균형적 설정**: threshold = 0.35~0.40 (일반적 권장)
- **공격적 설정**: threshold = 0.30 (False Negative 최소화, 보안 위험)

### 로그인 검증 예시 코드
```python
import numpy as np
from src import generate_token_from_camera

# 1. 사용자 등록 시 (Enroll)
enrollment_embedding = generate_token_from_camera(max_frames=5)
# DB에 저장: user_id, enrollment_embedding

# 2. 로그인 시 (Verify)
login_embedding = generate_token_from_camera(max_frames=3)

# 3. 유사도 계산
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

similarity = cosine_similarity(
    np.array(enrollment_embedding, dtype=np.float32),
    np.array(login_embedding, dtype=np.float32)
)

# 4. 판정
THRESHOLD = 0.40
if similarity >= THRESHOLD:
    print("로그인 성공")
else:
    print(f"로그인 실패 (유사도: {similarity:.3f})")
```

### 유틸리티 스크립트 사용
```powershell
# 두 임베딩 파일 비교
python compare_embeddings.py emb1.txt emb2.txt
# 출력: cosine_similarity=0.7523, 동일인 가능성 높음
```

---

## 보안 및 프라이버시

### 데이터 최소화
- **카메라 캡처**: 메모리에서만 처리, 디스크 미저장 (단, 디버그 모드에서는 옵션으로 저장 가능)
- **디렉토리 이미지**: 읽기 전용, 원본 복제/이동/삭제 없음
- **임베딩**: 원본 이미지 복원 불가능한 고차원 특징 벡터 (512 floats)

### 전송 보안
- **HTTP 모드**: 반드시 HTTPS(TLS) 사용 권장 (프로덕션 환경)
- **자체 서명 인증서** 또는 Let's Encrypt 인증서 적용
- **API Key 인증**: FastAPI middleware에 Bearer Token 추가 (별도 구현 필요)

### 저장 보안
- **임베딩 암호화**: 데이터베이스 저장 시 AES-256-GCM으로 암호화 권장
- **키 관리**: Azure Key Vault, AWS KMS, 또는 Windows DPAPI 사용
- **접근 제어**: DB 계정 최소 권한 원칙 (Least Privilege)

### GDPR/개인정보보호법 준수
- 임베딩은 생체정보 → 동의 필요, 명시적 목적 제한
- 사용자 삭제 요청 시 임베딩도 즉시 삭제
- 로그에 임베딩 값 출력 금지 (민감 정보)

---

## 에러 처리 및 트러블슈팅

### 자주 발생하는 에러

#### 1. `카메라를 열 수 없습니다.`
**원인**:
- 카메라 인덱스 불일치 (다른 프로그램이 점유 중)
- 카메라 드라이버 미설치
- 권한 부족 (Windows 카메라 프라이버시 설정)

**해결**:
```python
# 카메라 인덱스 변경
from src.config import AppConfig
config = AppConfig(camera_index=1)  # 또는 2, 3...
```
- Windows 설정 → 개인정보 → 카메라 → 앱 액세스 허용

#### 2. `얼굴 임베딩을 추출하지 못했습니다.`
**원인**:
- 모든 이미지에서 얼굴 미검출 (각도/조명 불량)
- 다중 얼굴 검출로 필터링됨

**해결**:
- 더 많은 이미지 제공 (`max_images` 증가)
- 정면 얼굴 사진 사용
- `detection_threshold` 하향 조정 (예: 0.6 → 0.4)

#### 3. CUDA DLL 로딩 실패 (`cublasLt64_12.dll` 등)
**원인**:
- CUDA 런타임 미설치 또는 버전 불일치
- PATH 환경변수 미설정

**해결**:
```powershell
# CPU 모드로 전환
pip uninstall onnxruntime-gpu
pip install onnxruntime
```
- 또는 CUDA 12.x 설치: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

#### 4. DirectML 사용 불가
**원인**:
- InsightFace가 DirectML 공식 미지원
- onnxruntime-directml 호환성 문제

**해결**:
```python
# config에서 DirectML 비활성화
config = AppConfig(use_gpu=False, use_directml=False)
```

### 로그 레벨 조정
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 상세 로그 출력
```

---

## 성능 최적화

### GPU 가속 활성화
- **NVIDIA**: onnxruntime-gpu 설치, CUDA 드라이버 최신화
- **AMD**: onnxruntime-directml 시도 (제한적)
- 성능 비교: GPU (RTX 3060) ~50ms vs CPU (i7) ~200ms (단일 이미지 기준)

### 배치 처리
- 현재 구현은 순차 처리, 대량 이미지 처리 시 배치 추론으로 개선 가능
- InsightFace는 배치 입력 지원 (`get(images)` → 내부 배치화)

### 모델 경량화
- `buffalo_s` 사용 (정확도 약간 하락, 속도 2배 향상)
- 양자화: ONNX INT8 양자화 (정확도 0.5~1% 하락, 속도 30% 향상)

### 프로세스 재사용
- HTTP 서버 모드 사용 → 모델 1회만 로딩, 다중 요청 처리
- CLI 모드는 매번 모델 재로딩 (~2초 오버헤드)

### 스레드 안전성 (Thread Safety)
**문제**: InsightFace의 onnxruntime 세션은 스레드 안전하지 않아 다중 스레드 환경에서 경쟁 조건 발생 가능

**해결**: ThreadSafeEncoderManager 사용
- `threading.local`과 `contextvars`를 함께 사용
- 각 스레드/비동기 태스크는 독립적인 FaceEncoder 인스턴스 사용
- 자동으로 스레드별 인코더 생성 및 캐싱

**지원 환경**:
- ✅ FastAPI async def (비동기 엔드포인트)
- ✅ FastAPI def (동기 엔드포인트)
- ✅ gunicorn -w N (다중 워커)
- ✅ uvicorn --workers N (다중 워커)
- ✅ threading.Thread (수동 멀티스레딩)
- ❌ multiprocessing (프로세스 간 공유 불가, 각 프로세스별 독립 생성)

**메모리 고려사항**:
- 스레드당 ~2GB (모델 메모리)
- 워커 수 제한 권장: `gunicorn -w 4` (4개 워커)
- 메모리 부족 시 워커 수 감소 또는 CPU 모드 전환

**사용 예시**:
```python
# 자동 처리 (권장)
from src import generate_token_from_camera
embedding = generate_token_from_camera(max_frames=5)  # 스레드 안전

# 수동 관리 (고급)
from src import encoder_manager
encoder = encoder_manager.get_encoder(config)  # 현재 스레드 전용 인코더
encoder_manager.clear_encoder()  # 명시적 정리 (일반적으로 불필요)
```

---

## FAQ

### Q1. 동일인의 여러 사진에서 임베딩이 다릅니다. 정상인가요?
**A**: 정상입니다. 조명, 각도, 표정 변화로 임베딩이 약간씩 달라지나, 코사인 유사도는 0.7~0.9 범위로 높게 유지됩니다. 평균화를 통해 일관성을 높일 수 있습니다.

### Q2. 임베딩을 DB에 저장할 때 어떤 자료형을 써야 하나요?
**A**: 
- PostgreSQL: `VECTOR(512)` (pgvector 확장)
- MySQL: `JSON` 또는 `BLOB` (직렬화 후)
- MongoDB: `array` 필드
- 권장: pgvector + 코사인 유사도 인덱스 (빠른 검색)

### Q3. 얼굴 사진 대신 영상(동영상)을 사용할 수 있나요?
**A**: 가능합니다. 영상에서 프레임을 추출해 리스트로 전달하면 됩니다.
```python
import cv2
video = cv2.VideoCapture("video.mp4")
frames = []
while len(frames) < 10:
    ret, frame = video.read()
    if not ret: break
    frames.append(frame)
video.release()
# frames를 extract_face_embeddings()에 전달
```

### Q4. 마스크 착용 시 인식이 안 됩니다.
**A**: ArcFace는 전체 얼굴 특징을 사용하므로 마스크 착용 시 정확도 급락합니다. 마스크 인식 전용 모델(예: FaceMask-Net) 사용을 권장합니다.

### Q5. 쌍둥이를 구분할 수 있나요?
**A**: 일란성 쌍둥이는 유전적으로 매우 유사해 코사인 유사도 0.5~0.7로 높게 나타납니다. 완벽 구분은 어려우며, 추가 생체인증(지문/홍채) 병행이 필요합니다.

### Q6. 프로덕션 배포 시 권장 구성은?
**A**:
- **아키텍처**: FastAPI 서버 + Nginx 리버스 프록시 + HTTPS
- **프로세스 관리**: systemd (Linux) 또는 Windows Service
- **로드 밸런싱**: Nginx upstream + 다중 uvicorn 워커
- **모니터링**: Prometheus + Grafana (응답 시간, 에러율)
- **로깅**: 구조화 로그 (JSON) + ELK Stack

### Q7. 라이센스는?
**A**: 본 프로젝트는 MIT 라이센스입니다. InsightFace는 별도 라이센스(Apache 2.0)를 따르므로 상업적 사용 시 확인 필요.

---

## 프로덕션 배포 체크리스트
- [ ] HTTPS/TLS 적용
- [ ] API Key 인증 추가
- [ ] 임베딩 DB 암호화
- [ ] 로그에서 민감 정보 제거
- [ ] 에러 로깅 및 모니터링 설정
- [ ] 부하 테스트 (예: Locust)
- [ ] 백업 및 재해 복구 계획
- [ ] GDPR/개인정보보호법 준수 확인
- [ ] 사용자 동의 문구 추가

---

**Author**: 최진호  
**Date**: 2025-12-16  
**Version**: 1.0.0  
**Contact**: jinho.von.choi@nerdvana.kr
