# 기여 가이드 (Contributing Guide)

프로젝트에 기여해 주셔서 감사합니다!

## 기여 방법

### 1. 이슈 제보
- 버그, 기능 요청, 문서 개선 등 모든 이슈를 환영합니다
- [Issues](https://github.com/JinHo-von-Choi/face/issues)에서 새 이슈 생성
- 가능한 자세히 설명해 주세요 (환경, 재현 방법, 스크린샷 등)

### 2. Pull Request
1. Fork 및 Clone
```bash
git clone https://github.com/JinHo-von-Choi/face.git
cd face
```

2. 브랜치 생성
```bash
git checkout -b feature/your-feature-name
# 또는
git checkout -b fix/your-bug-fix
```

3. 환경 설정
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

4. 변경 사항 작성
- 코드 스타일 준수 (주석, 타입 힌트, 네이밍)
- 엔터프라이즈급 주석 작성
- 가능한 테스트 코드 추가

5. 커밋
```bash
git add .
git commit -m "feat: 새 기능 설명" # 또는 fix: / docs: / refactor: 등
```

6. Push 및 PR
```bash
git push origin feature/your-feature-name
```
- GitHub에서 Pull Request 생성
- 변경 사항 상세 설명

## 코드 스타일

### 변수 정렬
```python
camera_index: int          = 0
max_frames: int            = 5
detection_threshold: float = 0.6
```

### 주석 형식
```python
"""
<summary>함수 요약</summary>
<param name="param">파라미터 설명</param>
<returns>반환값 설명</returns>
<remarks>
상세 설명, 전제 조건, 후행 조건, 사용 예시 등
</remarks>
"""
```

### 타입 힌트
- 모든 함수 파라미터와 반환값에 타입 힌트 명시
- `from typing import List, Optional, Dict` 등 활용

## 테스트
- 단위 테스트 작성 권장 (pytest)
- 기존 기능 영향 확인

## 라이센스
기여하신 코드는 MIT 라이센스로 배포됩니다.

## 질문
궁금한 점은 이슈로 남겨주세요.

감사합니다!

