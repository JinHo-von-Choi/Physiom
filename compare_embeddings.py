"""
<summary>텍스트 파일 두 개에 저장된 임베딩 배열을 읽어 코사인 유사도를 계산하는 스크립트</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>ArcFace 정규화 임베딩 기준으로 코사인 유사도를 출력한다.</remarks>
"""

import argparse
import ast
import sys
from typing import List

import numpy as np


def read_vector_from_file(path: str) -> np.ndarray:
    """
    <summary>텍스트 파일에서 배열 문자열을 읽어 numpy 벡터로 변환한다.</summary>
    <param name="path">배열이 저장된 텍스트 파일 절대/상대 경로</param>
    <returns>float32 numpy 벡터 (shape: (N,))</returns>
    <remarks>
    - 배열 표기는 파이썬 리스트(예: [0.1, 0.2, ...]) 형태를 가정한다.
    - ast.literal_eval로 파싱하여 임의 코드 실행을 방지한다 (보안).
    - 잘못된 형식일 경우 ValueError를 발생시켜 호출자가 즉시 알 수 있게 한다.
    
    파일 형식:
        - Python list 표기: [0.014, -0.015, 0.052, ...]
        - 줄바꿈 있어도 무방
        - UTF-8 인코딩 사용
    
    전제 조건:
        - path는 읽기 가능한 텍스트 파일
        - 파일 내용은 유효한 Python 리스트 리터럴
    
    후행 조건:
        - 반환된 벡터는 dtype=float32, 1차원
    
    예외:
        - FileNotFoundError: 파일 미존재
        - ValueError: 파싱 실패 (잘못된 형식)
        - PermissionError: 읽기 권한 부족
    
    사용 예시:
        # 파일 내용: [0.1, 0.2, 0.3]
        vec = read_vector_from_file("embedding.txt")
        print(vec.shape)  # (3,)
    
    보안:
        - ast.literal_eval 사용으로 코드 인젝션 방지
        - eval() 사용 금지 (취약)
    </remarks>
    """
    with open(path, "r", encoding="utf-8") as file:
        content: str = file.read().strip()
    try:
        data: List[float] = ast.literal_eval(content)
    except Exception as exc:
        raise ValueError(f"파일을 배열로 파싱할 수 없습니다: {path}") from exc
    vector: np.ndarray = np.array(data, dtype=np.float32)
    return vector


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    <summary>두 벡터의 코사인 유사도를 계산한다.</summary>
    <param name="vec_a">첫 번째 임베딩 벡터 (shape: (N,))</param>
    <param name="vec_b">두 번째 임베딩 벡터 (shape: (N,))</param>
    <returns>코사인 유사도 값 (-1.0 ~ 1.0 범위)</returns>
    <remarks>
    - ArcFace 정규화 임베딩의 경우 내적이 곧 코사인 유사도다.
    - 노름이 0인 입력은 잘못된 데이터로 간주하고 ValueError를 발생시킨다.
    
    공식:
        cosine_similarity = (A · B) / (||A|| * ||B||)
        - A · B: 내적 (dot product)
        - ||A||: A의 L2 노름 (벡터 길이)
    
    전제 조건:
        - vec_a와 vec_b는 같은 shape
        - 노름이 0이 아님 (제로 벡터 불가)
    
    후행 조건:
        - 반환값은 -1.0 ~ 1.0 범위
        - 1.0: 완전 동일 방향
        - 0.0: 직교
        - -1.0: 정반대 방향
    
    예외:
        - ValueError: 노름이 0인 벡터 입력
    
    사용 예시:
        emb1 = np.array([0.1, 0.2, 0.3])
        emb2 = np.array([0.1, 0.2, 0.3])
        sim = cosine_similarity(emb1, emb2)  # 1.0 (동일)
        
        emb3 = np.array([-0.1, -0.2, -0.3])
        sim2 = cosine_similarity(emb1, emb3)  # -1.0 (정반대)
    
    ArcFace 임베딩 특성:
        - 이미 L2 정규화됨 (||emb|| = 1.0)
        - 내적만으로도 유사도 계산 가능: np.dot(vec_a, vec_b)
        - 본 함수는 범용성을 위해 정규화되지 않은 벡터도 지원
    
    성능:
        - O(N) 시간 복잡도 (N: 벡터 차원)
        - 512차원 기준 ~0.01ms (numpy 최적화)
    </remarks>
    """
    norm_a: float = float(np.linalg.norm(vec_a))
    norm_b: float = float(np.linalg.norm(vec_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("벡터 노름이 0입니다.")
    similarity: float = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return similarity


def main() -> int:
    """
    <summary>CLI 진입점</summary>
    <returns>종료 코드 (0=성공, 1=실패)</returns>
    <remarks>
    - 두 텍스트 파일에서 임베딩을 읽어 코사인 유사도를 계산한다.
    - 길이가 다르면 실패로 간주하고 종료 코드 1을 반환한다.
    - 판정 메시지는 경험적 임계값(0.40/0.30)으로 안내한다.
    
    종료 코드:
        0: 정상 계산 완료
        1: 오류 발생 (파일 미존재, 형식 오류, 길이 불일치 등)
    
    출력:
        stdout: cosine_similarity=값, 판정 메시지
        stderr: 오류 메시지 (실패 시)
    
    사용 예시:
        # 직접 실행
        python compare_embeddings.py emb1.txt emb2.txt
        
        # 출력 예시 (동일인)
        cosine_similarity=0.752314
        판정: 동일인 가능성 높음
        
        # 출력 예시 (타인)
        cosine_similarity=0.123456
        판정: 타인 가능성 높음
    
    판정 임계값:
        - 0.40 이상: 동일인 가능성 높음
        - 0.30~0.40: 경계 구간 (추가 캡처 권장)
        - 0.30 미만: 타인 가능성 높음
        * 임계값은 환경에 따라 조정 필요
    
    파일 형식:
        - Python list 표기: [0.014, -0.015, ...]
        - UTF-8 인코딩
        - 512 floats (ArcFace 임베딩 기준)
    
    에러 처리:
        - 파일 미존재: FileNotFoundError → stderr 출력, 종료 코드 1
        - 형식 오류: ValueError → stderr 출력, 종료 코드 1
        - 길이 불일치: stderr 출력, 종료 코드 1
    
    프로덕션 활용:
        - 로그인 검증 시 임계값 비교용
        - 중복 가입 방지 시 유사도 체크
        - 테스트/디버깅용 빠른 검증
    </remarks>
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="임베딩 텍스트 파일 코사인 유사도 계산")
    parser.add_argument("file1", type=str, help="첫 번째 임베딩 텍스트 파일 경로")
    parser.add_argument("file2", type=str, help="두 번째 임베딩 텍스트 파일 경로")
    args = parser.parse_args()

    vec1: np.ndarray = read_vector_from_file(args.file1)
    vec2: np.ndarray = read_vector_from_file(args.file2)

    if vec1.shape != vec2.shape:
        print(f"벡터 길이가 다릅니다: {vec1.shape} vs {vec2.shape}", file=sys.stderr)
        return 1

    cosine: float = cosine_similarity(vec1, vec2)
    print(f"cosine_similarity={cosine:.6f}")

    if cosine >= 0.40:
        print("판정: 동일인 가능성 높음")
    elif cosine >= 0.30:
        print("판정: 경계 구간(추가 캡처 권장)")
    else:
        print("판정: 타인 가능성 높음")

    return 0


if __name__ == "__main__":
    sys.exit(main())

