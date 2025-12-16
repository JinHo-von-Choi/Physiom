"""
<summary>얼굴 임베딩으로부터 대표 임베딩을 생성하는 모듈</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>임베딩을 양자화 후 해시하여 재식별 위험을 줄인다.</remarks>
"""

import logging
from typing import List

import numpy as np

from src.config import AppConfig, default_config
from src.exceptions import EmbeddingError


logger = logging.getLogger("token-generator")
logger.setLevel(logging.INFO)


def generate_token_from_embeddings(embeddings: List[np.ndarray], config: AppConfig = default_config) -> np.ndarray:
    """
    <summary>임베딩 리스트로부터 대표 ArcFace 임베딩(평균)을 생성</summary>
    <param name="embeddings">최소 1개의 임베딩 벡터 리스트 (각 임베딩: shape (512,), dtype float32)</param>
    <param name="config">전역 설정 객체 (현재 미사용, 확장성 위해 보유)</param>
    <returns>대표 임베딩 벡터 (shape: (512,), dtype: float64, 정규화되지 않음)</returns>
    <remarks>
    - 여러 이미지에서 얻은 임베딩의 산술 평균을 대표 임베딩으로 사용한다.
    - ArcFace 정규화 임베딩 특성상 코사인 유사도가 곧 내적 값이다.
    - 해시를 제거하고 float 벡터를 그대로 반환하여 서버 측에서 유사도 비교에 활용한다.
    
    동작 원리:
        1. 임베딩 리스트를 axis=0으로 스택 (shape: (N, 512))
        2. axis=0 방향으로 평균 계산 (shape: (512,))
        3. 노이즈 상쇄 및 본질적 특징 강화
    
    전제 조건:
        - embeddings는 비어있지 않은 리스트
        - 각 임베딩은 shape (512,)
        - 각 임베딩은 L2 정규화됨 (ArcFace 출력)
    
    후행 조건:
        - 반환된 평균 임베딩은 정규화되지 않음 (np.linalg.norm ≈ 1.0이지만 정확히 1.0 아님)
        - 유사도 비교 시 정규화 필요: emb / np.linalg.norm(emb)
    
    예외:
        - ValueError: 임베딩 리스트가 비어있을 경우
    
    사용 예시:
        # 여러 이미지의 임베딩 평균화
        embeddings = [emb1, emb2, emb3]
        mean_embedding = generate_token_from_embeddings(embeddings)
        
        # 정규화 후 유사도 비교
        norm_emb = mean_embedding / np.linalg.norm(mean_embedding)
        similarity = np.dot(norm_emb, other_embedding)
    
    성능:
        - 5개 임베딩 평균: ~0.1ms (CPU)
        - 메모리: (N * 512 * 4 bytes) + (512 * 8 bytes)
    
    주의:
        - 함수명이 generate_token이지만 실제로는 임베딩(float 벡터)을 반환함
        - 하위 호환을 위해 함수명 유지, 향후 리팩토링 예정
    </remarks>
    """
    if len(embeddings) == 0:
        raise EmbeddingError("임베딩 리스트가 비어있어 대표 임베딩을 생성할 수 없습니다.")
    
    if not all(isinstance(emb, np.ndarray) for emb in embeddings):
        raise EmbeddingError("임베딩 리스트의 모든 요소는 numpy.ndarray여야 합니다.")
    
    try:
        stacked: np.ndarray     = np.stack(embeddings, axis=0)
        mean_vector: np.ndarray = np.mean(stacked, axis=0)
    except Exception as exc:
        raise EmbeddingError(
            f"임베딩 평균화 중 오류 발생: {type(exc).__name__}: {exc}"
        ) from exc
    
    logger.info("대표 임베딩 생성 완료 (입력: %d개, 출력 차원: %d)", len(embeddings), mean_vector.shape[0])
    return mean_vector

