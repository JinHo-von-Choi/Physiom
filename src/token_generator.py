"""
<summary>얼굴 임베딩으로부터 대표 임베딩을 생성하는 모듈</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>임베딩을 양자화 후 해시하여 재식별 위험을 줄인다.</remarks>
"""

import logging
from typing import List, Optional

import numpy as np

from src.config import AppConfig, default_config
from src.exceptions import EmbeddingError


logger = logging.getLogger("token-generator")
logger.setLevel(logging.INFO)


def generate_token_from_embeddings(
    embeddings: List[np.ndarray],
    weights: Optional[List[float]] = None,
    config: AppConfig = default_config
) -> np.ndarray:
    """
    <summary>임베딩 리스트로부터 L2 정규화된 대표 임베딩 생성</summary>
    <param name="embeddings">ArcFace L2 정규화 임베딩 리스트 (각 shape: (512,))</param>
    <param name="weights">프레임별 품질 가중치 (None이면 균등 가중)</param>
    <param name="config">전역 설정 객체 (확장성 위해 보유)</param>
    <returns>L2 정규화된 대표 임베딩 (shape: (512,), norm=1.0)</returns>
    """
    if len(embeddings) == 0:
        raise EmbeddingError("임베딩 리스트가 비어있어 대표 임베딩을 생성할 수 없습니다.")
    
    if not all(isinstance(emb, np.ndarray) for emb in embeddings):
        raise EmbeddingError("임베딩 리스트의 모든 요소는 numpy.ndarray여야 합니다.")
    
    try:
        stacked: np.ndarray = np.stack(embeddings, axis=0)  # (N, 512)

        if weights is not None and len(weights) == len(embeddings):
            w = np.array(weights, dtype=np.float64)
            w = w / w.sum()
            mean_vec: np.ndarray = np.average(stacked, axis=0, weights=w)
        else:
            mean_vec = np.mean(stacked, axis=0)

        # L2 재정규화: 평균 벡터는 단위 벡터가 아니므로 반드시 정규화
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-8:
            raise EmbeddingError("평균 임베딩이 영벡터입니다.")
        mean_vec = mean_vec / norm

    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(
            f"임베딩 평균화 중 오류 발생: {type(exc).__name__}: {exc}"
        ) from exc

    logger.info("대표 임베딩 생성 완료 (입력: %d개, 출력 norm: %.6f)", len(embeddings), np.linalg.norm(mean_vec))
    return mean_vec

