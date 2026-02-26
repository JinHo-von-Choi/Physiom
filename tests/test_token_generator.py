import pytest
import numpy as np
from src.token_generator import generate_token_from_embeddings
from src.exceptions import EmbeddingError


def _make_unit_vec(dim=512, seed=None):
    """L2 정규화된 랜덤 벡터 생성."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_output_is_l2_normalized():
    """반환 벡터는 반드시 L2 정규화(노름=1)되어야 한다."""
    embs = [_make_unit_vec(seed=i) for i in range(5)]
    result = generate_token_from_embeddings(embs)
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-6, f"norm={norm}, not unit vector"


def test_weighted_mean_differs_from_uniform():
    """가중치 적용 시 균등 평균과 다른 결과가 나와야 한다."""
    embs = [_make_unit_vec(seed=i) for i in range(3)]
    weights = [0.1, 0.1, 0.8]
    result_weighted = generate_token_from_embeddings(embs, weights=weights)
    result_uniform = generate_token_from_embeddings(embs)
    cosine = float(np.dot(result_weighted, result_uniform))
    assert cosine < 0.9999, "가중 평균과 균등 평균이 동일해서는 안 됨"


def test_single_embedding_unchanged():
    """단일 임베딩 입력 시 정규화된 동일 벡터 반환."""
    emb = _make_unit_vec(seed=42)
    result = generate_token_from_embeddings([emb])
    cosine = float(np.dot(result, emb))
    assert abs(cosine - 1.0) < 1e-6


def test_empty_raises():
    with pytest.raises(EmbeddingError):
        generate_token_from_embeddings([])
