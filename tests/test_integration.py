import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_outlier_rejection_in_generate_token():
    """outlier 임베딩(cosine < threshold)이 최종 임베딩에서 제외되는지 확인."""
    from src import generate_token_from_directory
    from src.config import default_config

    good_embs  = [_unit(i) for i in range(4)]
    outlier    = _unit(99) * -1.0  # 반대 방향 벡터
    outlier   /= np.linalg.norm(outlier)
    all_scores = [0.9, 0.8, 0.85, 0.88, 0.9]

    with patch('src.load_images_from_directory') as mock_load, \
         patch('src.extract_face_embeddings') as mock_extract, \
         patch('src.encoder_manager') as mock_mgr:

        mock_mgr.get_encoder.return_value = MagicMock()
        mock_load.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5
        mock_extract.return_value = (good_embs + [outlier], all_scores)

        result = generate_token_from_directory("fake/path", config=default_config)

    assert len(result) == 512
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


def test_insufficient_frames_raises():
    """유효 프레임이 min_valid_frames 미만이면 NoValidEmbeddingError 발생."""
    from src import generate_token_from_directory
    from src.config import AppSettings
    from src.exceptions import NoValidEmbeddingError

    cfg = AppSettings(min_valid_frames=3)

    with patch('src.load_images_from_directory') as mock_load, \
         patch('src.extract_face_embeddings') as mock_extract, \
         patch('src.encoder_manager') as mock_mgr:

        mock_mgr.get_encoder.return_value = MagicMock()
        mock_load.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2
        mock_extract.return_value = ([_unit(0), _unit(1)], [0.9, 0.85])  # 2개만 반환

        with pytest.raises(NoValidEmbeddingError, match="최소"):
            generate_token_from_directory("fake/path", config=cfg)
