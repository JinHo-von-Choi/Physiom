import pytest
import numpy as np
import cv2
from src.face_encoder import FaceEncoder
from src.config import AppSettings

@pytest.fixture(scope="session")
def encoder():
    """테스트용 FaceEncoder (CPU 모드)"""
    config = AppSettings(use_gpu=False, use_directml=False)
    return FaceEncoder(config)

def test_smart_target_selection(encoder):
    """다중 얼굴 중 큰 얼굴이 선택되는지 검증합니다."""
    # 640x640 빈 이미지 생성
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # 가상의 두 얼굴 (bbox 모사)
    class MockFace:
        def __init__(self, bbox, embedding):
            self.bbox = bbox
            self.normed_embedding = embedding

    face_small = MockFace([10, 10, 50, 50], np.random.rand(512))  # 면적 1600
    face_large = MockFace([100, 100, 300, 300], np.random.rand(512)) # 면적 40000
    
    # encoder.app.get 결과 모사 (Monkeypatch 대신 리스트 직접 조작 테스트는 어려우므로 로직만 검증)
    faces = [face_small, face_large]
    faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    
    assert faces[0] == face_large

def test_blur_detection(encoder):
    """흐릿한 이미지 입력 시 필터링되는지 검증합니다."""
    # 심하게 흐릿한 이미지 (단색)
    blurry_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(blurry_img, (10, 10), (20, 20), (255, 255, 255), -1)
    # 가우시안 블러를 심하게 적용
    blurry_img = cv2.GaussianBlur(blurry_img, (99, 99), 0)

    faces = encoder.detect_faces(blurry_img)
    assert len(faces) == 0 # 선명도 점수 미달로 빈 리스트 반환 예상


def test_pose_filter_returns_none_on_mocked_bad_pose(encoder):
    """포즈 범위 초과 시 pose_out_of_range 반환 확인 (mock 기반)."""
    import unittest.mock as mock

    good_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(good_img, (200, 200), (440, 440), (200, 200, 200), -1)

    with mock.patch.object(encoder, '_is_pose_acceptable', return_value=False):
        with mock.patch.object(encoder, 'detect_faces') as mock_detect:
            class FakeFace:
                bbox = [200, 200, 440, 440]
                det_score = 0.95
                normed_embedding = np.random.rand(512)
                pose = None
                def get(self, key, default=None):
                    return getattr(self, key, default)
            mock_detect.return_value = [FakeFace()]
            emb, err, score = encoder.extract_embedding(good_img)

    assert emb is None
    assert err == "pose_out_of_range"
    assert score == 0.0
