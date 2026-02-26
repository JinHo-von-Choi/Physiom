import pytest
from src.config import AppSettings, default_config
from pathlib import Path

def test_config_immutability():
    """설정 객체가 불변(frozen)인지 확인합니다."""
    with pytest.raises(Exception):
        # frozen=True 이므로 속성 수정 시 예외가 발생해야 함
        default_config.max_frames = 999

def test_config_env_override(monkeypatch):
    """환경 변수가 설정을 올바르게 덮어쓰는지 확인합니다."""
    monkeypatch.setenv("FACER_MAX_FRAMES", "15")
    # 새로운 인스턴스 생성 시 환경 변수 반영 확인
    new_config = AppSettings()
    assert new_config.max_frames == 15

def test_config_default_values():
    """기본 설정값이 올바르게 설정되었는지 확인합니다."""
    assert default_config.model_name == "buffalo_l"
    assert default_config.server_port == 23535
    assert isinstance(default_config.temp_dir, Path)


def test_new_performance_fields():
    """추가된 성능 강화 필드 기본값 검증."""
    assert default_config.allowed_modules == ["detection", "landmark_3d_68", "recognition"]
    assert default_config.require_pose is True
    assert default_config.det_size == 320
    assert default_config.min_valid_frames == 3
    assert default_config.min_quality_score == 0.4
    assert default_config.outlier_cosine_threshold == 0.45
    assert default_config.pose_yaw_limit == 25.0
    assert default_config.pose_pitch_limit == 20.0
    assert default_config.camera_warmup_frames == 5
    assert default_config.frame_interval_ms == 200.0


def test_config_env_override_det_size(monkeypatch):
    """환경 변수로 det_size가 오버라이드되는지 확인."""
    monkeypatch.setenv("FACER_DET_SIZE", "640")
    cfg = AppSettings()
    assert cfg.det_size == 640
