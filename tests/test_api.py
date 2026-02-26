import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient
from src.api_http import app, state


@pytest.mark.asyncio
async def test_health_check():
    """헬스체크 엔드포인트를 검증합니다."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "starting"]
    assert "det_size" in data


@pytest.mark.asyncio
async def test_generate_token_invalid_file():
    """유효하지 않은 파일 업로드 시 에러 응답을 검증합니다."""
    # httpx AsyncClient는 lifespan을 트리거하지 않으므로 encoder를 직접 mock
    state.encoder = MagicMock()
    try:
        async with AsyncClient(app=app, base_url="http://test") as ac:
            files = {"file": ("test.txt", b"not an image", "text/plain")}
            response = await ac.post("/token/generate", files=files)
        assert response.status_code == 400
    finally:
        state.encoder = None
