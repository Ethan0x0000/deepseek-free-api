import time
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.credentials import CredentialsManager, credentials_manager


def test_token_pool_round_robin(tmp_path):
    mgr = CredentialsManager()
    mgr.project_file = tmp_path / "credentials.json"
    mgr.user_file = tmp_path / "user_credentials.json"
    mgr.env_file = tmp_path / ".env"

    mgr.save(["token_A", "token_B"], provider="deepseek")
    assert mgr.is_authenticated("deepseek") is True
    assert mgr.get_all_tokens("deepseek") == ["token_A", "token_B"]

    t1 = mgr.get_token("deepseek", rotate=True)
    t2 = mgr.get_token("deepseek", rotate=True)
    t3 = mgr.get_token("deepseek", rotate=True)
    t4 = mgr.get_token("deepseek", rotate=True)

    assert [t1, t2, t3, t4] == ["token_A", "token_B", "token_A", "token_B"]


def test_token_pool_cooldown_isolation(tmp_path):
    mgr = CredentialsManager()
    mgr.project_file = tmp_path / "credentials.json"
    mgr.user_file = tmp_path / "user_credentials.json"
    mgr.env_file = tmp_path / ".env"

    mgr.save(["token_A", "token_B"], provider="deepseek")

    # 标记 token_A 冷却 3600 秒
    mgr.mark_token_status("deepseek", "token_A", cooldown_seconds=3600, error="Banned")

    # 后续调用应自动跳过 token_A，始终返回健康的 token_B
    assert mgr.get_token("deepseek", rotate=True) == "token_B"
    assert mgr.get_token("deepseek", rotate=True) == "token_B"

    # 标记 token_A 恢复成功
    mgr.mark_token_status("deepseek", "token_A", is_success=True)

    # 恢复后重新参与轮询
    assert mgr.get_token("deepseek", rotate=True) == "token_A"
    assert mgr.get_token("deepseek", rotate=True) == "token_B"


def test_token_pool_all_cooling_fallback(tmp_path):
    mgr = CredentialsManager()
    mgr.project_file = tmp_path / "credentials.json"
    mgr.user_file = tmp_path / "user_credentials.json"
    mgr.env_file = tmp_path / ".env"

    mgr.save(["token_slow", "token_fast"], provider="deepseek")

    # token_slow 冷却 1000 秒，token_fast 冷却 60 秒
    mgr.mark_token_status("deepseek", "token_slow", cooldown_seconds=1000)
    mgr.mark_token_status("deepseek", "token_fast", cooldown_seconds=60)

    # 当所有 Token 均处于冷却时，必须兜底返回离解封最近的那个 (token_fast)
    selected = mgr.get_token("deepseek", rotate=True)
    assert selected == "token_fast"


def test_token_pool_append_and_status(tmp_path):
    mgr = CredentialsManager()
    mgr.project_file = tmp_path / "credentials.json"
    mgr.user_file = tmp_path / "user_credentials.json"
    mgr.env_file = tmp_path / ".env"

    mgr.save("token_1", provider="deepseek")
    assert mgr.get_all_tokens("deepseek") == ["token_1"]

    mgr.save("token_2", provider="deepseek", append=True)
    assert mgr.get_all_tokens("deepseek") == ["token_1", "token_2"]

    status = mgr.get_pool_status("deepseek")
    assert status["total"] == 2
    assert status["healthy"] == 2
    assert status["cooling"] == 0
    assert len(status["tokens"]) == 2


@pytest.mark.asyncio
async def test_auth_api_multi_token(monkeypatch, tmp_path):
    fake_proj = tmp_path / "credentials.json"
    fake_user = tmp_path / "user_credentials.json"
    fake_env = tmp_path / ".env"
    monkeypatch.setattr(credentials_manager, "project_file", fake_proj)
    monkeypatch.setattr(credentials_manager, "user_file", fake_user)
    monkeypatch.setattr(credentials_manager, "env_file", fake_env)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/api/v1/auth/token", json={
            "provider": "deepseek",
            "tokens": ["test_tok_1", "test_tok_2"],
            "action": "replace"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["total_tokens"] == 2
        assert data["healthy_tokens"] == 2

        status_resp = await ac.get("/api/v1/auth/status")
        assert status_resp.status_code == 200
        st_data = status_resp.json()
        assert st_data["authenticated"] is True
        assert st_data["pools"]["deepseek"]["total"] == 2
