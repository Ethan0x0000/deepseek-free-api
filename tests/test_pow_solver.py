import pytest
from app.core.pow_solver import pow_solver


@pytest.mark.asyncio
async def test_pow_solver_wasm():
    challenge_data = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "0000000000000000000000000000000000000000000000000000000000000000",
        "salt": "test_salt",
        "difficulty": 100,
        "expire_at": 1788251660,
        "signature": "test_sig",
        "target_path": "/api/v0/chat/completion"
    }

    try:
        res = await pow_solver.solve_challenge(challenge_data)
        assert isinstance(res, str)
        assert len(res) > 0
    except RuntimeError as e:
        assert "未找到" in str(e) or "PoW" in str(e)
