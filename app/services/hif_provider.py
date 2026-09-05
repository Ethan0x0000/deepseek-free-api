"""
HIF (High-Integrity Framework) token provider for DeepSeek.
Fetches and caches x-hif-leim and x-hif-dliq tokens from DeepSeek's signing service.
Tokens are required for Vision and certain advanced operations.
Includes automatic IPv4 fallback for hif-dliq (which only has IPv6 AAAA DNS records).
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
import httpx
import anyio

logger = logging.getLogger(__name__)

HIF_LEIM_URL = "https://hif-leim.deepseek.com/query"
HIF_DLIQ_URL = "https://hif-dliq.deepseek.com/query"


class HIFProvider:
    """Manages cached HIF tokens (leim, dliq) with ~10 min TTL."""

    def __init__(self, ttl: float = 600.0):
        self.ttl = ttl
        self._leim: Optional[str] = None
        self._dliq: Optional[str] = None
        self._expires_at: float = 0
        self._lock = asyncio.Lock()

    async def get_headers(self, client: httpx.AsyncClient, token: str) -> Dict[str, str]:
        """Returns dict containing x-hif-leim and x-hif-dliq headers."""
        now = time.time()
        if now < self._expires_at and self._leim and self._dliq:
            return {
                "x-hif-leim": self._leim,
                "x-hif-dliq": self._dliq,
            }

        async with self._lock:
            # Double-check after acquiring lock
            if time.time() < self._expires_at and self._leim and self._dliq:
                return {
                    "x-hif-leim": self._leim,
                    "x-hif-dliq": self._dliq,
                }

            headers = {
                "Authorization": f"Bearer {token}",
                "Origin": "https://chat.deepseek.com",
                "Referer": "https://chat.deepseek.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
                ),
            }

            # Note: hif-dliq.deepseek.com only has IPv6 AAAA records on many public DNS,
            # which fails on IPv4-only networks/docker containers.
            # Both leim and dliq point to the exact same CloudFront/TencentCloud edge IP distribution.
            # We intercept anyio.connect_tcp to connect to hif-leim's IPv4 address when querying hif-dliq.
            _orig_connect_tcp = anyio.connect_tcp

            async def _patched_connect_tcp(remote_host, remote_port, **kwargs):
                if isinstance(remote_host, str) and remote_host == "hif-dliq.deepseek.com":
                    remote_host = "hif-leim.deepseek.com"
                return await _orig_connect_tcp(remote_host, remote_port, **kwargs)

            anyio.connect_tcp = _patched_connect_tcp
            try:
                leim_resp = await client.get(HIF_LEIM_URL, headers=headers, timeout=15.0)
                dliq_resp = await client.get(HIF_DLIQ_URL, headers=headers, timeout=15.0)

                leim_val = leim_resp.json().get("data", {}).get("biz_data", {}).get("value")
                dliq_val = dliq_resp.json().get("data", {}).get("biz_data", {}).get("value")

                if leim_val and dliq_val:
                    self._leim = str(leim_val)
                    self._dliq = str(dliq_val)
                    self._expires_at = time.time() + (self.ttl * 0.8)
                    logger.debug(f"HIF tokens refreshed successfully (expires in {int(self.ttl*0.8)}s)")
                    return {
                        "x-hif-leim": self._leim,
                        "x-hif-dliq": self._dliq,
                    }
                else:
                    logger.warning("Failed to extract HIF tokens from responses")
            except Exception as e:
                logger.warning(f"Error fetching HIF tokens: {e}")
            finally:
                anyio.connect_tcp = _orig_connect_tcp

        # Return whatever we had cached if fresh request failed
        out = {}
        if self._leim:
            out["x-hif-leim"] = self._leim
        if self._dliq:
            out["x-hif-dliq"] = self._dliq
        return out


hif_provider = HIFProvider()
