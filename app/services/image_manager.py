"""
Image Manager for DeepSeek Vision multimodal integration.
Extracts images from OpenAI/Anthropic messages, uploads them to chat.deepseek.com,
forks them to the Vision model, and caches them to avoid duplicate uploads.
"""
import asyncio
import base64
import hashlib
import logging
import mimetypes
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from app.core.config import settings
from app.core.credentials import credentials_manager
from app.core.pow_solver import pow_solver

logger = logging.getLogger(__name__)

BASE_URL = settings.DEEPSEEK_BASE_URL


class ImageManager:
    """Handles image extraction, uploading, vision forking, and caching."""

    def __init__(self):
        # Cache image_hash -> vision_file_id
        self._cache: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    def extract_images_from_messages(
        self, messages: List[Any]
    ) -> List[Tuple[bytes, str, str]]:
        """
        Extracts all images from a list of OpenAI or Anthropic messages.
        Returns a list of (image_bytes, mime_type, filename).
        """
        extracted = []
        for msg in messages:
            content = getattr(msg, "content", None)
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type", "")

                # 1. OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                if part_type == "image_url":
                    img_info = part.get("image_url", {})
                    url = img_info.get("url", "")
                    if url:
                        img_bytes, mime = self._parse_image_data_url(url)
                        if img_bytes:
                            ext = mimetypes.guess_extension(mime) or ".png"
                            extracted.append((img_bytes, mime, f"image{ext}"))

                # 2. Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
                elif part_type == "image":
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        b64_data = source.get("data", "")
                        mime = source.get("media_type", "image/png")
                        try:
                            img_bytes = base64.b64decode(b64_data)
                            ext = mimetypes.guess_extension(mime) or ".png"
                            extracted.append((img_bytes, mime, f"image{ext}"))
                        except Exception as e:
                            logger.warning(f"Failed to decode Anthropic base64 image: {e}")

        return extracted

    def _parse_image_data_url(self, url: str) -> Tuple[Optional[bytes], str]:
        """Parses a data URL like data:image/png;base64,iVBORw... or raw base64."""
        if url.startswith("data:"):
            # Format: data:<mime>;base64,<data>
            match = re.match(r"^data:([^;]+);base64,(.+)$", url, re.DOTALL)
            if match:
                mime_type = match.group(1).strip()
                b64_str = match.group(2).strip()
                try:
                    return base64.b64decode(b64_str), mime_type
                except Exception as e:
                    logger.warning(f"Error decoding base64 from data URL: {e}")
                    return None, "image/png"
        elif len(url) > 100 and not url.startswith("http"):
            # Could be raw base64
            try:
                data = base64.b64decode(url)
                return data, "image/png"
            except Exception:
                pass
        return None, "image/png"

    async def upload_image_for_vision(
        self,
        client: httpx.AsyncClient,
        image_bytes: bytes,
        filename: str = "image.png",
        mime_type: str = "image/png",
    ) -> str:
        """
        Uploads image to DeepSeek, polls OCR status, forks to vision, and returns vision_file_id.
        Reuses cached file_id if image was already uploaded.
        """
        img_hash = hashlib.sha256(image_bytes).hexdigest()
        if img_hash in self._cache:
            logger.info(f"Using cached vision_file_id for image {img_hash[:8]}: {self._cache[img_hash]}")
            return self._cache[img_hash]

        async with self._lock:
            # Re-check inside lock
            if img_hash in self._cache:
                return self._cache[img_hash]

            token = credentials_manager.get_token("deepseek")
            if not token:
                raise RuntimeError("DeepSeek authentication token not configured.")

            # Step 1: PoW for file upload
            target_path = "/api/v0/file/upload_file"
            pow_header = await pow_solver.get_pow_header(client, target_path)

            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": settings.USER_AGENT,
                "Accept": "application/json",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
                "x-client-platform": "web",
                "x-client-version": settings.CLIENT_VERSION,
                "x-ds-pow-response": pow_header,
            }

            files = {"file": (filename, image_bytes, mime_type)}

            logger.info(f"Uploading image ({len(image_bytes)} bytes) to {target_path}...")
            upload_resp = await client.post(
                f"{BASE_URL}{target_path}",
                headers=headers,
                files=files,
                timeout=45.0,
            )

            if upload_resp.status_code != 200:
                raise RuntimeError(
                    f"Image upload failed with status {upload_resp.status_code}: {upload_resp.text[:200]}"
                )

            res_data = upload_resp.json()
            if res_data.get("code") != 0:
                raise RuntimeError(f"Image upload rejected: {res_data}")

            raw_file_id = res_data["data"]["biz_data"]["id"]
            logger.info(f"Image uploaded successfully, file_id: {raw_file_id}")

            # Step 2: Poll file until SUCCESS or timeout (up to 20s)
            status_headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
            }
            for i in range(20):
                await asyncio.sleep(1.0)
                poll_resp = await client.get(
                    f"{BASE_URL}/api/v0/file/fetch_files?file_ids={raw_file_id}",
                    headers=status_headers,
                    timeout=15.0,
                )
                if poll_resp.status_code == 200:
                    files_list = poll_resp.json().get("data", {}).get("biz_data", {}).get("files", [])
                    if files_list:
                        status = files_list[0].get("status", "").upper()
                        if status == "SUCCESS" or status == "CONTENT_EMPTY":
                            # Ready for forking (CONTENT_EMPTY is normal for pure binary images before OCR)
                            break
                        if status in ["FAILED", "ERROR"]:
                            raise RuntimeError(f"File upload parsing failed: {status}")

            # Step 3: Fork to Vision
            logger.info(f"Forking file {raw_file_id} to vision model...")
            fork_headers = {
                **status_headers,
                "Content-Type": "application/json",
            }
            fork_resp = await client.post(
                f"{BASE_URL}/api/v0/file/fork_file_task",
                headers=fork_headers,
                json={"file_id": raw_file_id, "to_model_type": "vision"},
                timeout=20.0,
            )

            if fork_resp.status_code != 200:
                raise RuntimeError(f"Fork to vision failed HTTP {fork_resp.status_code}: {fork_resp.text[:200]}")

            fork_data = fork_resp.json()
            if fork_data.get("code") != 0:
                raise RuntimeError(f"Fork to vision rejected: {fork_data}")

            vision_file_id = fork_data["data"]["biz_data"]["id"]
            logger.info(f"Forked to vision, vision_file_id: {vision_file_id}")

            # Step 4: Poll vision file until SUCCESS (up to 25s)
            for i in range(25):
                await asyncio.sleep(1.0)
                poll_resp = await client.get(
                    f"{BASE_URL}/api/v0/file/fetch_files?file_ids={vision_file_id}",
                    headers=status_headers,
                    timeout=15.0,
                )
                if poll_resp.status_code == 200:
                    files_list = poll_resp.json().get("data", {}).get("biz_data", {}).get("files", [])
                    if files_list:
                        v_status = files_list[0].get("status", "").upper()
                        if v_status == "SUCCESS":
                            logger.info(f"✓ Vision file {vision_file_id} is ready for inference!")
                            self._cache[img_hash] = vision_file_id
                            return vision_file_id
                        if v_status in ["FAILED", "ERROR"]:
                            raise RuntimeError(f"Vision file processing failed: {v_status}")

            # Fallback return vision_file_id even if poll timed out (sometimes DeepSeek processes concurrently)
            self._cache[img_hash] = vision_file_id
            return vision_file_id

    async def process_images(
        self, client: httpx.AsyncClient, messages: List[Any]
    ) -> List[str]:
        """
        Extracts and uploads all images from messages concurrently.
        Returns a list of vision_file_id strings.
        """
        extracted = self.extract_images_from_messages(messages)
        if not extracted:
            return []

        logger.info(f"Detected {len(extracted)} image(s) in messages. Processing for DeepSeek Vision...")
        tasks = [
            self.upload_image_for_vision(client, img_bytes, filename, mime)
            for img_bytes, mime, filename in extracted
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_file_ids = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Failed to upload image #{idx+1}: {res}")
            else:
                valid_file_ids.append(res)

        return valid_file_ids


image_manager = ImageManager()
