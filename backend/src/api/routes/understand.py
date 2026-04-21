import logging

import httpx
from fastapi import APIRouter, HTTPException

from src.config import get_settings
from src.models.schemas import ImageUnderstandRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/understand")


async def call_vllm(image_url: str, question: str, model: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(proxy=None) as client:
        resp = await client.post(
            f"{settings.VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            logger.warning("VL backend %s returned %s: %s",
                           settings.VLLM_BASE_URL, resp.status_code, resp.text[:200])
            raise HTTPException(
                status_code=502,
                detail=f"VL backend error ({resp.status_code}): {resp.text[:200]}",
            )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "model": data.get("model", model)}


@router.post("/image")
async def understand_image(req: ImageUnderstandRequest):
    settings = get_settings()
    model = req.model or settings.VL_MODEL
    try:
        return await call_vllm(req.image_url, req.question, model)
    except HTTPException:
        raise
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"VL backend unreachable: {e}")
