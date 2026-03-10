import httpx
from fastapi import APIRouter

from src.config import get_settings
from src.models.schemas import ImageUnderstandRequest

router = APIRouter(prefix="/api/v1/understand")


async def call_vllm(image_url: str, question: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": "Qwen2.5-VL-7B-Instruct",
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
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "model": data.get("model", "qwen25-vl")}


@router.post("/image")
async def understand_image(req: ImageUnderstandRequest):
    result = await call_vllm(req.image_url, req.question)
    return result
