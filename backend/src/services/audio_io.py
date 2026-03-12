"""Client for nous-core audio IO service."""

import httpx


class AudioIOClient:
    """Async HTTP client wrapping nous-core /audio/* endpoints.

    Reuses a single httpx.AsyncClient for connection pooling.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def info(self, path: str) -> dict:
        resp = await self._client.post("/audio/info", json={"path": path})
        resp.raise_for_status()
        return resp.json()

    async def resample(self, input_path: str, output_path: str, target_sample_rate: int) -> dict:
        resp = await self._client.post("/audio/resample", json={
            "input_path": input_path,
            "output_path": output_path,
            "target_sample_rate": target_sample_rate,
        })
        resp.raise_for_status()
        return resp.json()

    async def concat(self, input_paths: list[str], output_path: str) -> dict:
        resp = await self._client.post("/audio/concat", json={
            "input_paths": input_paths,
            "output_path": output_path,
        })
        resp.raise_for_status()
        return resp.json()

    async def split(self, input_path: str, output_dir: str, split_points_ms: list[int]) -> dict:
        resp = await self._client.post("/audio/split", json={
            "input_path": input_path,
            "output_dir": output_dir,
            "split_points_ms": split_points_ms,
        })
        resp.raise_for_status()
        return resp.json()

    async def convert(self, input_path: str, output_path: str, target_format: str = "wav") -> dict:
        resp = await self._client.post("/audio/convert", json={
            "input_path": input_path,
            "output_path": output_path,
            "target_format": target_format,
        })
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()


# Default singleton
audio_io = AudioIOClient()
