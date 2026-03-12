import shutil
from pathlib import Path

from src.config import get_settings


class StorageService:
    def __init__(self, outputs_path: str | None = None):
        self._base = Path(outputs_path or get_settings().NAS_OUTPUTS_PATH)

    def save(self, content: bytes, task_id: str, filename: str) -> str:
        task_dir = self._base / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        file_path = task_dir / filename
        file_path.write_bytes(content)
        return str(file_path)

    def get_url(self, task_id: str, filename: str) -> str:
        return str(self._base / task_id / filename)

    def list_files(self, task_id: str) -> list[str]:
        task_dir = self._base / task_id
        if not task_dir.exists():
            return []
        return [str(f) for f in task_dir.iterdir() if f.is_file()]

    def delete_task(self, task_id: str) -> None:
        task_dir = self._base / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
