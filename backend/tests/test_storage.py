import tempfile
from pathlib import Path

from src.storage.nas import StorageService


def test_save_and_get_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageService(outputs_path=tmpdir)
        content = b"fake image data"
        file_path = storage.save(content, task_id="abc123", filename="output.png")
        assert Path(file_path).exists()
        assert storage.get_url("abc123", "output.png") == f"{tmpdir}/abc123/output.png"


def test_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageService(outputs_path=tmpdir)
        storage.save(b"data1", task_id="t1", filename="a.png")
        storage.save(b"data2", task_id="t1", filename="b.png")
        files = storage.list_files("t1")
        assert len(files) == 2


def test_delete_task_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageService(outputs_path=tmpdir)
        storage.save(b"data", task_id="t2", filename="x.png")
        storage.delete_task("t2")
        assert not Path(tmpdir, "t2").exists()
