import uuid
from datetime import datetime, timezone

from src.models.task import TaskStatus, TaskRecord


def test_task_record_creation():
    task = TaskRecord(
        id=uuid.uuid4(),
        task_type="image",
        status=TaskStatus.PENDING,
        params={"prompt": "a cat"},
    )
    assert task.status == TaskStatus.PENDING
    assert task.task_type == "image"
    assert task.params["prompt"] == "a cat"
    assert task.result is None


def test_task_status_values():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
