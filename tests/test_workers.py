from src.workers.image_worker import generate_image_task
from src.workers.tts_worker import generate_tts_task
from src.workers.video_worker import generate_video_task


def test_image_task_is_celery_task():
    assert hasattr(generate_image_task, "delay")
    assert generate_image_task.name == "src.workers.image_worker.generate_image_task"


def test_tts_task_is_celery_task():
    assert hasattr(generate_tts_task, "delay")


def test_video_task_is_celery_task():
    assert hasattr(generate_video_task, "delay")
