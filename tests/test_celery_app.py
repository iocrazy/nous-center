from src.workers.celery_app import celery_app


def test_celery_app_configured():
    assert celery_app.main == "mind-center"


def test_celery_queues():
    routes = celery_app.conf.task_routes
    assert routes["src.workers.image_worker.*"]["queue"] == "image"
    assert routes["src.workers.tts_worker.*"]["queue"] == "tts"
    assert routes["src.workers.video_worker.*"]["queue"] == "video"
