from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="src.workers.video_worker.generate_video_task")
def generate_video_task(self, task_id: str, params: dict):
    """Generate video using Wan2.1. Requires exclusive dual-GPU access."""
    self.update_state(state="RUNNING", meta={"progress": 0})
    # TODO: Unload all models, load Wan2.1 on dual GPU, run inference, restore models
    return {"task_id": task_id, "status": "completed", "file": f"{task_id}/output.mp4"}
