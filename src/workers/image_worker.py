from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="src.workers.image_worker.generate_image_task")
def generate_image_task(self, task_id: str, params: dict):
    """Generate image using diffusers. GPU model loading handled here."""
    self.update_state(state="RUNNING", meta={"progress": 0})
    # TODO: Load model via ModelManager, run inference, save to NAS
    return {"task_id": task_id, "status": "completed", "file": f"{task_id}/output.png"}
