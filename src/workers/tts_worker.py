from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="src.workers.tts_worker.generate_tts_task")
def generate_tts_task(self, task_id: str, params: dict):
    """Generate speech using CosyVoice2 or Qwen TTS."""
    self.update_state(state="RUNNING", meta={"progress": 0})
    # TODO: Load model, run inference, save audio to NAS
    return {"task_id": task_id, "status": "completed", "file": f"{task_id}/output.wav"}
