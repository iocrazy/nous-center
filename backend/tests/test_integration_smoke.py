"""Smoke tests: verify all new modules wire together without import errors."""
import pytest


async def test_app_starts_without_errors(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_apps_endpoint_exists(db_client):
    resp = await db_client.get("/api/v1/apps")
    assert resp.status_code == 200


async def test_inference_imports():
    from src.services.inference import InferenceAdapter, InferenceResult, ModelRegistry
    from src.services.inference.llm_vllm import VLLMAdapter
    from src.services.model_manager import ModelManager
    from src.services.gpu_allocator import GPUAllocator
    from src.services.task_queue import TaskQueue
    from src.models.workflow_app import WorkflowApp
    assert True
