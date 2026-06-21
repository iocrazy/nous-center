"""OpenAI-compatible endpoints: chat/completions, audio/speech, models."""

import asyncio
import io
import json
import logging
import os
import tempfile
import time
from typing import Literal

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token_any
from src.config import get_settings
from src.errors import APIError, InvalidRequestError, NotFoundError, NousError
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.inference.vllm_endpoint import (
    VLLMNoEndpoint,
    VLLMNotLoaded,
    get_vllm_base_url,
)
from src.services.model_resolver import ModelNotFound, resolve_target_service
from src.services.prompt_composer import (
    AgentLoadFailed,
    AgentNotFound,
)
from src.services.prompt_composer import (
    compose as compose_agent_prompt,
)
from src.services.skill_tools import skill_tool_schema

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai-compat"])

# round3 #1:持有 fire-and-forget 结算 task 的强引用直到完成。CPython 事件循环只持 task
# 弱引用,未被引用的 task 可能在跑完前被 GC → 结算(记账+扣配额)协程中途消失。
# done_callback 在完成时移除引用,避免 set 无界增长。
_settle_tasks: set[asyncio.Task] = set()


async def sse_with_error_envelope(inner):
    """Wrap an SSE async generator so any NousError/Exception is emitted as an
    OpenAI-style error chunk followed by exactly one `data: [DONE]`.

    - Strips stray `data: [DONE]` markers emitted by the inner generator so we
      always emit exactly one terminator from the wrapper.
    - Converts NousError via to_dict(); any other Exception becomes a generic
      APIError (no traceback leak).
    """
    try:
        async for chunk in inner:
            if chunk.strip() == "data: [DONE]":
                # wrapper owns the terminator
                continue
            yield chunk
    except NousError as e:
        yield f"data: {json.dumps(e.to_dict())}\n\n"
    except Exception:
        logger.exception("SSE stream failure")
        err = APIError("Internal server error", code="internal_error")
        yield f"data: {json.dumps(err.to_dict())}\n\n"
    finally:
        yield "data: [DONE]\n\n"


# --- thinking-mode model whitelist ---
# Models whose chat template honors `chat_template_kwargs.enable_thinking`.
# Match is by case-insensitive substring on the engine name. If a model is not
# listed, the `extra_body.thinking` field is silently ignored (per Step 2 spec
# decision C+A: whitelist with silent fallback).
_THINKING_MODEL_PATTERNS = (
    "qwen3",  # qwen3.5-35b, qwen3-8b, etc.
    "deepseek-r1",
    "deepseek-v3",
    "doubao-seed-1.8",
    "doubao-seed-2",
)


def _supports_thinking(engine_name: str) -> bool:
    n = (engine_name or "").lower()
    return any(p in n for p in _THINKING_MODEL_PATTERNS)


async def _post_consume_quota(api_key_id: int, service_id: int, units: int) -> None:
    """Charge `units` against the (api_key, service) grant post-inference.

    Best-effort: legacy keys (no grant) are silently skipped; exhaustion is
    logged but does not fail the already-completed request. A pre-flight
    check belongs in a follow-up; for now the next call fails at
    pre-flight once that lands.
    """
    if units <= 0:
        return
    from src.models.database import get_session_factory
    from src.services.quota_gate import NoActiveGrant, consume_for_request
    from src.services.resource_pack import QuotaExhausted

    sf = get_session_factory()
    async with sf() as s:
        try:
            await consume_for_request(
                s, api_key_id=api_key_id, service_id=service_id, units=units,
            )
            await s.commit()
        except NoActiveGrant:
            return
        except QuotaExhausted:
            logger.warning(
                "grant exhausted post-inference for api_key=%s service=%s",
                api_key_id, service_id,
            )


def _maybe_inject_thinking(body: dict, engine_name: str) -> None:
    """Translate `body['thinking'] = {'type': enabled|disabled|auto}` into
    `body['chat_template_kwargs']['enable_thinking'] = bool` for vLLM.

    - Pops `thinking` from body either way (vLLM rejects unknown top-level fields).
    - If model isn't whitelisted, silently drop (per Ark `extra_body` semantics:
      non-standard fields are best-effort, not hard contract).
    - `auto` = leave unset, let model default.
    """
    thinking = body.pop("thinking", None)
    if not isinstance(thinking, dict):
        return
    t = thinking.get("type")
    if t not in ("enabled", "disabled", "auto"):
        return
    if not _supports_thinking(engine_name):
        return
    if t == "auto":
        return
    kwargs = body.setdefault("chat_template_kwargs", {})
    kwargs["enable_thinking"] = (t == "enabled")


# --- /v1/chat/completions ---

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
    session: AsyncSession = Depends(get_async_session),
):
    """OpenAI-compatible chat completions with token metering.

    Dispatch:
      - Legacy keys (instance_id bound) → use that instance, ignore body.model
      - M:N keys → look up active ApiKeyGrant by body.model (ServiceInstance.name)

    After resolution, instance.source_type selects the handler:
      - "model"    → vllm subprocess (below)
      - "workflow" → WorkflowExecutor (not yet wired; 501)
      - "app"      → 501 (v3: app == workflow-backed service; routed via /v1/apps)
    """
    instance, api_key = auth

    body = await request.json()
    requested_model = body.get("model") or None

    # Resolve target service. Legacy 1:1 keys (instance set by the auth dep)
    # short-circuit; M:N keys use the v3 grant lookup.
    if instance is None:
        try:
            instance = await resolve_target_service(
                session, api_key=api_key, requested_model=requested_model,
            )
        except ModelNotFound as e:
            raise NotFoundError(str(e), code="model_not_found")
        if instance.status != "active":
            raise HTTPException(403, detail="Instance is inactive")
        # M:N key 在 auth 层(verify_bearer_token_any 返回 None instance)没限流,
        # 解析出目标 instance 后必须补占坑,否则 M:N key 完全绕过 RPM/TPM。
        from src.api.deps_auth import enforce_instance_rate_limit
        await enforce_instance_rate_limit(instance)
        # v3: dispatch needs the snapshot if the service is workflow-backed.
        # Force-load deferred columns now so handlers below see real data
        # (covered by test_services_dispatch.py SQL counter assertion).
        await session.refresh(
            instance,
            attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"],
        )

    # Dispatch by source_type
    if instance.source_type == "workflow":
        raise HTTPException(
            501,
            detail="workflow-backed chat/completions not yet implemented",
        )
    if instance.source_type == "app":
        raise HTTPException(
            501,
            detail="app-backed chat/completions not yet implemented",
        )
    if instance.source_type != "model":
        raise HTTPException(
            400,
            detail=f"Unsupported instance source_type: {instance.source_type}",
        )

    engine_name = instance.source_name or str(instance.source_id)
    # spec §4.5 D6/D8: direct-to-vLLM HTTP. base-URL lookup via single source of truth.
    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise HTTPException(503, detail=str(e)) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e

    # Adapter handle still needed downstream for max_model_len clamp (line ~283).
    adapter = model_mgr.get_adapter(engine_name)

    body["model"] = ""  # vLLM uses its own model path

    # Resolve agent (top-level or extra_body.agent). vLLM rejects unknown
    # top-level fields, so always pop — even when injection is disabled.
    agent_id = body.pop("agent", None)
    if not agent_id and isinstance(body.get("extra_body"), dict):
        agent_id = body["extra_body"].pop("agent", None)
        if not body["extra_body"]:
            body.pop("extra_body", None)

    # Compose agent system message (chat/completions has no session concept,
    # so there's no binding check — every request is independent).
    settings = get_settings()
    agent_sys: str | None = None
    if settings.NOUS_ENABLE_AGENT_INJECTION and agent_id:
        try:
            agent_sys = compose_agent_prompt(agent_id, None)
        except AgentNotFound:
            raise InvalidRequestError(
                f"agent not found: {agent_id}",
                code="agent_not_found",
            )
        except AgentLoadFailed as e:
            logger.error("agent load failed: %s", e)
            raise APIError(
                f"failed to load agent {agent_id}",
                code="agent_load_failed",
            )

    if agent_sys is not None:
        messages = list(body.get("messages") or [])
        messages.insert(0, {"role": "system", "content": agent_sys})
        body["messages"] = messages
        # Inject Skill tool schema when an agent is active.
        tools_list = list(body.get("tools") or [])
        tools_list.insert(0, skill_tool_schema())
        body["tools"] = tools_list

    # Resolve context_id (top-level or extra_body.context_id)
    context_id = body.pop("context_id", None)
    if not context_id and isinstance(body.get("extra_body"), dict):
        context_id = body["extra_body"].pop("context_id", None)
        if not body["extra_body"]:
            body.pop("extra_body", None)

    if context_id:
        from src.models.database import get_session_factory as _csf
        from src.services.context_cache_service import (
            increment_hit_and_extend as _ihe,
        )
        from src.services.context_cache_service import (
            resolve_for_request,
        )

        sf = _csf()
        async with sf() as cache_session:
            cached_messages, cached_ttl = await resolve_for_request(
                cache_session,
                context_id=context_id,
                owner_key_id=api_key.id,
                engine_name=engine_name,
            )
        if cached_messages:
            body["messages"] = cached_messages + list(body.get("messages", []))

        # Fire-and-forget hit-count update; loop persists across requests under uvicorn.
        async def _bump(cid: str = context_id, ttl: int = cached_ttl, kid: int = api_key.id):
            try:
                async with _csf()() as s2:
                    await _ihe(s2, cid, ttl, owner_key_id=kid)
            except Exception:
                logger.exception("hit_count update failed for %s", cid)
        asyncio.create_task(_bump())

    # OpenAI SDK extra_body.thinking → vLLM chat_template_kwargs.enable_thinking
    # Whitelist-driven; silent ignore for unsupported models (Step 2 spec).
    _maybe_inject_thinking(body, engine_name)

    # Clamp max_tokens
    max_model_len = getattr(adapter, "max_model_len", 4096) or 4096
    if body.get("max_tokens") and body["max_tokens"] > max_model_len - 512:
        body["max_tokens"] = max(max_model_len - 512, max_model_len // 2)

    is_stream = body.get("stream", False)
    start_ms = time.monotonic()

    if is_stream:
        # Streaming: inject include_usage, proxy SSE chunks
        body.setdefault("stream_options", {})["include_usage"] = True

        async def _stream_proxy():
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                async with httpx.AsyncClient(timeout=300, proxy=None) as client:
                    async with client.stream(
                        "POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=body
                    ) as resp:
                        if resp.status_code != 200:
                            error_text = (await resp.aread()).decode(errors="replace")
                            # Map upstream status to a NousError so the wrapper
                            # formats it uniformly.
                            if resp.status_code == 404:
                                raise NotFoundError(error_text[:500], code="upstream_not_found")
                            if 400 <= resp.status_code < 500:
                                raise InvalidRequestError(error_text[:500], code="upstream_bad_request")
                            raise APIError("Upstream LLM error", code="upstream_error")
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            yield line + "\n"
                            # Extract usage from final chunk
                            if line.startswith("data: ") and line[6:] != "[DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    if "usage" in chunk and chunk["usage"]:
                                        usage = chunk["usage"]
                                except Exception:
                                    pass
                        yield "\n"
            finally:
                # round2 #7:记账/扣配额放 finally —— 否则客户端中途断连(generator 被取消)
                # 时,流后的结算代码不执行 = 已生成的 token 不记账、不扣配额(漏收入)。用
                # create_task 把结算跟 generator 取消解耦(record/consume 各自开 session);只在
                # 拿到 usage(含 token 数的末帧已到)时结算 —— 纯错误/早断连无计数,跳过免噪音。
                tok = usage.get("total_tokens", 0) or usage.get("completion_tokens", 0)
                if tok > 0:
                    duration = int((time.monotonic() - start_ms) * 1000)
                    _u = dict(usage)

                    async def _settle() -> None:
                        try:
                            from src.services.usage_service import record_llm_usage
                            await record_llm_usage(
                                model=engine_name,
                                prompt_tokens=_u.get("prompt_tokens", 0),
                                completion_tokens=_u.get("completion_tokens", 0),
                                duration_ms=duration,
                                instance_id=instance.id,
                                api_key_id=api_key.id,
                                agent_id=agent_id if settings.NOUS_ENABLE_AGENT_INJECTION else None,
                            )
                            await _post_consume_quota(
                                api_key.id, instance.id, _u.get("total_tokens", 0),
                            )
                        except Exception as e:  # noqa: BLE001 — 结算失败不该崩流
                            logger.warning("stream billing settle failed: %s", e)

                    # 持强引用直到完成,否则正常跑完的流式也可能被 GC 掉结算 task(round3 #1)。
                    _t = asyncio.create_task(_settle())
                    _settle_tasks.add(_t)
                    _t.add_done_callback(_settle_tasks.discard)

        return StreamingResponse(
            sse_with_error_envelope(_stream_proxy()),
            media_type="text/event-stream",
        )

    else:
        # Non-streaming: proxy request, extract usage
        async with httpx.AsyncClient(timeout=300, proxy=None) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body)

        duration = int((time.monotonic() - start_ms) * 1000)

        if resp.status_code != 200:
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

        data = resp.json()
        usage = data.get("usage", {})

        # Record usage
        from src.services.usage_service import record_llm_usage
        await record_llm_usage(
            model=engine_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_ms=duration,
            instance_id=instance.id,
            api_key_id=api_key.id,
            agent_id=agent_id if settings.NOUS_ENABLE_AGENT_INJECTION else None,
        )
        await _post_consume_quota(
            api_key.id, instance.id, usage.get("total_tokens", 0),
        )

        return Response(content=resp.content, media_type="application/json")


# --- /v1/audio/speech ---

class SpeechRequest(BaseModel):
    model: str = "cosyvoice2"
    input: str
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal["wav", "mp3", "opus", "flac"] = "wav"


CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


@router.post("/v1/audio/speech")
async def create_speech(
    req: SpeechRequest,
    # PR-5a:legacy verify_bearer_token(只认 1:1 key,M:N 实际 403)→ verify_bearer_token_any。
    # handler 不用 instance(只按 req.model 取 engine),M:N 有效 key 即可。
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
):
    """Generate audio from text (OpenAI TTS compatible)."""
    from src.workers.tts_engines import registry

    engine = registry._ENGINE_INSTANCES.get(req.model)
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            409,
            detail=f"Model '{req.model}' is not loaded. Load it first via POST /api/v1/engines/{req.model}/load",
        )

    try:
        # to_thread:同步阻塞 CUDA 调用,直接 await 卡死事件循环(round2 低)。
        result = await asyncio.to_thread(
            engine.synthesize,
            text=req.input,
            voice=req.voice,
            speed=req.speed,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    audio_bytes = result.audio_bytes

    # Format conversion if needed (engine returns wav by default)
    if req.response_format != "wav" and req.response_format != result.format:
        try:
            audio_bytes = _convert_audio(audio_bytes, result.format, req.response_format, result.sample_rate)
        except Exception:
            # If conversion fails, return wav
            pass

    content_type = CONTENT_TYPE_MAP.get(req.response_format, "audio/wav")
    return Response(content=audio_bytes, media_type=content_type)


def _convert_audio(audio_bytes: bytes, src_fmt: str, dst_fmt: str, sample_rate: int) -> bytes:
    """Convert audio format using soundfile."""
    import soundfile as sf

    buf_in = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf_in, dtype="float32")

    buf_out = io.BytesIO()
    fmt_map = {"wav": "WAV", "flac": "FLAC", "opus": "OGG"}
    sf_fmt = fmt_map.get(dst_fmt)
    if sf_fmt is None:
        raise ValueError(f"Unsupported output format: {dst_fmt}")

    sf.write(buf_out, data, sr, format=sf_fmt)
    buf_out.seek(0)
    return buf_out.read()


# --- /v1/embeddings ---


@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
    session: AsyncSession = Depends(get_async_session),
):
    """OpenAI 兼容 embeddings(2026-06-12 embedding 模态接入)。

    解析链与 chat/completions 同款:body.model → 服务(M:N grant / legacy 1:1)→
    source_type=model → vLLM 子进程(models.yaml `vllm_runner: pooling` 起的
    pooling 实例,如 qwen3_embedding_4b/8b)→ 透传 `/v1/embeddings`。
    body.model 置空让 vLLM 用自己的 served 模型(同 chat 的处理);input/
    encoding_format/dimensions 等字段原样透传。usage 计 prompt_tokens
    (embedding 无 completion)。
    """
    instance, api_key = auth

    body = await request.json()
    requested_model = body.get("model") or None

    if instance is None:
        try:
            instance = await resolve_target_service(
                session, api_key=api_key, requested_model=requested_model,
            )
        except ModelNotFound as e:
            raise NotFoundError(str(e), code="model_not_found")
        if instance.status != "active":
            raise HTTPException(403, detail="Instance is inactive")
        from src.api.deps_auth import enforce_instance_rate_limit
        await enforce_instance_rate_limit(instance)

    if instance.source_type != "model":
        raise HTTPException(
            400,
            detail=f"embeddings 只支持 model-backed 服务(got source_type={instance.source_type})",
        )

    engine_name = instance.source_name or str(instance.source_id)
    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise HTTPException(503, detail=str(e)) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e

    body["model"] = ""  # vLLM uses its own model path(同 chat)

    start_ms = time.monotonic()
    async with httpx.AsyncClient(timeout=120, proxy=None) as client:
        resp = await client.post(f"{base_url.rstrip('/')}/v1/embeddings", json=body)

    duration = int((time.monotonic() - start_ms) * 1000)
    if resp.status_code != 200:
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    data = resp.json()
    usage = data.get("usage", {}) or {}
    from src.services.usage_service import record_llm_usage
    await record_llm_usage(
        model=engine_name,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=0,
        duration_ms=duration,
        instance_id=instance.id,
        api_key_id=api_key.id,
        agent_id=None,
    )
    await _post_consume_quota(
        api_key.id, instance.id, usage.get("total_tokens", 0),
    )
    # 响应里的 model 回填服务名(对外契约:caller 看到自己请求的 model 名,不暴露本地路径)
    data["model"] = requested_model or engine_name
    return data


# --- /v1/audio/transcriptions ---


async def _ffmpeg_to_wav16k(raw: bytes) -> bytes:
    """任意音频 → 16kHz/单声道/PCM-s16le WAV。

    vLLM 的 ASR 端点先用 soundfile、回退 PyAV 解码上传音频;非常规格式(IEEE-Float
    WAV 等)易被拒。统一 ffmpeg 归一化成标准 PCM 再转发,兼容各种上传格式且确保稳解码。

    **输入/输出都用可 seek 的临时文件**(2026-06-21 真机踩,两类坑):
    - 输入若从不可 seek 的 pipe:0 读,moov-atom 在末尾的 m4a/mp4 等需 seek 的容器
      读不到音轨 → 输出空音频 → vLLM 处理器报 `audio=[array([], dtype=float32)]`。
    - 输出若写不可 seek 的 pipe:1,WAV 头的 RIFF/data size 回填不了(写 0xFFFFFFFF 占位)。
    临时文件可 seek,两者都解决。

    空音频(无有效音轨/无声/损坏)→ 直接报清晰 400,**不**把空数组转发给 vLLM
    (否则用户只看到晦涩的 Qwen3ASRProcessor empty-array 报错)。
    """
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as in_tf:
        in_tf.write(raw)
        in_path = in_tf.name
    out_path = in_path + ".wav"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", in_path, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", out_path,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            msg = err.decode("utf-8", "ignore")[:200] if err else "unknown"
            raise InvalidRequestError(f"音频解码失败(ffmpeg): {msg}")
        with open(out_path, "rb") as f:
            out = f.read()
        # WAV header(含 ffmpeg 的 LIST/INFO chunk)~80-120 字节;有任何可用音频都是 KB 级。
        # < 1KB 视为空(header-only)→ ffmpeg 读到了文件但解出 0 采样。
        if len(out) < 1024:
            raise InvalidRequestError("音频无有效音轨或为空(可能不是有效音频文件、无声、或格式损坏)")
        return out
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


async def _auth_transcriptions(
    request: Request,
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance | None, InstanceApiKey | None]:
    """转写端点 auth:Bearer 优先(走 key+grant+quota,外部调用);否则 admin session
    旁路(给 in-app Playground 用 —— 同 /v1/apps 的 _auth_apps_run,否则 cookie 调
    会被 verify_bearer_token_any 的必填 Authorization 卡 400)。(None,None)=admin。"""
    if authorization:
        return await verify_bearer_token_any(authorization, session)
    from src.api.admin_session import request_is_authed
    if request_is_authed(request):
        return None, None
    raise HTTPException(401, detail="Missing API key or admin session")


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str | None = Form(None),
    auth: tuple[ServiceInstance | None, InstanceApiKey | None] = Depends(_auth_transcriptions),
    session: AsyncSession = Depends(get_async_session),
):
    """OpenAI 兼容语音转写(ASR,2026-06-20 接入;spec asr-qwen3-integration)。

    multipart:`file`(音频)+ `model`(服务名)。解析链同 /v1/embeddings:model → 服务
    (M:N grant)→ source_type=model → vLLM ASR 子进程(Qwen3-ASR,models.yaml type:asr)
    → 透传 `/v1/audio/transcriptions`。上传音频先 ffmpeg 归一化(PR-0 spike:vLLM PyAV 拒
    非常规格式)。usage 按 vLLM 返回的 audio 秒数计量。Bearer 走 grant+quota;admin cookie
    (in-app Playground)按 model 名直查服务、跳 grant/quota。
    """
    instance, api_key = auth
    requested_model = model or None
    admin_run = api_key is None  # admin session 旁路(Playground)

    if admin_run:
        # admin:按服务名直查(单管理员隐式授权,跳 grant/quota),同 /v1/apps execute_service。
        from sqlalchemy import select
        instance = (
            await session.execute(
                select(ServiceInstance).where(ServiceInstance.name == requested_model)
            )
        ).scalar_one_or_none()
        if instance is None:
            raise NotFoundError(f"service '{requested_model}' not found", code="service_not_found")
    else:
        try:
            instance = await resolve_target_service(
                session, api_key=api_key, requested_model=requested_model,
            )
        except ModelNotFound as e:
            raise NotFoundError(str(e), code="model_not_found")
        if instance.status != "active":
            raise HTTPException(403, detail="Instance is inactive")
        from src.api.deps_auth import enforce_instance_rate_limit
        await enforce_instance_rate_limit(instance)

    if instance.source_type != "model":
        raise HTTPException(
            400,
            detail=f"transcriptions 只支持 model-backed 服务(got source_type={instance.source_type})",
        )

    engine_name = instance.source_name or str(instance.source_id)
    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise HTTPException(503, detail=str(e)) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e

    raw = await file.read()
    if not raw:
        raise InvalidRequestError("空音频文件")
    wav = await _ffmpeg_to_wav16k(raw)

    start_ms = time.monotonic()
    async with httpx.AsyncClient(timeout=300, proxy=None) as client:
        files = {"file": ("audio.wav", wav, "audio/wav")}
        form: dict[str, str] = {"model": ""}  # vLLM 用自己的 served 模型(同 chat/embeddings)
        if language:
            form["language"] = language
        if response_format:
            form["response_format"] = response_format
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/audio/transcriptions", files=files, data=form,
        )

    duration_ms = int((time.monotonic() - start_ms) * 1000)
    if resp.status_code != 200:
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    out = resp.json()
    usage = out.get("usage", {}) or {}
    audio_seconds = int(usage.get("seconds", 0)) if isinstance(usage, dict) else 0
    from src.services.usage_service import record_llm_usage
    await record_llm_usage(
        model=engine_name,
        prompt_tokens=0,
        completion_tokens=0,
        duration_ms=duration_ms,
        instance_id=instance.id,
        api_key_id=api_key.id if api_key else None,
        agent_id=None,
    )
    # 计量:按音频秒数扣(ASR 无 token 概念);至少 1。admin(Playground)跳过 grant/quota。
    if api_key is not None:
        await _post_consume_quota(api_key.id, instance.id, max(1, audio_seconds))
    return out


# --- /v1/images/generations ---


class ImageGenerationRequest(BaseModel):
    # extra="allow":火山式额外参数(如 SeedVR2 的 resolution)随 body 透传,
    # 按服务 exposed_inputs 的 key 通用合并注入(见 handler)。
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="已发布的 image 服务名(= ServiceInstance.name)")
    # prompt 可选:图生图/编辑有 prompt,但纯超分(SeedVR2 细节增强)无 prompt。
    prompt: str | None = None
    # 输入图(图生图/编辑/超分):base64 data URI('data:image/...;base64,...')。
    # 对齐火山 Seedream:image 字段接 URL 或 base64;本轮先吃 base64(URL 下载留 follow-up)。
    # 多图传 list(火山多参考),当前 image_input 节点只消费单图。
    image: str | list[str] | None = None
    n: int = 1
    # OpenAI 兼容字段。当前工作流用自身固定尺寸,size 暂作占位(不注入);
    # response_format 先只支持 url(b64_json 留待后续读图字节编码)。
    size: str | None = None
    response_format: Literal["url", "b64_json"] = "url"


def _pick_prompt_input_key(exposed_inputs: list | None) -> str | None:
    """选承接 prompt 的 exposed input key:优先 string/text 类型,否则第一个。"""
    if not exposed_inputs:
        return None
    for p in exposed_inputs:
        if str(p.get("type", "")).lower() in ("string", "text", "str"):
            return p.get("key") or p.get("api_name")
    first = exposed_inputs[0]
    return first.get("key") or first.get("api_name")


# 输入源节点:其 output 含 image_url 但那是上传图的回显(image_input executor 落盘签 URL),
# 不是生成结果。外部端点扫 result 捞图时必须跳过,否则把输入图误当输出返回(#372 的外部路径版)。
_INPUT_SOURCE_NODE_TYPES = {"image_input"}


def _input_source_node_ids(snapshot: dict | None) -> set[str]:
    """从 snapshot 找出输入源节点 id(api-shape dict / editor-shape list 都认)。"""
    if not isinstance(snapshot, dict):
        return set()
    ids: set[str] = set()
    nodes = snapshot.get("nodes")
    if isinstance(nodes, dict):
        for nid, n in nodes.items():
            t = (n.get("class_type") or n.get("type")) if isinstance(n, dict) else None
            if t in _INPUT_SOURCE_NODE_TYPES:
                ids.add(str(nid))
    elif isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if (n.get("type") or n.get("class_type")) in _INPUT_SOURCE_NODE_TYPES:
                ids.add(str(n.get("id")))
    return ids


def _extract_image_urls(
    result: dict,
    snapshot: dict | None = None,
    exposed_outputs: list | None = None,
) -> list[str]:
    """从 executor result 捞产图终端的 image_url。

    1) 优先按 exposed_outputs 声明的 node_id 取(发布契约把输出指向产图终端
       dec=flux2_vae_decode / up=seedvr2_upscale)——精确、不依赖遍历顺序。
    2) 兜底扫全部节点 output,但跳过 image_input 类型节点的 echo(否则把上传图
       当输出返回,#372 外部路径版)。snapshot 缺省时退化为「扫全部」,与老服务兼容。
    """
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    if not isinstance(outputs, dict):
        return []

    declared = [
        str(p.get("node_id")) for p in (exposed_outputs or [])
        if isinstance(p, dict) and p.get("node_id") is not None
    ]
    # batch(num_images>1):节点 output 带 image_urls 列表(全部 N 张);否则单 image_url。
    def _node_urls(node_out: object) -> list[str]:
        if not isinstance(node_out, dict):
            return []
        many = node_out.get("image_urls")
        if isinstance(many, list) and many:
            return [u for u in many if isinstance(u, str) and u]
        one = node_out.get("image_url")
        return [one] if isinstance(one, str) and one else []

    urls: list[str] = []
    for nid in declared:
        urls.extend(_node_urls(outputs.get(nid)))
    if urls:
        return urls

    input_ids = _input_source_node_ids(snapshot)
    for nid, node_out in outputs.items():
        if str(nid) in input_ids:
            continue
        urls.extend(_node_urls(node_out))
    return urls


@router.post("/v1/images/generations")
async def images_generations(
    body: ImageGenerationRequest,
    request: Request,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
    session: AsyncSession = Depends(get_async_session),
):
    """OpenAI / 火山(Ark)兼容的图像生成端点。

    统一模态端点 + body.model 选模型 + API key grant 做 scope —— 跟
    /v1/chat/completions 同一套设计(对齐火山:不是每个出图工作流一个 URL
    路径,而是 model 参数指定服务 + key 的授权范围决定可访问哪些)。

    内部 dispatch:body.model = 已发布 image 工作流服务名;body 里命中服务
    exposed_inputs key 的字段(prompt/image/resolution...)通用合并注入(文生图
    =prompt;编辑/角度=image+prompt;超分=image+resolution 无 prompt);经共享
    执行核心 run_published_workflow(带 GPU runner_clients)跑出图;产图终端
    节点的 image_url 转成 OpenAI {data:[{url}]}。
    """
    from sqlalchemy import select
    from sqlalchemy.orm import undefer

    from src.models.api_gateway import ApiKeyGrant
    from src.services.workflow_service_runner import run_published_workflow

    _instance, api_key = auth
    if api_key is None:
        raise NotFoundError("request requires an API key", code="model_not_found")

    # resolve image 服务:(key 的 active grant, model == service name) —— 与
    # chat/completions 同款 M:N scope,这里直接 join + undefer 工作流快照。
    stmt = (
        select(ServiceInstance)
        .options(
            undefer(ServiceInstance.workflow_snapshot),
            undefer(ServiceInstance.exposed_inputs),
            undefer(ServiceInstance.exposed_outputs),
        )
        .join(ApiKeyGrant, ApiKeyGrant.service_id == ServiceInstance.id)
        .where(
            ApiKeyGrant.api_key_id == api_key.id,
            ApiKeyGrant.status == "active",
            ServiceInstance.name == body.model,
        )
    )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise NotFoundError(
            f"no active grant for model '{body.model}' on this key",
            code="model_not_found",
        )

    # 通用参数合并(火山式):body 里任意字段命中服务 exposed_inputs 的 key → 注入对应
    # 节点。prompt / image / resolution / negative_prompt 走同一套,SeedVR2 无 prompt 也
    # 不报错。这取代了原「只塞单个文本 prompt」的逻辑(那条让带图/无 prompt 的服务发不出去)。
    exposed = svc.exposed_inputs or []
    exposed_keys = {(p.get("key") or p.get("api_name")) for p in exposed}
    exposed_keys.discard(None)
    body_fields = body.model_dump(exclude_none=True)  # 含 extra 透传字段(resolution 等)
    inputs: dict = {k: body_fields[k] for k in exposed_keys if k in body_fields}

    # OpenAI 兼容兜底:服务的文本输入 key 不字面叫 'prompt' 时,把 body.prompt 注进文本输入。
    if body.prompt is not None:
        prompt_key = _pick_prompt_input_key(exposed)
        if prompt_key and prompt_key not in inputs:
            inputs[prompt_key] = body.prompt

    if not inputs:
        raise InvalidRequestError(
            f"service '{body.model}' received no inputs matching its exposed schema "
            f"(exposed keys: {sorted(k for k in exposed_keys)})",
            code="no_matching_input",
        )

    # 在执行前抓出 snapshot / exposed_outputs —— run_published_workflow 内部多次 commit
    # 会 expire ORM 属性,事后再访问会触发 lazy 重载 → MissingGreenlet。
    snapshot = svc.workflow_snapshot
    out_params = svc.exposed_outputs

    # OpenAI n:一次出 N 张 —— 注入到喂输出的末段采样节点 num_images(batch,B1 全栈)。
    # 段路(非 euler 采样器手写循环)暂只 1 张(引擎层 follow-up),其余路径真出 N 张。
    result = await run_published_workflow(
        request, session, svc, inputs, api_key, num_images=max(1, int(body.n)),
    )

    urls = _extract_image_urls(result, snapshot, out_params)
    if not urls:
        raise APIError(
            f"service '{body.model}' did not produce an image",
            code="no_image_output",
        )

    base = str(request.base_url).rstrip("/")
    abs_urls = [u if u.startswith("http") else base + u for u in urls]
    # 真出了 N 张就返 N 张;截到 n 作上限保护(避免工作流意外多产)。
    data = [{"url": u} for u in abs_urls[: max(1, int(body.n))]]
    return {"created": int(time.time()), "data": data}


# --- /v1/models ---

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "nous-center"
    type: str = "model"   # 服务类目:llm / embedding / image / app / tts / vl ...


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


async def _granted_services(session: AsyncSession, api_key: InstanceApiKey):
    """该 key active-grant 的全部服务(ServiceInstance),按类目+名排序 —— 与
    /v1/chat·/v1/embeddings·/v1/images 同款 M:N scope。"""
    from sqlalchemy import select  # noqa: PLC0415

    from src.models.api_gateway import ApiKeyGrant  # noqa: PLC0415

    rows = await session.execute(
        select(ServiceInstance)
        .join(ApiKeyGrant, ApiKeyGrant.service_id == ServiceInstance.id)
        .where(
            ApiKeyGrant.api_key_id == api_key.id,
            ApiKeyGrant.status == "active",
        )
        .order_by(ServiceInstance.category, ServiceInstance.name)
    )
    return rows.scalars().all()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    type: str | None = None,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
    session: AsyncSession = Depends(get_async_session),
):
    """List the services THIS key can call (OpenAI 兼容发现端点)。

    返回该 key **active-grant 的全部服务**(LLM / embedding / 图像工作流 / app / TTS…),
    `id` = **服务名**(与 /v1/chat·/v1/embeddings·/v1/images 的 `model` 字段完全一致),
    `type` = 类目(客户端据此选端点)。即「发现到的 == 能调的」—— 对齐 Doubao 式
    一链多服务自选。可选 `?type=llm` 过滤类目。
    """
    _instance, api_key = auth
    if api_key is None:
        raise NotFoundError("request requires an API key", code="model_not_found")
    services = await _granted_services(session, api_key)
    data = [
        ModelObject(id=s.name, type=(s.category or "model"))
        for s in services
        if not type or (s.category or "model") == type
    ]
    return ModelListResponse(data=data)


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
    session: AsyncSession = Depends(get_async_session),
):
    """Get one service the key can call (OpenAI 兼容)。按服务名查,未授权 → 404。"""
    _instance, api_key = auth
    if api_key is None:
        raise NotFoundError("request requires an API key", code="model_not_found")
    svc = next((s for s in await _granted_services(session, api_key) if s.name == model_id), None)
    if svc is None:
        raise NotFoundError(
            f"model '{model_id}' not found or no active grant on this key",
            code="model_not_found")
    return ModelObject(id=svc.name, type=svc.category or "model")
