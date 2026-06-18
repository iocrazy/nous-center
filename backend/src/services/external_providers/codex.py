"""CodexProvider — OpenAI codex CLI 转发(图像模态)。

事件解析对齐 modern codex `exec --json`(参考 paperclip codex-local/parse.ts):
  thread.started → thread_id;item.completed(item.type=agent_message)→ text;
  turn.completed → usage;error / turn.failed → 错误。
图像产物提取沿用「workspace 扫描 + agent_message 里的 markdown/路径」(参考 T8 codexCliRunner)
—— paperclip 把 codex 当编码 agent 不收图,我们收图,所以这块是自己的。

**重要**:codex 的 image_generation feature 当前可能是 under-development(被 gate)。本 provider
按 `codex features list` 探测,可用才 `--enable image_generation`;不可用则 codex 只会吐文本/提示词,
generate 会明确报错(退回提示词在 message 里),而不是假装出了图。鉴权走 ~/.codex(ChatGPT 登录),
封号护栏由 governor 负责。
"""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from pathlib import Path

from src.services.external_providers.base import (
    ArtifactRef,
    ExternalCliProvider,
    ExternalGenRequest,
    ExternalGenResult,
    ProviderError,
    ProviderStatus,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}
_IMAGE_URL_RE = re.compile(r"\.(png|jpe?g|webp|gif|bmp|avif)(?:[?#].*)?$", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)>\s]+)>?\s*\)")
_LOOSE_PATH_RE = re.compile(
    r"(?:https?://[^\s\"'<>),，。；]+|/[^\s\"'<>),，。；]+\.(?:png|jpe?g|webp|gif|bmp|avif))",
    re.IGNORECASE,
)


def parse_feature_list(raw: str) -> list[dict]:
    """解析 `codex features list` 输出(每行 `name  stage  enabled`)。"""
    out: list[dict] = []
    for line in str(raw or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        enabled = parts[-1].strip().lower() == "true"
        stage = " ".join(parts[1:-1])
        if not re.match(r"^[a-z0-9_/-]+$", name, re.IGNORECASE):
            continue
        out.append({"name": name, "stage": stage, "enabled": enabled})
    return out


def available_feature_names(features: list[dict]) -> set[str]:
    return {f["name"] for f in features if f.get("enabled")}


def parse_codex_jsonl(stdout: str) -> dict:
    """对齐 paperclip parseCodexJsonl:取 session/final agent_message/usage/error。"""
    session_id = ""
    final_message = ""
    error_message = ""
    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type") or "")
        if etype == "thread.started":
            session_id = str(event.get("thread_id") or session_id)
        elif etype == "error":
            msg = str(event.get("message") or "").strip()
            if msg:
                error_message = msg
        elif etype == "item.completed":
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            if str(item.get("type") or "") == "agent_message":
                text = str(item.get("text") or "")
                if text:
                    final_message = text
        elif etype == "turn.failed":
            err = event.get("error") if isinstance(event.get("error"), dict) else {}
            msg = str(err.get("message") or "").strip()
            if msg:
                error_message = msg
    return {"session_id": session_id, "text": final_message.strip(), "error": error_message}


def extract_artifacts_from_text(text: str) -> list[str]:
    """从 agent_message 里抽图片链接/路径(markdown + 裸 URL/路径)。"""
    candidates: list[str] = []
    for m in _MD_LINK_RE.finditer(str(text or "")):
        candidates.append(m.group(1))
    for m in _LOOSE_PATH_RE.finditer(str(text or "")):
        candidates.append(m.group(0))
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        clean = c.strip().strip("<>").rstrip(".,;，。；")
        if _IMAGE_URL_RE.search(clean) and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def build_exec_args(
    *,
    images: list[str],
    available: set[str],
    image_generation: bool,
    reasoning_effort: str = "",
) -> list[str]:
    args = ["exec", "--json", "--skip-git-repo-check", "--sandbox", "workspace-write"]
    args += ["-c", 'approval_policy="never"']
    if reasoning_effort:
        safe = reasoning_effort.replace('"', '\\"')
        args += ["-c", f'model_reasoning_effort="{safe}"']
    # image_generation 仅在 CLI 真提供该 feature 时才开,否则别传(避免 unknown feature flag)。
    if image_generation and "image_generation" in available:
        args += ["--enable", "image_generation"]
    for img in images:
        if str(img).strip():
            args += ["-i", str(img)]
    args.append("-")  # prompt 走 stdin
    return args


def make_creator_prompt(req: ExternalGenRequest, image_generation_available: bool) -> str:
    parts = [
        "你是图像生成助手。基于下面的描述生成一张图片。",
    ]
    if req.input_images:
        parts.append(
            f"已附 {len(req.input_images)} 张参考图,生成时继承其主体/构图/风格,除非要求重绘。"
        )
    if image_generation_available:
        parts.append(
            "必须用 image_generation 工具直接生成图片文件,并在最终回复里给出该文件的本地路径或 "
            "Markdown 图片链接,方便收集产物。不要只输出提示词。"
        )
    else:
        parts.append(
            "若 image_generation 工具不可用,请明确说明工具不可用,并退回输出一段可投喂图像模型的完整英文提示词。"
        )
    if req.negative_prompt:
        parts.append(f"负面约束:{req.negative_prompt}")
    parts.append(f"图像描述:\n{req.prompt}")
    return "\n\n".join(parts)


def _unknown_feature_flag_error(message: str) -> bool:
    return bool(
        re.search(
            r"unknown feature flag|unrecognized feature|unsupported feature|unexpected argument.*--enable",
            str(message or ""),
            re.IGNORECASE,
        )
    )


class CodexProvider(ExternalCliProvider):
    name = "codex"
    default_executable = "codex"
    modalities = {"image"}

    def __init__(self, *, executable: str = "", reasoning_effort: str = "", cwd: str | None = None) -> None:
        super().__init__(executable=executable, cwd=cwd)
        self.reasoning_effort = reasoning_effort

    # ---- 状态 / 登录 -----------------------------------------------------

    async def _list_features(self, timeout: float = 12.0) -> list[dict]:
        code, out, _err = await self._run(["features", "list"], timeout=timeout)
        return parse_feature_list(out) if code == 0 else []

    async def probe_status(self) -> ProviderStatus:
        status = ProviderStatus(name=self.name, modalities=sorted(self.modalities))
        if not self.is_installed():
            status.message = "未找到 codex CLI(npm i -g @openai/codex)。"
            return status
        code, out, err = await self._run(["--version"], timeout=15)
        if code != 0:
            status.message = (err or out or "codex --version 失败").strip()[:500]
            return status
        status.available = True
        vm = re.search(r"(\d+\.\d+\.\d+)", f"{out} {err}")
        status.version = vm.group(1) if vm else ""
        l_code, l_out, l_err = await self._run(["login", "status"], timeout=15)
        login_text = (l_out or l_err or "").strip()
        status.logged_in = l_code == 0
        features = await self._list_features()
        avail = available_feature_names(features)
        img_ok = "image_generation" in avail
        status.quota = None
        if status.logged_in:
            status.message = (
                f"codex 可用且已登录({login_text[:80]})。"
                + ("image_generation 可用。" if img_ok else "image_generation 当前被 gate(under-development),暂出不了图。")
            )
        else:
            status.message = login_text or "codex 已安装但未登录,请运行 codex login。"
        return status

    async def login_start(self) -> dict:
        if not self.is_installed():
            raise ProviderError("未找到 codex CLI(npm i -g @openai/codex)。", status_code=400)
        # codex login 走浏览器 OAuth;detached 触发,完成后回面板刷新状态。
        try:
            proc = await asyncio.create_subprocess_exec(
                self.resolve_executable(),
                "login",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=self._cwd,
            )
        except OSError as exc:
            raise ProviderError(f"codex login 启动失败:{exc}", status_code=502) from exc
        return {
            "started": True,
            "pid": proc.pid,
            "message": "已触发 codex login(浏览器 OAuth);完成后回面板点刷新。",
            "executable": self.resolve_executable(),
        }

    # ---- 生成 ------------------------------------------------------------

    async def generate(self, req: ExternalGenRequest) -> ExternalGenResult:
        started = time.monotonic()
        features = await self._list_features()
        avail = available_feature_names(features)
        img_available = "image_generation" in avail
        workspace = Path(tempfile.mkdtemp(prefix="codex_"))
        prompt = make_creator_prompt(req, img_available)
        args = build_exec_args(
            images=req.input_images,
            available=avail,
            image_generation=True,
            reasoning_effort=self.reasoning_effort,
        )
        timeout = req.timeout_s or 600.0
        code, out, err = await self._run(
            args,
            timeout=timeout,
            cwd=str(workspace),
            stdin_data=prompt,
            env={"T8_CODEX_WORKSPACE": str(workspace)},
        )
        # 个别 CLI 版本对 --enable 不认 → 剥离重试一次。
        if code != 0 and _unknown_feature_flag_error(err) and "--enable" in args:
            retry = [a for i, a in enumerate(args) if a != "--enable" and args[i - 1] != "--enable"]
            code, out, err = await self._run(
                retry, timeout=timeout, cwd=str(workspace), stdin_data=prompt,
                env={"T8_CODEX_WORKSPACE": str(workspace)},
            )
        parsed = parse_codex_jsonl(out)
        if code != 0:
            raise ProviderError(
                f"codex CLI 调用失败:{(parsed['error'] or err or out or f'exit={code}')[:800]}"
            )
        artifacts = self._collect_artifacts(parsed["text"], workspace, started)
        if not artifacts:
            hint = (
                "codex 当前未提供 image_generation(under-development 被 gate),只产出了文本,未出图。"
                if not img_available
                else "codex 本轮未产出图片文件。"
            )
            raise ProviderError(f"{hint} 文本回复:{parsed['text'][:500]}", status_code=502)
        return ExternalGenResult(
            artifacts=artifacts,
            text=parsed["text"],
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def _collect_artifacts(self, text: str, workspace: Path, started_at: float) -> list[ArtifactRef]:
        seen: set[str] = set()
        out: list[ArtifactRef] = []
        # 1) agent_message 里的本地路径(workspace 内的)。
        for cand in extract_artifacts_from_text(text):
            p = Path(cand)
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                resolved = str(p.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(ArtifactRef(kind="image", local_path=resolved, title=p.name))
        # 2) 扫 workspace(codex 把生成文件写在 cwd)。
        if workspace.is_dir():
            for p in sorted(workspace.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
                    continue
                if p.name in {"config.json", "config.toml"}:
                    continue
                resolved = str(p.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                out.append(ArtifactRef(kind="image", local_path=resolved, title=p.name))
        return out
