"""DreaminaProvider — 即梦 dreamina CLI 转发(移植 Infinite-Canvas/main.py 的 run_jimeng_cli
+ 提交/轮询/收图逻辑,Node 无关,纯 Python)。

CLI 形态(来自 dreamina text2image/image2image -h):
  dreamina text2image  --prompt=.. --ratio=16:9 --resolution_type=2k --poll=N [--model_version=4.0]
  dreamina image2image --images=PATH --prompt=.. --resolution_type=2k --poll=N [--model_version=4.0]
  dreamina query_result --submit_id=.. --download_dir=DIR
  dreamina --version | login | user_credit

CLI 自带 --poll 轮询并把产物下载到 --download_dir;返回的 JSON 里含本地路径/URL。
本 provider 把下载到的本地文件作为 ArtifactRef 返回(转签名 URL 是节点的事)。
"""
from __future__ import annotations

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

_MEDIA_EXT_RE = re.compile(r"\.(png|jpe?g|webp|gif|bmp|mp4|webm|mov|m4v|avi|mkv)(\?|#|$)", re.IGNORECASE)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# 官方 dreamina 支持的图片模型(来自 text2image/image2image -h)。image2image 不支持 3.0/3.1。
_TEXT2IMAGE_MODELS = {"3.0", "3.1", "4.0", "4.1", "4.5", "4.6", "5.0"}
_IMAGE2IMAGE_MODELS = {"4.0", "4.1", "4.5", "4.6", "5.0"}
_DEFAULT_MODEL = "4.0"

_RATIO_CHOICES = [(21, 9), (16, 9), (3, 2), (4, 3), (1, 1), (3, 4), (2, 3), (9, 16)]


def extract_json(text: str) -> dict:
    """从混杂输出里提取最像「结果」的 JSON 对象(移植 jimeng_extract_json)。"""
    text = str(text or "").strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    parsed: list[tuple[int, object]] = []
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if not text[:i].strip():
            return obj  # type: ignore[return-value]
        parsed.append((i, obj))

    def score(item: tuple[int, object]) -> int:
        _idx, obj = item
        if not isinstance(obj, dict):
            return 1
        keys = {str(k).lower() for k in obj}
        weight = 0
        for key in ("submit_id", "gen_status", "result_json", "images", "videos", "data", "total_credit"):
            if key in keys:
                weight += 10
        return weight

    return max(parsed, key=score)[1] if parsed else {"text": text}  # type: ignore[return-value]


def submit_id_of(raw: object) -> str:
    found: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"submit_id", "submitid", "task_id", "taskid"} and item:
                    found.append(str(item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(raw)
    return found[0] if found else ""


def failure_reason(raw: object) -> str:
    found: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            status = str(value.get("gen_status") or value.get("status") or "").strip().lower()
            reason = (
                value.get("fail_reason")
                or value.get("failReason")
                or value.get("error")
                or value.get("message")
                or value.get("msg")
            )
            if reason and (status in {"fail", "failed", "error"} or "fail" in str(reason).lower()):
                found.append(str(reason))
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(raw)
    return found[0] if found else ""


def collect_media_values(value: object, outputs: list[str]) -> None:
    """递归收集媒体路径/URL(移植 jimeng_collect_media_values)。"""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        if text.startswith(("http://", "https://", "file://", "/")) or _MEDIA_EXT_RE.search(text):
            outputs.append(text)
        return
    if isinstance(value, list):
        for item in value:
            collect_media_values(item, outputs)
        return
    if isinstance(value, dict):
        for key in (
            "url", "urls", "image", "images", "image_url", "image_urls",
            "video", "videos", "video_url", "video_urls", "output", "outputs",
            "result", "results", "file", "files", "path", "paths",
            "download_url", "download_urls", "downloadUrl", "file_path", "filePath",
        ):
            if key in value:
                collect_media_values(value.get(key), outputs)
        for item in value.values():
            if isinstance(item, (dict, list)):
                collect_media_values(item, outputs)


def output_values(raw: object) -> list[str]:
    outputs: list[str] = []
    collect_media_values(raw, outputs)
    deduped: list[str] = []
    for value in outputs:
        if value not in deduped:
            deduped.append(value)
    return deduped


def ratio_from_size(width: int, height: int, fallback: str = "1:1") -> str:
    if not width or not height:
        return fallback
    ratio = width / max(1, height)
    left, right = min(_RATIO_CHOICES, key=lambda item: abs(ratio - item[0] / item[1]))
    return f"{left}:{right}"


def normalize_model(model: str | None) -> str:
    match = re.search(r"(\d+\.\d+)", str(model or ""))
    return match.group(1) if match else ""


def image_model_version(model: str | None, mode: str = "text2image") -> str:
    version = normalize_model(model) or _DEFAULT_MODEL
    allowed = _IMAGE2IMAGE_MODELS if mode == "image2image" else _TEXT2IMAGE_MODELS
    return version if version in allowed else _DEFAULT_MODEL


def image_resolution(model: str | None, width: int, height: int, mode: str = "text2image") -> str:
    text = str(model or "").lower()
    if "4k" in text:
        desired = "4k"
    elif "1k" in text:
        desired = "1k"
    elif "2k" in text:
        desired = "2k"
    else:
        desired = "4k" if max(width, height) > 2048 else "2k"
    version = normalize_model(model)
    if mode == "image2image":
        return "4k" if desired == "4k" else "2k"
    if version in ("3.0", "3.1"):
        return "1k" if desired == "1k" else "2k"
    return "4k" if desired == "4k" else "2k"


class DreaminaProvider(ExternalCliProvider):
    name = "dreamina"
    default_executable = "dreamina"
    modalities = {"image"}

    def __init__(self, *, executable: str = "", poll_seconds: int = 5, cwd: str | None = None) -> None:
        super().__init__(executable=executable, cwd=cwd)
        self.poll_seconds = max(1, min(3600, int(poll_seconds)))

    # ---- 状态 / 登录 -----------------------------------------------------

    async def probe_status(self) -> ProviderStatus:
        status = ProviderStatus(name=self.name, modalities=sorted(self.modalities))
        if not self.is_installed():
            status.message = "未找到 dreamina CLI。安装:curl -fsSL https://jimeng.jianying.com/cli | bash"
            return status
        code, out, err = await self._run(["--version"], timeout=15)
        if code != 0:
            status.message = (err or out or "dreamina --version 失败").strip()[:500]
            return status
        status.available = True
        version_match = re.search(r"(\d+\.\d+\.\d+)", f"{out} {err}")
        status.version = version_match.group(1) if version_match else ""
        # user_credit 成功 = 已登录;额度文本回填 quota。
        credit_code, credit_out, credit_err = await self._run(["user_credit"], timeout=20)
        if credit_code == 0:
            status.logged_in = True
            status.quota = (credit_out or "").strip()[:200] or None
            status.message = "dreamina 可用且已登录"
        else:
            status.message = (credit_err or credit_out or "dreamina 已安装但未登录,请运行 dreamina login").strip()[:500]
        return status

    async def login_start(self) -> dict:
        """触发 dreamina login。登录是交互扫码,这里 best-effort 捕获初始输出里的二维码/URL。

        说明:headless 后端上更推荐用户在终端自行 `dreamina login` 扫码;此端点供前端面板
        展示引导。完整的后台登录会话管理留 PR-4。
        """
        if not self.is_installed():
            raise ProviderError(
                "未找到 dreamina CLI。安装:curl -fsSL https://jimeng.jianying.com/cli | bash",
                status_code=400,
            )
        return {
            "started": False,
            "message": "请在本机终端运行 `dreamina login` 扫码登录(后台登录会话管理待 PR-4)。",
            "executable": self.resolve_executable(),
        }

    # ---- 生成 ------------------------------------------------------------

    async def generate(self, req: ExternalGenRequest) -> ExternalGenResult:
        started = time.monotonic()
        download_dir = Path(tempfile.mkdtemp(prefix="dreamina_"))
        timeout = req.timeout_s or (self.poll_seconds + 120)
        if req.input_images:
            mode = "image2image"
            args = [
                "image2image",
                f"--images={req.input_images[0]}",
                f"--prompt={req.prompt}",
                f"--resolution_type={image_resolution(req.model, req.width, req.height, mode)}",
                f"--poll={self.poll_seconds}",
                f"--download_dir={download_dir}",
                f"--model_version={image_model_version(req.model, mode)}",
            ]
        else:
            mode = "text2image"
            args = [
                "text2image",
                f"--prompt={req.prompt}",
                f"--ratio={ratio_from_size(req.width, req.height)}",
                f"--resolution_type={image_resolution(req.model, req.width, req.height, mode)}",
                f"--poll={self.poll_seconds}",
                f"--download_dir={download_dir}",
                f"--model_version={image_model_version(req.model, mode)}",
            ]
        code, out, err = await self._run(args, timeout=timeout)
        if code != 0:
            raise ProviderError(f"即梦 CLI 调用失败:{(err or out or f'exit={code}')[:800]}")
        raw = extract_json(f"{out}\n{err}".strip())
        artifacts = self._collect_artifacts(raw, download_dir, started)
        if not artifacts:
            # 提交成功但本轮没下到媒体:尝试 query_result 兜底一次。
            submit_id = submit_id_of(raw)
            failure = failure_reason(raw)
            if failure:
                raise ProviderError(f"即梦生成失败:{failure}")
            if submit_id:
                q_code, q_out, q_err = await self._run(
                    [
                        "query_result",
                        f"--submit_id={submit_id}",
                        f"--download_dir={download_dir}",
                    ],
                    timeout=min(300, self.poll_seconds + 60),
                )
                if q_code == 0:
                    artifacts = self._collect_artifacts(
                        extract_json(f"{q_out}\n{q_err}".strip()), download_dir, started
                    )
            if not artifacts:
                raise ProviderError(
                    f"即梦 CLI 未返回可用媒体结果(submit_id={submit_id or '?'})", status_code=502
                )
        return ExternalGenResult(
            artifacts=artifacts,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def _collect_artifacts(self, raw: object, download_dir: Path, started_at: float) -> list[ArtifactRef]:
        """先从 JSON 输出值里取本地文件,再扫 download_dir 兜底(CLI 自动下载落点)。"""
        seen: set[str] = set()
        out: list[ArtifactRef] = []
        for value in output_values(raw):
            path = Path(value)
            if path.is_file():
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(ArtifactRef(kind="image", local_path=resolved, title=path.name))
        if download_dir.is_dir():
            for path in sorted(download_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                out.append(ArtifactRef(kind="image", local_path=resolved, title=path.name))
        return out
