"""真机 smoke:导入 H3 模板 → 异步 prediction → 校验 mp4 + 音轨。

前置:sidecar 已跑(enginectl status)、H3 INT8 权重已入 ComfyUI models、
GSP 缓解脚本已上机(infra/gpu/setup-gpu-mitigations.sh)。跑法:
  cd backend && uv run python tests/manual/smoke_comfy_h3.py \
      --base http://127.0.0.1:8000 --admin-token $ADMIN_TOKEN \
      --workflow /path/to/minimax-h3-t2v-api.json --mapping /path/to/mapping.json
workflow 取自 ComfyUI 模板库(Workflow → Export (API));mapping 为
exposed_params JSON(至少暴露 prompt 与时长,时长压到最短以省时——本脚本对
prediction 只发空 `input`,所以要在 exposed_params 里给这两个 key 直接写
`default`;`comfy_bridge.py::ComfyUIWorkflowNode.invoke` 是 `data.get(key,
m.get("default"))`,mapping 的 `default` 优先,只有 mapping 也没设
`default` 时才落回 workflow 的冻结原值)。

鉴权分两段(与 src/api/routes/predictions.py 的鉴权语义对齐 —— ADMIN_TOKEN 只认
`/api/v1/*` 管理端点;`/v1/services/*/predictions` 走 M:N `InstanceApiKey`,一旦带了
`Authorization` header 就不会走 admin-session 兜底):
  1. --admin-token 建模板 / 改映射 / 查 service id / 铸一把临时 InstanceApiKey(全走
     `/api/v1/*`)。
  2. 用这把临时 key 的 secret 去调 `/v1/services/{name}/predictions`(`/v1/*`,不是
     `/api/v1/*`)。跑完(非 --keep)删 key + 删模板/服务。

流程:POST /api/v1/comfy-templates → PUT /api/v1/comfy-templates/{id}/mapping →
GET /api/v1/services 找 service 数字 id → POST /api/v1/keys 铸临时 key 授权给该 service →
POST /v1/services/{name}/predictions(Prefer: respond-async)→ 202 → 轮询
GET /v1/predictions/{id}(每 5s,上限 --timeout,默认 NOUS_COMFY_TIMEOUT 或 14400s)→
succeeded 后下载首个 video_url → ffprobe 断言:至少一路 video 流 + 一路 audio 流 →
打印 PASS + 耗时 + 文件路径。任何断言失败 sys.exit(1);缺 ffprobe sys.exit(2)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

OUT_DIR = Path(__file__).parent / "_smoke_out"
POLL_INTERVAL_S = 5.0
TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


def _die(msg: str, code: int = 1) -> None:
    print(f"[smoke_comfy_h3] FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def _admin_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _load_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except OSError as e:
        _die(f"读文件失败 {path}: {e}")
    except json.JSONDecodeError as e:
        _die(f"JSON 解析失败 {path}: {e}")
    raise AssertionError("unreachable")  # 让类型检查器满意,_die 已 exit


def _load_mapping(path: str) -> dict:
    data = _load_json(path)
    if isinstance(data, list):
        return {"exposed_params": data}
    if "exposed_params" not in data:
        _die(f"mapping 文件缺 exposed_params 字段: {path}")
    return data


def register_template(client: httpx.Client, headers: dict, name: str, workflow: dict) -> tuple[int, str]:
    r = client.post("/api/v1/comfy-templates", json={"name": name, "workflow": workflow}, headers=headers)
    if r.status_code != 201:
        _die(f"注册模板失败 [{r.status_code}]: {r.text}")
    body = r.json()
    print(f"[smoke_comfy_h3] 模板已注册 id={body['id']} service_name={body['service_name']} "
          f"node_count={body['node_count']}")
    return int(body["id"]), body["service_name"]


def put_mapping(client: httpx.Client, headers: dict, template_id: int, mapping: dict) -> None:
    r = client.put(f"/api/v1/comfy-templates/{template_id}/mapping", json=mapping, headers=headers)
    if r.status_code != 200:
        _die(f"设置 mapping 失败 [{r.status_code}]: {r.text}")
    print(f"[smoke_comfy_h3] mapping 已设置,{len(mapping['exposed_params'])} 个 exposed_params")


def find_service_id(client: httpx.Client, headers: dict, name: str) -> int:
    r = client.get("/api/v1/services", headers=headers)
    if r.status_code != 200:
        _die(f"查 services 列表失败 [{r.status_code}]: {r.text}")
    for svc in r.json():
        if svc.get("name") == name:
            return int(svc["id"])
    _die(f"services 列表里没找到刚注册的 {name!r}")
    raise AssertionError("unreachable")


def mint_key(client: httpx.Client, headers: dict, service_id: int, label: str) -> tuple[int, str]:
    r = client.post("/api/v1/keys", json={"label": label, "service_ids": [service_id]}, headers=headers)
    if r.status_code != 201:
        _die(f"铸临时 API key 失败 [{r.status_code}]: {r.text}")
    body = r.json()
    print(f"[smoke_comfy_h3] 临时 key 已铸 id={body['id']}")
    return int(body["id"]), body["secret"]


def delete_key(client: httpx.Client, headers: dict, key_id: int) -> None:
    # cleanup 路径:吞掉一切异常(包括 httpx 传输层的 ConnectError/TimeoutException),
    # 不然一个清理步骤炸了会连累 finally 里排在它后面的清理步骤(以及 client.close())
    # 都跑不到——恰好败在它本该防的场景上。
    try:
        r = client.delete(f"/api/v1/keys/{key_id}", headers=headers)
        if r.status_code not in (204, 404):
            print(f"[smoke_comfy_h3] 警告:删除临时 key 失败 [{r.status_code}]: {r.text}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_comfy_h3] 警告:删除临时 key 时异常: {e}", file=sys.stderr)


def delete_template(client: httpx.Client, headers: dict, template_id: int) -> None:
    try:
        r = client.delete(f"/api/v1/comfy-templates/{template_id}", headers=headers)
        if r.status_code not in (204, 404):
            print(f"[smoke_comfy_h3] 警告:删除模板失败 [{r.status_code}]: {r.text}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_comfy_h3] 警告:删除模板时异常: {e}", file=sys.stderr)


def create_prediction(client: httpx.Client, secret: str, service_name: str) -> str:
    r = client.post(
        f"/v1/services/{service_name}/predictions", json={"input": {}},
        headers={"Authorization": f"Bearer {secret}", "Prefer": "respond-async"})
    if r.status_code != 202:
        _die(f"创建 prediction 失败 [{r.status_code}]: {r.text}")
    body = r.json()
    print(f"[smoke_comfy_h3] prediction 已提交 id={body['id']} status={body['status']}")
    return str(body["id"])


def poll_prediction(client: httpx.Client, secret: str, pred_id: str, timeout_s: float) -> dict:
    headers = {"Authorization": f"Bearer {secret}"}
    deadline = time.monotonic() + timeout_s
    t0 = time.monotonic()
    body: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/v1/predictions/{pred_id}", headers=headers)
        if r.status_code != 200:
            _die(f"轮询 prediction 失败 [{r.status_code}]: {r.text}")
        body = r.json()
        elapsed = time.monotonic() - t0
        print(f"[smoke_comfy_h3] t+{elapsed:6.1f}s status={body['status']}")
        if body["status"] in TERMINAL_STATUSES:
            return body
        time.sleep(POLL_INTERVAL_S)
    _die(f"prediction {pred_id} 在 {timeout_s:.0f}s 内未到终态,last={body}")
    raise AssertionError("unreachable")


def extract_video_url(output: dict | None) -> str | None:
    """`output` 形态 = `{"outputs": {node_id: {video_url, items, thumbnails, seed}, ...}}`
    (`src/services/nodes/comfy_bridge.py` 每个节点结果的信封)。取第一个非空 video_url。"""
    for node_result in (output or {}).get("outputs", {}).values():
        if isinstance(node_result, dict) and node_result.get("video_url"):
            return node_result["video_url"]
    return None


def download_video(client: httpx.Client, base: str, video_url: str, dest: Path) -> None:
    full = video_url if video_url.startswith("http") else base.rstrip("/") + video_url
    # 独立 client:签名 URL 已自带鉴权(HMAC token+expires),不需要 admin/service 头。
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(30.0, read=120.0)) as dl:
        r = dl.get(full)
        if r.status_code != 200:
            _die(f"下载视频失败 [{r.status_code}] {full}: {r.text[:200]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
    print(f"[smoke_comfy_h3] 视频已下载 → {dest} ({dest.stat().st_size} bytes)")


def ffprobe_check(path: Path) -> None:
    if shutil.which("ffprobe") is None:
        _die("系统缺 ffprobe(装 ffmpeg 套件)", code=2)
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        _die("ffprobe 超时(30s)")
    if r.returncode != 0:
        _die(f"ffprobe 失败: {r.stderr.strip()}")
    try:
        streams = json.loads(r.stdout).get("streams", [])
    except json.JSONDecodeError as e:
        _die(f"ffprobe 输出解析失败: {e}")
        return
    kinds = {s.get("codec_type") for s in streams}
    if "video" not in kinds:
        _die(f"产物无 video 流,streams codec_type={kinds}")
    if "audio" not in kinds:
        _die(f"产物无 audio 流,streams codec_type={kinds}")
    print(f"[smoke_comfy_h3] ffprobe 校验通过:{kinds}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--admin-token", default=None, help="缺省读环境变量 ADMIN_TOKEN")
    ap.add_argument("--workflow", required=True, help="ComfyUI API-format workflow JSON 路径")
    ap.add_argument("--mapping", required=True, help="exposed_params JSON 路径")
    ap.add_argument("--name", default=None, help="缺省 smoke-comfy-h3-<timestamp>,避免 409")
    ap.add_argument("--timeout", type=float, default=None,
                     help="轮询上限秒数,缺省读 NOUS_COMFY_TIMEOUT 环境变量或 14400")
    ap.add_argument("--keep", action="store_true", help="跑完不删模板/服务/临时 key")
    args = ap.parse_args()

    admin_token = args.admin_token or os.environ.get("ADMIN_TOKEN") or None
    timeout_s = args.timeout if args.timeout is not None else float(os.environ.get("NOUS_COMFY_TIMEOUT", "14400"))
    name = args.name or f"smoke-comfy-h3-{int(time.time())}"

    workflow = _load_json(args.workflow)
    mapping = _load_mapping(args.mapping)

    headers = _admin_headers(admin_token)
    client = httpx.Client(base_url=args.base, trust_env=False, timeout=httpx.Timeout(15.0, read=30.0))

    template_id: int | None = None
    key_id: int | None = None
    t_start = time.monotonic()
    try:
        template_id, service_name = register_template(client, headers, name, workflow)
        put_mapping(client, headers, template_id, mapping)
        service_id = find_service_id(client, headers, service_name)
        key_id, secret = mint_key(client, headers, service_id, label=f"smoke-{name}")

        pred_id = create_prediction(client, secret, service_name)
        final = poll_prediction(client, secret, pred_id, timeout_s)

        if final["status"] != "succeeded":
            _die(f"prediction 未成功:status={final['status']} error={final.get('error')}")

        video_url = extract_video_url(final.get("output"))
        if not video_url:
            _die(f"succeeded 但没找到 video_url,output={final.get('output')}")

        dest = OUT_DIR / f"{name}.mp4"
        download_video(client, args.base, video_url, dest)
        ffprobe_check(dest)

        elapsed = time.monotonic() - t_start
        print(f"[smoke_comfy_h3] PASS 耗时 {elapsed:.1f}s → {dest}")
    finally:
        if not args.keep:
            if key_id is not None:
                delete_key(client, headers, key_id)
            if template_id is not None:
                delete_template(client, headers, template_id)
        else:
            print(f"[smoke_comfy_h3] --keep:保留 template_id={template_id} key_id={key_id}")
        try:
            client.close()
        except Exception as e:  # noqa: BLE001
            print(f"[smoke_comfy_h3] 警告:关闭 client 时异常: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
