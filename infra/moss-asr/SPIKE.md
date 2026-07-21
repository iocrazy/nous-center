# PR-0 Spike — SGLang Omni serving MOSS-Transcribe-Diarize (真机验证)

- Date: 2026-07-20
- Machine: cu130 生态 / driver 595.71.05 / 3 GPUs (2× RTX 3090, 1× RTX PRO 6000 Blackwell)
- GPU used: **idle 3090, index0, `GPU-2fd7c91c-af39-7b02-66b9-988331ce3bd7`** (Pro 6000 绝不碰)
- Venv: `infra/moss-asr/.venv` (uv, py3.12.13) — **完全隔离,未碰 `backend/.venv` / `backend/pyproject.toml`**

## 判定:PASS ✅

三件事全部拿到硬证据:

1. **装得起跑得起** — sglang-omni 在本机 cu130 装通(9.8GB,565 包),`sgl-omni serve` 成功起 MOSS,`/v1/models` 返回 200,uvicorn on `127.0.0.1:8003`。**但需要 CUDA 工具链对齐(见 §5 坑),README 的裸装法在本机起不来。**
2. **verbose_json 真回说话人+时间戳分段** — 短 wav 单段 `[S01]`;7.7 分钟双人播客 **53 段**、**S01/S02 两个说话人**(S01=主持人 21 段 / S02=嘉宾 32 段),时间戳单调、末端≈时长。
3. **响应 JSON 结构 + 显存/启动实测记录完整**(见 §3、§4)。

---

## 1. 完整 serve 命令(可复现)

```bash
VENV=infra/moss-asr/.venv
CU=$VENV/lib/python3.12/site-packages/nvidia/cu13   # CUDA_HOME 指向 venv 内 cu13
MODEL=/media/heygo/program/models/nous/speech/MOSS-Transcribe-Diarize

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-2fd7c91c-af39-7b02-66b9-988331ce3bd7 \
CUDA_HOME=$CU \
PATH=$VENV/bin:$CU/bin:$PATH \                       # .venv/bin 上 PATH → ninja;$CU/bin → nvcc
LD_LIBRARY_PATH=$CU/lib:$LD_LIBRARY_PATH \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \            # 模型本地,避免代理 stall
NO_PROXY=127.0.0.1,localhost \
$VENV/bin/sgl-omni serve \
  --model-path $MODEL \
  --host 127.0.0.1 --port 8003 \
  --mem-fraction-static 0.25
```

- MOSS 是**单 stage `asr` pipeline**(无 talker),`--text-only`/`--colocate` 不适用,不要加。
- 架构 `MossTranscribeDiarizeForConditionalGeneration` 被 sglang-omni **原生注册**,无需 `--trust-remote-code`。
- 脚本形式见 `start_spike_serve.sh`(spike 用,--config YAML 注入 server_args_overrides)/
  `start_serve.sh`(PR-1 生产件)。

## 2. curl 验证命令

```bash
# verbose_json(要分段用这个)
curl -s -X POST http://127.0.0.1:8003/v1/audio/transcriptions \
  -F "model=$MODEL" -F "file=@audio.wav" -F "response_format=verbose_json"

# 长音频:给 max_new_tokens,但必须 ≤ KV 容量(见 §5.5)
curl ... -F "response_format=verbose_json" -F "max_new_tokens=16384"

# json 对照(只回纯文本)
curl ... -F "response_format=json"
```

## 3. 响应 JSON 精确结构(PR-2 按此解析)

### verbose_json(短 wav 全文,原始响应)
```json
{"task":"transcribe","duration":3.48,
 "text":"[0.26][S01]希望你以后能够做得比我还好哟。[3.24]",
 "segments":[{"id":0,"start":0.26,"end":3.24,"text":"[S01]希望你以后能够做得比我还好哟。"}],
 "usage":{"type":"duration","seconds":4}}
```

### verbose_json(播客,字段结构 + 前后段)
```json
{"task":"transcribe","duration":460.2,
 "text":"...(完整 [start][Sxx]text[end] markup)...",
 "segments":[
   {"id":0,"start":0.29,"end":0.99,"text":"[S01]哈喽大家好，"},
   {"id":1,"start":0.99,"end":6.28,"text":"[S01]欢迎收听我们的播客啊，今天咱们来聊一聊尽量去做人生当中回报率高的事情啊，"},
   {"id":3,"start":15.72,"end":18.0,"text":"[S02]嗯，好的，那我们就直接开始今天的主题吧。"},
   ...
   {"id":52,"start":454.14,"end":454.44,"text":"[S02]拜拜。"}
 ],
 "usage":{"type":"duration","seconds":461}}
```

### 字段清单 + PR-2 注意点

| 字段 | 类型 | 说明 |
|---|---|---|
| `task` | str | 恒为 `"transcribe"` |
| `duration` | float | 音频秒数(round 2) |
| `text` | str | **原始 markup** `[start][Sxx]正文[end]…`,拼接全段 |
| `segments[]` | list | diar 分段 |
| `segments[].id` | int | 0-based 序号 |
| `segments[].start`/`.end` | float | 秒,round 2,单调 |
| `segments[].text` | str | **`"[Sxx]正文"`** — 说话人是 text 前缀 |
| `usage` | obj | `{"type":"duration","seconds":int}` |

- ⚠️ **没有独立 `speaker` 字段** — 说话人标签 `[S01]/[S02]` 塞在 `segments[].text` 前缀里。PR-2 要用正则 `^\[(S\d+)\]` 从 text 抠说话人 + 去前缀取正文。(源码 `serve/transcription_adapters/moss_transcribe_diarize.py` + `TranscriptionSegment` in `serve/protocol.py`)
- ⚠️ **没有 `language` 字段** — 实测响应里根本不含 `language` 键(不是 null,是缺失)。回答 spec 悬置项:**MOSS 经此端点不回语种**。PR-2 `segments`/`language` 对外若要保留 language,自己填 null。
- `json` 格式:`{"text":"<原始 markup>","usage":{...}}`,只有纯文本、无 segments。

## 4. 实测:显存 / 启动 / 时延(mem_fraction_static=0.25)

| 项 | 值 |
|---|---|
| 模型权重 | **1.80 GB** |
| KV cache @0.25 | **33428 tokens, 3.58 GB**(K 1.79 + V 1.79) |
| **稳态显存占用(stage worker)** | **≈ 8.3 GB**(8278 MiB;含权重+KV+cuda graph+encoder graph+overhead) |
| 启动(**warm** JIT cache) | 权重加载 0.28s;CUDA graph capture 24.98s;coordinator→ready **≈ 36s** |
| 启动(**cold**,空 JIT cache) | **数分钟**(flashinfer + sgl-kernel 的 nvcc JIT 首次编译);之后 `~/.cache/flashinfer`、`~/.cache/tvm-ffi` 落盘,后续都是 warm |
| 短 wav 推理 | 首次 ~19s(decode-path JIT 预热),之后快 |
| 播客 7.7min(460s)推理 | **11s wall**,53 段 |

**`--mem-fraction-static` 定值建议:0.25 可用**(共享 3090 不抢卡,稳态 ~8.3GB)。代价:KV 只有 33428 token,`input_tokens + max_new_tokens` 必须 ≤ ~33k。播客(input 6095)配 max_new_tokens=16384 正好。若要吃 90 分钟长音频,input token 本身会撑爆,需调高 mem_fraction 或分段——这是 PR-2 的动态 max_new_tokens 问题,不是本 spike 的定值问题。

## 5. 踩坑记录(PR-1 setup.sh / PR-2 必读)

**装 sglang-omni**(先于以下所有):`uv pip install -e .`,两个 extra-index:`https://download.pytorch.org/whl/cu130` + `https://sgl-project.github.io/whl/cu130/`,`--index-strategy unsafe-best-match`。大 CUDA wheel 经 mihomo 代理反复断流,`UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 UV_LINK_MODE=copy` 才稳;中断拷贝会污染 uv 缓存(`uv cache clean` 修)。`nvidia-*` 大 wheel 从 `pypi.nvidia.com` 传递拉取(不用显式加 index)。

以下是 **README 裸装跑不起来、本 spike 逐个啃出来的运行时坑**:

1. **flashinfer/sgl-kernel 运行时 JIT 需 nvcc + ninja**。启动时 CUDA graph capture 会用 nvcc 现编 attention kernel。必须:`CUDA_HOME=$VENV/.../nvidia/cu13`、`$CU/bin` 上 PATH、`.venv/bin` 上 PATH(ninja 在这)。否则报 `Could not find nvcc` / `FileNotFoundError: 'ninja'`。

2. **CUDA 编译器 4 件套必须同一 minor,且必须 = 13.3(不是 13.0)**。这是最大的坑:
   - torch cu130 传递装的 `nvidia-cuda-runtime` 头文件是 **13.0**(`CUDA_VERSION 13000`);主循环 install 时 nvcc/nvvm/crt 装成了 **13.3** → 版本错配。
   - **nvcc 13.3 + runtime 头 13.0** → flashinfer 自带 cccl 守卫报 `CUDA compiler and CUDA toolkit headers are incompatible`,capture 失败。
   - 反过来把 nvcc 降到 **13.0** → flashinfer 过了,但 sgl-kernel qknorm JIT 撞本机**超新 glibc**(`features.h` = `_POSIX_C_SOURCE 202405L`)报 `rsqrtf ... exception specification is incompatible`(CUDA 13.0 的 `crt/math_functions.h` 对新 glibc 有 bug,13.3 修了)。
   - **解法 = 把 nvcc/nvvm/crt/runtime 全部钉 13.3**,四者一致:
     ```bash
     uv pip install --reinstall \
       'nvidia-cuda-nvcc==13.3.*' 'nvidia-nvvm==13.3.*' \
       'nvidia-cuda-crt==13.3.*' 'nvidia-cuda-runtime==13.3.*' \
       --extra-index-url https://download.pytorch.org/whl/cu130 \
       --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
     ```
     结果 `CUDA_VERSION 13030` == nvcc 13.3 → cccl 守卫过,且 13.3 crt 兼容新 glibc。改版本后 **`rm -rf ~/.cache/flashinfer ~/.cache/tvm-ffi`** 清掉半成品 JIT 再起。
   - ⚠️ 注意包名:`nvidia-cuda-nvcc`(不是 `-cu13` 后缀,那是 PyPI 空壳 0.0.1);`--reinstall` + unsafe-best-match 会顺手改 runtime 版本,**四个都要显式钉**否则 resolver 又给你错配。

3. **非 wav 输入(mp3)需 nvidia-npp**。torchcodec CUDA 解码链接 `libnppicc.so.13`,不装报 `Could not load libtorchcodec / libnppicc.so.13`。装 `nvidia-npp==13.*`(并把其 lib 加 LD_LIBRARY_PATH)。**但生产走 wav**:后端 `_ffmpeg_to_wav16k` 先归一化,wav 解码不经 npp CUDA 路径(本 spike 短 wav + 播客转 wav16k 都没 npp 也成)。→ npp 可选;若后端保证 wav 入模,PR-1 可不装。

4. **`max_new_tokens > KV 容量 → HTTP 500(不 clamp)**。播客配 65536 直接被拒:`required_tokens=71631 > kv_capacity=33427`。PR-2 必须按时长动态给 max_new_tokens 且 ≤ KV 容量(0.25 时 ≈ 33k − input)。

5. **热词经 `prompt` form 字段,但要 append 不要 replace**。端点收 OpenAI 风格 `prompt`;MOSS request_builder 用 `params["prompt"]` 顶掉默认 prompt。实测传裸 `prompt="热词提示：…"` → **输出丢掉 `[Sxx]`/时间戳 markup**(退化成纯文本)。PR-2 映射 `context` 必须是 `DEFAULT_TRANSCRIBE_DIARIZE_PROMPT + "热词提示：{context}"`(默认 prompt 见 `models/moss_transcribe_diarize/request_builders.py:41`),保留 diar 格式。

## 6. 关键包版本(装通态)

```
sglang-omni        0.1.0
sglang             0.5.12.post1
torch              2.11.0+cu130
transformers       5.6.0
flashinfer-python  0.6.11.post1
torchcodec         0.11.1+cu130
nvidia-cuda-nvcc   13.3.73     nvidia-nvvm 13.3.73  nvidia-cuda-crt 13.3.73
nvidia-cuda-runtime 13.3.29    nvidia-npp 13.1.2.81
```

## 6.5 补充坑(主循环并行验证,与上文互补)

- **triton attention 在 sm_86 数值是坏的**:`attention_backend: triton` 编译/运行都不报错,
  但输出退化为满屏重复 token(实测短 wav 全是 "people")。**别用 triton 绕 flashinfer**;
  绕就绕到 `torch_native`(数值正确,播客 31.5s)。
- 顶层 CLI 没有 `--disable-cuda-graph` 开关(pydantic extra_forbidden)。要关 CUDA graph /
  改 attention backend 只能走 `--config` YAML 的 `stages[].factory_args.server_args_overrides`
  (见 `moss_spike_config.yaml`)。
- 中断的 uv 拷贝会**永久污染缓存**:z3_solver 解包目录混入 `.tmp` 文件后 RECORD 校验永远失败,
  `uv cache clean z3-solver` 定点清除才好。
- sglang worker 子进程 cmdline 不含 config 名,只杀主进程会留孤儿囤显存(实测累计 22GB);
  手工清理按 `nvidia-smi --query-compute-apps` 找 pid;生产 systemd cgroup 收整组无此虞。

## 8. 生产选型仲裁(主循环定,PR-1 已固化)

两条可用通路,**生产走 ①**:

| | ① torch_native(PR-1 采用) | ② flashinfer + CUDA graph(本文件 §1-4 实测) |
|---|---|---|
| 配置 | `disable_cuda_graph: true` + `attention_backend: torch_native` + mem 0.5 | 全默认 + mem 0.25,要求 nvcc/nvvm/crt/**runtime** 四件套全钉 13.3 |
| 播客 7.7min | 31.5s | 11s |
| 启动 | ~3s(无 capture) | warm 36s / cold 数分钟 JIT |
| 活动件 | 无运行时 JIT 依赖 | 依赖 JIT 缓存健康 + 工具链四钉不漂移 |
| 验证 | 主循环 + moss-pr1 两次独立验通 | moss-spike 一次验通 |

理由:单管理员低并发下 31.5s 足够,①启动快、无 JIT 活动件,鲁棒优先。真需要 3× 吞吐时
按 ② 切换(改 moss_config.yaml 删两行 override + mem 降 0.25),风险点=四钉漂移与 JIT 缓存。

## 7. 收尾

serve 进程已杀,GPU0 显存已释放回 baseline(见对话末验证)。JIT 缓存(`~/.cache/flashinfer`、`~/.cache/tvm-ffi`)保留 → 下次 warm 启动 ~36s。

原始响应存于 `infra/moss-asr/logs/`:`short_verbose.json`、`short_json.json`、`podcast_verbose.json`;serve 日志 `logs/serve.log`。
