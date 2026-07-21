# PR-0 Spike 报告 — SGLang Omni 服务 MOSS-Transcribe-Diarize(2026-07-20)

**判定:PASS。** 三件闸门事全部真机验证通过。spec:
`docs/superpowers/specs/2026-07-20-moss-asr-sglang-serving-design.md`。

## 判定证据

1. **装得起、起得来**(cu130 本机):`infra/moss-asr/.venv`(py3.12,565 包/9.8GB,
   torch 2.11.0+cu130 + sglang 0.5.12.post1 + sglang-omni 0.1.0 editable)。
   `sgl-omni serve --config moss_spike_config.yaml` 起服务,health 200。
2. **verbose_json 真回说话人分段**:7.7 分钟双人播客(16k WAV)→ **53 段,S01/S02**,
   时间戳 0.29→454.43s(音频 460.2s),转写与 transformers 路径 smoke 一致。
   耗时 **31.5s**(torch_native attention);短句 5.3s。
3. **响应 JSON 结构**(PR-2 按此解析):
   ```json
   {"task":"transcribe","duration":3.48,
    "text":"[0.26][S01]希望你以后能够做得比我还好哟。[3.24]",
    "segments":[{"id":0,"start":0.26,"end":3.24,"text":"[S01]希望你以后能够做得比我还好哟。"}],
    "usage":{"type":"duration","seconds":4}}
   ```
   **注意:segment 无独立 speaker 字段**,说话人是 `segments[].text` 的 `[Sxx]` 前缀,
   后端要正则抠出。`json` 格式则只回 text。

## 关键结论(接入必读)

- **attention 必须 `torch_native`**:默认 flashinfer 在 sm_86(3090)要 JIT(需完整 CUDA
  工具链);**triton backend 数值坏**——编译能过但输出乱码(满屏 "people" 重复)。
  `torch_native` 正确且 31.5s/7.7min 完全够用。⇒ 若未来迁 Blackwell 卡再评 flashinfer。
- **CUDA 工具链要 pip 补装**(本机无 /usr/local/cuda;sglang 的 fused_rope JIT 内核在
  sm_86 无预编译,必须现场 nvcc 编译):
  ```
  uv pip install "cuda-toolkit[nvcc]==13.0.2"        # nvcc 落 nvidia/cu13/bin/
  uv pip install "nvidia-cuda-nvcc==13.3.*" "nvidia-cuda-crt==13.3.*" \
                 "nvidia-nvvm==13.3.*" "nvidia-cuda-cccl==13.3.*"
  # 13.0 头文件与 glibc 2.41+ 冲突(rsqrt noexcept),13.3 已修;整链同版本(PTX 版本要配)。
  ln -s lib .venv/.../nvidia/cu13/lib64              # JIT 链接找 lib64
  ln -s libcudart.so.13 .venv/.../nvidia/cu13/lib/libcudart.so
  ```
  环境变量:`CUDA_HOME=<venv>/nvidia/cu13`、`PATH` 加 `<venv>/bin`(ninja)。
- **CLI 无 disable-cuda-graph 顶层开关**(pydantic extra_forbidden;`--thinker-cuda-graph`
  是别的 pipeline 的)⇒ 走 `--config` YAML,`server_args_overrides` 注入
  `disable_cuda_graph: true` + `attention_backend: torch_native`(见 `moss_spike_config.yaml`)。
- **mp3 直传不行**:sgl-omni 用 torchcodec 解码,本机缺 libav 共享库 → 500。
  **生产无影响**:backend `_ffmpeg_to_wav16k` 先归一化,微服务只见 16k WAV(已验可用)。
- **显存**:`mem_fraction_static 0.25` 起得来但 KV 只 1143 token(权重+4GB encoder cache
  挤占)→ 单请求都不够;**0.5(≈13.4GB 实测占用)跑播客稳**。生产共享 3090 建议 0.5;
  高并发/长音频再专卡拉高。
- **杀进程要杀全家**:sglang 有 worker 子进程(cmdline 不含 config 名),只杀主进程会
  留孤儿占显存(实测累积 5 个孤儿吃掉 22GB)。systemd `KillMode=control-group`(默认)
  天然解决 ⇒ 生产 unit 无此问题;手工操作按 GPU compute-apps 清。
- 安装网络坑:大 CUDA wheel 经 mihomo 代理断流(UV_HTTP_TIMEOUT=600 + 低并发过);
  中断拷贝会污染 uv 缓存(z3_solver 混入 .tmp,`uv cache clean z3-solver` 修);
  跨盘 hardlink 退化(UV_LINK_MODE=copy)。

## 版本清单

| 组件 | 版本 |
|---|---|
| sglang-omni | 0.1.0(git main,editable) |
| sglang / sglang-kernel | 0.5.12.post1 / 0.4.2.post2+cu130 |
| torch | 2.11.0+cu130 |
| transformers | 5.6.0(venv 内独立,MOSS adapter 原生内置) |
| nvcc/crt/nvvm/cccl | 13.3.73 系(nvcc 13.3.73) |

## 复现

```bash
./start_spike_serve.sh          # CUDA_HOME/PATH/GPU UUID 都在脚本里
curl -X POST http://127.0.0.1:8003/v1/audio/transcriptions \
  -F model=moss-transcribe-diarize -F file=@audio_16k.wav \
  -F response_format=verbose_json -F max_new_tokens=16384
```
