#!/usr/bin/env bash
# nous-moss-asr 微服务一次性搭建:独立 venv(sglang-omni,与 backend 完全隔离)。
# 以 heygo 身份跑(不用 root)。重建 prod 检出 / 升级 sglang-omni 时重跑。
# 幂等:重跑不炸(venv/clone 存在则复用,pip install 已满足则 no-op,软链用 -sfn)。
#
# 配方来自 PR-0 spike(infra/moss-asr/SPIKE.md)+ 主循环把 serve 真正调起来时补的
# CUDA 版本对齐结论(见 §4 注释;SPIKE.md 待回填 runtime==13.3 一项):
#   - torch 2.11.0+cu130 全家 → PyTorch cu130 index;sglang-kernel/sgl-deep-gemm +cu130
#     → SGLang cu130 index(两个 --extra-index-url,即"cu130 双 index")。
#   - --index-strategy unsafe-best-match:允许跨 index 取最优版本(cu130 本地版本号
#     +cu130 要压过 PyPI 纯版本)。
#   - UV_HTTP_TIMEOUT=600 + UV_CONCURRENT_DOWNLOADS=2:大 CUDA wheel(torch 506MB、
#     cublas 403MB…)经 mihomo 代理易断流,默认超时/并发都过不去;拉长超时 + 压低并发是
#     spike 里唯一稳过的配方(SPIKE.md「低并发过」)。UV_LINK_MODE=copy:缓存与目标常跨盘,
#     hardlink 退化会告警/失败,强制拷贝。
#   - sm_86(3090)无预编译 JIT 内核,要现场 nvcc 编译 → 本机无 /usr/local/cuda,pip 补装
#     CUDA 工具链(nvcc 落 venv 内 nvidia/cu13/bin/):cuda-toolkit[nvcc]==13.0.2 拿 nvcc,
#     再整链升 13.3.*(13.0 头文件与 glibc 2.41+ 的 rsqrt noexcept 冲突,13.3 已修;整链
#     同版本因 PTX 版本要配对),最后补两个软链让 JIT 链接期找到 lib64/libcudart。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
PY="$VENV/bin/python"
CLONE="$DIR/sglang-omni"

# PR-0 spike 钉死的 sglang-omni commit(experimental,API 会漂;固定住 blast radius)。
SGLANG_OMNI_COMMIT="f916b86cb324f21d479e798ad36e2a992cfd010a"
SGLANG_OMNI_REPO="https://github.com/sgl-project/sglang-omni.git"

# cu130 双 extra-index(spike verbatim 确认;pypi.nvidia.com 是 torch→cuda-toolkit→nvidia-*
# 的传递依赖 host,uv 解析器自己抓,不用显式加):
PYTORCH_CU130="https://download.pytorch.org/whl/cu130"
SGLANG_CU130="https://sgl-project.github.io/whl/cu130/"

export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=2
export UV_LINK_MODE=copy

# 统一的 uv pip install 包装:钉 venv python + 双 cu130 index + unsafe-best-match。
uvpip() {
  uv pip install --python "$PY" \
    --extra-index-url "$PYTORCH_CU130" \
    --extra-index-url "$SGLANG_CU130" \
    --index-strategy unsafe-best-match \
    "$@"
}

echo "==> 1/5 建 moss-asr venv: $VENV (py3.12)"
if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$VENV"
else
  echo "    venv 已存在,复用。"
fi

echo "==> 2/5 clone sglang-omni 并钉到 $SGLANG_OMNI_COMMIT"
if [ ! -d "$CLONE/.git" ]; then
  git clone "$SGLANG_OMNI_REPO" "$CLONE"
fi
git -C "$CLONE" fetch --quiet origin
git -C "$CLONE" checkout --quiet "$SGLANG_OMNI_COMMIT"
echo "    HEAD = $(git -C "$CLONE" rev-parse HEAD)"

echo "==> 3/5 装 sglang-omni(editable,~9.8GB:torch+cu130 / sglang / flashinfer 全家)"
# 中断拷贝会把 z3_solver 半个 wheel 混进 uv 缓存(RECORD 不匹配 .tmp),重装即炸;spike 实测
# `uv cache clean z3-solver` 清掉即可。故 install 失败先清该缓存再重试一次,不行才真失败。
if ! ( cd "$CLONE" && uvpip -e . ); then
  echo "    !! install 失败,清 z3-solver 缓存后重试一次(spike 已知坑)。"
  uv cache clean z3-solver || true
  ( cd "$CLONE" && uvpip -e . )
fi

echo "==> 4/5 CUDA 编译链钉到同一 minor=13.3(nvcc/nvvm/crt/runtime;sm_86 fused_rope JIT 要)"
# 关键坑(PR-0 主循环真机调通,现编 kernel 能过、serve 起得来的唯一配方):
# `-e .` 装完后 nvcc/nvvm/crt 是 13.3,但 torch 传递装的 nvidia-cuda-runtime 头停在 13.0
# (CUDA_VERSION 13000)。sglang fused_rope / flashinfer 在 sm_86 无预编译、要现场 nvcc 编译时,
# cccl 守卫比对 nvcc(13.3)与 runtime 头(13.0)不一致 → 报「CUDA compiler and CUDA toolkit
# headers are incompatible」,serve 起不来。必须把 runtime 也钉 13.3(头 CUDA_VERSION 13030
# == nvcc 13.3)。四件套全显式钉 + --reinstall:否则 unsafe-best-match 会把 runtime 又解回
# 错配的 13.0。别降到 13.0:其 crt/math_functions.h 撞本机超新 glibc 的 _POSIX_C_SOURCE
# 报 rsqrtf 异常规格冲突——只有 13.3 两头都满足。
# 包名是 nvidia-cuda-nvcc(无 -cu13 后缀;-cu13 在 PyPI 是 0.0.1 空壳)。runtime 等在
# pypi.nvidia.com,故这步显式带上该 index(它不在上面两个 cu130 index 里)。
# cccl 是第五钉:cuda_fp16.h 要 <nv/target>(cccl wheel 提供)。全新 venv 没它 fused_rope
# 编不过(fatal error: nv/target);现 venv 已有故 pr1/spike 复验暴露不了,别删。
uv pip install --python "$PY" --reinstall \
  'nvidia-cuda-nvcc==13.3.*' 'nvidia-nvvm==13.3.*' \
  'nvidia-cuda-crt==13.3.*' 'nvidia-cuda-runtime==13.3.*' \
  'nvidia-cuda-cccl==13.3.*' \
  --extra-index-url "$PYTORCH_CU130" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match

# JIT 链接期找 lib64 与非版本化 libcudart.so;补两个软链(idempotent: -sfn)。
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"
ln -sfn lib "$CU13/lib64"
ln -sfn libcudart.so.13 "$CU13/lib/libcudart.so"
echo "    软链: $CU13/lib64 -> lib ; $CU13/lib/libcudart.so -> libcudart.so.13"

# 刚改了 CUDA 版本 → 清半成品 JIT 缓存,免得 stale kernel 毒化首次冷启(主循环验)。
rm -rf "$HOME/.cache/flashinfer" "$HOME/.cache/tvm-ffi"

echo "==> 5/5 自检:导入 sglang_omni + CUDA 可见"
CUDA_HOME="$CU13" PATH="$VENV/bin:$CU13/bin:$PATH" LD_LIBRARY_PATH="$CU13/lib:${LD_LIBRARY_PATH:-}" "$PY" - <<'PY'
import sglang_omni  # noqa: F401
import torch
print("sglang_omni import OK, torch", torch.__version__,
      "cuda_available=", torch.cuda.is_available(),
      "n_gpu=", torch.cuda.device_count())
PY

echo "==> 完成。模型需先下(若没下):"
echo "    modelscope download --model MOSS/MOSS-Transcribe-Diarize \\"
echo "      --local_dir \$MODELS_ROOT/nous/speech/MOSS-Transcribe-Diarize"
echo "    启动: $DIR/start_serve.sh  (或 systemd: nous-moss-asr.service)"
