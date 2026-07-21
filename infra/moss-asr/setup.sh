#!/usr/bin/env bash
# nous-moss-asr 微服务一次性搭建:独立 venv(sglang-omni,与 backend 完全隔离)。
# 以 heygo 身份跑(不用 root)。重建 prod 检出 / 升级 sglang-omni 时重跑。
# 幂等:重跑不炸(venv/clone 存在则复用,pip install 已满足则 no-op,软链用 -sfn)。
#
# 全部配方来自 PR-0 spike(infra/moss-asr/SPIKE.md)真机验证结论:
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

echo "==> 4/5 补 CUDA 工具链(nvcc + 整链 13.3.*;sm_86 JIT 现场编译要)"
# cuda-toolkit[nvcc] 拿 nvcc,落 .venv/.../nvidia/cu13/bin/。
uvpip "cuda-toolkit[nvcc]==13.0.2"
# 整链升 13.3.*(修 glibc 2.41+ 头文件冲突;PTX 版本要整链一致)。
uvpip "nvidia-cuda-nvcc==13.3.*" "nvidia-cuda-crt==13.3.*" \
      "nvidia-nvvm==13.3.*" "nvidia-cuda-cccl==13.3.*"

# JIT 链接期找 lib64 与非版本化 libcudart.so;补两个软链(idempotent: -sfn)。
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"
ln -sfn lib "$CU13/lib64"
ln -sfn libcudart.so.13 "$CU13/lib/libcudart.so"
echo "    软链: $CU13/lib64 -> lib ; $CU13/lib/libcudart.so -> libcudart.so.13"

echo "==> 5/5 自检:导入 sglang_omni + CUDA 可见"
CUDA_HOME="$CU13" PATH="$VENV/bin:$PATH" "$PY" - <<'PY'
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
