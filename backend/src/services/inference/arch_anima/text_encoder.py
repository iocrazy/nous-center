"""Anima text encoder wrapper — qwen3-0.6b base + 可选 t5xxl tokenizer。

源:`comfy/text_encoders/anima.py`(63 行)+ `comfy/text_encoders/llama.py` 的 Qwen3_06B class。

nous 不 port ComfyUI 的 `sd1_clip.SDClipModel/SDTokenizer` framework(它依赖 comfy 内部框架);
直接用 transformers `AutoTokenizer` + `AutoModel`,这跟 PR-anima-1 `build_bridged_text_encoder`
一致(Flux2 用 Qwen3-8B 同样套路)。

## 接口

```python
te = AnimaTextEncoder(
    qwen_weights_path="/path/to/qwen_3_06b_base.safetensors",
    qwen_tokenizer_dir="/path/to/qwen25_tokenizer/",   # ComfyUI 提供
    t5_tokenizer_dir="/path/to/t5_tokenizer/",         # ComfyUI 提供;None → 不走 t5xxl
    device="cuda:0", dtype=torch.bfloat16,
)
out = te.encode("a fox")
# out = {"context": (1,N,1024), "t5xxl_ids": (1,L)|None, "t5xxl_weights": (1,L)|None}
```

t5xxl_ids 仅在 `t5_tokenizer_dir` 提供时返回 —— Anima.preprocess_text_embeds 跟着启用
LLMAdapter 路径(spec 决策点 3)。

## 单文件加载

跟 PR-anima-1 `build_bridged_text_encoder` 一致:`init_empty_weights` → safetensors
load_state_dict → tie_weights → materialize_meta → `.to(device, dtype)`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class AnimaTextEncoder:
    """qwen3-0.6b base 文本编码器(可选 t5xxl tokenizer 走 LLMAdapter 路径)。

    懒加载:`__init__` 只存 paths,`load()` 才真装 weight + tokenizer。让 caller(后续
    AnimaPipeline / model_manager)决定何时把它落到 device。
    """

    def __init__(
        self,
        qwen_weights_path: str | Path,
        qwen_tokenizer_dir: str | Path,
        t5_tokenizer_dir: Optional[str | Path] = None,
        device: str = "cpu",
        dtype: Any = None,
    ) -> None:
        self.qwen_weights_path = Path(qwen_weights_path)
        self.qwen_tokenizer_dir = Path(qwen_tokenizer_dir)
        self.t5_tokenizer_dir = Path(t5_tokenizer_dir) if t5_tokenizer_dir is not None else None
        self.device = device
        self.dtype = dtype

        self._qwen_model: Any = None
        self._qwen_tokenizer: Any = None
        self._t5_tokenizer: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # Bundle config path(PR-anima-5):repo 内置 Qwen3-0.6B-Base config.json 副本,免运行时联网。
    _BUNDLE_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "image_arch" / "anima" / "qwen3_06b_base_config.json"

    def load(self) -> None:
        """加载 Qwen3-0.6B base 单文件 + Qwen2 tokenizer +(可选)T5 tokenizer。

        Qwen3 base 没 LM head(只编码),用 `AutoModel.from_config` + `load_state_dict`
        单文件路径。tie_weights / materialize_meta 跟 PR-anima-1 `build_bridged_text_encoder`
        一致(尾部 lm_head 没在 state_dict 时零初始化兜底)。
        """
        if self._loaded:
            return
        import json  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from accelerate import init_empty_weights  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoConfig,
            AutoModel,
            Qwen2Tokenizer,
            T5TokenizerFast,
        )

        if not self.qwen_weights_path.exists():
            raise FileNotFoundError(f"qwen weights not found: {self.qwen_weights_path}")
        if not self.qwen_tokenizer_dir.exists():
            raise FileNotFoundError(f"qwen tokenizer dir not found: {self.qwen_tokenizer_dir}")

        # Qwen2Tokenizer(ComfyUI 用这个 wrapper Qwen3 tokenizer 兼容)。
        self._qwen_tokenizer = Qwen2Tokenizer.from_pretrained(str(self.qwen_tokenizer_dir))

        # Qwen3-0.6B base:加载单文件 state_dict 到 AutoModel。
        sd = load_file(str(self.qwen_weights_path))
        # 关键修复(噪点根因):单文件权重 key 带 `model.` 前缀(Qwen3ForCausalLM 顶层),
        # 但 AutoModel.from_config(qwen3) 加载的是 inner Qwen3Model,期望 key 无 `model.` 前缀
        # → 不 strip 则 matched=0 / 312 个参数全落 meta → 下方 meta 兜底全填 torch.zeros →
        # text encoder 输出纯噪声 context → DiT 收到零 conditioning → 出**纯噪点**(Anima 一直
        # 没真正出过图,旧 smoke 的 "PASS" 只是没崩、没真看图)。strip `model.` 让 310 key 全 matched。
        if any(k.startswith("model.") for k in sd) and not any(
            k.startswith("layers.") for k in sd
        ):
            sd = {
                (k[len("model."):] if k.startswith("model.") else k): v
                for k, v in sd.items()
            }
        # config 来源(优先级):bundle config(免联网)→ HF 网络拉(兜底)。
        if self._BUNDLE_CONFIG.exists():
            with self._BUNDLE_CONFIG.open() as fh:
                cfg_dict = json.load(fh)
            cfg = AutoConfig.for_model(**cfg_dict)
        else:
            cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")
        with init_empty_weights():
            model = AutoModel.from_config(cfg)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if unexpected:
            import logging  # noqa: PLC0415
            logging.getLogger(__name__).warning(
                "AnimaTextEncoder: unexpected weight keys = %d", len(unexpected),
            )
        # tie_weights + materialize meta params(参考 PR-anima-1 build_bridged_text_encoder)。
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        for name, p in list(model.named_parameters()):
            if not p.is_meta:
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, torch.nn.Parameter(
                torch.zeros(p.shape, dtype=torch.bfloat16), requires_grad=False,
            ))
        dtype = self.dtype if self.dtype is not None else torch.bfloat16
        self._qwen_model = model.to(self.device, dtype=dtype)
        self._qwen_model.eval()

        # T5 tokenizer 走 t5xxl 桥接路径;可选(spec 决策点 3 — 首版可不实现)。
        if self.t5_tokenizer_dir is not None and self.t5_tokenizer_dir.exists():
            self._t5_tokenizer = T5TokenizerFast.from_pretrained(str(self.t5_tokenizer_dir))

        self._loaded = True

    def encode(self, text: str) -> dict:
        """编码文本 → 给 Anima.forward(context=..., t5xxl_ids=...) 用的 dict。

        Returns:
            {
              "context":        (1, N_qwen, hidden) — qwen3 隐状态
              "t5xxl_ids":      (1, N_t5) | None
              "t5xxl_weights":  (1, N_t5) | None — 全 1.0(ComfyUI 设计;权重在 LLMAdapter 内部混)
            }

        没启用 t5 tokenizer 时 t5xxl_ids/weights 是 None;Anima.forward 自动走 qwen 直传路径。
        """
        if not self._loaded:
            raise RuntimeError("AnimaTextEncoder not loaded; call .load() first")
        import torch  # noqa: PLC0415

        # Qwen3 编码:tokenizer → hidden_states(last_hidden_state)。
        qwen_input = self._qwen_tokenizer(text, return_tensors="pt", padding=False)
        input_ids = qwen_input["input_ids"].to(self.device)
        with torch.no_grad():
            qwen_out = self._qwen_model(input_ids=input_ids)
        context = qwen_out.last_hidden_state  # (1, N, hidden)

        result: dict = {"context": context, "t5xxl_ids": None, "t5xxl_weights": None}
        if self._t5_tokenizer is not None:
            t5_input = self._t5_tokenizer(text, return_tensors="pt", padding=False)
            t5_ids = t5_input["input_ids"]
            result["t5xxl_ids"] = t5_ids.to(self.device)
            # ComfyUI 设计:所有 weights = 1.0(逐 token 等权)。形状必须 (1, L, 1) —— LLMAdapter
            # 输出是 (1, L, 1024),weights 要按 token 维(L)广播,不是按 hidden 维。早先 (1, L) →
            # 广播到 hidden 维炸 "size 1024 vs 100"。unsqueeze 末维。
            result["t5xxl_weights"] = torch.ones(
                (t5_ids.shape[0], t5_ids.shape[1], 1), dtype=torch.float32, device=self.device,
            )
        return result

    def unload(self) -> None:
        """显式释放 GPU 内存(供 model_manager / pipeline 倒换组件时调用)。"""
        self._qwen_model = None
        self._qwen_tokenizer = None
        self._t5_tokenizer = None
        self._loaded = False
