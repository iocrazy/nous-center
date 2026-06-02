"""节点分流判定 —— dispatch 节点（GPU runner 串行队列）vs inline 节点（主进程 event loop）。

spec §2.1 step 9 / §4.5「Inline 执行点改道清单」。

- dispatch 节点：在 GPU runner 子进程内执行（image / tts），主进程经 RunnerClient.run_node 投递
- inline 节点：在主进程 event loop 内直接 await（CPU 逻辑节点；llm 节点本身已是 HTTP-调-vLLM）

DISPATCH_NODE_TYPES 是显式白名单 —— 新增任何需要 GPU runner 的节点类型，必须在此登记，
否则会被当作 inline 在主进程内执行（撞 GPU race，正是 V1.5 要消灭的问题）。
"""
from __future__ import annotations

from typing import Literal

ExecClass = Literal["inline", "dispatch"]

# GPU 节点白名单 —— 这些节点 dispatch 到对应 runner 的串行队列执行。
# flux2_vae_decode 是细粒度图的 dispatch 终端:整条 Load*→Encode→KSampler 链
# inline 累积描述符,末端 VAE Decode 把嵌套 latent 派发到 image runner,整模型在
# 所选卡执行(spec 2026-05-21 rev 2)。Family B 的 image_generate 已收敛删除(PR-4)。
# seedvr2_upscale 是图→图超分(SeedVR2),吃 GPU,跑在 image runner 组(SeedVR2 PR-3b)。
DISPATCH_NODE_TYPES: frozenset[str] = frozenset(
    {"tts_engine", "flux2_vae_decode", "seedvr2_upscale"})


def node_exec_class(node_type: str) -> ExecClass:
    """判定一个节点类型走 dispatch 还是 inline。

    未登记的类型（含第三方插件节点）默认 inline —— 保守策略：不假设未知节点
    需要 GPU runner。若某插件节点其实吃 GPU，需显式加进 DISPATCH_NODE_TYPES。
    """
    return "dispatch" if node_type in DISPATCH_NODE_TYPES else "inline"
