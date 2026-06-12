# lowvram 流式分块:大模型跑小卡(group offloading)

日期:2026-06-12 | 状态:spec(用户已拍「立项」)| 来源:ComfyUI 源码对照审计的 gap ①

## 动机

- **大模型跑小卡**:Ideogram-4 bf16(双 DiT 37G + TE 16G)、Qwen-Edit(54G)装不进
  24G 3090,目前只能挤 Pro 6000(与 vLLM 抢)或 OOM。ComfyUI 的 lowvram(部分驻卡 +
  逐层流式)能跑,nous 不能 —— memory 挂账已久的「lowvram 分块未做」。
- **精确腾位**(次要):ComfyUI `model_unload(memory_to_free)` 按层腾「刚好需要的量」;
  nous 整组件粒度会过度驱逐。流式模式落地后,该需求强度自然下降(装不下就流式,
  不必精确腾)。

## 地基(侦察结论,免手写 per-layer hook)

diffusers main 的 `diffusers.hooks.apply_group_offloading` = ComfyUI lowvram 的官方对应物:
- `block_level` / `leaf_level` 分组轮转;`use_stream=True` 异步预取(下一组边算边搬,
  即 ComfyUI 的 overlap);streamed 模式自动预 pin CPU 权重(DMA);
  `offload_to_disk_path`(RAM 也紧时落盘)。
- 已在 nous 钉的 diffusers commit(784fa626,#490)中可用。

## Spike 真机数据(2026-06-12,cuda:2 = 3090 24G)

Ideogram-4 bf16(54G)全权重驻 RAM,双 DiT `block_level + use_stream`,TE/VAE 驻卡:

| 指标 | 数据 |
|---|---|
| 峰值显存 | **22.0G**(< 24G,跑通) |
| 采样速度 | 6.34s/step(对照 Pro 6000 全驻卡 1.07s/step,**~6× 慢但从不能跑到能跑**) |
| 挂载耗时 | 35.5s(streamed 预 pin ~37G 权重) |
| 出图 | **正确**(已知好 caption 复跑:COMFY 海报文字渲染与全驻卡同档) |

坑(实测):
- `use_stream` 强制 `num_blocks_per_group=1`(警告自动改);
- **transformers 模型(Qwen3-VL TE)block 检测不稳**:embedding 留 CPU → device 错配崩。
  TE 不流式,驻卡(16G 塞 24G 可行)或后续用 `block_modules` 显式指定;
- streamed 预 pin = **常驻 ~37G pinned RAM**(不可换页!),必须与 RAM stash 水位
  (spec 2026-06-12-ram-stash)统一记账,且与 `NOUS_STASH_PIN_BUDGET_GB` 共享预算口径。

## 设计

### PR-1:offload 选项「流式分块」+ 引擎接入

- 节点(Load Checkpoint / Load Diffusion Model)offload 下拉加
  `{ value: stream, label: "流式分块(大模型跑小卡)" }`。
- 引擎 `_ensure_pipe`:offload=="stream" → 对 transformer 类组件(含
  unconditional_transformer)`apply_group_offloading(onload=目标卡, block_level,
  num_blocks_per_group=1, use_stream=True, record_stream=True)`;TE/VAE `.to(目标卡)`。
- 与 RAM stash 互斥天然成立:`ModularImageBackend.stash()` 对 offload!=none 已返回 False
  (hook pipe 不搬,#500 守卫)——流式 pipe 被驱逐时走销毁(其权重本来就在 RAM/pinned)。
- 显存守卫口径:stream 模式的 need ≠ 全模型文件大小,≈ TE+VAE+单组+激活(估算函数
  按 offload==stream 分支)。

### PR-2:auto 自动降级(装不下 → 流式,而非退 CPU/OOM)

- `_resolve_auto_card`:所有卡「真空闲+可驱逐」都装不下整模型时,先评估「stream 模式
  need」能否装进某卡 → 能则该卡 + 自动置 offload=stream(日志人话说明「已自动启用
  流式分块,速度约 1/6」),仍不行才退 CPU。
- 四态/UI:组件状态加「流式」标记(可后置)。

### PR-3:真机验收 + 服务化验证

- 3090 画布 e2e 出图(Ideogram-4 + Qwen-Edit 两架构);
- 与全驻卡同 seed 出图一致性(SSIM);
- RAM 水位联动测试(stash 池 + pinned 预算同机共存);
- manifest/memory 更新。

## 与 ComfyUI 的残余差异(记录,不在本 arc)

- ComfyUI 支持**部分驻留**(显存富余时多驻少搬,`lowvram_model_memory` 动态比例);
  diffusers group offload 是全轮转。富余显存利用率低 → 速度差距比 ComfyUI 大。
  后续可探 `block_modules` 参数做「前 N 块排除在 offload 外」的部分驻留。
- `offload_to_disk_path`(RAM 不足落盘)可作为 128G RAM 也吃紧时的逃生门,暂不接。
