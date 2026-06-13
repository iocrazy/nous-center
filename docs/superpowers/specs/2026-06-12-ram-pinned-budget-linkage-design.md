# RAM 水位 / pinned 预算联动(lowvram 流式 × RAM stash 记账统一)

日期:2026-06-12 | 状态:spec | 来源:lowvram spec(2026-06-12-lowvram-streaming)挂账的
「streamed 预 pin ~37G 必须与 RAM stash 水位 / `NOUS_STASH_PIN_BUDGET_GB` 统一记账」

## 动机(真机实锤,2026-06-12)

- **流式预 pin 不进账本**:diffusers group offloading 的预 pin 走 `tensor.pin_memory()`
  (cudaHostAlloc **拷贝**语义),双 DiT ~37G pinned RAM 完全绕过 `pinned_stash` 的
  全局账本(账本只记 stash 自己的 `cudaHostRegister`)。`NOUS_STASH_PIN_BUDGET_GB=64`
  形同虚设:stash pin + 流式 pin 实际可叠到远超预算。
- **流式挂载前无 RAM 水位检查**:125G 机器,流式 Ideogram-4 驻 RAM 后 available 仅 44G、
  **swap 7G 打满**(2026-06-12 实测)。stash 池装着大模型时再挂流式 = RAM 爆 / swap 风暴
  / OOM killer 杀 runner。
- **auto 降级(#510)选流式时不看 RAM**:`_stream_footprint_mb` 只算显存口径,RAM 装不
  装得下权重 + pin 拷贝没人管。
- pinned 内存**不可换页**,叠过头比显存 OOM 更恶性(拖死整机而非单进程)。

## 侦察结论(diffusers 784fa626,免手写)

- `apply_group_offloading(..., low_cpu_mem_usage=True)` = **不预 pin**:逐组用
  `_pinned_memory_tensors()` context 临时 pin(用完即释)→ pinned 占用有界(单块大小),
  代价 = 每步重 pin + H2D 退回 pageable 档(spike 实测 19.4 vs 53.3 GB/s)。
  官方现成的降级逃生门。
- 预 pin 是拷贝语义:挂载瞬间 DiT 权重**双份**(pageable 原件 + pinned 拷贝,原件随
  `param.data` 替换被 gc)→ 挂载瞬时 RAM 峰值按「+streamed 权重字节」计,不是零。

## 设计

降级哲学与 lowvram arc 一致:**永不拒绝,逐级变慢**(从不能跑到能跑)。

梯子:全量预 pin(最快)→ 先腾 stash 池 → `low_cpu_mem_usage=True` 逐块临时 pin(慢但
pinned 有界)。

### PR-1:pinned 账本统一 + 可观测

- `pinned_stash` 增 `register_external(nbytes) -> handle` / `release_external(handle)`:
  流式挂载完成后,实测 streamed 组件 `t.is_pinned()` 字节总和入账;`unload()` 销毁流式
  pipe 时出账。`total_pinned_bytes()` 自此 = stash pin + 流式 pin 的真实总量。
- `/api/v1/monitor/stats` 暴露 `pinned_ram_mb`(账本总量)+ `stash_ram_mb`(stash 池
  字节和)——排查 RAM 去哪了不再靠 `free -g` 猜。前端 Dashboard 展示可后置。

### PR-2:流式挂载 RAM 门禁 + 降级梯子

- **单一卡口在 manager**(能看见 stash 池、能 trim;engine 看不见):
  `get_or_load_image_adapter` 里 offload==stream(auto 降级与用户显式选共用)build 前:
  - `need_pin` = streamed 权重字节(diffusion model 组件文件大小;整模型 repo 路线用
    `_repo_total_mb` 减 TE/VAE 估)。
  - ① `psutil.available - need_pin < reserve` → `_trim_stash_lru(extra_need=need_pin)`
    (扩签名带目标量,现状只裁到水位线);
  - ② 仍不足 **或** `total_pinned_bytes() + need_pin > NOUS_STASH_PIN_BUDGET_GB` →
    降级 `low_cpu_mem_usage=True`,日志人话:「RAM 紧张,流式分块降级为逐块临时锁页
    (不预占 pinned),挂载更快但每步搬运更慢」。
  - ③ 永不拒绝。
- flag 传递:manager → backend 构造参数(如 `stream_low_ram: bool`),**不进
  ComponentSpec / 缓存键**(同 combo 两种 pin 策略出图相同,进键白白翻倍缓存;也避免
  #509 式 pydantic 白名单坑)。engine `_apply_stream_offload` 透传给
  `apply_group_offloading`。
- 真机先验证钉死 commit 下 `use_stream=True + low_cpu_mem_usage=True` 组合可用
  (源码读是支持的,`_process_tensors_from_modules(pinned_memory=...)` 流式路径接了
  context;spike 级确认再写死)。

### PR-3:真机验收

- 流式挂载后 monitor `pinned_ram_mb` ≈ 37G 入账,卸载归零(账本闭环);
- 压 RAM 场景(stash 池占满 + 挂流式):观察 ①trim 日志 → ②降级日志,出图正确;
- 降级路径速度数据(per-step 对照全量预 pin 的 6.4s/step)记入本 spec;
- memory 更新。

## 边界 / 非目标

- `offload_to_disk_path`(RAM 也吃紧落盘)仍不接 —— 降级梯子兜住后,125G 机器没有
  真实场景;留作 future 逃生门。
- stash 池与流式 pin 的「预算分配比例」不做策略化(谁先到谁用,LRU/降级兜底)——
  单管理员单机,复杂策略无收益。
- vLLM 进程的 RAM(独立进程)不入账 —— psutil available 口径天然含它的影响。
