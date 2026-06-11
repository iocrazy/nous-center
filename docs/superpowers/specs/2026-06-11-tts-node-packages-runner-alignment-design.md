# qwen3-tts / voxcpm2 节点包归位 TTS runner 架构

日期:2026-06-11 · 状态:spec(用户已拍板排期归位)· 来源:全节点包体检(project_full_node_workflow_audit)

## 问题

`backend/nodes/qwen3-tts/` 与 `backend/nodes/voxcpm2/` 两个节点包的 executor 是 **inline 节点 +
模块级 `_MODEL_CACHE`,在主 API 进程直接 `from_pretrained` 载 GPU 模型**。偏离既定 runner 架构:

1. **显存不受管**:不进 ModelManager `_models` —— 不计入 VRAM 守卫预算、不可被 evict_lru 驱逐、
   不上报 loaded_models_snapshot(Dashboard/ModelsOverlay 看不见)、unload 端点摸不到。
2. **稳定性**:模型 CUDA 调用卡死/段错误直接拖垮主 API 进程(runner 子进程隔离正是为防这个,
   见 in-use 守卫 #214 防 segfault 链)。
3. **双轨**:同一引擎两套加载 —— `workers/tts_engines/qwen3_tts.py` / `voxcpm2.py`(models.yaml
   注册,走 TTS runner)与节点包 `_MODEL_CACHE` 各载一份,显存翻倍且行为漂移
   (体检中 voxcpm2 device 修复就要改两处,#481)。
4. **无进度/取消**:inline 路径没有 dispatch 的 progress_callback/cancel_flag 签名探测注入。

对照:cosyvoice2 的 `tts_engine` 节点是 **dispatch 节点** —— runner `_build_request` 构
TTSRequest → `get_or_load` → `TTSEngine.infer`(progress + cancel + 显存全套)。这就是目标形态。

## 方案(对齐 tts_engine dispatch 路径)

models.yaml 条目已齐(qwen3_tts_base / qwen3_tts_customvoice / qwen3_tts_voicedesign /
voxcpm2,#476 起路径已对 speech/),引擎类已存在 —— 工作量在「节点包改 dispatch + 引擎参数对齐」。

### PR-1 qwen3-tts 三节点 → dispatch

- node.yaml 三节点(base 克隆 / custom voice / voice design)改 `dispatch: tts`(对齐 tts_engine
  的归类;runner 按 node_type 选 TTS runner client)。
- runner `_build_request` 新分支:node_type ∈ qwen3_tts_* → TTSRequest,model_key 映射
  `qwen3_tts_base` 等;widget(speaker/language/ref_text/voice_description/seed)进
  TTSRequest extras → `TTSEngine.synthesize` kwargs(引擎签名已收 reference_audio/
  reference_text/emotion,缺的补 kwargs 透传)。
- 参考音频:上游 audio 端口(data URI/URL)→ runner 落盘临时文件传路径(对齐现 executor 行为),
  用完 finally 清理。
- 删节点包 executor 的 `_get_qwen_model`/`_MODEL_CACHE`(executor.py 只留无 GPU 的参数整形,
  或整文件删除若 dispatch 全覆盖)。
- 引擎 `qwen3_tts.py`:`device_map=self.device` 保留("auto" 是 transformers 合法值,体检已核);
  对齐 #476 base 路径解析(已生效)。

### PR-2 voxcpm2 两节点 → dispatch

- `voxcpm2_load_model` 降级为 inline 配置节点(产 {model_name, device, load_denoiser} 描述符,
  不再真载模型 —— 同 seedvr2 loader 惰性模式),或直接删(widget 并进 generate 节点,二选一,
  倾向后者:VoxCPM2 单模型无多变体,load 节点没有存在价值)。
- `voxcpm2_generate` 改 dispatch:TTSRequest(model_key=voxcpm2)+ mode/voice_description/
  prompt_text/参考音频 进 synthesize kwargs(引擎 synthesize 已收 reference_audio/
  reference_text;mode/voice_description 补)。
- 删 executor `_MODEL_CACHE`。

### PR-3 收尾

- 体检 e2e 三链复跑:qwen3 base 克隆 / custom voice / voxcpm2 design,验进度帧(WS)+
  Dashboard 可见 + unload 可卸。
- 节点包 wiring 测试更新(EXECUTORS 集合变化);删两包的「主进程载模型」路径死代码。

## 不做 / 风险

- 不动 cosyvoice2/indextts2/moss(已在正轨)。
- TTS runner 当前单 GPU 组:三个 qwen3 变体 + voxcpm2 同时常驻会挤(1.7B×3 + VoxCPM ~8GB);
  靠 ModelManager LRU 驱逐天然解决(归位后才有这能力 —— 现状反而是无界的)。
- 节点 id/端口不变 → 既有画布工作流零迁移。
