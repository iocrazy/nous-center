# 生成图片节点 / 灯箱增强(对齐 Infinite-Canvas)

- 日期: 2026-06-10
- 状态: 设计
- 参考: Infinite-Canvas `static/js/image-preview.js`(灯箱缩放/平移)+ `canvas.js`
  (输出节点菜单 / 灯箱 prompt 面板 / 前后对比 / 下载)
- 相关: [[2026-06-09-run-history-and-artifacts-design]]

## 1. 背景:IC 生成图片节点功能 vs nous 现状

通读 IC 生成图片节点(画布上「节点即生成单元」范式)。nous 的画布是**工作流编辑器**,
生成走 Playground/run、出图回显在 `ImageOutputNode`(一节点一图)。对照后:

**nous 已有**:NodeResizer 缩放、下载、生成状态 phase、节点 caption 元信息
(seed/steps/cfg/分辨率/时长)、灯箱(`Lightbox.tsx` + `lightbox.ts`:跨工作流收图 +
←/→ 切图 + Esc 关)。

**用户要补的 4 项 → 适用性**:
| 项 | 适用性 |
|---|---|
| 灯箱缩放+平移 | ✅ 直接移植(IC image-preview.js)。**PR-1** |
| 灯箱元信息面板(prompt/分辨率/时长 + 重跑) | ✅ 适用,需扩 store 携带 per-image meta。**PR-2** |
| 灯箱前后对比滑块 | ✅ 适用(nous 另有 ImageCompareNode,灯箱内对比是新)。**PR-3** |
| 批量网格(一节点多图) | ⚠️ **不适用** —— nous 一节点一图,要网格得先支持 batch_size 批量生成(大改)。**defer**。可做的是「下载全部」(工作流内所有图)= **PR-4** |

## 2. 关键现状约束

- `lightbox.ts` store:`images: string[]`(纯 url)+ index;`openFromUrl` 经
  `collectWorkflowImages()` 扫当前工作流全部图 url(IMAGE_KEYS + data.images)。**无 meta**。
- `Lightbox.tsx`:fixed overlay + `<img contain>` + 左右切 + 计数;**无缩放/平移**。
- `ImageOutputNode`:单 `data.image_url` + meta 字段(seed/steps/cfg/width/height/duration_ms)
  已在 node.data。点图 → `openLightbox(url)`(只传 url,丢了 meta)。
- 重跑:服务页/任务面板已有「重跑(相同参数)」(`input_json` 回填,见 run-history arc)。
  灯箱重跑可复用同一导航(带 input → Playground 预填)。

## 3. PR 切分

### PR-1 灯箱缩放 + 平移(自包含,最小)
- `Lightbox.tsx` 内加本地 state:`scale`(1–6)、`offset{x,y}`。
- 滚轮缩放(以光标为中心,clamp 1–6;到 1 时归位);scale>1 时拖拽平移(夹边界);
  双击复位 scale=1。移植 IC `image-preview.js` 的 onWheel/onDown/onDblClick 数学。
- 切图 / 关闭时重置 scale/offset。多图导航、Esc 不变。
- 测试:缩放夹紧 1–6、scale=1 不平移、双击复位(纯函数抽出 clamp/center 便于单测)。

### PR-2 灯箱元信息面板 + 重跑
- 扩 store:`images: Array<{ url: string; meta?: ImageMeta }>`(或并行 `metas[]`);
  `openFromUrl` 收集时一并从 ImageOutputNode 抽 meta(prompt/seed/dims/steps/cfg/duration/
  run 上下文用于重跑)。**向后兼容**:无 meta 时面板不显示。
- `collectWorkflowImages` → 返回带 meta 的项;`ImageOutputNode` 点图时已能从 node.data 取。
- `Lightbox.tsx`:右/下侧可折叠面板 —— prompt(复制按钮)+ 分辨率(优先 meta,缺则
  `<img>` onLoad naturalWidth×naturalHeight)+ 时长 + seed/steps/cfg;**重跑**按钮(有 run
  上下文/workflow 时)→ 复用现有重跑导航。
- 测试:meta 渲染 + 缺 meta 不崩 + 分辨率回退 onLoad。

### PR-3 灯箱前后对比滑块
- 灯箱工具栏加「对比」切换:进入后用滑块在**当前图 vs 相邻图**(或指定 a/b)间左右擦。
- 复用/参考 `ImageCompareNode` 的滑块实现(clip-path inset)。
- 测试:对比模式渲染两图 + 滑块改 clip。

### PR-4 下载全部(workflow 内所有图)
- 灯箱工具栏「下载全部」:对 `images` 全部触发下载。**实现先用客户端顺序下载**
  (anchor click 循环,带原文件名/序号),无新 dep、无后端。若后续要 zip 再加后端端点。
- 批量网格(一节点多图)= defer,注明需 batch_size 批量生成。

> 依赖:PR-1/3/4 自包含;PR-2 扩 store(PR-3 的对比可复用 store 的多图)。各自独立 PR。

## 4. 真机验证(每 PR)
隔离 vite + Playwright / 真出图:PR-1 滚轮放大可平移;PR-2 灯箱见 prompt/分辨率 +
重跑跳 Playground 回填;PR-3 对比滑块;PR-4 下载全部触发多次下载。

## 5. 非目标 / follow-up
- 批量生成(batch_size 出 N 图)+ 节点内网格 —— 大改,单列 arc。
- IC 的画布内图像编辑器(裁剪/蒙版/画笔/外扩)、出图拖到输入做下一次生成的链式范式 ——
  属 IC「画布即生成」范式,nous 要不要那套是独立架构决策,本 arc 不含。
