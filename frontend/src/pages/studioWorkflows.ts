// 创作台四功能的「客户端搭工作流图」+「发布为服务的 exposed schema」。
// 从 Studio.tsx 抽出为纯模块:① 无 React 依赖,可被 vitest 测;② 与 Studio.tsx 解耦避免循环依赖;
// ③ 发布 schema 与 build 函数同文件,改图必同步改 schema(测试守住 node_id 一致,防 #372 类漂移)。
import type { Workflow } from '../models/workflow'
import type { ExposedParam } from '../api/services'

export type FeatureId = 'text2img' | 'enhance' | 'edit' | 'angle'

// --- 客户端搭工作流图(节点 id 与发布 schema 的 node_id 必须一致)---

/** 文生图(Z-Image):checkpoint[arch=z-image]→encode→ksampler→vae_decode→image_output。 */
export function buildZImageWorkflow(
  { ckpt, prompt, width, height, seed }: { ckpt: string; prompt: string; width: number; height: number; seed: number },
): Workflow {
  const nodes = [
    { id: 'ckpt', type: 'flux2_load_checkpoint' as const, position: { x: 0, y: 0 },
      data: { file: ckpt, weight_dtype: 'bfloat16', device: 'auto', offload: 'none', adapter_arch: 'z-image' } },
    { id: 'enc', type: 'flux2_encode_prompt' as const, position: { x: 320, y: 0 },
      data: { text: prompt, negative_prompt: '' } },
    { id: 'ks', type: 'flux2_ksampler' as const, position: { x: 640, y: 0 },
      data: { width, height, steps: 8, cfg_scale: 1.0, sampler_name: 'euler', scheduler: 'normal', seed: String(seed) } },
    { id: 'dec', type: 'flux2_vae_decode' as const, position: { x: 960, y: 0 }, data: {} },
    // image_output 是 executeWorkflow 必需的输出节点(否则抛「工作流缺少输出节点」)。
    { id: 'out', type: 'image_output' as const, position: { x: 1280, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'ckpt', sourceHandle: 'clip', target: 'enc', targetHandle: 'clip' },
    { id: 'e2', source: 'ckpt', sourceHandle: 'model', target: 'ks', targetHandle: 'model' },
    { id: 'e3', source: 'enc', sourceHandle: 'conditioning', target: 'ks', targetHandle: 'conditioning' },
    { id: 'e4', source: 'ckpt', sourceHandle: 'vae', target: 'dec', targetHandle: 'vae' },
    { id: 'e5', source: 'ks', sourceHandle: 'latent', target: 'dec', targetHandle: 'latent' },
    { id: 'e6', source: 'dec', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·文生图(Z-Image)', nodes, edges } as unknown as Workflow
}

/** 图片编辑(Flux2):image_input + checkpoint[arch=flux2]→encode→ksampler(image 端口)→vae_decode。 */
export function buildFlux2EditWorkflow(
  { ckpt, prompt, imageDataUri, width, height, seed }:
  { ckpt: string; prompt: string; imageDataUri: string; width: number; height: number; seed: number },
): Workflow {
  const nodes = [
    { id: 'img', type: 'image_input' as const, position: { x: 0, y: 240 },
      data: { image: imageDataUri } },
    { id: 'ckpt', type: 'flux2_load_checkpoint' as const, position: { x: 0, y: 0 },
      data: { file: ckpt, weight_dtype: 'bfloat16', device: 'auto', offload: 'none', adapter_arch: 'flux2' } },
    { id: 'enc', type: 'flux2_encode_prompt' as const, position: { x: 320, y: 0 },
      data: { text: prompt, negative_prompt: '' } },
    { id: 'ks', type: 'flux2_ksampler' as const, position: { x: 640, y: 0 },
      data: { width, height, steps: 20, cfg_scale: 4.0, sampler_name: 'euler', scheduler: 'normal', seed: String(seed) } },
    { id: 'dec', type: 'flux2_vae_decode' as const, position: { x: 960, y: 0 }, data: {} },
    { id: 'out', type: 'image_output' as const, position: { x: 1280, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'ckpt', sourceHandle: 'clip', target: 'enc', targetHandle: 'clip' },
    { id: 'e2', source: 'ckpt', sourceHandle: 'model', target: 'ks', targetHandle: 'model' },
    { id: 'e3', source: 'enc', sourceHandle: 'conditioning', target: 'ks', targetHandle: 'conditioning' },
    { id: 'e4', source: 'img', sourceHandle: 'image', target: 'ks', targetHandle: 'image' },
    { id: 'e5', source: 'ckpt', sourceHandle: 'vae', target: 'dec', targetHandle: 'vae' },
    { id: 'e6', source: 'ks', sourceHandle: 'latent', target: 'dec', targetHandle: 'latent' },
    { id: 'e7', source: 'dec', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·图片编辑(Flux2)', nodes, edges } as unknown as Workflow
}

/** 细节增强(SeedVR2 超分):image_input→seedvr2_upscale→image_output。dit/vae loader 不连用默认。 */
export function buildSeedVR2Workflow(
  { imageDataUri, resolution }: { imageDataUri: string; resolution: number },
): Workflow {
  const nodes = [
    { id: 'img', type: 'image_input' as const, position: { x: 0, y: 0 }, data: { image: imageDataUri } },
    { id: 'up', type: 'seedvr2_upscale' as const, position: { x: 320, y: 0 },
      data: { resolution, max_resolution: 0, color_correction: 'lab' } },
    { id: 'out', type: 'image_output' as const, position: { x: 640, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'img', sourceHandle: 'image', target: 'up', targetHandle: 'image' },
    { id: 'e2', source: 'up', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·细节增强(SeedVR2)', nodes, edges } as unknown as Workflow
}

/** 角度控制(Qwen-Image-Edit):image_input + checkpoint[arch=qwen-edit]→encode→ksampler(image 端口)→vae_decode。 */
export function buildQwenEditWorkflow(
  { ckpt, prompt, imageDataUri, width, height, seed }:
  { ckpt: string; prompt: string; imageDataUri: string; width: number; height: number; seed: number },
): Workflow {
  const nodes = [
    { id: 'img', type: 'image_input' as const, position: { x: 0, y: 240 }, data: { image: imageDataUri } },
    { id: 'ckpt', type: 'flux2_load_checkpoint' as const, position: { x: 0, y: 0 },
      data: { file: ckpt, weight_dtype: 'bfloat16', device: 'auto', offload: 'none', adapter_arch: 'qwen-edit' } },
    { id: 'enc', type: 'flux2_encode_prompt' as const, position: { x: 320, y: 0 },
      data: { text: prompt, negative_prompt: '' } },
    { id: 'ks', type: 'flux2_ksampler' as const, position: { x: 640, y: 0 },
      data: { width, height, steps: 40, cfg_scale: 4.0, sampler_name: 'euler', scheduler: 'normal', seed: String(seed) } },
    { id: 'dec', type: 'flux2_vae_decode' as const, position: { x: 960, y: 0 }, data: {} },
    { id: 'out', type: 'image_output' as const, position: { x: 1280, y: 0 }, data: {} },
  ]
  const edges = [
    { id: 'e1', source: 'ckpt', sourceHandle: 'clip', target: 'enc', targetHandle: 'clip' },
    { id: 'e2', source: 'ckpt', sourceHandle: 'model', target: 'ks', targetHandle: 'model' },
    { id: 'e3', source: 'enc', sourceHandle: 'conditioning', target: 'ks', targetHandle: 'conditioning' },
    { id: 'e4', source: 'img', sourceHandle: 'image', target: 'ks', targetHandle: 'image' },
    { id: 'e5', source: 'ckpt', sourceHandle: 'vae', target: 'dec', targetHandle: 'vae' },
    { id: 'e6', source: 'ks', sourceHandle: 'latent', target: 'dec', targetHandle: 'latent' },
    { id: 'e7', source: 'dec', sourceHandle: 'image', target: 'out', targetHandle: 'image' },
  ]
  return { name: '创作台·角度控制(Qwen-Image-Edit)', nodes, edges } as unknown as Workflow
}

// --- 链式采样(跨模型 A-ref,spec 2026-06-08-multi-sampling-cross-model)---
// stage1 文生图;stage2..N 把上一段出图作「参考编辑条件」喂下一段(引擎对 Flux2-Klein/Qwen-Edit
// 注入 image=,多参考编辑;Z-Image 无 image= 会忽略)。真机验过 Z-Image→Flux2-Klein→Flux2-Klein
// 三采样链(零引擎改:vae_decode.image_url → 下游 ks.image → input_image → 引擎 image= 注入)。

export interface ChainStage {
  ckpt: string   // 整模型 abs_path(flux2_load_checkpoint.file 必须绝对路径,非裸名)
  arch: string   // 'z-image' | 'flux2' | 'qwen-edit'
  prompt: string
  steps: number
  cfg: number
  // img2img 重绘强度(PR-A2):仅 z-image 段 + 有上游图(i>0)时生效 —— 引擎 _wants_img2img 门控
  // (arch 有 img2img 变体 + input_image + 0<strength<1)。<1=保留上段结构重绘,1/缺省=纯重生成(零回归)。
  strength?: number
  // 接力方式(PR-B3):i>0 段如何消费上一段。缺省 'image'。
  //   'image'  = A-ref 图接力(上段出图 → 本段 input_image,参考编辑 / z-image img2img)。
  //   'latent' = 同模型留噪续采(KSamplerAdvanced split):注入上段**带噪** latent + 不重加噪,
  //              base/refiner 共享一次去噪的总步数,在 split 步交接。仅同 latent 空间(同 z-image)有效。
  relay?: 'image' | 'latent'
  // latent 续采的劈分步(同时是上段 end_at_step 与本段 start_at_step);仅 relay==='latent' 用。
  split?: number
}

// 组首步数:latent 接力组里所有段共享一次去噪的总步数 = 组首(relay!=='latent' 的段)的 steps。
// 沿 relay==='latent' 往回走到组首(image 接力段或第 0 段)。
function chainGroupSteps(stages: ChainStage[], i: number): number {
  let j = i
  while (j > 0 && stages[j].relay === 'latent') j--
  return stages[j].steps
}

/** 搭链式采样图(跨模型 A-ref 图接力 + 同模型留噪 latent 接力)。返回 workflow + 每段终端
 *  (flux2_vae_decode)节点 id,供 UI 逐段收集出图(latent 中间段终端不出图,只产 latent_ref)。
 *  第 i 段(i>0):relay='image' → ks.image ← 上段 vae_decode.image;relay='latent' → ks.init_latent
 *  ← 上段 vae_decode.latent_ref(上段转 output_mode=latent + end_at_step 留噪导出带噪 latent)。 */
export function buildChainWorkflow(
  stages: ChainStage[],
  { width, height, seed }: { width: number; height: number; seed: number },
): { workflow: Workflow; stageTerminals: string[] } {
  // 留噪 latent 接力校验(派发前 fail loud,别搭出错图):同 latent 空间(同 z-image)+ split 边界。
  stages.forEach((st, i) => {
    if (st.relay !== 'latent') return
    const prev = stages[i - 1]
    if (!prev || st.arch !== prev.arch || st.arch !== 'z-image') {
      throw new Error(
        `第 ${i + 1} 段「latent 留噪续采」要求与上一段同为 z-image 架构(latent 空间一致才能接力);` +
        '跨模型 latent 物理不兼容,请改用「图参考编辑」。')
    }
    const total = chainGroupSteps(stages, i)
    if (st.split == null || st.split <= 0 || st.split >= total) {
      throw new Error(`第 ${i + 1} 段 split(劈分步)须在 1..${total - 1} 之间(组内总步数 ${total})。`)
    }
  })

  const nodes: unknown[] = []
  const edges: unknown[] = []
  const terminals: string[] = []
  stages.forEach((st, i) => {
    const c = `c${i}`, e = `e${i}`, k = `k${i}`, d = `d${i}`
    const y = i * 360
    const isLatentRefiner = st.relay === 'latent'
    const feedsLatentNext = stages[i + 1]?.relay === 'latent'
    const totalSteps = chainGroupSteps(stages, i)
    nodes.push(
      { id: c, type: 'flux2_load_checkpoint', position: { x: 0, y },
        data: { file: st.ckpt, weight_dtype: 'bfloat16', device: 'auto', offload: 'none', adapter_arch: st.arch } },
      { id: e, type: 'flux2_encode_prompt', position: { x: 320, y },
        data: { text: st.prompt, negative_prompt: '' } },
      { id: k, type: 'flux2_ksampler', position: { x: 640, y },
        data: {
          // latent 接力组共享总步数(base/refiner 同 steps,在 split 步交接)。
          width, height, steps: totalSteps, cfg_scale: st.cfg, sampler_name: 'euler', scheduler: 'normal',
          seed: String(seed + i),
          // 本段是 refiner:从 split 步起、不重加噪(注入上段带噪 latent 原样续采)。
          ...(isLatentRefiner ? { start_at_step: st.split, add_noise: false } : {}),
          // 本段是下段的 base:停在 split 步、保留余噪(带噪交接给下段 refiner)。
          ...(feedsLatentNext ? { end_at_step: stages[i + 1].split, return_with_leftover_noise: true } : {}),
          // strength 仅 image 接力的 z-image 段(i>0)有意义;latent 接力不走 img2img。
          ...(i > 0 && st.relay !== 'latent' && st.strength != null ? { strength: st.strength } : {}),
        } },
      // 下段要 latent 接力 → 本段终端不解码,导出带噪 latent(output_mode=latent → latent_ref)。
      { id: d, type: 'flux2_vae_decode', position: { x: 960, y },
        data: feedsLatentNext ? { output_mode: 'latent' } : {} },
    )
    edges.push(
      { id: `${c}-clip`, source: c, sourceHandle: 'clip', target: e, targetHandle: 'clip' },
      { id: `${c}-model`, source: c, sourceHandle: 'model', target: k, targetHandle: 'model' },
      { id: `${e}-cond`, source: e, sourceHandle: 'conditioning', target: k, targetHandle: 'conditioning' },
      { id: `${c}-vae`, source: c, sourceHandle: 'vae', target: d, targetHandle: 'vae' },
      { id: `${k}-latent`, source: k, sourceHandle: 'latent', target: d, targetHandle: 'latent' },
    )
    if (i > 0) {
      if (isLatentRefiner) {
        // 留噪 latent 接力:上段带噪 latent_ref → 本段 ksampler.init_latent(同空间真 latent 续采)。
        edges.push({ id: `chain-lat-${i}`, source: `d${i - 1}`, sourceHandle: 'latent_ref', target: k, targetHandle: 'init_latent' })
      } else {
        // A-ref 图接力:上段 decode 出图 → 本段 ksampler.image(参考编辑 / img2img 条件)。
        edges.push({ id: `chain-${i}`, source: `d${i - 1}`, sourceHandle: 'image', target: k, targetHandle: 'image' })
      }
    }
    terminals.push(d)
  })
  const last = terminals[terminals.length - 1]
  nodes.push({ id: 'out', type: 'image_output', position: { x: 1280, y: (stages.length - 1) * 360 }, data: {} })
  edges.push({ id: 'out-edge', source: last, sourceHandle: 'image', target: 'out', targetHandle: 'image' })
  return { workflow: { name: '创作台·链式采样', nodes, edges } as unknown as Workflow, stageTerminals: terminals }
}

/** 发布为模板工作流图(占位参数;exposed 输入在调用时由 caller 覆盖)。 */
export function buildFeatureWorkflow(feature: FeatureId, ckpt: string): Workflow {
  switch (feature) {
    case 'text2img':
      return buildZImageWorkflow({ ckpt, prompt: '', width: 1024, height: 1024, seed: 0 })
    case 'edit':
      return buildFlux2EditWorkflow({ ckpt, prompt: '', imageDataUri: '', width: 1024, height: 1024, seed: 0 })
    case 'enhance':
      return buildSeedVR2Workflow({ imageDataUri: '', resolution: 1024 })
    case 'angle':
      return buildQwenEditWorkflow({ ckpt, prompt: '', imageDataUri: '', width: 1024, height: 1024, seed: 0 })
  }
}

// --- 发布为服务的 exposed schema(node_id 指向上面 build 的节点;PR-1 真机验证已用同款建过 4 服务)---

export interface FeaturePublish {
  defaultName: string
  label: string
  // 文生图/编辑/角度的工作流含 flux2_vae_decode → 后端 _detect_category 自动判 image;
  // 细节增强(SeedVR2)无 dec 节点 → 必须显式传 category=image。
  category?: 'image'
  exposed_inputs: ExposedParam[]
  exposed_outputs: ExposedParam[]
}

// 输出全指**产图终端**(dec=flux2_vae_decode / up=seedvr2_upscale),非 image_output(它是 sink 无输出)。
export const FEATURE_PUBLISH: Record<FeatureId, FeaturePublish> = {
  text2img: {
    defaultName: 'studio-text-to-image',
    label: '创作台·文生图',
    exposed_inputs: [
      { node_id: 'enc', key: 'prompt', input_name: 'text', type: 'string', required: true, label: '提示词' },
    ],
    exposed_outputs: [{ node_id: 'dec', key: 'image_url', input_name: 'image_url', type: 'string' }],
  },
  edit: {
    defaultName: 'studio-image-edit',
    label: '创作台·图片编辑',
    exposed_inputs: [
      { node_id: 'img', key: 'image', input_name: 'image', type: 'image', required: true, label: '输入图' },
      { node_id: 'enc', key: 'prompt', input_name: 'text', type: 'string', required: true, label: '编辑指令' },
    ],
    exposed_outputs: [{ node_id: 'dec', key: 'image_url', input_name: 'image_url', type: 'string' }],
  },
  enhance: {
    defaultName: 'studio-upscale',
    label: '创作台·细节增强',
    category: 'image',
    exposed_inputs: [
      { node_id: 'img', key: 'image', input_name: 'image', type: 'image', required: true, label: '输入图' },
      { node_id: 'up', key: 'resolution', input_name: 'resolution', type: 'int', required: false, label: '目标分辨率(短边)' },
    ],
    exposed_outputs: [{ node_id: 'up', key: 'image_url', input_name: 'image_url', type: 'string' }],
  },
  angle: {
    defaultName: 'studio-angle',
    label: '创作台·角度控制',
    exposed_inputs: [
      { node_id: 'img', key: 'image', input_name: 'image', type: 'image', required: true, label: '输入图' },
      { node_id: 'enc', key: 'prompt', input_name: 'text', type: 'string', required: true, label: '视角描述' },
    ],
    exposed_outputs: [{ node_id: 'dec', key: 'image_url', input_name: 'image_url', type: 'string' }],
  },
}
