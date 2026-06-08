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
