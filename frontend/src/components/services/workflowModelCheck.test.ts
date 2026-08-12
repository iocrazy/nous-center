import { describe, it, expect } from 'vitest'
import { findInvalidModelRefs, applyModelFixes, issueKey } from './workflowModelCheck'
import type { ComfyWorkflow } from './comfyGraphLayout'

// object_info 片段,形状对齐真实增量的三个字段(VAELoader.vae_name / UNETLoader.unet_name
// / CLIPLoader.clip_name),取值表里同时放本机真实候选和几个无关干扰项,用来验证打分
// 不会瞎猜。
const OBJECT_INFO = {
  VAELoader: {
    input: {
      required: {
        vae_name: [
          [
            'ae.safetensors',
            'minimax-h3/minimax_h3_video_vae_fp16.safetensors',
            'flux/flux_vae.safetensors',
          ],
        ],
      },
    },
  },
  UNETLoader: {
    input: {
      required: {
        unet_name: [
          [
            'minimax-h3/minimax_h3_ref2va_nvfp4.safetensors',
            'flux/flux1-dev.safetensors',
            'sdxl/sd_xl_base_1.0.safetensors',
          ],
        ],
      },
    },
  },
  CLIPLoader: {
    input: {
      required: {
        clip_name: [
          [
            'minimax-h3/minimax_h3_clip_nvfp4.safetensors',
            'flux/clip_l.safetensors',
          ],
        ],
      },
    },
  },
  CLIPTextEncode: {
    input: {
      required: {
        text: ['STRING', { multiline: true }],
      },
    },
  },
}

function incidentWorkflow(): ComfyWorkflow {
  return {
    '119': { class_type: 'VAELoader', inputs: { vae_name: 'minimax_h3_video_vae_fp16.safetensors' } },
    '127': {
      class_type: 'UNETLoader',
      inputs: { unet_name: 'minimax_h3_ref2va_pruned_int8_convrot.safetensors' },
    },
    '128': { class_type: 'CLIPLoader', inputs: { clip_name: 'minimax_h3_clip_pruned_int8.safetensors' } },
  }
}

describe('findInvalidModelRefs', () => {
  it('全部合法 → 空数组', () => {
    const wf: ComfyWorkflow = {
      '119': { class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' } },
    }
    expect(findInvalidModelRefs(wf, OBJECT_INFO)).toEqual([])
  })

  it('真实事故三例:vae basename 精确匹配 → 直接给出正确子目录路径', () => {
    const issues = findInvalidModelRefs(incidentWorkflow(), OBJECT_INFO)
    const vae = issues.find((i) => i.nodeId === '119')!
    expect(vae.suggestion).toBe('minimax-h3/minimax_h3_video_vae_fp16.safetensors')
  })

  it('真实事故三例:unet 无 basename 匹配 → token 重合最多的候选压过无关选项', () => {
    const issues = findInvalidModelRefs(incidentWorkflow(), OBJECT_INFO)
    const unet = issues.find((i) => i.nodeId === '127')!
    expect(unet.suggestion).toBe('minimax-h3/minimax_h3_ref2va_nvfp4.safetensors')
  })

  it('真实事故三例:clip 同理 token 打分选中本机 minimax-h3 候选', () => {
    const issues = findInvalidModelRefs(incidentWorkflow(), OBJECT_INFO)
    const clip = issues.find((i) => i.nodeId === '128')!
    expect(clip.suggestion).toBe('minimax-h3/minimax_h3_clip_nvfp4.safetensors')
  })

  it('非 combo 字符串输入(prompt 文本)永不被标记', () => {
    const wf: ComfyWorkflow = {
      '1': { class_type: 'CLIPTextEncode', inputs: { text: 'a photo of a cat, not a real model filename' } },
    }
    expect(findInvalidModelRefs(wf, OBJECT_INFO)).toEqual([])
  })

  it('object_info 里查不到的 class_type(自定义节点)整节点跳过', () => {
    const wf: ComfyWorkflow = {
      '1': { class_type: 'SomeCustomNode', inputs: { model_name: 'whatever.safetensors' } },
    }
    expect(findInvalidModelRefs(wf, OBJECT_INFO)).toEqual([])
  })

  it('object_info 为 null(sidecar 离线)→ 空数组,不抛异常', () => {
    expect(findInvalidModelRefs(incidentWorkflow(), null)).toEqual([])
  })

  it('非字符串输入(数字/布尔/节点引用连线)永不被标记', () => {
    const wf: ComfyWorkflow = {
      '1': { class_type: 'VAELoader', inputs: { vae_name: ['2', 0], steps: 20, enabled: true } },
    }
    expect(findInvalidModelRefs(wf, OBJECT_INFO)).toEqual([])
  })

  it('值在取值表里 → 不标记(即便字符串形态相似)', () => {
    const wf: ComfyWorkflow = {
      '1': { class_type: 'VAELoader', inputs: { vae_name: 'flux/flux_vae.safetensors' } },
    }
    expect(findInvalidModelRefs(wf, OBJECT_INFO)).toEqual([])
  })
})

describe('applyModelFixes', () => {
  it('按 fixes 替换字段值,返回新对象,不改入参', () => {
    const wf = incidentWorkflow()
    const issues = findInvalidModelRefs(wf, OBJECT_INFO)
    const fixes = issues.map((i) => ({ nodeId: i.nodeId, inputName: i.inputName, newValue: i.suggestion! }))
    const fixed = applyModelFixes(wf, fixes)

    expect(fixed).not.toBe(wf)
    expect(fixed['119'].inputs.vae_name).toBe('minimax-h3/minimax_h3_video_vae_fp16.safetensors')
    expect(fixed['127'].inputs.unet_name).toBe('minimax-h3/minimax_h3_ref2va_nvfp4.safetensors')
    expect(fixed['128'].inputs.clip_name).toBe('minimax-h3/minimax_h3_clip_nvfp4.safetensors')
    // 原工作流对象未被就地改动
    expect(wf['119'].inputs.vae_name).toBe('minimax_h3_video_vae_fp16.safetensors')
    // 结果里对应输入已应用修复
    expect(findInvalidModelRefs(fixed, OBJECT_INFO)).toEqual([])
  })

  it('引用不存在的 nodeId 静默跳过,不抛异常', () => {
    const wf: ComfyWorkflow = { '1': { class_type: 'VAELoader', inputs: { vae_name: 'x' } } }
    const fixed = applyModelFixes(wf, [{ nodeId: '999', inputName: 'vae_name', newValue: 'y' }])
    expect(fixed).toEqual(wf)
  })
})

describe('issueKey', () => {
  it('nodeId+inputName 组合成唯一 key', () => {
    expect(issueKey({ nodeId: '119', inputName: 'vae_name' })).toBe('119::vae_name')
  })
})
