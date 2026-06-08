import { describe, it, expect } from 'vitest'
import { buildFeatureWorkflow, buildChainWorkflow, FEATURE_PUBLISH, type FeatureId, type ChainStage } from './studioWorkflows'

const FEATURES: FeatureId[] = ['text2img', 'edit', 'enhance', 'angle']

// 发布 schema 的每个 node_id 必须真实存在于对应功能搭出的工作流图里 —— 否则 publish 端点
// 422(node_id 不存在),或注入落空。这是 #372 类「schema 指向不存在节点」漂移的守门测试。
describe('studio 发布 schema ↔ 工作流图一致性', () => {
  for (const f of FEATURES) {
    const wf = buildFeatureWorkflow(f, '/m/ckpt') as unknown as { nodes: { id: string; type: string; data: Record<string, unknown> }[] }
    const byId = new Map(wf.nodes.map((n) => [n.id, n]))
    const schema = FEATURE_PUBLISH[f]

    it(`${f}: 每个 exposed node_id 都在工作流图里`, () => {
      for (const p of [...schema.exposed_inputs, ...schema.exposed_outputs]) {
        expect(byId.has(p.node_id), `${f} exposed node_id=${p.node_id} 不存在`).toBe(true)
      }
    })

    it(`${f}: exposed input 的 input_name 是该节点 data 的真实字段`, () => {
      // image_input→image / flux2_encode_prompt→text / seedvr2_upscale→resolution。
      // build 函数给这些字段都设了默认值,故 input_name 必在 node.data 的 key 里。
      for (const p of schema.exposed_inputs) {
        const node = byId.get(p.node_id)!
        expect(Object.keys(node.data), `${f} 节点 ${p.node_id} 无字段 ${p.input_name}`).toContain(p.input_name)
      }
    })

    it(`${f}: exposed output 指向产图终端(dec/up)而非 image_output sink`, () => {
      // image_output 是 sink 无输出;真正 emit image_url 的是 flux2_vae_decode(dec)/seedvr2_upscale(up)。
      for (const p of schema.exposed_outputs) {
        const node = byId.get(p.node_id)!
        expect(['flux2_vae_decode', 'seedvr2_upscale']).toContain(node.type)
      }
    })

    it(`${f}: 含 image_input 节点时必有 image 输入暴露(编辑/增强/角度)`, () => {
      const hasImageInput = wf.nodes.some((n) => n.type === 'image_input')
      if (hasImageInput) {
        const exposesImage = schema.exposed_inputs.some((p) => p.node_id === 'img' && p.input_name === 'image')
        expect(exposesImage, `${f} 有 image_input 却没暴露 image`).toBe(true)
      }
    })
  }

  it('文生图只暴露 prompt(无图输入);增强无 prompt(只 image+resolution)', () => {
    const t2i = FEATURE_PUBLISH.text2img.exposed_inputs
    expect(t2i.map((p) => p.key)).toEqual(['prompt'])
    const enh = FEATURE_PUBLISH.enhance.exposed_inputs.map((p) => p.key)
    expect(enh).toContain('image')
    expect(enh).toContain('resolution')
    expect(enh).not.toContain('prompt')
  })

  it('细节增强显式 category=image(无 dec 节点,后端探测不到);其余 auto', () => {
    expect(FEATURE_PUBLISH.enhance.category).toBe('image')
    expect(FEATURE_PUBLISH.text2img.category).toBeUndefined()
    expect(FEATURE_PUBLISH.edit.category).toBeUndefined()
    expect(FEATURE_PUBLISH.angle.category).toBeUndefined()
  })
})

// 链式采样图守门:跨段接力边、终端列表、节点 id 唯一 —— 真机已验 Z-Image→Flux2-Klein×2(零引擎改)。
describe('buildChainWorkflow 跨模型链式采样图', () => {
  const stages: ChainStage[] = [
    { ckpt: '/m/Z-Image-Turbo', arch: 'z-image', prompt: '山湖', steps: 8, cfg: 1 },
    { ckpt: '/m/Flux2-klein-9B', arch: 'flux2', prompt: '加风暴', steps: 20, cfg: 4 },
    { ckpt: '/m/Flux2-klein-9B', arch: 'flux2', prompt: '油画化', steps: 20, cfg: 4 },
  ]
  const { workflow, stageTerminals } = buildChainWorkflow(stages, { width: 1024, height: 1024, seed: 100 })
  const wf = workflow as unknown as {
    nodes: { id: string; type: string; data: Record<string, unknown> }[]
    edges: { id: string; source: string; sourceHandle: string; target: string; targetHandle: string }[]
  }

  it('每段产 4 节点(ckpt/enc/ks/dec)+ 一个 image_output sink', () => {
    expect(wf.nodes.filter((n) => n.type === 'flux2_load_checkpoint')).toHaveLength(3)
    expect(wf.nodes.filter((n) => n.type === 'flux2_ksampler')).toHaveLength(3)
    expect(wf.nodes.filter((n) => n.type === 'flux2_vae_decode')).toHaveLength(3)
    expect(wf.nodes.filter((n) => n.type === 'image_output')).toHaveLength(1)
  })

  it('节点 id 全唯一(无碰撞)', () => {
    const ids = wf.nodes.map((n) => n.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('stageTerminals 指向各段 vae_decode,顺序对', () => {
    expect(stageTerminals).toEqual(['d0', 'd1', 'd2'])
    for (const t of stageTerminals) {
      expect(wf.nodes.find((n) => n.id === t)?.type).toBe('flux2_vae_decode')
    }
  })

  it('跨段接力:第 i 段 ksampler.image ← 第 i-1 段 vae_decode.image', () => {
    // d0.image → k1.image
    expect(wf.edges).toContainEqual(
      expect.objectContaining({ source: 'd0', sourceHandle: 'image', target: 'k1', targetHandle: 'image' }))
    // d1.image → k2.image
    expect(wf.edges).toContainEqual(
      expect.objectContaining({ source: 'd1', sourceHandle: 'image', target: 'k2', targetHandle: 'image' }))
    // 第 0 段 ksampler 无 image 入边(纯文生图,不接力)
    expect(wf.edges.some((e) => e.target === 'k0' && e.targetHandle === 'image')).toBe(false)
  })

  it('最终段 vae_decode → image_output(整图唯一 sink)', () => {
    expect(wf.edges).toContainEqual(
      expect.objectContaining({ source: 'd2', sourceHandle: 'image', target: 'out', targetHandle: 'image' }))
  })

  it('每段 ckpt.file = 传入 abs_path(非裸名,否则 runner 找不到 transformer/)', () => {
    expect(wf.nodes.find((n) => n.id === 'c0')?.data.file).toBe('/m/Z-Image-Turbo')
    expect(wf.nodes.find((n) => n.id === 'c1')?.data.adapter_arch).toBe('flux2')
  })

  it('每段 seed 递增(seed+i),避免同 seed 同图', () => {
    expect(wf.nodes.find((n) => n.id === 'k0')?.data.seed).toBe('100')
    expect(wf.nodes.find((n) => n.id === 'k1')?.data.seed).toBe('101')
    expect(wf.nodes.find((n) => n.id === 'k2')?.data.seed).toBe('102')
  })
})
