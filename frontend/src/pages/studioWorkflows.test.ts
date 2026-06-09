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

  it('无 strength 时 ks.data 不含 strength(零回归:默认文生图/纯重生成)', () => {
    for (const k of ['k0', 'k1', 'k2']) {
      expect(wf.nodes.find((n) => n.id === k)?.data).not.toHaveProperty('strength')
    }
  })

  it('PR-A3:z-image 段(i>0)的 strength 透传进 ks.data;第 0 段无上游图不透传', () => {
    const st: ChainStage[] = [
      { ckpt: '/m/Z', arch: 'z-image', prompt: 'a', steps: 8, cfg: 1, strength: 0.6 },
      { ckpt: '/m/Z', arch: 'z-image', prompt: 'b', steps: 8, cfg: 1, strength: 0.6 },
    ]
    const { workflow: w } = buildChainWorkflow(st, { width: 512, height: 512, seed: 1 })
    const n = (w as unknown as { nodes: { id: string; data: Record<string, unknown> }[] }).nodes
    // 第 0 段无上游图 → strength 不写(i>0 才有意义)
    expect(n.find((x) => x.id === 'k0')?.data).not.toHaveProperty('strength')
    // 第 1 段有上游图 → strength 透传(引擎再按 arch/范围门控)
    expect(n.find((x) => x.id === 'k1')?.data.strength).toBe(0.6)
  })
})

describe('buildChainWorkflow 留噪 latent 接力(PR-B3,同模型双采)', () => {
  // Z-Image 双采:base(段0)→ latent 留噪续采(段1),split=5;再 flux2 图参考编辑(段2)。
  const stages: ChainStage[] = [
    { ckpt: '/m/Z', arch: 'z-image', prompt: '人像', steps: 12, cfg: 2.3 },
    { ckpt: '/m/Z', arch: 'z-image', prompt: '人像精修', steps: 12, cfg: 1.15, relay: 'latent', split: 5 },
    { ckpt: '/m/Flux2', arch: 'flux2', prompt: '风格化', steps: 20, cfg: 4 },
  ]
  const { workflow } = buildChainWorkflow(stages, { width: 1024, height: 1024, seed: 7 })
  const wf = workflow as unknown as {
    nodes: { id: string; type: string; data: Record<string, unknown> }[]
    edges: { id: string; source: string; sourceHandle: string; target: string; targetHandle: string }[]
  }
  const data = (id: string) => wf.nodes.find((n) => n.id === id)!.data

  it('latent 续采段:ks.init_latent ← 上段 vae_decode.latent_ref(不是 image 接力边)', () => {
    expect(wf.edges).toContainEqual(
      expect.objectContaining({ source: 'd0', sourceHandle: 'latent_ref', target: 'k1', targetHandle: 'init_latent' }))
    expect(wf.edges.some((e) => e.source === 'd0' && e.targetHandle === 'image')).toBe(false)
  })

  it('base 段(段0)vae_decode output_mode=latent(导出带噪 latent 给下段续采)', () => {
    expect(data('d0').output_mode).toBe('latent')
    expect(data('d1')).not.toHaveProperty('output_mode')  // 续采段终端出图,不设 latent
  })

  it('base 段 ks:end_at_step=split + 保留余噪;refiner 段 ks:start_at_step=split + 不重加噪', () => {
    expect(data('k0').end_at_step).toBe(5)
    expect(data('k0').return_with_leftover_noise).toBe(true)
    expect(data('k0')).not.toHaveProperty('start_at_step')
    expect(data('k1').start_at_step).toBe(5)
    expect(data('k1').add_noise).toBe(false)
    expect(data('k1')).not.toHaveProperty('end_at_step')
  })

  it('latent 接力组共享总步数:refiner 段 steps = base 段 steps', () => {
    expect(data('k0').steps).toBe(12)
    expect(data('k1').steps).toBe(12)  // 组首 base 的 steps,非 refiner 自己
  })

  it('latent 续采段不写 strength(不走 img2img)', () => {
    expect(data('k1')).not.toHaveProperty('strength')
  })

  it('跨模型段(段2 flux2)仍走图接力:d1.image → k2.image', () => {
    expect(wf.edges).toContainEqual(
      expect.objectContaining({ source: 'd1', sourceHandle: 'image', target: 'k2', targetHandle: 'image' }))
  })

  it('零回归:全 image 接力(无 relay)不产分段字段 / latent_ref 边', () => {
    const plain: ChainStage[] = [
      { ckpt: '/m/Z', arch: 'z-image', prompt: 'a', steps: 8, cfg: 1 },
      { ckpt: '/m/F', arch: 'flux2', prompt: 'b', steps: 20, cfg: 4 },
    ]
    const { workflow: w } = buildChainWorkflow(plain, { width: 512, height: 512, seed: 1 })
    const n = (w as unknown as { nodes: { id: string; data: Record<string, unknown> }[] }).nodes
    const e = (w as unknown as { edges: { sourceHandle: string }[] }).edges
    for (const id of ['k0', 'k1']) {
      const d = n.find((x) => x.id === id)!.data
      for (const f of ['start_at_step', 'end_at_step', 'add_noise', 'return_with_leftover_noise']) {
        expect(d).not.toHaveProperty(f)
      }
    }
    expect(n.find((x) => x.id === 'd0')!.data).not.toHaveProperty('output_mode')
    expect(e.some((x) => x.sourceHandle === 'latent_ref')).toBe(false)
  })

  it('校验:跨架构 latent 接力(flux2 续采 z-image)报错', () => {
    const bad: ChainStage[] = [
      { ckpt: '/m/Z', arch: 'z-image', prompt: 'a', steps: 12, cfg: 1 },
      { ckpt: '/m/F', arch: 'flux2', prompt: 'b', steps: 20, cfg: 4, relay: 'latent', split: 5 },
    ]
    expect(() => buildChainWorkflow(bad, { width: 512, height: 512, seed: 1 })).toThrow(/z-image|latent/)
  })

  it('校验:split 越界(>=总步数)报错', () => {
    const bad: ChainStage[] = [
      { ckpt: '/m/Z', arch: 'z-image', prompt: 'a', steps: 12, cfg: 1 },
      { ckpt: '/m/Z', arch: 'z-image', prompt: 'b', steps: 12, cfg: 1, relay: 'latent', split: 12 },
    ]
    expect(() => buildChainWorkflow(bad, { width: 512, height: 512, seed: 1 })).toThrow(/split/)
  })
})
