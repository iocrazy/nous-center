import { describe, it, expect } from 'vitest'
import { appendOutput } from './imageOutputGallery'

const img = (url: string) => ({ url, seed: 1, steps: 8, cfg: 1, width: 1024, height: 1024, durationMs: 100 })

describe('ImageOutputNode 累积画廊 appendOutput', () => {
  it('追加新图(对齐 IC OUTPUT:每次运行 append 不覆盖)', () => {
    const a = appendOutput([], img('u1'))
    const b = appendOutput(a, img('u2'))
    expect(b.map((x) => x.url)).toEqual(['u1', 'u2'])
  })

  it('同 url 去重 —— 返回原数组引用(node 据此跳过 updateNode)', () => {
    const a = appendOutput([], img('u1'))
    const b = appendOutput(a, img('u1'))
    expect(b).toBe(a) // 引用不变 = 不触发写回
    expect(b.length).toBe(1)
  })

  it('超出上限丢最旧(防 data 撑爆)', () => {
    let acc: ReturnType<typeof appendOutput> = []
    for (let i = 0; i < 5; i++) acc = appendOutput(acc, img(`u${i}`), 3)
    expect(acc.map((x) => x.url)).toEqual(['u2', 'u3', 'u4'])
  })

  it('保留每图元信息(seed/steps/cfg/分辨率/时长)', () => {
    const a = appendOutput([], img('u1'))
    expect(a[0]).toMatchObject({ url: 'u1', seed: 1, steps: 8, cfg: 1, width: 1024, height: 1024, durationMs: 100 })
  })

  it('batch:一次 node_complete 的 image_urls 列表逐张 fold append(PR-B1 后端一次出 N 张)', () => {
    // 模拟 handler 对 detail.image_urls=[u1,u2,u3] 的折叠累积
    let next = appendOutput([], img('existing'))
    for (const u of ['u1', 'u2', 'u3']) next = appendOutput(next, img(u))
    expect(next.map((x) => x.url)).toEqual(['existing', 'u1', 'u2', 'u3'])
  })
})
