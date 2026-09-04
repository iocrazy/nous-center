import { describe, it, expect, vi } from 'vitest'
import { buildGpuAssignSubmenu } from './gpuAssignMenu'

const DEVICES = [
  { index: 0, name: 'NVIDIA GeForce RTX 3090', vram_gb: 24 },
  { index: 1, name: 'NVIDIA RTX PRO 6000', vram_gb: 96 },
  { index: 2, name: 'NVIDIA GeForce RTX 3090', vram_gb: 24 },
]
const GROUPS = [
  { gpus: [0, 2], name: 'NVIDIA GeForce RTX 3090', nvlink: true, total_gb: 48 },
]

describe('GPU 分配子菜单（单卡 + 组合）', () => {
  it('没有可用组时只有单卡项，不出分隔线', () => {
    const items = buildGpuAssignSubmenu({
      devices: DEVICES, groups: [], currentGpu: 1, currentGroup: null,
      onPickGpu: vi.fn(), onPickGroup: vi.fn(),
    })
    expect(items).toHaveLength(3)
    expect(items.some((i) => i.divider)).toBe(false)
  })

  it('有组时追加分隔线 + 组合项，标注 NVLink 与总显存', () => {
    const items = buildGpuAssignSubmenu({
      devices: DEVICES, groups: GROUPS, currentGpu: 1, currentGroup: null,
      onPickGpu: vi.fn(), onPickGroup: vi.fn(),
    })
    expect(items).toHaveLength(5)  // 3 单卡 + 分隔线 + 1 组合
    expect(items[3].divider).toBe(true)
    expect(items[4].label).toContain('组合 GPU 0+2')
    expect(items[4].label).toContain('NVLink')
    expect(items[4].label).toContain('48GB')
  })

  it('点组合项回调拿到的是 gpus 数组（→ 请求体 {gpus}）', () => {
    const onPickGroup = vi.fn()
    const items = buildGpuAssignSubmenu({
      devices: DEVICES, groups: GROUPS, currentGpu: 1, currentGroup: null,
      onPickGpu: vi.fn(), onPickGroup,
    })
    items[4].onClick!()
    expect(onPickGroup).toHaveBeenCalledWith([0, 2])
  })

  it('当前就是该组 → 组合项置灰', () => {
    const items = buildGpuAssignSubmenu({
      devices: DEVICES, groups: GROUPS, currentGpu: 0, currentGroup: [0, 2],
      onPickGpu: vi.fn(), onPickGroup: vi.fn(),
    })
    expect(items[4].disabled).toBe(true)
  })

  it('已在组里时单卡项全部可点（= 退出组回到单卡）', () => {
    const items = buildGpuAssignSubmenu({
      devices: DEVICES, groups: GROUPS, currentGpu: 0, currentGroup: [0, 2],
      onPickGpu: vi.fn(), onPickGroup: vi.fn(),
    })
    expect(items.slice(0, 3).every((i) => i.disabled === false)).toBe(true)
  })

  it('没配组时当前单卡置灰', () => {
    const items = buildGpuAssignSubmenu({
      devices: DEVICES, groups: GROUPS, currentGpu: 1, currentGroup: null,
      onPickGpu: vi.fn(), onPickGroup: vi.fn(),
    })
    expect(items[1].disabled).toBe(true)
    expect(items[0].disabled).toBe(false)
  })

  it('PCIe（无 NVLink）组也列出来，只是标注不同', () => {
    const items = buildGpuAssignSubmenu({
      devices: DEVICES,
      groups: [{ gpus: [0, 2], name: 'RTX 3090', nvlink: false, total_gb: 48 }],
      currentGpu: 1, currentGroup: null,
      onPickGpu: vi.fn(), onPickGroup: vi.fn(),
    })
    expect(items[4].label).toContain('PCIe')
  })
})
