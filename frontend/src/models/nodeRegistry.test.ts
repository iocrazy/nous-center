import { describe, it, expect, vi, beforeEach } from 'vitest'
import { DECLARATIVE_NODES, loadPluginDefinitions } from './nodeRegistry'

describe('loadPluginDefinitions — componentRole 透传(PR-2)', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    // 用唯一节点类型避免与既有注册冲突
    delete DECLARATIVE_NODES['pr2_test_unet_loader']
  })

  it('把插件节点的 componentRole + widget role 透传进 DECLARATIVE_NODES', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        pr2_test_unet_loader: {
          label: 'Load Diffusion Model',
          category: 'image',
          badge: 'Loader',
          badgeColor: 'var(--accent)',
          componentRole: 'unet',
          widgets: [
            { name: 'file', label: '文件', widget: 'component_select', role: 'unet' },
            { name: 'weight_dtype', label: '精度', widget: 'select', options: [] },
          ],
          outputs: [{ id: 'model', type: 'MODEL', label: 'MODEL' }],
        },
      }),
    }))

    await loadPluginDefinitions()

    const def = DECLARATIVE_NODES['pr2_test_unet_loader']
    expect(def).toBeDefined()
    // 四态头靠 declDef.componentRole 触发(DeclarativeNode.tsx:554)
    expect(def.componentRole).toBe('unet')
    // component_select 靠 widget.role 拉对应组件列表(DeclarativeNode.tsx:368)
    expect(def.widgets.find((w) => w.name === 'file')?.role).toBe('unet')
  })
})
