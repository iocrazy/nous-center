import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ReactFlowProvider } from '@xyflow/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import DeclarativeNode from './DeclarativeNode'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'
import { NODE_DEFS } from '../../models/workflow'

// 回归:plugin node.yaml 的 select 常写成纯字符串 options(['default','bfloat16',...]),
// 之前前端只认 [{value,label}] → 对字符串取 .value/.label = undefined → 选项全空白
// (细粒度图 loader 的 精度/显卡/架构 下拉打开没任何选项)。
beforeAll(() => {
  DECLARATIVE_NODES['t_stropts'] = {
    type: 't_stropts', label: 'StrOpts', category: 'image', badge: 'T', badgeColor: 'x',
    widgets: [
      { name: 'weight_dtype', label: '精度', widget: 'select',
        options: ['default', 'bfloat16', 'fp8_e4m3'] as unknown as string[], default: 'default' },
    ],
  }
  NODE_DEFS['t_stropts'] = { type: 't_stropts', label: 'StrOpts', inputs: [], outputs: [] }
})

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><ReactFlowProvider>{ui}</ReactFlowProvider></QueryClientProvider>)
}

describe('select widget 字符串 options', () => {
  it('字符串列表 options 渲染出可选项(非空白)', () => {
    wrap(<DeclarativeNode id="n" type="t_stropts" data={{}} selected={false} {...({} as any)} />)
    for (const v of ['default', 'bfloat16', 'fp8_e4m3']) {
      const opt = screen.getByRole('option', { name: v }) as HTMLOptionElement
      expect(opt.value).toBe(v)
    }
  })
})
