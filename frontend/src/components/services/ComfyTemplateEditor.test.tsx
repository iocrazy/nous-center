import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import ComfyTemplateEditor from './ComfyTemplateEditor'
import * as api from '../../api/comfyTemplates'

// React Flow rendering is exercised separately (comfyGraphLayout unit tests + real
// usage precedent in WorkflowAppEditor); mock the graph child here so editor tests
// stay fast/robust in jsdom and focus on popover/table/save logic (per task brief).
vi.mock('./ComfyTemplateGraph', () => ({
  default: ({ nodes, onNodeClick }: { nodes: Array<{ id: string; class_type: string }>; onNodeClick: (id: string) => void }) => (
    <div data-testid="graph-mock">
      {nodes.map((n) => (
        <button key={n.id} type="button" onClick={() => onNodeClick(n.id)}>
          {n.class_type} #{n.id}
        </button>
      ))}
    </div>
  ),
}))

vi.mock('../../api/comfyTemplates', () => ({
  getComfyTemplate: vi.fn(),
  getComfyHealth: vi.fn(),
  getObjectInfo: vi.fn(),
  putMapping: vi.fn(),
  reuploadComfyTemplate: vi.fn(),
}))

const WORKFLOW = {
  '1': { class_type: 'LoadImage', inputs: { image: 'photo.png' } },
  '2': { class_type: 'KSampler', inputs: { pixels: ['1', 0], steps: 20, seed: 42 } },
}

function baseDetail(exposed: api.ComfyExposedParam[] = []): api.ComfyTemplateDetail {
  return {
    id: '7',
    name: 'tpl',
    service_name: 'svc-7',
    workflow_json: WORKFLOW,
    exposed_params: exposed,
  }
}

beforeEach(() => {
  vi.mocked(api.getComfyTemplate).mockResolvedValue(baseDetail())
  vi.mocked(api.getComfyHealth).mockResolvedValue({
    online: true, queue_depth: 0, version: '0.3.1', base_url: 'http://x', timeout_s: 30,
  })
  vi.mocked(api.getObjectInfo).mockResolvedValue({
    KSampler: {
      input: {
        required: {
          steps: ['INT', { default: 20, min: 1, max: 150 }],
          seed: ['INT', { default: 0, min: 0, max: 999999 }],
        },
      },
    },
  })
  vi.mocked(api.putMapping).mockResolvedValue({ ok: true })
  vi.mocked(api.reuploadComfyTemplate).mockResolvedValue({ stale_keys: [] })
})

describe('ComfyTemplateEditor', () => {
  it('渲染每个节点一张卡', async () => {
    render(<ComfyTemplateEditor templateId="7" />)
    expect(await screen.findByText('LoadImage #1')).toBeInTheDocument()
    expect(screen.getByText('KSampler #2')).toBeInTheDocument()
  })

  it('点卡片开弹窗,列出该节点原始 inputs(跳过节点引用连线)', async () => {
    render(<ComfyTemplateEditor templateId="7" />)
    fireEvent.click(await screen.findByText('KSampler #2'))
    expect(await screen.findByText('steps')).toBeInTheDocument()
    expect(screen.getByText('seed')).toBeInTheDocument()
    expect(screen.queryByText('pixels')).not.toBeInTheDocument()
  })

  it('弹窗内勾选暴露 → 汇总表出现该行', async () => {
    render(<ComfyTemplateEditor templateId="7" />)
    fireEvent.click(await screen.findByText('KSampler #2'))
    const stepsRow = (await screen.findByText('steps')).closest('[data-input-row]') as HTMLElement
    fireEvent.click(within(stepsRow).getByRole('checkbox', { name: /暴露/ }))
    await screen.findByTestId('exposed-summary-table')
    expect(screen.getByTestId('exposed-row-steps')).toBeInTheDocument()
  })

  it('object_info 预填数值类型 min/max(未勾选也先展示,勾选后才可编辑)', async () => {
    render(<ComfyTemplateEditor templateId="7" />)
    fireEvent.click(await screen.findByText('KSampler #2'))
    const stepsRow = (await screen.findByText('steps')).closest('[data-input-row]') as HTMLElement
    expect(within(stepsRow).getByDisplayValue('1')).toBeInTheDocument() // min
    expect(within(stepsRow).getByDisplayValue('150')).toBeInTheDocument() // max
  })

  it('seed 类整数出「随机」勾选', async () => {
    render(<ComfyTemplateEditor templateId="7" />)
    fireEvent.click(await screen.findByText('KSampler #2'))
    const seedRow = (await screen.findByText('seed')).closest('[data-input-row]') as HTMLElement
    expect(within(seedRow).getByRole('checkbox', { name: /随机/ })).toBeInTheDocument()
  })

  it('保存调用 putMapping,payload 形状正确(comfy_node_id/comfy_input)', async () => {
    render(<ComfyTemplateEditor templateId="7" />)
    fireEvent.click(await screen.findByText('KSampler #2'))
    const stepsRow = (await screen.findByText('steps')).closest('[data-input-row]') as HTMLElement
    fireEvent.click(within(stepsRow).getByRole('checkbox', { name: /暴露/ }))
    fireEvent.click(screen.getByRole('button', { name: /保存配置/ }))
    await waitFor(() => expect(api.putMapping).toHaveBeenCalled())
    const [id, params] = vi.mocked(api.putMapping).mock.calls[0]
    expect(id).toBe('7')
    expect(params).toEqual([
      expect.objectContaining({
        key: 'steps',
        comfy_node_id: '2',
        comfy_input: 'steps',
        type: 'integer',
        min: 1,
        max: 150,
      }),
    ])
  })

  it('sidecar 离线 → 降级提示', async () => {
    vi.mocked(api.getComfyHealth).mockResolvedValue({
      online: false, queue_depth: 0, version: '', base_url: 'http://x', timeout_s: 30,
    })
    render(<ComfyTemplateEditor templateId="7" />)
    expect(await screen.findByText(/离线/)).toBeInTheDocument()
  })

  it('已保存的 exposed_params 回显进汇总表', async () => {
    vi.mocked(api.getComfyTemplate).mockResolvedValue(baseDetail([
      { key: 'steps', label: '步数', type: 'integer', comfy_node_id: '2', comfy_input: 'steps', min: 1, max: 150 },
    ]))
    render(<ComfyTemplateEditor templateId="7" />)
    await screen.findByTestId('exposed-summary-table')
    expect(screen.getByTestId('exposed-row-steps')).toBeInTheDocument()
  })

  it('上传替换:重新上传工作流后高亮 stale_keys', async () => {
    vi.mocked(api.getComfyTemplate).mockResolvedValue(baseDetail([
      { key: 'gone', label: 'gone', type: 'string', comfy_node_id: '99', comfy_input: 'x' },
    ]))
    vi.mocked(api.reuploadComfyTemplate).mockResolvedValue({ stale_keys: ['gone'] })
    render(<ComfyTemplateEditor templateId="7" />)
    await screen.findByTestId('exposed-summary-table')
    expect(screen.getByTestId('exposed-row-gone')).toHaveAttribute('data-stale', 'false')

    const file = new File([JSON.stringify(WORKFLOW)], 'wf.json', { type: 'application/json' })
    fireEvent.change(screen.getByTestId('comfy-reupload-input'), { target: { files: [file] } })
    await waitFor(() => expect(api.reuploadComfyTemplate).toHaveBeenCalledWith('7', WORKFLOW))
    await waitFor(() => {
      expect(screen.getByTestId('exposed-row-gone')).toHaveAttribute('data-stale', 'true')
    })
  })
})
