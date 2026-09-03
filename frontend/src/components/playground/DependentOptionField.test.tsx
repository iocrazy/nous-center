import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import SchemaDrivenForm from './SchemaDrivenForm'
import type { ExposedParam } from '../../api/services'
import type { ComfyStyleOption } from '../../api/comfyTemplates'
import { apiFetch } from '../../api/client'

// 只 mock 最底下的 HTTP 层 —— getComfyStyles / useComfyStyles / React Query 全跑真的,
// 要测的正是「切 pack → queryKey 变 → 重新拉 → 清单换掉」这条链。
// (mock 上层的 getComfyStyles 是够不着的:useComfyStyles 引用的是模块内部的绑定。)
vi.mock('../../api/client', () => ({ apiFetch: vi.fn() }))

const fetchMock = vi.mocked(apiFetch)

/** 本次渲染里实际请求过的风格包(按顺序)。 */
function requestedPacks(): string[] {
  return fetchMock.mock.calls
    .map(([path]) => String(path))
    .filter((path) => path.startsWith('/api/v1/comfy/styles'))
    .map((path) => decodeURIComponent(path.split('pack=')[1] ?? ''))
}

const PACKS: Record<string, ComfyStyleOption[]> = {
  fooocus_styles: [
    { value: 'Fooocus Enhance', label: 'Fooocus-优化增强', image: 'https://x/a.jpg' },
    { value: 'sai-anime', label: 'SAI-动漫', image: 'https://x/b.jpg' },
  ],
  krea2_anime: [
    { value: 'krea-cel', label: 'Krea-赛璐璐', image: 'https://x/c.jpg' },
    { value: 'krea-ink', label: 'Krea-水墨', image: 'https://x/d.jpg' },
  ],
}

/** krea2 的两个字段:风格包(下拉)+ 风格(缩略图网格,清单依赖风格包)。 */
function inputs(): ExposedParam[] {
  return [
    {
      node_id: 'bridge',
      key: 'style_pack',
      input_name: 'style_pack',
      label: '风格包',
      type: 'string',
      default: 'fooocus_styles',
      constraints: { enum: ['fooocus_styles', 'krea2_anime'] },
    },
    {
      node_id: 'bridge',
      key: 'styles',
      input_name: 'styles',
      label: '风格(可多选,随风格包切换)',
      type: 'string',
      default: '',
      constraints: {
        enum: ['Fooocus Enhance', 'sai-anime'],
        option_meta: PACKS.fooocus_styles,
        multiple: true,
        options_depends_on: 'style_pack',
        options_source: 'comfy_styles',
      },
    },
  ]
}

function renderForm() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onSubmit = vi.fn()
  render(
    <QueryClientProvider client={qc}>
      <SchemaDrivenForm inputs={inputs()} onSubmit={onSubmit} />
    </QueryClientProvider>,
  )
  return { onSubmit }
}

/** 切风格包:下拉是 NodeSelectPopover(按钮 + 弹层),点开再点目标项。 */
async function switchPack(to: string) {
  fireEvent.click(await screen.findByText('fooocus_styles'))
  fireEvent.click(await screen.findByText(to))
}

beforeEach(() => {
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (path: string) => {
    const pack = decodeURIComponent(path.split('pack=')[1] ?? '')
    return { pack, options: PACKS[pack] ?? [] }
  })
})

describe('SchemaDrivenForm — 选项依赖', () => {
  it('按默认包拉清单,切包后换成新包的选项', async () => {
    renderForm()
    // 初始:依赖参数的 default = fooocus_styles
    await waitFor(() => expect(requestedPacks()).toContain('fooocus_styles'))
    expect(await screen.findByText('SAI-动漫')).toBeInTheDocument()

    await switchPack('krea2_anime')

    await waitFor(() => expect(requestedPacks()).toContain('krea2_anime'))
    expect(await screen.findByText('Krea-赛璐璐')).toBeInTheDocument()
    // 旧包的风格不再出现在网格里
    await waitFor(() => expect(screen.queryByText('SAI-动漫')).not.toBeInTheDocument())
  })

  it('切包后把不在新清单里的已选值清掉(否则提交必 422)', async () => {
    const { onSubmit } = renderForm()
    // 先在默认包里选两个
    fireEvent.click(await screen.findByText('SAI-动漫'))
    fireEvent.click(await screen.findByText('Fooocus-优化增强'))

    await switchPack('krea2_anime')
    await waitFor(() => expect(requestedPacks()).toContain('krea2_anime'))
    // 新包里选一个
    fireEvent.click(await screen.findByText('Krea-水墨'))

    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({
      style_pack: 'krea2_anime',
      styles: 'krea-ink',
    })
  })

  it('拉取失败时退回静态清单并给一行提示', async () => {
    fetchMock.mockRejectedValue(new Error('sidecar 掉线'))
    renderForm()

    expect(await screen.findByText(/选项清单加载失败/)).toBeInTheDocument()
    // 静态 option_meta 仍然渲得出来 —— 加载失败不该让用户面对一片空白
    expect(screen.getByText('SAI-动漫')).toBeInTheDocument()
  })

  it('拉取失败时不动用户已选的值', async () => {
    fetchMock.mockRejectedValue(new Error('sidecar 掉线'))
    const { onSubmit } = renderForm()

    fireEvent.click(await screen.findByText('SAI-动漫'))
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({
      style_pack: 'fooocus_styles',
      styles: 'sai-anime',
    })
  })

  it('没声明依赖的字段一次都不拉(老映射零回归)', async () => {
    const plain = inputs()
    delete (plain[1].constraints as Record<string, unknown>).options_depends_on
    delete (plain[1].constraints as Record<string, unknown>).options_source
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <SchemaDrivenForm inputs={plain} onSubmit={vi.fn()} />
      </QueryClientProvider>,
    )
    expect(await screen.findByText('SAI-动漫')).toBeInTheDocument()
    expect(requestedPacks()).toEqual([])
  })
})
