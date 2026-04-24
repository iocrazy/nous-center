import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import SettingsOverlay from './SettingsOverlay'

// 验证 m16 sub-nav 切换：通用 / 节点包 / 关于。

vi.mock('../../api/settings', () => ({
  useServerSettings: () => ({ data: undefined }),
  useUpdateServerSettings: () => ({ mutate: vi.fn(), isPending: false }),
}))

vi.mock('../../api/nodes', () => ({
  useNodePackages: () => ({ data: {}, isLoading: false }),
  useRescanPackages: () => ({ mutate: vi.fn(), isPending: false }),
  useInstallPackageZip: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useInstallPackageGit: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUninstallPackage: () => ({ mutateAsync: vi.fn(), isPending: false, variables: undefined }),
  useInstallPackageDeps: () => ({ mutateAsync: vi.fn(), isPending: false, variables: undefined }),
}))

function withQuery(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>
}

describe('SettingsOverlay sub-nav', () => {
  it('default shows 通用 panel', () => {
    render(withQuery(<SettingsOverlay />))
    expect(screen.getByText('路径配置')).toBeTruthy()
  })

  it('switching to 节点包 shows packages panel header', () => {
    render(withQuery(<SettingsOverlay />))
    fireEvent.click(screen.getByText('节点包'))
    expect(screen.getByText(/0 个已安装/)).toBeTruthy()
  })

  it('switching to 关于 shows about page', () => {
    render(withQuery(<SettingsOverlay />))
    fireEvent.click(screen.getByText('关于'))
    expect(screen.getByText('Nous Center')).toBeTruthy()
    expect(screen.getByText(/协议兼容/)).toBeTruthy()
  })
})
