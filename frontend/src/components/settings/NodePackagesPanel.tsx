import { useRef, useState } from 'react'
import {
  GitBranch,
  Loader2,
  Package,
  Pause,
  Play,
  RefreshCw,
  Trash2,
  Upload,
} from 'lucide-react'
import {
  useInstallPackageDeps,
  useInstallPackageGit,
  useInstallPackageZip,
  useNodePackages,
  useRescanPackages,
  useTogglePackage,
  useUninstallPackage,
} from '../../api/nodes'
import { useToastStore } from '../../stores/toast'

// m12 内嵌版 — 同 NodePackagesOverlay 但去掉了 absolute inset-0 容器，
// 让 SettingsOverlay 把它作为子页嵌进去。

export default function NodePackagesPanel() {
  const { data: packages, isLoading } = useNodePackages()
  const rescan = useRescanPackages()
  const installZip = useInstallPackageZip()
  const installGit = useInstallPackageGit()
  const uninstall = useUninstallPackage()
  const installDeps = useInstallPackageDeps()
  const togglePkg = useTogglePackage()
  const fileInput = useRef<HTMLInputElement>(null)
  const [gitUrl, setGitUrl] = useState('')
  const toast = useToastStore.getState().add

  const onPickZip = async (f: File | null) => {
    if (!f) return
    try {
      const r = await installZip.mutateAsync({ file: f })
      toast(`已安装节点包 ${r.installed}`, 'success')
    } catch (e) {
      toast(`安装失败: ${(e as Error).message}`, 'error')
    } finally {
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const onInstallGit = async () => {
    const url = gitUrl.trim()
    if (!url) return
    try {
      const r = await installGit.mutateAsync({ repo_url: url })
      toast(`已克隆节点包 ${r.installed}`, 'success')
      setGitUrl('')
    } catch (e) {
      toast(`克隆失败: ${(e as Error).message}`, 'error')
    }
  }

  const onUninstall = async (name: string) => {
    if (!confirm(`确定卸载节点包 "${name}" 吗？`)) return
    try {
      await uninstall.mutateAsync(name)
      toast(`已卸载 ${name}`, 'success')
    } catch (e) {
      toast(`卸载失败: ${(e as Error).message}`, 'error')
    }
  }

  const onInstallDeps = async (name: string) => {
    try {
      const r = await installDeps.mutateAsync(name)
      toast(`${name} 依赖${r.status === 'no_requirements' ? '（无需安装）' : '安装完成'}`, 'success')
    } catch (e) {
      toast(`依赖安装失败: ${(e as Error).message}`, 'error')
    }
  }

  const onToggle = async (name: string, enabled: boolean) => {
    try {
      await togglePkg.mutateAsync({ name, enabled })
      toast(`${name} 已${enabled ? '启用' : '禁用'}`, 'success')
    } catch (e) {
      toast(`切换失败: ${(e as Error).message}`, 'error')
    }
  }

  const list = packages ? Object.values(packages) : []

  return (
    <div>
      <div className="flex items-center gap-3" style={{ marginBottom: 12 }}>
        <Package size={16} style={{ color: 'var(--accent-2)' }} />
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', margin: 0 }}>
          节点包
        </h3>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
          {list.length} 个已安装
        </span>
        <div className="flex-1" />
        <ToolbarBtn
          icon={<RefreshCw size={11} />}
          label="重新扫描"
          onClick={() => rescan.mutate()}
          loading={rescan.isPending}
        />
      </div>

      <div
        style={{
          background: 'var(--bg-accent)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          padding: 14,
          marginBottom: 14,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: 'var(--muted)',
            textTransform: 'uppercase',
            letterSpacing: 0.5,
            marginBottom: 10,
          }}
        >
          安装新包
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>
              <Upload size={11} className="inline mr-1" />
              上传 .zip（含 node.yaml）
            </div>
            <input
              ref={fileInput}
              type="file"
              accept=".zip"
              onChange={(e) => onPickZip(e.target.files?.[0] ?? null)}
              style={{ fontSize: 11, color: 'var(--text)' }}
            />
            {installZip.isPending && (
              <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                <Loader2 size={10} className="inline mr-1 animate-spin" /> 上传中...
              </div>
            )}
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>
              <GitBranch size={11} className="inline mr-1" />
              Git 仓库 URL
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={gitUrl}
                onChange={(e) => setGitUrl(e.target.value)}
                placeholder="https://github.com/user/repo.git"
                style={{
                  flex: 1,
                  fontSize: 11,
                  padding: '4px 8px',
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  color: 'var(--text)',
                }}
              />
              <ToolbarBtn
                label="克隆安装"
                onClick={onInstallGit}
                loading={installGit.isPending}
                disabled={!gitUrl.trim()}
              />
            </div>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div style={{ fontSize: 11, color: 'var(--muted)', textAlign: 'center', padding: 20 }}>
          加载中...
        </div>
      ) : list.length === 0 ? (
        <div
          style={{
            fontSize: 12,
            color: 'var(--muted)',
            textAlign: 'center',
            padding: 32,
            border: '1px dashed var(--border)',
            borderRadius: 6,
          }}
        >
          暂无已安装的节点包 — 上传 zip 或克隆 git 仓库开始
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2">
          {list.map((p) => (
            <PackageCard
              key={p.name}
              pkg={p}
              onUninstall={() => onUninstall(p.name)}
              onInstallDeps={() => onInstallDeps(p.name)}
              onToggle={() => onToggle(p.name, !p.enabled)}
              installingDeps={installDeps.isPending && installDeps.variables === p.name}
              uninstalling={uninstall.isPending && uninstall.variables === p.name}
              toggling={
                togglePkg.isPending && togglePkg.variables?.name === p.name
              }
            />
          ))}
        </div>
      )}
    </div>
  )
}

function PackageCard({
  pkg,
  onUninstall,
  onInstallDeps,
  onToggle,
  installingDeps,
  uninstalling,
  toggling,
}: {
  pkg: {
    name: string
    version: string
    description: string
    node_count: number
    nodes: string[]
    enabled: boolean
  }
  onUninstall: () => void
  onInstallDeps: () => void
  onToggle: () => void
  installingDeps: boolean
  uninstalling: boolean
  toggling: boolean
}) {
  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        padding: '10px 12px',
        borderRadius: 6,
        opacity: pkg.enabled ? 1 : 0.65,
      }}
    >
      <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>{pkg.name}</span>
        <span
          style={{
            fontSize: 9,
            padding: '1px 6px',
            borderRadius: 3,
            background: pkg.enabled
              ? 'rgba(34,197,94,0.15)'
              : 'var(--bg-accent)',
            color: pkg.enabled ? 'var(--accent-2, #22c55e)' : 'var(--muted)',
            border: pkg.enabled ? 'none' : '1px solid var(--border)',
          }}
        >
          {pkg.enabled ? '已启用' : '已禁用'}
        </span>
        <span
          style={{
            fontSize: 9,
            padding: '1px 5px',
            borderRadius: 3,
            background: 'color-mix(in srgb, var(--accent-2) 18%, transparent)',
            color: 'var(--accent-2)',
          }}
        >
          v{pkg.version}
        </span>
        <span style={{ fontSize: 10, color: 'var(--muted)' }}>{pkg.node_count} 节点</span>
        <div className="flex-1" />
        <ToolbarBtn label="安装依赖" onClick={onInstallDeps} loading={installingDeps} />
        <ToolbarBtn
          icon={pkg.enabled ? <Pause size={11} /> : <Play size={11} />}
          label={pkg.enabled ? '禁用' : '启用'}
          onClick={onToggle}
          loading={toggling}
        />
        <ToolbarBtn
          icon={<Trash2 size={11} />}
          label="卸载"
          onClick={onUninstall}
          loading={uninstalling}
          danger
        />
      </div>
      {pkg.description && (
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>
          {pkg.description}
        </div>
      )}
      <div className="flex flex-wrap gap-1">
        {pkg.nodes.map((n) => (
          <span
            key={n}
            style={{
              fontSize: 9,
              padding: '1px 5px',
              borderRadius: 3,
              background: 'var(--bg)',
              color: 'var(--muted)',
              fontFamily: 'var(--mono)',
            }}
          >
            {n}
          </span>
        ))}
      </div>
    </div>
  )
}

function ToolbarBtn({
  label,
  icon,
  onClick,
  loading,
  disabled,
  danger,
}: {
  label: string
  icon?: React.ReactNode
  onClick?: () => void
  loading?: boolean
  disabled?: boolean
  danger?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        fontSize: 10,
        padding: '3px 8px',
        background: danger ? 'color-mix(in srgb, var(--accent) 12%, transparent)' : 'var(--bg)',
        color: danger ? 'var(--accent)' : 'var(--text)',
        border: '1px solid var(--border)',
        borderRadius: 3,
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
      }}
    >
      {loading ? <Loader2 size={11} className="animate-spin" /> : icon}
      {label}
    </button>
  )
}
