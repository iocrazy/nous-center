import { Loader2 } from 'lucide-react'
import { useHealth } from '../../api/health'

/**
 * 启动/加载提示(用户要的「启动提示」c):重启后 resident 模型在后台重载的窗口里,
 * 顶部挂一条细横幅「系统启动中 · 模型加载 M/N」,加载完自动消失。全局(MainLayout)。
 */
export default function StartupBanner() {
  const { data } = useHealth()
  const s = data?.startup
  if (!s || !s.preloading) return null

  return (
    <div
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
        padding: '6px 12px', fontSize: 12, fontWeight: 500,
        background: 'var(--accent-subtle, rgba(99,102,241,0.12))',
        color: 'var(--accent, #6366f1)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
      系统启动中 · 模型加载 {s.resident_loaded}/{s.resident_total}…(刚重启,常驻模型重载中,稍候可用)
      <style>{'@keyframes spin{to{transform:rotate(360deg)}}'}</style>
    </div>
  )
}
