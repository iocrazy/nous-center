/**
 * Image page placeholder(PR-2,任务面板重置)。
 *
 * 真正的图像生成 UI(prompt 输入 / 参数面板 / gallery)在后续 PR 实施。本页只占
 * 路由 `/image`,让 GlobalTopbar 上的 Image tab 可被选中,UX 不留 404。
 */
import { Image as ImageIcon } from 'lucide-react'

export default function ImagePage() {
  return <ServicePlaceholder icon={<ImageIcon size={32} />} title="Image" sub="图像生成" />
}

function ServicePlaceholder({ icon, title, sub }: {
  icon: React.ReactNode; title: string; sub: string
}) {
  return (
    <div
      className="flex-1 flex flex-col items-center justify-center"
      style={{ background: 'var(--tp-bg-base)', color: 'var(--tp-text)' }}
    >
      <div style={{ color: 'var(--type-image)' }}>{icon}</div>
      <h1 className="mt-4 text-2xl font-semibold">{title}</h1>
      <p className="mt-1 text-sm" style={{ color: 'var(--tp-text-muted)' }}>{sub} · 即将上线</p>
    </div>
  )
}
