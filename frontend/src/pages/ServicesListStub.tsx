import { Wrench } from 'lucide-react'

export default function ServicesListStub() {
  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16] flex items-center justify-center"
      style={{ background: 'var(--bg)' }}
    >
      <div
        style={{
          maxWidth: 480,
          padding: 32,
          textAlign: 'center',
          color: 'var(--muted)',
        }}
      >
        <div
          className="inline-flex items-center justify-center mb-4"
          style={{
            width: 56,
            height: 56,
            borderRadius: 12,
            background: 'var(--accent-glow)',
            color: 'var(--accent)',
          }}
        >
          <Wrench size={24} />
        </div>
        <h2
          className="text-[18px] font-semibold mb-2"
          style={{ color: 'var(--fg)' }}
        >
          v3 服务页面重构中
        </h2>
        <p className="text-[13px] leading-relaxed">
          IA 重构 v3：服务列表 / 详情（Playground / API 文档 / Key 授权 / 用量
          5 tabs）将随 PR-B 上线。后端契约 (PR-A) 已就绪。
        </p>
        <p className="text-[12px] mt-3" style={{ color: 'var(--muted)' }}>
          docs/designs/2026-04-22-ia-rebuild-v3.md
        </p>
      </div>
    </div>
  )
}
