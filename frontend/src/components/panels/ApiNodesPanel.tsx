import FloatingPanel from '../layout/FloatingPanel'

const MOCK_APIS = [
  { method: 'POST', path: '/v1/tts/synthesize', published: true },
  { method: 'POST', path: '/v1/collections/voices', published: true },
  { method: 'GET', path: '/v1/presets', published: false },
]

export default function ApiNodesPanel() {
  return (
    <FloatingPanel title="API Nodes">
      {MOCK_APIS.map((api) => (
        <div
          key={api.path}
          className="rounded-md mb-1.5"
          style={{
            background: 'var(--card)',
            border: '1px solid var(--border)',
            padding: '8px 10px',
            transition: 'all 0.12s',
            opacity: api.published ? 1 : 0.6,
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-strong)' }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
        >
          <div className="flex items-center gap-2" style={{ marginBottom: 3 }}>
            <span
              style={{
                fontSize: 9,
                fontWeight: 700,
                fontFamily: 'var(--mono)',
                padding: '1px 5px',
                borderRadius: 3,
                background: api.method === 'POST' ? 'rgba(34,197,94,0.15)' : 'rgba(59,130,246,0.12)',
                color: api.method === 'POST' ? 'var(--ok)' : 'var(--info)',
              }}
            >
              {api.method}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>
              {api.path}
            </span>
          </div>
          <div className="flex items-center gap-1.5" style={{ fontSize: 10, color: 'var(--muted)' }}>
            <div
              className="relative cursor-pointer"
              style={{
                width: 24,
                height: 12,
                borderRadius: 6,
                background: api.published ? 'var(--ok)' : 'var(--muted-strong)',
                transition: 'background 0.15s',
              }}
            >
              <div
                className="absolute rounded-full bg-white"
                style={{
                  width: 8,
                  height: 8,
                  top: 2,
                  left: api.published ? 14 : 2,
                  transition: 'left 0.15s',
                }}
              />
            </div>
            <span style={{ color: api.published ? 'var(--ok)' : 'var(--muted)', fontSize: 9 }}>
              {api.published ? '已发布' : '未发布'}
            </span>
          </div>
        </div>
      ))}
      <div
        className="rounded-md mb-1.5 text-center cursor-pointer"
        style={{
          border: '1px dashed var(--border)',
          padding: '8px 10px',
          fontSize: 10,
          color: 'var(--muted)',
          transition: 'border-color 0.12s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-strong)' }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
      >
        + 添加 API 端点
      </div>
    </FloatingPanel>
  )
}
