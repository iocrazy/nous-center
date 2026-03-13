import { useEngines, useLoadEngine, useUnloadEngine, type EngineInfo } from '../../api/engines'

export default function ModelsOverlay() {
  const { data: engines, isLoading, isError } = useEngines()
  const loadEngine = useLoadEngine()
  const unloadEngine = useUnloadEngine()

  // Group engines by display_name prefix (e.g. "CosyVoice2-0.5B" → "CosyVoice2")
  const groups = (engines ?? []).reduce<Record<string, EngineInfo[]>>((acc, e) => {
    const group = e.display_name.replace(/-[^-]+$/, '').replace(/ .*$/, '')
    ;(acc[group] ??= []).push(e)
    return acc
  }, {})

  const handleToggle = (engine: EngineInfo) => {
    if (engine.status === 'loaded') {
      unloadEngine.mutate(engine.name)
    } else {
      loadEngine.mutate(engine.name)
    }
  }

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ padding: 16 }}>

        {isLoading && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
        )}

        {isError && !engines && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            无法连接后端服务，等待重试...
          </div>
        )}

        {Object.entries(groups).map(([groupName, models]) => (
          <div key={groupName} className="mb-4">
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: 'var(--accent-2)',
                marginBottom: 8,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              {groupName}
            </div>

            {models.map((model) => {
              const busy =
                (loadEngine.isPending && loadEngine.variables === model.name) ||
                (unloadEngine.isPending && unloadEngine.variables === model.name)

              return (
                <div
                  key={model.name}
                  className="flex items-center gap-3 rounded-md mb-1.5"
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    padding: '10px 12px',
                  }}
                >
                  <div className="flex-1">
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}>
                      {model.display_name}
                    </div>
                    <div className="flex gap-3 mt-1" style={{ fontSize: 10, color: 'var(--muted)' }}>
                      <span>{model.vram_gb} GB</span>
                      <span>GPU {model.gpu}</span>
                      {model.resident && <span style={{ color: 'var(--warn)' }}>resident</span>}
                    </div>
                  </div>

                  <StatusBadge status={model.status} />

                  <button
                    disabled={busy}
                    onClick={() => handleToggle(model)}
                    style={{
                      padding: '4px 10px',
                      fontSize: 10,
                      borderRadius: 4,
                      border: '1px solid var(--border)',
                      background: model.status === 'loaded' ? 'none' : 'var(--accent)',
                      color: model.status === 'loaded' ? 'var(--muted)' : '#fff',
                      cursor: busy ? 'wait' : 'pointer',
                      opacity: busy ? 0.6 : 1,
                    }}
                  >
                    {busy ? '...' : model.status === 'loaded' ? 'Unload' : 'Load'}
                  </button>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const isLoaded = status === 'loaded'
  return (
    <span
      className="flex items-center gap-1"
      style={{ fontSize: 10, color: isLoaded ? 'var(--ok)' : 'var(--muted-strong)' }}
    >
      <span
        className="inline-block rounded-full"
        style={{
          width: 6,
          height: 6,
          background: isLoaded ? 'var(--ok)' : 'var(--muted-strong)',
        }}
      />
      {isLoaded ? 'loaded' : 'idle'}
    </span>
  )
}
