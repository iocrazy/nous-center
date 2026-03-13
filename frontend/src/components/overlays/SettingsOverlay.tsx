import { useState, useEffect } from 'react'
import { useSettingsStore, type SettingsState } from '../../stores/settings'
import { useServerSettings, useUpdateServerSettings } from '../../api/settings'

type SettingsKey = keyof Omit<SettingsState, 'update' | 'reset'>

const SECTIONS: {
  title: string
  fields: { key: SettingsKey; serverKey?: string; label: string; placeholder?: string }[]
}[] = [
  {
    title: '路径配置',
    fields: [
      { key: 'localModelsPath', serverKey: 'local_models_path', label: '本地模型目录', placeholder: '/media/heygo/Program/models' },
      { key: 'cosyvoiceRepoPath', serverKey: 'cosyvoice_repo_path', label: 'CosyVoice 仓库', placeholder: '/path/to/CosyVoice' },
      { key: 'indexttsRepoPath', serverKey: 'indextts_repo_path', label: 'IndexTTS 仓库', placeholder: '/path/to/index-tts' },
    ],
  },
  {
    title: 'GPU 分配',
    fields: [
      { key: 'gpuImage', serverKey: 'gpu_image', label: '图像生成 GPU' },
      { key: 'gpuTts', serverKey: 'gpu_tts', label: 'TTS GPU' },
    ],
  },
  {
    title: '服务地址',
    fields: [
      { key: 'redisUrl', serverKey: 'redis_url', label: 'Redis URL', placeholder: 'redis://localhost:6379/0' },
      { key: 'apiBaseUrl', label: 'API Base URL', placeholder: 'http://localhost:8000' },
    ],
  },
]

export default function SettingsOverlay() {
  const store = useSettingsStore()
  const { data: serverSettings } = useServerSettings()
  const updateServer = useUpdateServerSettings()

  const [draft, setDraft] = useState<Record<SettingsKey, string | number>>(() => buildDraft(store))
  const [saved, setSaved] = useState(false)

  // Sync from server when data arrives
  useEffect(() => {
    if (!serverSettings) return
    const merged: Partial<Record<SettingsKey, string | number>> = {}
    for (const section of SECTIONS) {
      for (const f of section.fields) {
        if (f.serverKey && f.serverKey in serverSettings) {
          merged[f.key] = (serverSettings as unknown as Record<string, string | number>)[f.serverKey]
        }
      }
    }
    setDraft((prev) => ({ ...prev, ...merged }))
  }, [serverSettings])

  const handleSave = () => {
    // Save to local store
    const localValues: Record<string, string | number> = {}
    for (const section of SECTIONS) {
      for (const f of section.fields) {
        const v = draft[f.key]
        localValues[f.key] = typeof store[f.key] === 'number' ? Number(v) : v
      }
    }
    store.update(localValues)

    // Save to server
    const serverValues: Record<string, string | number> = {}
    for (const section of SECTIONS) {
      for (const f of section.fields) {
        if (f.serverKey) {
          const v = draft[f.key]
          serverValues[f.serverKey] = typeof store[f.key] === 'number' ? Number(v) : v
        }
      }
    }
    updateServer.mutate(serverValues)

    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleReset = () => {
    store.reset()
    setDraft(buildDraft(store))
  }

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ padding: 16, maxWidth: 600 }}>
        {SECTIONS.map((section) => (
          <div key={section.title} className="mb-5">
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
              {section.title}
            </div>

            {section.fields.map((field) => (
              <div key={field.key} className="mb-3">
                <label
                  style={{ display: 'block', fontSize: 10, color: 'var(--muted)', marginBottom: 4 }}
                >
                  {field.label}
                </label>
                <input
                  type={typeof store[field.key] === 'number' ? 'number' : 'text'}
                  value={draft[field.key] ?? ''}
                  placeholder={field.placeholder}
                  onChange={(e) =>
                    setDraft((prev) => ({ ...prev, [field.key]: e.target.value }))
                  }
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: 11,
                    fontFamily: 'var(--mono)',
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                    color: 'var(--text-strong)',
                    outline: 'none',
                  }}
                />
              </div>
            ))}
          </div>
        ))}

        <div className="flex gap-2 mt-4 items-center">
          <button
            onClick={handleSave}
            disabled={updateServer.isPending}
            style={{
              padding: '6px 16px',
              fontSize: 11,
              borderRadius: 4,
              border: '1px solid var(--accent)',
              background: 'var(--accent)',
              color: '#fff',
              cursor: updateServer.isPending ? 'wait' : 'pointer',
              opacity: updateServer.isPending ? 0.6 : 1,
            }}
          >
            {updateServer.isPending ? 'Saving...' : 'Save'}
          </button>
          <button
            onClick={handleReset}
            style={{
              padding: '6px 16px',
              fontSize: 11,
              borderRadius: 4,
              border: '1px solid var(--border)',
              background: 'none',
              color: 'var(--muted)',
              cursor: 'pointer',
            }}
          >
            Reset
          </button>
          {saved && (
            <span style={{ fontSize: 10, color: 'var(--ok)' }}>已保存</span>
          )}
        </div>
      </div>
    </div>
  )
}

function buildDraft(store: SettingsState): Record<SettingsKey, string | number> {
  const d: Record<string, string | number> = {}
  for (const section of SECTIONS) {
    for (const f of section.fields) {
      d[f.key] = store[f.key]
    }
  }
  return d as Record<SettingsKey, string | number>
}
