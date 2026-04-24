import { useEffect, useState } from 'react'
import { Cog, Info, Package } from 'lucide-react'
import { useServerSettings, useUpdateServerSettings } from '../../api/settings'
import { useSettingsStore, type SettingsState } from '../../stores/settings'
import NodePackagesPanel from '../settings/NodePackagesPanel'

// m16 v3: Settings 改成左侧 sub-nav。原"通用 设置 + 节点包"两个独立
// overlay 收敛到这里；"关于" 是新增的版本/链接子页。

type Section = 'general' | 'packages' | 'about'

const NAV: { id: Section; label: string; icon: typeof Cog }[] = [
  { id: 'general', label: '通用', icon: Cog },
  { id: 'packages', label: '节点包', icon: Package },
  { id: 'about', label: '关于', icon: Info },
]

export default function SettingsOverlay() {
  const [section, setSection] = useState<Section>('general')

  return (
    <div className="absolute inset-0 flex z-[16]" style={{ background: 'var(--bg)' }}>
      <SubNav section={section} onChange={setSection} />
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{ maxWidth: 720, padding: 24, margin: '0 auto' }}>
          {section === 'general' && <GeneralPanel />}
          {section === 'packages' && <NodePackagesPanel />}
          {section === 'about' && <AboutPanel />}
        </div>
      </div>
    </div>
  )
}

function SubNav({ section, onChange }: { section: Section; onChange: (s: Section) => void }) {
  return (
    <nav
      style={{
        width: 200,
        borderRight: '1px solid var(--border)',
        background: 'var(--bg-accent)',
        padding: '16px 8px',
        flexShrink: 0,
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.6,
          padding: '0 8px',
          marginBottom: 8,
        }}
      >
        设置
      </div>
      {NAV.map(({ id, label, icon: Icon }) => {
        const active = id === section
        return (
          <button
            key={id}
            type="button"
            onClick={() => onChange(id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              width: '100%',
              padding: '7px 8px',
              borderRadius: 4,
              border: 'none',
              background: active ? 'var(--accent-subtle, rgba(99,102,241,0.1))' : 'transparent',
              color: active ? 'var(--accent)' : 'var(--text)',
              fontSize: 12,
              cursor: 'pointer',
              textAlign: 'left',
              marginBottom: 2,
            }}
          >
            <Icon size={13} />
            {label}
          </button>
        )
      })}
    </nav>
  )
}

// ---------- 通用 ----------

type SettingsKey = keyof Omit<SettingsState, 'update' | 'reset'>

const SECTIONS: {
  title: string
  fields: { key: SettingsKey; serverKey?: string; label: string; placeholder?: string }[]
}[] = [
  {
    title: '路径配置',
    fields: [
      {
        key: 'localModelsPath',
        serverKey: 'local_models_path',
        label: '本地模型目录',
        placeholder: '/media/heygo/Program/models',
      },
      {
        key: 'cosyvoiceRepoPath',
        serverKey: 'cosyvoice_repo_path',
        label: 'CosyVoice 仓库',
        placeholder: '/path/to/CosyVoice',
      },
      {
        key: 'indexttsRepoPath',
        serverKey: 'indextts_repo_path',
        label: 'IndexTTS 仓库',
        placeholder: '/path/to/index-tts',
      },
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
      {
        key: 'redisUrl',
        serverKey: 'redis_url',
        label: 'Redis URL',
        placeholder: 'redis://localhost:6379/0',
      },
      { key: 'apiBaseUrl', label: 'API Base URL', placeholder: 'http://localhost:8000' },
    ],
  },
]

function GeneralPanel() {
  const store = useSettingsStore()
  const { data: serverSettings } = useServerSettings()
  const updateServer = useUpdateServerSettings()

  const [draft, setDraft] = useState<Record<SettingsKey, string | number>>(() => buildDraft(store))
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (!serverSettings) return
    const merged: Partial<Record<SettingsKey, string | number>> = {}
    for (const s of SECTIONS) {
      for (const f of s.fields) {
        if (f.serverKey && f.serverKey in serverSettings) {
          merged[f.key] = (serverSettings as unknown as Record<string, string | number>)[f.serverKey]
        }
      }
    }
    setDraft((prev) => ({ ...prev, ...merged }))
  }, [serverSettings])

  const handleSave = () => {
    const localValues: Record<string, string | number> = {}
    for (const s of SECTIONS) {
      for (const f of s.fields) {
        const v = draft[f.key]
        localValues[f.key] = typeof store[f.key] === 'number' ? Number(v) : v
      }
    }
    store.update(localValues)

    const serverValues: Record<string, string | number> = {}
    for (const s of SECTIONS) {
      for (const f of s.fields) {
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
    <>
      {SECTIONS.map((s) => (
        <div key={s.title} style={{ marginBottom: 22 }}>
          <h3
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: 'var(--muted)',
              marginBottom: 10,
              textTransform: 'uppercase',
              letterSpacing: 0.5,
            }}
          >
            {s.title}
          </h3>
          {s.fields.map((field) => (
            <div key={field.key} style={{ marginBottom: 12 }}>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: 'var(--text)',
                  marginBottom: 4,
                }}
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
                  padding: '6px 9px',
                  fontSize: 12,
                  fontFamily: 'var(--mono, monospace)',
                  background: 'var(--bg-accent)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  color: 'var(--text)',
                  outline: 'none',
                }}
              />
            </div>
          ))}
        </div>
      ))}

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8 }}>
        <button
          onClick={handleSave}
          disabled={updateServer.isPending}
          style={{
            padding: '7px 16px',
            fontSize: 12,
            borderRadius: 4,
            border: 'none',
            background: 'var(--accent)',
            color: '#fff',
            cursor: updateServer.isPending ? 'wait' : 'pointer',
            opacity: updateServer.isPending ? 0.6 : 1,
          }}
        >
          {updateServer.isPending ? 'Saving...' : '保存'}
        </button>
        <button
          onClick={handleReset}
          style={{
            padding: '7px 16px',
            fontSize: 12,
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'transparent',
            color: 'var(--muted)',
            cursor: 'pointer',
          }}
        >
          重置
        </button>
        {saved && <span style={{ fontSize: 11, color: 'var(--ok)' }}>已保存</span>}
      </div>
    </>
  )
}

function buildDraft(store: SettingsState): Record<SettingsKey, string | number> {
  const d: Record<string, string | number> = {}
  for (const s of SECTIONS) {
    for (const f of s.fields) {
      d[f.key] = store[f.key]
    }
  }
  return d as Record<SettingsKey, string | number>
}

// ---------- 关于 ----------

function AboutPanel() {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <div
          style={{
            width: 48,
            height: 48,
            borderRadius: 8,
            background: 'var(--accent)',
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 22,
            fontWeight: 700,
          }}
        >
          N
        </div>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>
            Nous Center
          </div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>
            推理 infra 控制台 · v3 IA
          </div>
        </div>
      </div>

      <Section title="关于">
        <p style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.7 }}>
          Nous Center 是单管理员的推理 infra 控制台 — 模型 / Workflow /
          服务 / API Key 一站管理。Agent / Skill 等上层应用由 mediahub
          这类消费方实现，nous-center 只负责把"能调用的服务"暴露成稳定
          endpoint。
        </p>
      </Section>

      <Section title="协议兼容">
        <ul
          style={{
            fontSize: 12,
            color: 'var(--text)',
            lineHeight: 1.8,
            paddingLeft: 18,
            margin: 0,
          }}
        >
          <li>OpenAI 兼容：<code>/v1/chat/completions</code></li>
          <li>Ollama 兼容：<code>/api/chat</code></li>
          <li>Anthropic 兼容：<code>/v1/messages</code></li>
        </ul>
      </Section>

      <Section title="文档与反馈">
        <ul
          style={{
            fontSize: 12,
            color: 'var(--text)',
            lineHeight: 1.8,
            paddingLeft: 18,
            margin: 0,
          }}
        >
          <li>设计文档：<code>docs/designs/2026-04-22-ia-rebuild-v3.md</code></li>
          <li>API 文档：在服务详情页 → "API 文档" tab 查看</li>
        </ul>
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h3
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          marginBottom: 8,
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  )
}
