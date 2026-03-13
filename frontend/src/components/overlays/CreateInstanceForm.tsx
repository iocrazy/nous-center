import { useState, useEffect } from 'react'
import { useVoicePresets } from '../../api/voices'
import { useCreateInstance } from '../../api/instances'
import type { ApiManagementOptions } from '../../stores/panel'

interface Props {
  defaultOptions?: ApiManagementOptions
  onCreated: (instanceId: string) => void
  onCancel: () => void
}

export default function CreateInstanceForm({ defaultOptions, onCreated, onCancel }: Props) {
  const { data: presets } = useVoicePresets()
  const createInstance = useCreateInstance()

  const [sourceType] = useState<'preset'>('preset') // Only preset for now
  const [sourceId, setSourceId] = useState(defaultOptions?.presetId ?? '')
  const [name, setName] = useState('')
  const [instanceType, setInstanceType] = useState('tts')

  // Pre-fill source_id from options
  useEffect(() => {
    if (defaultOptions?.presetId) {
      setSourceId(defaultOptions.presetId)
    }
  }, [defaultOptions?.presetId])

  const handleSubmit = async () => {
    if (!name.trim() || !sourceId) return
    const result = await createInstance.mutateAsync({
      source_type: sourceType,
      source_id: sourceId,
      name: name.trim(),
    })
    onCreated(result.id)
  }

  return (
    <div style={{ padding: '16px 20px', maxWidth: 480 }}>
      <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-strong)', margin: '0 0 16px' }}>
        新建服务实例
      </h2>

      {/* Instance Type */}
      <FormField label="服务类型">
        <select
          value={instanceType}
          onChange={(e) => setInstanceType(e.target.value)}
          style={selectStyle}
        >
          <option value="tts">语音合成 TTS</option>
          <option value="image">图像生成 Image</option>
          <option value="inference">文本推理 LLM</option>
        </select>
      </FormField>

      {/* Source Type (read-only for now) */}
      <FormField label="来源类型">
        <div style={{ ...inputStyle, background: 'var(--bg-accent)', color: 'var(--muted)' }}>
          预设 (Preset)
        </div>
      </FormField>

      {/* Source Selection */}
      <FormField label="选择预设">
        <select
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value)}
          style={selectStyle}
        >
          <option value="">-- 请选择 --</option>
          {presets?.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.engine})
            </option>
          ))}
        </select>
      </FormField>

      {/* Instance Name */}
      <FormField label="实例名称">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          placeholder="如：播客语音包、新闻TTS"
          autoFocus
          style={inputStyle}
        />
      </FormField>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8, marginTop: 20 }}>
        <button
          onClick={handleSubmit}
          disabled={createInstance.isPending || !name.trim() || !sourceId}
          style={{
            padding: '8px 24px',
            fontSize: 12,
            borderRadius: 5,
            border: 'none',
            background: 'var(--accent)',
            color: '#fff',
            cursor: 'pointer',
            fontWeight: 500,
            opacity: createInstance.isPending || !name.trim() || !sourceId ? 0.5 : 1,
          }}
        >
          {createInstance.isPending ? '创建中...' : '创建实例'}
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: '8px 16px',
            fontSize: 12,
            borderRadius: 5,
            border: '1px solid var(--border)',
            background: 'none',
            color: 'var(--muted)',
            cursor: 'pointer',
          }}
        >
          取消
        </button>
      </div>

      {createInstance.isError && (
        <div style={{ marginTop: 10, fontSize: 11, color: '#f87171' }}>
          创建失败：{createInstance.error?.message ?? '未知错误'}
        </div>
      )}
    </div>
  )
}

function FormField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4, fontWeight: 500 }}>
        {label}
      </label>
      {children}
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '7px 10px',
  fontSize: 12,
  background: 'var(--card)',
  border: '1px solid var(--border)',
  borderRadius: 5,
  color: 'var(--text-strong)',
  outline: 'none',
  boxSizing: 'border-box',
}

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: 'pointer',
}
