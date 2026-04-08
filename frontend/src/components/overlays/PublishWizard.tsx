// frontend/src/components/overlays/PublishWizard.tsx
import { useState } from 'react'
import { X, ChevronRight, ChevronLeft, Copy, Check } from 'lucide-react'
import { usePublishApp, type ExposedParam } from '../../api/apps'
import { useWorkspaceStore } from '../../stores/workspace'
import { useToastStore } from '../../stores/toast'

interface Props {
  workflowId: string
  onClose: () => void
}

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 100)
}

export default function PublishWizard({ workflowId, onClose }: Props) {
  const [step, setStep] = useState(1)
  const [name, setName] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [description, setDescription] = useState('')
  const [inputs, setInputs] = useState<ExposedParam[]>([])
  const [outputs, setOutputs] = useState<ExposedParam[]>([])
  const [copied, setCopied] = useState(false)
  const toast = useToastStore((s) => s.add)
  const publish = usePublishApp()
  const workflow = useWorkspaceStore((s) => s.getActiveWorkflow())

  const nodes = workflow?.nodes || []

  const handleDisplayNameChange = (v: string) => {
    setDisplayName(v)
    if (!name || name === slugify(displayName)) {
      setName(slugify(v))
    }
  }

  const toggleInput = (nodeId: string, paramKey: string, label: string) => {
    const exists = inputs.find((i) => i.node_id === nodeId && i.param_key === paramKey)
    if (exists) {
      setInputs(inputs.filter((i) => !(i.node_id === nodeId && i.param_key === paramKey)))
    } else {
      setInputs([...inputs, {
        node_id: nodeId,
        param_key: paramKey,
        api_name: paramKey,
        param_type: 'string',
        description: label,
        required: true,
        default: null,
      }])
    }
  }

  const handlePublish = async () => {
    try {
      await publish.mutateAsync({
        workflowId,
        body: { name, display_name: displayName, description, exposed_inputs: inputs, exposed_outputs: outputs },
      })
      toast('App 发布成功', 'success')
      onClose()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '发布失败'
      toast(msg, 'error')
    }
  }

  const curlExample = `curl -X POST http://localhost:8000/v1/apps/${name} \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(Object.fromEntries(inputs.map((i) => [i.api_name, i.default || ''])), null, 2)}'`

  const handleCopy = () => {
    navigator.clipboard.writeText(curlExample)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
    }}>
      <div style={{
        background: 'var(--bg-elevated)', borderRadius: 12, width: 600, maxHeight: '80vh',
        overflow: 'auto', padding: 24,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
          <h2 style={{ margin: 0, fontSize: 18 }}>发布 App — 步骤 {step}/3</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)' }}>
            <X size={20} />
          </button>
        </div>

        {step === 1 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label>
              显示名称
              <input value={displayName} onChange={(e) => handleDisplayNameChange(e.target.value)}
                style={{ width: '100%', padding: 8, marginTop: 4, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)', boxSizing: 'border-box' }} />
            </label>
            <label>
              URL 名称
              <input value={name} onChange={(e) => setName(e.target.value)}
                style={{ width: '100%', padding: 8, marginTop: 4, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)', fontFamily: 'monospace', boxSizing: 'border-box' }} />
            </label>
            <label>
              描述
              <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3}
                style={{ width: '100%', padding: 8, marginTop: 4, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)', resize: 'vertical', boxSizing: 'border-box' }} />
            </label>
          </div>
        )}

        {step === 2 && (
          <div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>选择要暴露给外部调用的节点参数：</p>
            {nodes.map((node) => (
              <div key={node.id} style={{ marginBottom: 12, padding: 8, border: '1px solid var(--border)', borderRadius: 8 }}>
                <strong>{node.type}</strong> <span style={{ color: 'var(--muted)', fontSize: 11, fontFamily: 'monospace' }}>({node.id})</span>
                {Object.keys(node.data || {}).filter((k) => !k.startsWith('_')).map((key) => {
                  const checked = inputs.some((i) => i.node_id === node.id && i.param_key === key)
                  return (
                    <label key={key} style={{ display: 'flex', gap: 8, padding: '4px 0', cursor: 'pointer', alignItems: 'center' }}>
                      <input type="checkbox" checked={checked}
                        onChange={() => toggleInput(node.id, key, `${node.type}.${key}`)} />
                      <span style={{ fontFamily: 'monospace', fontSize: 13 }}>{key}</span>
                    </label>
                  )
                })}
                {Object.keys(node.data || {}).filter((k) => !k.startsWith('_')).length === 0 && (
                  <span style={{ fontSize: 12, color: 'var(--muted)' }}>（无可暴露参数）</span>
                )}
              </div>
            ))}
          </div>
        )}

        {step === 3 && (
          <div>
            <h3 style={{ fontSize: 14, marginBottom: 8 }}>API 端点</h3>
            <code style={{ display: 'block', padding: 8, background: 'var(--bg)', borderRadius: 6, marginBottom: 16, fontSize: 13 }}>
              POST /v1/apps/{name}
            </code>
            <h3 style={{ fontSize: 14, marginBottom: 8 }}>curl 示例</h3>
            <div style={{ position: 'relative' }}>
              <pre style={{ padding: 12, background: 'var(--bg)', borderRadius: 6, overflow: 'auto', fontSize: 12, margin: 0 }}>
                {curlExample}
              </pre>
              <button onClick={handleCopy} style={{
                position: 'absolute', top: 8, right: 8, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)',
              }}>
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>
            {inputs.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ fontSize: 14, marginBottom: 8 }}>暴露参数</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--muted)' }}>参数名</th>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--muted)' }}>节点</th>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--muted)' }}>字段</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inputs.map((p) => (
                      <tr key={`${p.node_id}-${p.param_key}`} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{p.api_name}</td>
                        <td style={{ padding: '4px 8px', fontFamily: 'monospace', color: 'var(--muted)' }}>{p.node_id.slice(0, 8)}…</td>
                        <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{p.param_key}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 20 }}>
          <button onClick={() => setStep(Math.max(1, step - 1))} disabled={step === 1}
            style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '8px 16px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)', cursor: step === 1 ? 'not-allowed' : 'pointer', opacity: step === 1 ? 0.5 : 1 }}>
            <ChevronLeft size={16} /> 上一步
          </button>
          {step < 3 ? (
            <button onClick={() => setStep(step + 1)}
              style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '8px 16px', borderRadius: 6, border: 'none', background: 'var(--accent)', color: '#fff', cursor: 'pointer' }}>
              下一步 <ChevronRight size={16} />
            </button>
          ) : (
            <button onClick={handlePublish} disabled={publish.isPending || !name}
              style={{ padding: '8px 16px', borderRadius: 6, border: 'none', background: 'var(--green, #22c55e)', color: '#fff', cursor: publish.isPending || !name ? 'not-allowed' : 'pointer', opacity: publish.isPending || !name ? 0.6 : 1 }}>
              {publish.isPending ? '发布中...' : '发布'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
