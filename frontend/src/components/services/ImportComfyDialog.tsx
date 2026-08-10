import { useEffect, useRef, useState } from 'react'
import { Upload, X } from 'lucide-react'
import { createComfyTemplate } from '../../api/comfyTemplates'
import { NAME_RE } from '../../api/services'

export interface ImportComfyDialogProps {
  open: boolean
  onClose: () => void
  /** Called with the newly created service's name on success. */
  onImported: (serviceName: string) => void
}

const INVALID_FORMAT_MSG =
  '不是 ComfyUI API 格式导出，请在 ComfyUI 用 Export (API) 导出'

export default function ImportComfyDialog({ open, onClose, onImported }: ImportComfyDialogProps) {
  const [fileName, setFileName] = useState('')
  const [workflow, setWorkflow] = useState<Record<string, unknown> | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const dropRef = useRef<HTMLLabelElement>(null)
  // FileReader in jsdom (and real browsers) resolves asynchronously; a click on
  // 导入 can land before parsing finishes, so submit() awaits this instead of
  // trusting the `workflow` state snapshot at click time.
  const parseRef = useRef<Promise<Record<string, unknown> | null>>(Promise.resolve(null))
  // Monotonic token so a stale FileReader callback (superseded by picking another
  // file, or still in flight when the dialog closes/reopens) can't clobber state
  // that belongs to a newer selection — the closure captures its own token and
  // compares against the current one before calling any setState.
  const readTokenRef = useRef(0)

  useEffect(() => {
    if (open) return
    readTokenRef.current += 1
    setFileName('')
    setWorkflow(null)
    setFileError(null)
    setName('')
    setSubmitting(false)
    setSubmitError(null)
    parseRef.current = Promise.resolve(null)
  }, [open])

  if (!open) return null

  const handleFile = (file: File) => {
    const token = (readTokenRef.current += 1)
    setFileName(file.name)
    setWorkflow(null)
    setFileError(null)
    setSubmitError(null)
    parseRef.current = new Promise((resolve) => {
      const reader = new FileReader()
      reader.onerror = () => {
        if (readTokenRef.current === token) setFileError('读取文件失败')
        resolve(null)
      }
      reader.onload = () => {
        let parsed: unknown
        try {
          parsed = JSON.parse(String(reader.result))
        } catch {
          if (readTokenRef.current === token) setFileError(INVALID_FORMAT_MSG)
          resolve(null)
          return
        }
        if (!isApiFormatWorkflow(parsed)) {
          if (readTokenRef.current === token) setFileError(INVALID_FORMAT_MSG)
          resolve(null)
          return
        }
        if (readTokenRef.current === token) setWorkflow(parsed)
        resolve(parsed)
      }
      reader.readAsText(file)
    })
  }

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--accent)'
  }

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--border)'
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--border)'
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const nameValid = NAME_RE.test(name)

  const submit = async () => {
    if (submitting) return
    const wf = workflow ?? (await parseRef.current)
    if (!wf) {
      setSubmitError('请先选择合法的 ComfyUI API 格式工作流文件')
      return
    }
    if (!nameValid) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const res = await createComfyTemplate(name, wf)
      onImported(res.service_name)
      onClose()
    } catch (e) {
      setSubmitError((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div onClick={onClose} style={overlayStyle}>
      <div onClick={(e) => e.stopPropagation()} style={modalStyle}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ flex: 1, fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>
            导入 ComfyUI 工作流
          </h2>
          <button
            onClick={onClose}
            type="button"
            style={{ background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        <Section label="工作流 JSON（ComfyUI · Export (API)）">
          <label
            ref={dropRef}
            style={dropZoneStyle}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <Upload size={16} style={{ color: 'var(--muted)', flexShrink: 0 }} />
            <span style={{ fontSize: 12, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {fileName || '点击选择 API 格式导出的 workflow.json'}
            </span>
            <input
              data-testid="comfy-file-input"
              type="file"
              accept=".json,application/json"
              onChange={onFileInputChange}
              style={{ display: 'none' }}
            />
          </label>
          {fileError && <ErrorText>{fileError}</ErrorText>}
          {workflow && !fileError && (
            <div style={{ fontSize: 11, color: 'var(--accent-2, #22c55e)', marginTop: 6 }}>
              已解析 {Object.keys(workflow).length} 个节点
            </div>
          )}
        </Section>

        <Section label="服务名称 (对外 endpoint key)">
          <input
            value={name}
            onChange={(e) => setName(e.target.value.trim())}
            placeholder="服务名，例如：minimax-h3-r2v"
            style={inputStyle}
          />
          {name && !nameValid && (
            <ErrorText>
              必须匹配 {NAME_RE.source}（小写字母开头，只允许 a-z 0-9 -，2-63 字符）
            </ErrorText>
          )}
        </Section>

        {submitError && <ErrorText>{submitError}</ErrorText>}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 12 }}>
          <button onClick={onClose} type="button" style={btnGhost}>
            取消
          </button>
          <button onClick={submit} disabled={submitting} type="button" style={btnPrimary(!submitting)}>
            {submitting ? '导入中…' : '导入'}
          </button>
        </div>
      </div>
    </div>
  )
}

function isApiFormatWorkflow(parsed: unknown): parsed is Record<string, unknown> {
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return false
  const values = Object.values(parsed as Record<string, unknown>)
  if (values.length === 0) return false
  return values.every(
    (v) => !!v && typeof v === 'object' && !Array.isArray(v) && 'class_type' in (v as object),
  )
}

// ---------- shared bits (mirrors CreateServiceDialog's visual language) ----------

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          fontSize: 11,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      {children}
    </div>
  )
}

function ErrorText({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 11, color: 'var(--error, #ef4444)', marginTop: 6 }}>{children}</div>
  )
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(0,0,0,0.55)',
  zIndex: 50,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

const modalStyle: React.CSSProperties = {
  background: 'var(--bg-elevated, var(--bg))',
  border: '1px solid var(--border)',
  borderRadius: 8,
  width: 480,
  maxWidth: '92vw',
  maxHeight: '88vh',
  overflow: 'auto',
  padding: 20,
  boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
}

const dropZoneStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  width: '100%',
  background: 'var(--bg)',
  color: 'var(--text)',
  border: '1px dashed var(--border)',
  borderRadius: 4,
  padding: '10px 12px',
  fontSize: 12,
  cursor: 'pointer',
  transition: 'border-color 0.15s',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--bg)',
  color: 'var(--text)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  padding: '7px 9px',
  fontSize: 12,
}

const btnGhost: React.CSSProperties = {
  padding: '7px 14px',
  fontSize: 12,
  background: 'transparent',
  color: 'var(--muted)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  cursor: 'pointer',
}

const btnPrimary = (enabled: boolean): React.CSSProperties => ({
  padding: '7px 14px',
  fontSize: 12,
  background: 'var(--accent)',
  color: '#fff',
  border: 'none',
  borderRadius: 4,
  cursor: enabled ? 'pointer' : 'not-allowed',
  opacity: enabled ? 1 : 0.5,
})
