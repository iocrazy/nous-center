import { useState, type FormEvent } from 'react'
import { Fingerprint, ShieldCheck, Trash2 } from 'lucide-react'
import {
  isWebAuthnSupported,
  useDeletePasskey,
  usePasskeyList,
  usePasskeyStatus,
  useRegisterPasskey,
} from '../../api/passkey'
import {
  useDeleteTotp,
  useTotpList,
  useTotpSetup,
  useTotpStatus,
  useTotpVerify,
  type TotpSetupOut,
} from '../../api/totp'

export default function SecurityPanel() {
  const passkeyStatus = usePasskeyStatus()
  const totpStatus = useTotpStatus()

  const adminEnabled =
    passkeyStatus.data?.enabled === true || totpStatus.data?.enabled === true

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="text-sm font-medium" style={{ color: 'var(--text)' }}>
          安全 & 登录
        </div>
        <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
          除了密码登录，还可以绑定 Passkey（生物识别 / 硬件 Key）和 TOTP 动态码作为额外的登录方式。
        </div>
      </div>

      {!adminEnabled && (
        <Card>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            未启用管理员登录（<code>ADMIN_PASSWORD</code> 为空）。Passkey / TOTP 在 dev 模式下没有意义。设密码后此页才有作用。
          </div>
        </Card>
      )}

      {adminEnabled && <PasskeySection />}
      {adminEnabled && <TotpSection />}
    </div>
  )
}

// ---------- Passkey ----------

function PasskeySection() {
  const list = usePasskeyList()
  const register = useRegisterPasskey()
  const remove = useDeletePasskey()
  const [label, setLabel] = useState('')
  const [error, setError] = useState<string | null>(null)
  const supported = isWebAuthnSupported()

  function onRegister(e: FormEvent) {
    e.preventDefault()
    if (!label.trim()) return
    setError(null)
    register.mutate(label.trim(), {
      onSuccess: () => setLabel(''),
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Passkey 注册失败'
        setError(msg)
      },
    })
  }

  return (
    <Card>
      <SectionHeader
        icon={<Fingerprint size={14} />}
        title="Passkey (WebAuthn)"
        desc="绑定 TouchID / Windows Hello / YubiKey 等。下次登录可一键通过，不用敲密码。"
      />

      {!supported && (
        <Note color="warn">这个浏览器不支持 WebAuthn。需要 https 或 localhost。</Note>
      )}

      {supported && (
        <form onSubmit={onRegister} className="flex gap-2 mt-3">
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder='给这个 Passkey 起个名字 (如 "MacBook TouchID")'
            className="flex-1 px-3 py-1.5 rounded text-xs outline-none"
            style={{
              background: 'var(--bg)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
            }}
          />
          <button
            type="submit"
            disabled={register.isPending || !label.trim()}
            className="px-3 py-1.5 rounded text-xs font-medium disabled:opacity-50"
            style={{ background: 'var(--accent)', color: 'var(--accent-fg, #fff)' }}
          >
            {register.isPending ? '等待验证…' : '绑定 Passkey'}
          </button>
        </form>
      )}

      {error && <Note color="err">{error}</Note>}

      <div className="mt-4 flex flex-col gap-1">
        {list.data?.length === 0 && (
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            还没有绑定的 Passkey。
          </div>
        )}
        {list.data?.map((row) => (
          <Row
            key={row.id}
            label={row.label}
            sub={`${row.transports || '未知传输'} · ${
              row.last_used_at ? `上次使用 ${new Date(row.last_used_at).toLocaleString()}` : '未使用'
            } · ${row.backup_eligible ? '已同步' : '本设备'}`}
            onDelete={() => remove.mutate(row.id)}
            disabled={remove.isPending}
          />
        ))}
      </div>
    </Card>
  )
}

// ---------- TOTP ----------

function TotpSection() {
  const list = useTotpList()
  const setup = useTotpSetup()
  const verify = useTotpVerify()
  const remove = useDeleteTotp()
  const [label, setLabel] = useState('')
  const [pending, setPending] = useState<TotpSetupOut | null>(null)
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)

  function onSetup(e: FormEvent) {
    e.preventDefault()
    if (!label.trim()) return
    setError(null)
    setup.mutate(label.trim(), {
      onSuccess: (data) => {
        setPending(data)
        setLabel('')
        setCode('')
      },
    })
  }

  function onVerify(e: FormEvent) {
    e.preventDefault()
    if (!pending || code.length < 6) return
    setError(null)
    verify.mutate(
      { id: pending.id, code },
      {
        onSuccess: () => {
          setPending(null)
          setCode('')
        },
        onError: (err: unknown) => {
          setError(err instanceof Error ? err.message : '验证失败')
        },
      },
    )
  }

  return (
    <Card>
      <SectionHeader
        icon={<ShieldCheck size={14} />}
        title="TOTP 动态码 (RFC 6238)"
        desc="兼容 Authy / Google Authenticator / 1Password。备份用 — Passkey 失灵时还能登。"
      />

      {!pending && (
        <form onSubmit={onSetup} className="flex gap-2 mt-3">
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder='给这个 TOTP 起个名字 (如 "Authy")'
            className="flex-1 px-3 py-1.5 rounded text-xs outline-none"
            style={{
              background: 'var(--bg)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
            }}
          />
          <button
            type="submit"
            disabled={setup.isPending || !label.trim()}
            className="px-3 py-1.5 rounded text-xs font-medium disabled:opacity-50"
            style={{ background: 'var(--accent)', color: 'var(--accent-fg, #fff)' }}
          >
            {setup.isPending ? '生成中…' : '生成秘密'}
          </button>
        </form>
      )}

      {pending && (
        <div
          className="mt-3 p-3 rounded text-xs"
          style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
        >
          <div className="font-medium" style={{ color: 'var(--text)' }}>
            扫描或粘贴到 Authy / 1Password / Google Authenticator
          </div>
          <div
            className="mt-2 font-mono select-all break-all"
            style={{ color: 'var(--text-secondary)' }}
          >
            {pending.otpauth_url}
          </div>
          <div className="mt-2" style={{ color: 'var(--text-secondary)' }}>
            如果你的客户端不支持 URL 扫描，秘密为 <code style={{ color: 'var(--text)' }}>{pending.secret}</code>
          </div>
          <form onSubmit={onVerify} className="flex gap-2 mt-3">
            <input
              autoFocus
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))}
              placeholder="输入 App 显示的 6 位码"
              maxLength={8}
              className="flex-1 px-3 py-1.5 rounded text-xs outline-none tracking-widest"
              style={{
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
              }}
            />
            <button
              type="submit"
              disabled={verify.isPending || code.length < 6}
              className="px-3 py-1.5 rounded text-xs font-medium disabled:opacity-50"
              style={{ background: 'var(--accent)', color: 'var(--accent-fg, #fff)' }}
            >
              确认绑定
            </button>
            <button
              type="button"
              onClick={() => {
                setPending(null)
                setCode('')
                setError(null)
              }}
              className="px-3 py-1.5 rounded text-xs"
              style={{
                background: 'transparent',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
              }}
            >
              取消
            </button>
          </form>
          {error && <Note color="err">{error}</Note>}
        </div>
      )}

      <div className="mt-4 flex flex-col gap-1">
        {list.data?.length === 0 && (
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            还没有绑定的 TOTP。
          </div>
        )}
        {list.data?.map((row) => (
          <Row
            key={row.id}
            label={row.label}
            sub={`${
              row.verified_at ? '已激活' : '⚠ 未完成激活'
            } · ${row.last_used_at ? `上次使用 ${new Date(row.last_used_at).toLocaleString()}` : '未使用'}`}
            onDelete={() => remove.mutate(row.id)}
            disabled={remove.isPending}
          />
        ))}
      </div>
    </Card>
  )
}

// ---------- shared bits ----------

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="p-4 rounded"
      style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
      }}
    >
      {children}
    </div>
  )
}

function SectionHeader({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode
  title: string
  desc: string
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 text-sm font-medium" style={{ color: 'var(--text)' }}>
        {icon}
        {title}
      </div>
      <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        {desc}
      </div>
    </div>
  )
}

function Note({ children, color }: { children: React.ReactNode; color: 'warn' | 'err' }) {
  return (
    <div
      className="text-xs mt-2 px-2 py-1 rounded"
      style={{
        background: color === 'err' ? 'rgba(239,68,68,0.1)' : 'rgba(234,179,8,0.1)',
        color: color === 'err' ? 'var(--error, #ef4444)' : 'var(--warn, #eab308)',
      }}
    >
      {children}
    </div>
  )
}

function Row({
  label,
  sub,
  onDelete,
  disabled,
}: {
  label: string
  sub: string
  onDelete: () => void
  disabled?: boolean
}) {
  return (
    <div
      className="flex items-center justify-between px-3 py-2 rounded"
      style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
    >
      <div className="flex flex-col gap-0.5">
        <div className="text-xs font-medium" style={{ color: 'var(--text)' }}>
          {label}
        </div>
        <div className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
          {sub}
        </div>
      </div>
      <button
        onClick={onDelete}
        disabled={disabled}
        title="删除"
        className="p-1 rounded disabled:opacity-50"
        style={{ color: 'var(--text-secondary)', background: 'transparent' }}
      >
        <Trash2 size={12} />
      </button>
    </div>
  )
}
