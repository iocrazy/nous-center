import { useState, type FormEvent } from 'react'
import { useAdminLogin } from '../api/admin'

export default function Login() {
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const login = useAdminLogin()

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    login.mutate(password, {
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : '登录失败'
        setError(msg.includes('401') ? '密码错误' : msg)
      },
    })
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ background: 'var(--bg)' }}
    >
      <form
        onSubmit={onSubmit}
        className="w-[320px] p-8 rounded-lg flex flex-col gap-4"
        style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
      >
        <div>
          <div className="text-lg font-medium" style={{ color: 'var(--text)' }}>
            nous-center
          </div>
          <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
            管理员登录
          </div>
        </div>

        <input
          type="password"
          autoFocus
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="管理员密码"
          className="w-full px-3 py-2 rounded text-sm outline-none"
          style={{
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            color: 'var(--text)',
          }}
        />

        {error && (
          <div className="text-xs" style={{ color: 'var(--error, #ef4444)' }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={login.isPending || !password}
          className="w-full px-3 py-2 rounded text-sm font-medium disabled:opacity-50"
          style={{ background: 'var(--accent)', color: 'var(--accent-fg, #fff)' }}
        >
          {login.isPending ? '登录中…' : '登录'}
        </button>
      </form>
    </div>
  )
}
