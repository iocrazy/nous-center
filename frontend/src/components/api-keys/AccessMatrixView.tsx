import { useMemo } from 'react'
import { Check } from 'lucide-react'
import {
  useAccessMatrix, useAddGrant, useRemoveGrant,
  type MatrixGrant,
} from '../../api/keys'

/**
 * 对外出口控制台 —— 服务 × Key 访问矩阵(spec 2026-06-19)。
 * 行=服务(按类目分组,带类型/背后/今日调用),列=key,格=授权态,点击切换(grant/revoke)。
 * 某 key 那一列点亮的服务 = 这把 key /v1/models 看到、能 model= 调的清单。
 */

const CAT_ORDER = ['llm', 'embedding', 'image', 'app', 'vl', 'tts']
const CAT_COLOR: Record<string, string> = {
  llm: '#3b82f6', embedding: '#22c55e', image: '#a855f7',
  app: '#f59e0b', vl: '#06b6d4', tts: '#ec4899',
}
const CAT_LABEL: Record<string, string> = {
  llm: 'LLM', embedding: 'Embedding', image: '图像', app: 'App', vl: '视觉', tts: '语音',
}

function catColor(c: string | null) { return CAT_COLOR[c ?? ''] ?? 'var(--muted)' }

export default function AccessMatrixView({ search = '' }: { search?: string }) {
  const { data, isLoading, error } = useAccessMatrix()
  const addGrant = useAddGrant()
  const removeGrant = useRemoveGrant()
  const pending = addGrant.isPending || removeGrant.isPending

  // grant 查表:`${keyId}:${serviceId}` → grant
  const grantMap = useMemo(() => {
    const m = new Map<string, MatrixGrant>()
    for (const g of data?.grants ?? []) m.set(`${g.key_id}:${g.service_id}`, g)
    return m
  }, [data])

  const q = search.trim().toLowerCase()
  const keys = (data?.keys ?? []).filter((k) => !q || k.label.toLowerCase().includes(q) || k.key_prefix.toLowerCase().includes(q))

  // 服务按类目分组 + 组内按名;受 search 过滤(搜服务名 或 搜到 key 时全显)。
  const groups = useMemo(() => {
    const svcs = (data?.services ?? []).filter((s) => !q || s.name.toLowerCase().includes(q) || (s.backing ?? '').toLowerCase().includes(q) || keys.length > 0)
    const by: Record<string, typeof svcs> = {}
    for (const s of svcs) (by[s.category ?? 'other'] ??= []).push(s)
    const cats = Object.keys(by).sort((a, b) => {
      const ia = CAT_ORDER.indexOf(a), ib = CAT_ORDER.indexOf(b)
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib)
    })
    for (const c of cats) by[c].sort((a, b) => a.name.localeCompare(b.name))
    return cats.map((c) => ({ cat: c, services: by[c] }))
  }, [data, q, keys.length])

  if (isLoading) return <div style={{ color: 'var(--muted)', fontSize: 13 }}>加载中…</div>
  if (error) return <div style={{ color: 'var(--err, #ef4444)', fontSize: 13 }}>矩阵加载失败:{(error as Error).message}</div>
  if (!data) return null

  const toggle = (keyId: string, serviceId: string) => {
    if (pending) return
    const g = grantMap.get(`${keyId}:${serviceId}`)
    if (g) removeGrant.mutate(g.id)
    else addGrant.mutate({ keyId, serviceId })
  }

  const cellW = 116
  const nameW = 240

  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 12, minWidth: nameW + keys.length * cellW }}>
        <thead>
          <tr>
            <th style={{
              position: 'sticky', left: 0, zIndex: 2, width: nameW, minWidth: nameW,
              background: 'var(--bg-accent, #1a1a1a)', color: 'var(--muted)', textAlign: 'left',
              padding: '10px 12px', borderBottom: '1px solid var(--border)', fontWeight: 600,
            }}>服务 \ Key</th>
            {keys.map((k) => (
              <th key={k.id} title={k.key_prefix} style={{
                width: cellW, minWidth: cellW, padding: '8px 6px', textAlign: 'center',
                background: 'var(--bg-accent, #1a1a1a)', borderBottom: '1px solid var(--border)',
                borderLeft: '1px solid var(--border)',
              }}>
                <div style={{ color: 'var(--text)', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: cellW - 12 }}>{k.label}</div>
                <div style={{ color: 'var(--muted)', fontSize: 10 }}>{k.today_calls > 0 ? `今日 ${k.today_calls}` : '—'}{k.is_active ? '' : ' · 禁用'}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {groups.map(({ cat, services }) => (
            <>
              <tr key={`h-${cat}`}>
                <td colSpan={1 + keys.length} style={{
                  background: 'var(--bg)', padding: '6px 12px', borderBottom: '1px solid var(--border)',
                  color: catColor(cat), fontWeight: 600, fontSize: 11, letterSpacing: '.04em',
                  position: 'sticky', left: 0,
                }}>
                  {CAT_LABEL[cat] ?? cat}({services.length})
                </td>
              </tr>
              {services.map((s) => (
                <tr key={s.id}>
                  <td style={{
                    position: 'sticky', left: 0, zIndex: 1, width: nameW, minWidth: nameW,
                    background: 'var(--card, #171a21)', padding: '8px 12px',
                    borderBottom: '1px solid var(--border)',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: catColor(s.category), flex: 'none' }} />
                      <span style={{ color: 'var(--text)', fontWeight: 500 }}>{s.name}</span>
                    </div>
                    <div style={{ color: 'var(--muted)', fontSize: 10, marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: nameW - 24 }}>
                      {s.backing ?? '—'}{s.today_calls > 0 ? ` · 今日 ${s.today_calls}` : ''}
                    </div>
                  </td>
                  {keys.map((k) => {
                    const g = grantMap.get(`${k.id}:${s.id}`)
                    const on = g?.status === 'active'
                    const paused = g?.status === 'paused'
                    const col = catColor(s.category)
                    return (
                      <td key={k.id} style={{ textAlign: 'center', borderLeft: '1px solid var(--border)', borderBottom: '1px solid var(--border)', padding: 0 }}>
                        <button
                          type="button"
                          onClick={() => toggle(k.id, s.id)}
                          disabled={pending}
                          title={on ? '已授权 — 点击撤销' : paused ? '已暂停' : '未授权 — 点击授权'}
                          style={{
                            width: '100%', height: 40, border: 'none', cursor: pending ? 'wait' : 'pointer',
                            background: on ? `${col}22` : 'transparent',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                          }}
                        >
                          {on ? (
                            <span style={{ width: 18, height: 18, borderRadius: '50%', background: col, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
                              <Check size={12} color="#fff" />
                            </span>
                          ) : paused ? (
                            <span style={{ width: 14, height: 14, borderRadius: '50%', background: '#f59e0b', opacity: 0.7 }} />
                          ) : (
                            <span style={{ width: 14, height: 14, borderRadius: '50%', border: '1px solid var(--border)' }} />
                          )}
                        </button>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </>
          ))}
        </tbody>
      </table>
      {keys.length === 0 && <div style={{ padding: 16, color: 'var(--muted)', fontSize: 12 }}>没有 key。先新建一把 Key。</div>}
    </div>
  )
}
