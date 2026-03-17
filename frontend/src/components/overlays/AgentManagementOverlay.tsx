import { useState, useEffect } from 'react'
import { Plus, Trash2, Save } from 'lucide-react'
import {
  useAgents,
  useAgent,
  useCreateAgent,
  useUpdateAgent,
  useDeleteAgent,
  useSavePrompt,
} from '../../api/agents'
import { useSkills, type SkillSummary } from '../../api/skills'

const PROMPT_TABS = ['AGENT.md', 'SOUL.md', 'IDENTITY.md'] as const

export default function AgentManagementOverlay() {
  const { data: agents, isLoading } = useAgents()
  const { data: allSkills } = useSkills()
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [showCreateForm, setShowCreateForm] = useState(false)

  const handleSelect = (name: string) => {
    setSelectedName(name)
    setShowCreateForm(false)
  }

  const handleCreated = (name: string) => {
    setShowCreateForm(false)
    setSelectedName(name)
  }

  return (
    <div
      className="absolute inset-0 z-[16] flex"
      style={{ background: 'var(--bg)' }}
    >
      {/* Sidebar */}
      <div
        className="flex flex-col shrink-0"
        style={{
          width: 240,
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-accent)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '10px 12px',
            borderBottom: '1px solid var(--border)',
            fontSize: 12,
            fontWeight: 600,
            color: 'var(--text-strong)',
          }}
        >
          Agents
        </div>

        {/* Agent list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
          {isLoading && (
            <div style={{ padding: 12, fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
          )}
          {agents?.map((agent) => (
            <AgentCard
              key={agent.name}
              name={agent.name}
              displayName={agent.display_name}
              status={agent.status}
              skillCount={agent.skills.length}
              selected={agent.name === selectedName}
              onClick={() => handleSelect(agent.name)}
            />
          ))}
          {!isLoading && agents?.length === 0 && (
            <div style={{ padding: 12, fontSize: 11, color: 'var(--muted)' }}>
              暂无 Agent
            </div>
          )}
        </div>

        {/* New agent button */}
        <div style={{ padding: '8px 12px', borderTop: '1px solid var(--border)' }}>
          <button
            onClick={() => { setShowCreateForm(true); setSelectedName(null) }}
            style={{
              width: '100%',
              border: '1px dashed var(--border)',
              borderRadius: 5,
              padding: 8,
              textAlign: 'center',
              fontSize: 10,
              color: 'var(--muted)',
              background: 'none',
              cursor: 'pointer',
            }}
          >
            + 新建 Agent
          </button>
        </div>
      </div>

      {/* Detail area */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {showCreateForm ? (
          <CreateAgentForm onCreated={handleCreated} onCancel={() => setShowCreateForm(false)} />
        ) : selectedName ? (
          <AgentDetail
            name={selectedName}
            allSkills={allSkills ?? []}
            onDeleted={() => setSelectedName(null)}
          />
        ) : (
          <div
            className="flex items-center justify-center"
            style={{ height: '100%', color: 'var(--muted)', fontSize: 12 }}
          >
            选择一个 Agent 查看详情，或创建新 Agent
          </div>
        )}
      </div>
    </div>
  )
}

/* --- Agent Card (sidebar) --- */

function AgentCard({
  name,
  displayName,
  status,
  skillCount,
  selected,
  onClick,
}: {
  name: string
  displayName: string
  status: string
  skillCount: number
  selected: boolean
  onClick: () => void
}) {
  return (
    <div
      onClick={onClick}
      style={{
        margin: '0 8px 4px',
        background: selected ? 'var(--accent-subtle)' : 'var(--card)',
        border: `1px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 5,
        padding: '7px 10px',
        cursor: 'pointer',
        opacity: status !== 'active' ? 0.45 : 1,
        transition: 'all 0.12s',
      }}
    >
      <div className="flex items-center gap-1.5" style={{ marginBottom: 2 }}>
        <div
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: status === 'active' ? 'var(--ok)' : 'var(--warn)',
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-strong)' }}>
          {displayName || name}
        </span>
      </div>
      <div style={{ fontSize: 9, color: 'var(--muted)', paddingLeft: 12 }}>
        {name} · {skillCount} skills
      </div>
    </div>
  )
}

/* --- Create Agent Form --- */

function CreateAgentForm({
  onCreated,
  onCancel,
}: {
  onCreated: (name: string) => void
  onCancel: () => void
}) {
  const createAgent = useCreateAgent()
  const [name, setName] = useState('')
  const [displayName, setDisplayName] = useState('')

  const handleSubmit = async () => {
    if (!name.trim()) return
    await createAgent.mutateAsync({ name: name.trim(), display_name: displayName.trim() || undefined })
    onCreated(name.trim())
  }

  return (
    <div style={{ padding: '24px 28px', maxWidth: 480 }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-strong)', margin: '0 0 16px' }}>
        新建 Agent
      </h3>
      <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4 }}>
        名称 (唯一标识)
      </label>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="如：narrator"
        autoFocus
        style={{
          width: '100%',
          padding: '6px 10px',
          fontSize: 12,
          background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: 5,
          color: 'var(--text-strong)',
          outline: 'none',
          marginBottom: 12,
          boxSizing: 'border-box',
        }}
      />
      <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4 }}>
        显示名称
      </label>
      <input
        type="text"
        value={displayName}
        onChange={(e) => setDisplayName(e.target.value)}
        placeholder="如：旁白叙述者"
        style={{
          width: '100%',
          padding: '6px 10px',
          fontSize: 12,
          background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: 5,
          color: 'var(--text-strong)',
          outline: 'none',
          marginBottom: 16,
          boxSizing: 'border-box',
        }}
      />
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleSubmit}
          disabled={createAgent.isPending || !name.trim()}
          className="flex items-center gap-1"
          style={{
            padding: '6px 16px',
            fontSize: 11,
            borderRadius: 5,
            border: 'none',
            background: 'var(--accent)',
            color: '#fff',
            cursor: 'pointer',
            opacity: createAgent.isPending || !name.trim() ? 0.5 : 1,
          }}
        >
          <Plus size={12} /> 创建
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: '6px 16px',
            fontSize: 11,
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
    </div>
  )
}

/* --- Agent Detail --- */

function AgentDetail({
  name,
  allSkills,
  onDeleted,
}: {
  name: string
  allSkills: SkillSummary[]
  onDeleted: () => void
}) {
  const { data: agent } = useAgent(name)
  const updateAgent = useUpdateAgent()
  const deleteAgent = useDeleteAgent()
  const savePrompt = useSavePrompt()

  const [displayName, setDisplayName] = useState('')
  const [activeTab, setActiveTab] = useState<(typeof PROMPT_TABS)[number]>('AGENT.md')
  const [promptDrafts, setPromptDrafts] = useState<Record<string, string>>({})
  const [confirmDelete, setConfirmDelete] = useState(false)

  // Sync when agent data loads
  useEffect(() => {
    if (agent) {
      setDisplayName(agent.display_name)
      const drafts: Record<string, string> = {}
      for (const tab of PROMPT_TABS) {
        drafts[tab] = agent.prompts[tab] ?? ''
      }
      setPromptDrafts(drafts)
    }
  }, [agent])

  if (!agent) {
    return (
      <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
    )
  }

  const handleToggleStatus = () => {
    updateAgent.mutate({
      name: agent.name,
      status: agent.status === 'active' ? 'inactive' : 'active',
    })
  }

  const handleSaveDisplayName = () => {
    if (displayName !== agent.display_name) {
      updateAgent.mutate({ name: agent.name, display_name: displayName })
    }
  }

  const handleToggleSkill = (skillName: string) => {
    const current = agent.skills
    const next = current.includes(skillName)
      ? current.filter((s) => s !== skillName)
      : [...current, skillName]
    updateAgent.mutate({ name: agent.name, skills: next })
  }

  const handleSavePrompt = () => {
    const content = promptDrafts[activeTab] ?? ''
    savePrompt.mutate({ name: agent.name, filename: activeTab, content })
  }

  const handleDelete = async () => {
    await deleteAgent.mutateAsync(agent.name)
    onDeleted()
  }

  const handlePromptChange = (value: string) => {
    setPromptDrafts((prev) => ({ ...prev, [activeTab]: value }))
  }

  const currentPromptDirty = (promptDrafts[activeTab] ?? '') !== (agent.prompts[activeTab] ?? '')

  return (
    <div style={{ padding: '16px 24px', maxWidth: 960 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-strong)', margin: 0 }}>
          {agent.display_name || agent.name}
        </h2>
        <button
          onClick={handleToggleStatus}
          disabled={updateAgent.isPending}
          style={{
            fontSize: 10,
            padding: '2px 10px',
            borderRadius: 10,
            border: 'none',
            cursor: 'pointer',
            background: agent.status === 'active' ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
            color: agent.status === 'active' ? 'var(--ok)' : 'var(--warn)',
            fontWeight: 500,
          }}
        >
          {agent.status}
        </button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 20 }}>
        {agent.name}
      </div>

      {/* Display name edit */}
      <SectionTitle>显示名称</SectionTitle>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <input
          type="text"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          onBlur={handleSaveDisplayName}
          onKeyDown={(e) => e.key === 'Enter' && handleSaveDisplayName()}
          style={{
            flex: 1,
            maxWidth: 320,
            padding: '6px 10px',
            fontSize: 12,
            background: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: 5,
            color: 'var(--text-strong)',
            outline: 'none',
          }}
        />
      </div>

      {/* Skills */}
      <SectionTitle>Skills ({agent.skills.length})</SectionTitle>
      <div
        style={{
          background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          padding: '8px 12px',
          marginBottom: 20,
          maxWidth: 480,
        }}
      >
        {allSkills.length === 0 && (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>暂无可用 Skill</div>
        )}
        {allSkills.map((skill) => (
          <label
            key={skill.name}
            className="flex items-center gap-2"
            style={{
              padding: '4px 0',
              fontSize: 11,
              color: 'var(--text-strong)',
              cursor: 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={agent.skills.includes(skill.name)}
              onChange={() => handleToggleSkill(skill.name)}
              style={{ accentColor: 'var(--accent)' }}
            />
            <span style={{ fontWeight: 500 }}>{skill.name}</span>
            {skill.description && (
              <span style={{ color: 'var(--muted)', fontSize: 10 }}>
                — {skill.description}
              </span>
            )}
          </label>
        ))}
      </div>

      {/* Prompt tabs */}
      <SectionTitle>Prompts</SectionTitle>
      <div style={{ display: 'flex', gap: 0, marginBottom: 0 }}>
        {PROMPT_TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '6px 14px',
              fontSize: 10,
              fontWeight: activeTab === tab ? 600 : 400,
              border: '1px solid var(--border)',
              borderBottom: activeTab === tab ? '1px solid var(--card)' : '1px solid var(--border)',
              borderRadius: '5px 5px 0 0',
              background: activeTab === tab ? 'var(--card)' : 'transparent',
              color: activeTab === tab ? 'var(--text-strong)' : 'var(--muted)',
              cursor: 'pointer',
              marginRight: -1,
              position: 'relative',
              zIndex: activeTab === tab ? 1 : 0,
            }}
          >
            {tab}
          </button>
        ))}
      </div>
      <div
        style={{
          background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: '0 6px 6px 6px',
          padding: 12,
          marginBottom: 20,
          maxWidth: 720,
        }}
      >
        <textarea
          value={promptDrafts[activeTab] ?? ''}
          onChange={(e) => handlePromptChange(e.target.value)}
          style={{
            width: '100%',
            minHeight: 240,
            padding: '8px 10px',
            fontSize: 12,
            lineHeight: 1.6,
            fontFamily: 'var(--mono)',
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            color: 'var(--text-strong)',
            outline: 'none',
            resize: 'vertical',
            boxSizing: 'border-box',
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
          <button
            onClick={handleSavePrompt}
            disabled={savePrompt.isPending || !currentPromptDirty}
            className="flex items-center gap-1"
            style={{
              padding: '5px 14px',
              fontSize: 10,
              borderRadius: 4,
              border: 'none',
              background: currentPromptDirty ? 'var(--accent)' : 'var(--border)',
              color: currentPromptDirty ? '#fff' : 'var(--muted)',
              cursor: currentPromptDirty ? 'pointer' : 'default',
            }}
          >
            <Save size={11} />
            {savePrompt.isPending ? '保存中...' : '保存'}
          </button>
        </div>
      </div>

      {/* Delete */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, marginTop: 8 }}>
        {confirmDelete ? (
          <div className="flex items-center gap-3">
            <span style={{ fontSize: 11, color: 'var(--warn)' }}>
              确认删除 Agent "{agent.name}"？此操作不可撤销。
            </span>
            <button
              onClick={handleDelete}
              disabled={deleteAgent.isPending}
              style={{
                fontSize: 10,
                padding: '4px 12px',
                borderRadius: 4,
                border: 'none',
                background: 'rgba(248,113,113,0.15)',
                color: 'var(--warn)',
                cursor: 'pointer',
                fontWeight: 500,
              }}
            >
              {deleteAgent.isPending ? '删除中...' : '确认删除'}
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              style={{
                fontSize: 10,
                padding: '4px 12px',
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: 'none',
                color: 'var(--muted)',
                cursor: 'pointer',
              }}
            >
              取消
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmDelete(true)}
            className="flex items-center gap-1"
            style={{
              fontSize: 10,
              padding: '4px 12px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              background: 'none',
              color: 'var(--muted)',
              cursor: 'pointer',
            }}
          >
            <Trash2 size={11} /> 删除 Agent
          </button>
        )}
      </div>
    </div>
  )
}

/* --- Helpers --- */

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: 'var(--muted)',
        marginBottom: 6,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
      }}
    >
      {children}
    </div>
  )
}
