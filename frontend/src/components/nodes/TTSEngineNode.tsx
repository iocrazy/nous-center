import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { getEngineConfig, ENGINE_PARAMS } from '../../config/engineParams'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeInput, NodeSelect, NodeNumberDrag } from './BaseNode'

export default function TTSEngineNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const engine = (data.engine as string) ?? 'cosyvoice2'
  const cfg = getEngineConfig(engine)
  const def = NODE_DEFS.tts_engine

  return (
    <BaseNode
      title={cfg.displayName}
      badge={{ label: 'Engine', bg: 'var(--accent-subtle)', color: 'var(--accent)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="engine">
        <NodeSelect
          value={engine}
          onChange={(e) => updateNode(id, { engine: e.target.value })}
        >
          {Object.entries(ENGINE_PARAMS).map(([k, v]) => (
            <option key={k} value={k}>{v.displayName}</option>
          ))}
        </NodeSelect>
      </NodeWidgetRow>

      {cfg.supportsSpeed && (
        <NodeWidgetRow label="speed">
          <NodeNumberDrag
            value={(data.speed as number) ?? 1.0}
            onChange={(v) => updateNode(id, { speed: v })}
            min={0.5}
            max={2.0}
            step={0.1}
            precision={1}
          />
        </NodeWidgetRow>
      )}

      {cfg.supportsVoice && (
        <NodeWidgetRow label="voice">
          <NodeInput
            value={(data.voice as string) ?? 'default'}
            onChange={(e) => updateNode(id, { voice: e.target.value })}
          />
        </NodeWidgetRow>
      )}

      {cfg.supportsSampleRate && (
        <NodeWidgetRow label="sample_rate">
          <NodeSelect
            value={(data.sampleRate as number) ?? cfg.defaultSampleRate}
            onChange={(e) => updateNode(id, { sampleRate: parseInt(e.target.value) })}
          >
            <option value={16000}>16000</option>
            <option value={22050}>22050</option>
            <option value={24000}>24000</option>
            <option value={44100}>44100</option>
          </NodeSelect>
        </NodeWidgetRow>
      )}

      <NodeWidgetRow label="emotion">
        <NodeInput
          value={(data.emotion as string) ?? ''}
          onChange={(e) => updateNode(id, { emotion: e.target.value })}
          placeholder="情感描述（可选）"
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}
