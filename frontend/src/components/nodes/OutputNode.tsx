import type { NodeProps } from '@xyflow/react'
import { NODE_DEFS } from '../../models/workflow'
import { useExecutionStore } from '../../stores/execution'
import WavePlayer from '../common/WavePlayer'
import BaseNode from './BaseNode'

export default function OutputNode({ data, selected }: NodeProps) {
  const def = NODE_DEFS.output
  const { isRunning, progress, error } = useExecutionStore()

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <div style={{ padding: '4px 10px' }}>
        {data.audioBase64 ? (
          <WavePlayer
            audioBase64={data.audioBase64 as string}
            sampleRate={data.sampleRate as number}
            duration={data.duration as number}
          />
        ) : isRunning ? (
          <div
            className="flex flex-col items-center gap-1 rounded"
            style={{
              height: 32,
              background: 'var(--bg)',
              borderRadius: 3,
              padding: '4px 5px',
              justifyContent: 'center',
            }}
          >
            <div
              style={{
                width: '100%',
                height: 4,
                background: 'var(--border)',
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${progress}%`,
                  height: '100%',
                  background: 'var(--accent)',
                  borderRadius: 2,
                  transition: 'width 0.3s',
                }}
              />
            </div>
            <span style={{ fontSize: 8, color: 'var(--muted)' }}>
              生成中... {progress > 0 ? `${progress}%` : ''}
            </span>
          </div>
        ) : error ? (
          <div
            className="flex items-center gap-1 rounded"
            style={{
              height: 24,
              background: 'rgba(239,68,68,0.1)',
              borderRadius: 3,
              padding: '0 5px',
              fontSize: 9,
              color: 'var(--warn)',
              justifyContent: 'center',
            }}
          >
            {error}
          </div>
        ) : (
          <div
            className="flex items-center gap-1 rounded"
            style={{
              height: 24,
              background: 'var(--bg)',
              borderRadius: 3,
              padding: '0 5px',
              fontSize: 9,
              color: 'var(--muted-strong)',
              justifyContent: 'center',
            }}
          >
            等待生成...
          </div>
        )}
      </div>
    </BaseNode>
  )
}
