import type { NodeTypes } from '@xyflow/react'
import TextInputNode from './TextInputNode'
import TextOutputNode from './TextOutputNode'
import RefAudioNode from './RefAudioNode'
import TTSEngineNode from './TTSEngineNode'
import OutputNode from './OutputNode'
import ResampleNode from './ResampleNode'
import ConcatNode from './ConcatNode'
import MixerNode from './MixerNode'
import BgmMixNode from './BgmMixNode'
import DeclarativeNode from './DeclarativeNode'
import { DECLARATIVE_NODES, onPluginDefsLoaded } from '../../models/nodeRegistry'

const handwrittenTypes: NodeTypes = {
  text_input: TextInputNode,
  text_output: TextOutputNode,
  ref_audio: RefAudioNode,
  tts_engine: TTSEngineNode,
  output: OutputNode,
  resample: ResampleNode,
  concat: ConcatNode,
  mixer: MixerNode,
  bgm_mix: BgmMixNode,
}

function buildNodeTypes(): NodeTypes {
  const declarativeTypes = Object.fromEntries(
    Object.keys(DECLARATIVE_NODES).map((type) => [type, DeclarativeNode])
  )
  return { ...handwrittenTypes, ...declarativeTypes }
}

export let nodeTypes: NodeTypes = buildNodeTypes()

// When plugin definitions are loaded, rebuild the nodeTypes map
onPluginDefsLoaded(() => {
  const rebuilt = buildNodeTypes()
  // Mutate in place so existing references pick up changes
  for (const key of Object.keys(rebuilt)) {
    nodeTypes[key] = rebuilt[key]
  }
})
