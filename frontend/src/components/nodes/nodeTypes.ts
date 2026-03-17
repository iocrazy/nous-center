import type { NodeTypes } from '@xyflow/react'
import TextInputNode from './TextInputNode'
import RefAudioNode from './RefAudioNode'
import TTSEngineNode from './TTSEngineNode'
import OutputNode from './OutputNode'
import ResampleNode from './ResampleNode'
import ConcatNode from './ConcatNode'
import MixerNode from './MixerNode'
import BgmMixNode from './BgmMixNode'
import DeclarativeNode from './DeclarativeNode'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'

const handwrittenTypes: NodeTypes = {
  text_input: TextInputNode,
  ref_audio: RefAudioNode,
  tts_engine: TTSEngineNode,
  output: OutputNode,
  resample: ResampleNode,
  concat: ConcatNode,
  mixer: MixerNode,
  bgm_mix: BgmMixNode,
}

const declarativeTypes = Object.fromEntries(
  Object.keys(DECLARATIVE_NODES).map((type) => [type, DeclarativeNode])
)

export const nodeTypes: NodeTypes = { ...handwrittenTypes, ...declarativeTypes }
