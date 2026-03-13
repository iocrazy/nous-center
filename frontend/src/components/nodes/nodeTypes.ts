import type { NodeTypes } from '@xyflow/react'
import TextInputNode from './TextInputNode'
import RefAudioNode from './RefAudioNode'
import TTSEngineNode from './TTSEngineNode'
import OutputNode from './OutputNode'
import ResampleNode from './ResampleNode'
import ConcatNode from './ConcatNode'
import MixerNode from './MixerNode'
import BgmMixNode from './BgmMixNode'

export const nodeTypes: NodeTypes = {
  text_input: TextInputNode,
  ref_audio: RefAudioNode,
  tts_engine: TTSEngineNode,
  output: OutputNode,
  resample: ResampleNode,
  concat: ConcatNode,
  mixer: MixerNode,
  bgm_mix: BgmMixNode,
}
