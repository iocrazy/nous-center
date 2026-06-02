import type { NodeTypes } from '@xyflow/react'
import TextInputNode from './TextInputNode'
import TextOutputNode from './TextOutputNode'
import MultimodalInputNode from './MultimodalInputNode'
import RefAudioNode from './RefAudioNode'
import TTSEngineNode from './TTSEngineNode'
import OutputNode from './OutputNode'
import ResampleNode from './ResampleNode'
import ConcatNode from './ConcatNode'
import MixerNode from './MixerNode'
import BgmMixNode from './BgmMixNode'
import DeclarativeNode from './DeclarativeNode'
import ImageOutputNode from './ImageOutputNode'
import ImageCompareNode from './ImageCompareNode'
import { DECLARATIVE_NODES, onPluginDefsLoaded } from '../../models/nodeRegistry'

const handwrittenTypes: NodeTypes = {
  text_input: TextInputNode,
  text_output: TextOutputNode,
  multimodal_input: MultimodalInputNode,
  ref_audio: RefAudioNode,
  tts_engine: TTSEngineNode,
  output: OutputNode,
  resample: ResampleNode,
  concat: ConcatNode,
  mixer: MixerNode,
  bgm_mix: BgmMixNode,
  image_output: ImageOutputNode,
  // image_compare 是插件节点(nodes/image-io),但要自定义滑动对比组件 —— 靠下方
  // handwritten-wins 顺序覆盖默认 DeclarativeNode。
  image_compare: ImageCompareNode,
}

function buildNodeTypes(): NodeTypes {
  const declarativeTypes = Object.fromEntries(
    Object.keys(DECLARATIVE_NODES).map((type) => [type, DeclarativeNode])
  )
  // handwritten 放最后:bespoke 组件**覆盖**声明式默认(否则插件节点拿不到自定义组件)。
  // 当前无 key 同时在两边,纯语义修正 + 解锁插件节点自定义组件。
  return { ...declarativeTypes, ...handwrittenTypes }
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
