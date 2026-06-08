import { describe, it, expect } from 'vitest'
import {
  choicesAcceptingInput,
  choicesProvidingOutput,
  firstInputHandle,
  firstOutputHandle,
  getAllChoices,
} from './nodeChoices'

describe('nodeChoices port-type filtering', () => {
  it('getAllChoices returns built-in nodes with labels (deduped)', () => {
    const all = getAllChoices()
    const types = all.map((c) => c.type)
    expect(types).toContain('text_input')
    expect(types).toContain('llm')
    expect(types).toContain('tts_engine')
    // 去重:同一 type 不重复
    expect(new Set(types).size).toBe(types.length)
    // label 来自 NODE_DEFS
    expect(all.find((c) => c.type === 'text_input')?.label).toBe('文本输入')
  })

  it('choicesAcceptingInput(text): nodes with a text INPUT port (from an out-port drag)', () => {
    const types = choicesAcceptingInput('text').map((c) => c.type)
    // tts_engine(text 输入)/ llm(prompt:text)/ prompt_template / agent / if_else / python_exec / text_output 都吃 text
    expect(types).toContain('tts_engine')
    expect(types).toContain('llm')
    expect(types).toContain('prompt_template')
    expect(types).toContain('text_output')
    // text_input 无输入口 → 不应出现
    expect(types).not.toContain('text_input')
    // image_output 只吃 image → 不应出现在 text 候选
    expect(types).not.toContain('image_output')
  })

  it('choicesAcceptingInput(audio): only nodes with an audio INPUT port', () => {
    const types = choicesAcceptingInput('audio').map((c) => c.type)
    expect(types).toContain('tts_engine') // ref_audio:audio
    expect(types).toContain('resample')
    expect(types).toContain('output') // 输出播放 吃 audio
    expect(types).not.toContain('text_output')
  })

  it('choicesProvidingOutput(audio): nodes with an audio OUTPUT port (from an in-port drag)', () => {
    const types = choicesProvidingOutput('audio').map((c) => c.type)
    expect(types).toContain('ref_audio')
    expect(types).toContain('tts_engine')
    expect(types).toContain('resample')
    // text_output 无输出口 → 不出现
    expect(types).not.toContain('text_output')
  })

  it('firstInputHandle / firstOutputHandle resolve the matching handle id', () => {
    // tts_engine 输入: text(text) + ref_audio(audio);第一个 text 输入是 'text'
    expect(firstInputHandle('tts_engine', 'text')).toBe('text')
    expect(firstInputHandle('tts_engine', 'audio')).toBe('ref_audio')
    // llm 输出: text 口 id 'text'
    expect(firstOutputHandle('llm', 'text')).toBe('text')
    // 不匹配的类型 → undefined
    expect(firstInputHandle('text_input', 'text')).toBeUndefined() // 无输入口
    expect(firstOutputHandle('text_output', 'text')).toBeUndefined() // 无输出口
  })
})
