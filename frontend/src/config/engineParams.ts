/** Per-engine parameter definitions for the TTS Playground UI. */

export interface ParamDef {
  key: string
  label: string
  type: 'number' | 'range' | 'text' | 'select'
  default: number | string
  min?: number
  max?: number
  step?: number
  options?: { value: string | number; label: string }[]
  placeholder?: string
  description?: string
  /** Maps to a different request field name (e.g. key='tokens' maps to request field 'tokens') */
  requestKey?: string
}

export interface EngineParamConfig {
  /** Human-readable name */
  displayName: string
  /** Which standard fields this engine supports */
  supportsSpeed: boolean
  supportsVoice: boolean
  supportsRefAudio: boolean
  supportsRefText: boolean
  supportsSampleRate: boolean
  /** Default sample rate for this engine */
  defaultSampleRate: number
  /** Extra engine-specific parameters */
  extraParams: ParamDef[]
}

export const SAMPLE_RATE_OPTIONS = [
  { value: 16000, label: '16000 Hz' },
  { value: 22050, label: '22050 Hz' },
  { value: 24000, label: '24000 Hz' },
  { value: 44100, label: '44100 Hz' },
  { value: 48000, label: '48000 Hz' },
]

export const ENGINE_PARAMS: Record<string, EngineParamConfig> = {
  cosyvoice2: {
    displayName: 'CosyVoice2-0.5B',
    supportsSpeed: true,
    supportsVoice: true,
    supportsRefAudio: true,
    supportsRefText: true,
    supportsSampleRate: true,
    defaultSampleRate: 24000,
    extraParams: [],
  },
  indextts2: {
    displayName: 'IndexTTS-2',
    supportsSpeed: false,
    supportsVoice: false,
    supportsRefAudio: true,
    supportsRefText: false,
    supportsSampleRate: true,
    defaultSampleRate: 24000,
    extraParams: [],
  },
  qwen3_tts_base: {
    displayName: 'Qwen3-TTS Base',
    supportsSpeed: false,
    supportsVoice: false,
    supportsRefAudio: true,
    supportsRefText: true,
    supportsSampleRate: true,
    defaultSampleRate: 24000,
    extraParams: [],
  },
  qwen3_tts_customvoice: {
    displayName: 'Qwen3-TTS CustomVoice',
    supportsSpeed: false,
    supportsVoice: false,
    supportsRefAudio: true,
    supportsRefText: false,
    supportsSampleRate: true,
    defaultSampleRate: 24000,
    extraParams: [],
  },
  qwen3_tts_voicedesign: {
    displayName: 'Qwen3-TTS VoiceDesign',
    supportsSpeed: false,
    supportsVoice: false,
    supportsRefAudio: false,
    supportsRefText: false,
    supportsSampleRate: true,
    defaultSampleRate: 24000,
    extraParams: [
      {
        key: 'voice_description',
        label: '音色描述',
        type: 'text',
        default: '',
        placeholder: '用文字描述想要的音色，如：成熟稳重的男性声音',
        description: '通过文字描述生成音色，无需参考音频',
      },
    ],
  },
  moss_tts: {
    displayName: 'MOSS-TTS 8B',
    supportsSpeed: false,
    supportsVoice: false,
    supportsRefAudio: true,
    supportsRefText: false,
    supportsSampleRate: false,
    defaultSampleRate: 24000,
    extraParams: [
      {
        key: 'tokens',
        label: '目标时长 (tokens)',
        type: 'number',
        default: 0,
        min: 0,
        max: 4096,
        step: 25,
        placeholder: '留空自动，1秒 ≈ 12.5 tokens',
        description: '控制生成音频长度，0 = 自动',
      },
      {
        key: 'audio_temperature',
        label: 'Temperature',
        type: 'range',
        default: 1.7,
        min: 0.1,
        max: 3.0,
        step: 0.1,
        description: '越高变化越大，越低越稳定',
      },
      {
        key: 'audio_top_p',
        label: 'Top P',
        type: 'range',
        default: 0.8,
        min: 0.1,
        max: 1.0,
        step: 0.05,
      },
      {
        key: 'audio_top_k',
        label: 'Top K',
        type: 'number',
        default: 25,
        min: 1,
        max: 100,
        step: 1,
      },
      {
        key: 'audio_repetition_penalty',
        label: '重复惩罚',
        type: 'range',
        default: 1.0,
        min: 0.8,
        max: 2.0,
        step: 0.1,
        description: '>1.0 减少重复',
      },
    ],
  },
}

/** Get config for an engine, with sensible fallback */
export function getEngineConfig(name: string): EngineParamConfig {
  return ENGINE_PARAMS[name] ?? {
    displayName: name,
    supportsSpeed: true,
    supportsVoice: true,
    supportsRefAudio: false,
    supportsRefText: false,
    supportsSampleRate: true,
    defaultSampleRate: 24000,
    extraParams: [],
  }
}
