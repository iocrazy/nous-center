// src/wasm/index.ts
import init, {
  ping,
  decode_audio,
  compute_waveform,
  resample,
  mix_tracks,
  concat_tracks,
  encode_wav,
  type DecodedAudio,
} from './pkg/audio_engine.js'

let initialized = false

export async function initWasm(): Promise<void> {
  if (initialized) return
  await init()
  initialized = true
}

export {
  ping,
  decode_audio,
  compute_waveform,
  resample,
  mix_tracks,
  concat_tracks,
  encode_wav,
  type DecodedAudio,
}
