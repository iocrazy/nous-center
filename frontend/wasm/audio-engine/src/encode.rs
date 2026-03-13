// wasm/audio-engine/src/encode.rs
use std::io::Cursor;
use hound::{SampleFormat, WavSpec, WavWriter};
use wasm_bindgen::prelude::*;

/// Encode PCM float samples to WAV bytes.
#[wasm_bindgen]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, JsError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buf, spec)
            .map_err(|e| JsError::new(&format!("wav writer: {e}")))?;

        for &s in samples {
            let sample_i16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| JsError::new(&format!("wav write: {e}")))?;
        }

        writer
            .finalize()
            .map_err(|e| JsError::new(&format!("wav finalize: {e}")))?;
    }

    Ok(buf.into_inner())
}
