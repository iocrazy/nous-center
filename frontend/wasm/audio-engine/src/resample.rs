// wasm/audio-engine/src/resample.rs
use rubato::{FftFixedIn, Resampler};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, JsError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        1024,
        2,
        1,
    )
    .map_err(|e| JsError::new(&format!("resampler init: {e}")))?;

    let mut output = Vec::new();
    let chunk_size = resampler.input_frames_next();

    for chunk in samples.chunks(chunk_size) {
        let input = if chunk.len() < chunk_size {
            let mut padded = chunk.to_vec();
            padded.resize(chunk_size, 0.0);
            vec![padded]
        } else {
            vec![chunk.to_vec()]
        };

        let result = resampler
            .process(&input, None)
            .map_err(|e| JsError::new(&format!("resample: {e}")))?;
        output.extend_from_slice(&result[0]);
    }

    Ok(output)
}
