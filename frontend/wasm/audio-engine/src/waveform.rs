// wasm/audio-engine/src/waveform.rs
use wasm_bindgen::prelude::*;

/// Compute waveform peaks for display.
/// Returns pairs of [min, max] per bucket, flattened: [min0, max0, min1, max1, ...]
#[wasm_bindgen]
pub fn compute_waveform(samples: &[f32], display_width: usize) -> Vec<f32> {
    if samples.is_empty() || display_width == 0 {
        return vec![];
    }

    let bucket_size = (samples.len() as f64 / display_width as f64).ceil() as usize;
    let bucket_size = bucket_size.max(1);
    let mut peaks = Vec::with_capacity(display_width * 2);

    for chunk in samples.chunks(bucket_size) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &s in chunk {
            if s < min { min = s; }
            if s > max { max = s; }
        }
        peaks.push(min);
        peaks.push(max);
    }

    peaks
}
