// wasm/audio-engine/src/mixer.rs
use wasm_bindgen::prelude::*;

/// Mix multiple audio tracks with given volumes.
/// `tracks` is flat: [track1_samples..., track2_samples...]
/// `track_lengths` is length of each track.
/// `volumes` is volume for each track (0.0 - 1.0).
/// Output length = max track length, shorter tracks zero-padded.
#[wasm_bindgen]
pub fn mix_tracks(tracks: &[f32], track_lengths: &[u32], volumes: &[f32]) -> Vec<f32> {
    let n_tracks = track_lengths.len();
    if n_tracks == 0 {
        return vec![];
    }

    let max_len = track_lengths.iter().copied().max().unwrap_or(0) as usize;
    let mut output = vec![0.0f32; max_len];

    let mut offset = 0usize;
    for i in 0..n_tracks {
        let len = track_lengths[i] as usize;
        let vol = volumes.get(i).copied().unwrap_or(1.0);
        for j in 0..len.min(max_len) {
            output[j] += tracks[offset + j] * vol;
        }
        offset += len;
    }

    // Clamp to [-1, 1]
    for s in &mut output {
        *s = s.clamp(-1.0, 1.0);
    }

    output
}
