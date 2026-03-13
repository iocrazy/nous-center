// wasm/audio-engine/src/concat.rs
use wasm_bindgen::prelude::*;

/// Concatenate audio tracks with silence gap between them.
/// `tracks` is flat: [track1..., track2...]
/// `track_lengths` is length of each track.
/// `gap_samples` is number of silence samples between tracks.
#[wasm_bindgen]
pub fn concat_tracks(tracks: &[f32], track_lengths: &[u32], gap_samples: u32) -> Vec<f32> {
    let n_tracks = track_lengths.len();
    if n_tracks == 0 {
        return vec![];
    }

    let total: usize = track_lengths.iter().map(|&l| l as usize).sum::<usize>()
        + (n_tracks - 1) * gap_samples as usize;
    let mut output = Vec::with_capacity(total);

    let mut offset = 0usize;
    for (i, &len) in track_lengths.iter().enumerate() {
        let len = len as usize;
        output.extend_from_slice(&tracks[offset..offset + len]);
        offset += len;
        if i < n_tracks - 1 {
            output.extend(std::iter::repeat(0.0f32).take(gap_samples as usize));
        }
    }

    output
}
