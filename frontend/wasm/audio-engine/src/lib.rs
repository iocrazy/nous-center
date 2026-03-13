// wasm/audio-engine/src/lib.rs
use wasm_bindgen::prelude::*;

mod decode;
mod waveform;
mod resample;
mod mixer;
mod concat;
mod encode;

pub use decode::{decode_audio, DecodedAudio};
pub use waveform::compute_waveform;
pub use resample::resample;
pub use mixer::mix_tracks;
pub use concat::concat_tracks;
pub use encode::encode_wav;

#[wasm_bindgen]
pub fn ping() -> String {
    "audio-engine wasm ok".to_string()
}
