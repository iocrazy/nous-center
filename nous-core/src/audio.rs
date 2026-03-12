use axum::Json;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use std::path::Path;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Deserialize)]
pub struct AudioInfoRequest {
    pub path: String,
}

#[derive(Serialize)]
pub struct AudioInfoResponse {
    pub sample_rate: u32,
    pub channels: u32,
    pub duration_ms: u64,
    pub format: String,
    pub file_size_bytes: u64,
}

#[derive(Serialize)]
pub struct AudioErrorResponse {
    pub error: String,
}

pub async fn audio_info(Json(req): Json<AudioInfoRequest>) -> Result<Json<AudioInfoResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    let path = Path::new(&req.path);
    if !path.exists() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(AudioErrorResponse { error: format!("File not found: {}", req.path) }),
        ));
    }

    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Cannot open file: {e}") }),
        )),
    };

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(&ext);

    let probed = match symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(p) => p,
        Err(e) => return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: format!("Cannot probe audio: {e}") }),
        )),
    };

    let track = match probed.format.default_track() {
        Some(t) => t,
        None => return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: "No audio track found".into() }),
        )),
    };

    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track.codec_params.channels.map(|c| c.count() as u32).unwrap_or(0);
    let n_frames = track.codec_params.n_frames.unwrap_or(0);
    let duration_ms = if sample_rate > 0 {
        (n_frames as u64 * 1000) / sample_rate as u64
    } else {
        0
    };

    Ok(Json(AudioInfoResponse {
        sample_rate,
        channels,
        duration_ms,
        format: ext,
        file_size_bytes: file_size,
    }))
}

#[derive(Deserialize)]
pub struct ResampleRequest {
    pub input_path: String,
    pub output_path: String,
    pub target_sample_rate: u32,
}

#[derive(Serialize)]
pub struct ResampleResponse {
    pub output_path: String,
    pub original_sample_rate: u32,
    pub target_sample_rate: u32,
    pub duration_ms: u64,
}

pub async fn audio_resample(Json(req): Json<ResampleRequest>) -> Result<Json<ResampleResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    let input = Path::new(&req.input_path);
    if !input.exists() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(AudioErrorResponse { error: format!("Input not found: {}", req.input_path) }),
        ));
    }

    let reader = WavReader::open(input).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        Json(AudioErrorResponse { error: format!("Cannot read WAV: {e}") }),
    ))?;

    let spec = reader.spec();
    let original_sr = spec.sample_rate;
    let channels = spec.channels;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        SampleFormat::Int => reader.into_samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / 32768.0).collect(),
    };

    let ch = channels as usize;
    let frames_in = samples.len() / ch;
    let frames_out = (frames_in as f64 * req.target_sample_rate as f64 / original_sr as f64) as usize;
    let mut resampled = Vec::with_capacity(frames_out * ch);

    for i in 0..frames_out {
        let src_frame = i as f64 * original_sr as f64 / req.target_sample_rate as f64;
        let f0 = src_frame.floor() as usize;
        let f1 = (f0 + 1).min(frames_in - 1);
        let frac = (src_frame - f0 as f64) as f32;
        for c in 0..ch {
            let s0 = samples[f0 * ch + c];
            let s1 = samples[f1 * ch + c];
            resampled.push(s0 * (1.0 - frac) + s1 * frac);
        }
    }

    let out_spec = WavSpec {
        channels,
        sample_rate: req.target_sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&req.output_path, out_spec).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Cannot create output: {e}") }),
    ))?;

    for s in &resampled {
        let val = (*s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(val).map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Write error: {e}") }),
        ))?;
    }

    writer.finalize().map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Finalize error: {e}") }),
    ))?;

    let duration_ms = if req.target_sample_rate > 0 && channels > 0 {
        (resampled.len() as u64 * 1000) / (req.target_sample_rate as u64 * channels as u64)
    } else {
        0
    };

    Ok(Json(ResampleResponse {
        output_path: req.output_path,
        original_sample_rate: original_sr,
        target_sample_rate: req.target_sample_rate,
        duration_ms,
    }))
}
