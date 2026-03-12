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

#[derive(Deserialize)]
pub struct ConcatRequest {
    pub input_paths: Vec<String>,
    pub output_path: String,
}

#[derive(Serialize)]
pub struct ConcatResponse {
    pub output_path: String,
    pub total_duration_ms: u64,
    pub file_count: usize,
}

pub async fn audio_concat(Json(req): Json<ConcatRequest>) -> Result<Json<ConcatResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    if req.input_paths.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: "No input files provided".into() }),
        ));
    }

    let first_reader = WavReader::open(&req.input_paths[0]).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        Json(AudioErrorResponse { error: format!("Cannot read first WAV: {e}") }),
    ))?;
    let spec = first_reader.spec();

    let mut all_samples: Vec<i16> = first_reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();

    for path in &req.input_paths[1..] {
        let reader = WavReader::open(path).map_err(|e| (
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: format!("Cannot read {path}: {e}") }),
        ))?;
        let file_spec = reader.spec();
        if file_spec.sample_rate != spec.sample_rate || file_spec.channels != spec.channels {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(AudioErrorResponse {
                    error: format!(
                        "Sample rate/channel mismatch: expected {}Hz/{}ch, got {}Hz/{}ch in {path}",
                        spec.sample_rate, spec.channels, file_spec.sample_rate, file_spec.channels
                    ),
                }),
            ));
        }
        let samples: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();
        all_samples.extend(samples);
    }

    let mut writer = WavWriter::create(&req.output_path, spec).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Cannot create output: {e}") }),
    ))?;

    for s in &all_samples {
        writer.write_sample(*s).map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Write error: {e}") }),
        ))?;
    }
    writer.finalize().map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Finalize error: {e}") }),
    ))?;

    let total_duration_ms = if spec.sample_rate > 0 && spec.channels > 0 {
        (all_samples.len() as u64 * 1000) / (spec.sample_rate as u64 * spec.channels as u64)
    } else {
        0
    };

    Ok(Json(ConcatResponse {
        output_path: req.output_path,
        total_duration_ms,
        file_count: req.input_paths.len(),
    }))
}

#[derive(Deserialize)]
pub struct SplitRequest {
    pub input_path: String,
    pub output_dir: String,
    pub split_points_ms: Vec<u64>,
}

#[derive(Serialize)]
pub struct SplitResponse {
    pub output_paths: Vec<String>,
    pub durations_ms: Vec<u64>,
}

pub async fn audio_split(Json(req): Json<SplitRequest>) -> Result<Json<SplitResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    let reader = WavReader::open(&req.input_path).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        Json(AudioErrorResponse { error: format!("Cannot read WAV: {e}") }),
    ))?;
    let spec = reader.spec();
    let samples_per_ms = (spec.sample_rate as u64 * spec.channels as u64) / 1000;
    let all_samples: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();

    std::fs::create_dir_all(&req.output_dir).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Cannot create dir: {e}") }),
    ))?;

    let mut boundaries: Vec<usize> = vec![0];
    for &ms in &req.split_points_ms {
        boundaries.push((ms * samples_per_ms) as usize);
    }
    boundaries.push(all_samples.len());

    let mut output_paths = Vec::new();
    let mut durations_ms = Vec::new();

    for i in 0..boundaries.len() - 1 {
        let start = boundaries[i].min(all_samples.len());
        let end = boundaries[i + 1].min(all_samples.len());
        let chunk = &all_samples[start..end];

        let out_path = format!("{}/part_{:03}.wav", req.output_dir, i);
        let mut writer = WavWriter::create(&out_path, spec).map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Write error: {e}") }),
        ))?;
        for s in chunk {
            writer.write_sample(*s).map_err(|e| (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(AudioErrorResponse { error: format!("Write error: {e}") }),
            ))?;
        }
        writer.finalize().map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Finalize error: {e}") }),
        ))?;

        let dur = if samples_per_ms > 0 { chunk.len() as u64 / samples_per_ms } else { 0 };
        output_paths.push(out_path);
        durations_ms.push(dur);
    }

    Ok(Json(SplitResponse { output_paths, durations_ms }))
}

#[derive(Deserialize)]
pub struct ConvertRequest {
    pub input_path: String,
    pub output_path: String,
    pub target_format: String,
}

#[derive(Serialize)]
pub struct ConvertResponse {
    pub output_path: String,
    pub format: String,
}

pub async fn audio_convert(Json(req): Json<ConvertRequest>) -> Result<Json<ConvertResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    if req.target_format != "wav" {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: format!("Unsupported target format: {}. Currently only 'wav' is supported.", req.target_format) }),
        ));
    }

    let input = Path::new(&req.input_path);
    if !input.exists() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(AudioErrorResponse { error: format!("Input not found: {}", req.input_path) }),
        ));
    }

    std::fs::copy(input, &req.output_path).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Copy error: {e}") }),
    ))?;

    Ok(Json(ConvertResponse {
        output_path: req.output_path,
        format: req.target_format,
    }))
}
