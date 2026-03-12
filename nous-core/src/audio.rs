use axum::Json;
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
