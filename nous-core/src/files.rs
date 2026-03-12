use axum::{extract::Query, Json};
use serde::{Deserialize, Serialize};
use std::path::Path;
use walkdir::WalkDir;

#[derive(Deserialize)]
pub struct ModelQuery {
    pub path: Option<String>,
}

#[derive(Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub size_mb: u64,
    pub modified: String,
    pub is_directory: bool,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelEntry>,
    pub total_size_mb: u64,
}

pub async fn list_models(Query(q): Query<ModelQuery>) -> Json<ModelsResponse> {
    let base = q
        .path
        .unwrap_or_else(|| "/media/heygo/Program/models".to_string());

    let base_path = Path::new(&base);
    if !base_path.exists() {
        return Json(ModelsResponse {
            models: vec![],
            total_size_mb: 0,
        });
    }

    let mut models = Vec::new();
    let mut total_size: u64 = 0;

    // List top-level entries (depth 1) — each is a model dir or file
    for entry in WalkDir::new(&base).min_depth(1).max_depth(1).into_iter().flatten() {
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let size = if meta.is_dir() {
            dir_size(entry.path())
        } else {
            meta.len()
        };

        let size_mb = size / (1024 * 1024);
        total_size += size_mb;

        let modified = meta
            .modified()
            .ok()
            .and_then(|t| {
                let dt: chrono::DateTime<chrono::Utc> = t.into();
                Some(dt.format("%Y-%m-%d %H:%M").to_string())
            })
            .unwrap_or_default();

        models.push(ModelEntry {
            name: entry.file_name().to_string_lossy().to_string(),
            path: entry.path().to_string_lossy().to_string(),
            size_mb,
            modified,
            is_directory: meta.is_dir(),
        });
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));

    Json(ModelsResponse {
        models,
        total_size_mb: total_size,
    })
}

fn dir_size(path: &Path) -> u64 {
    WalkDir::new(path)
        .into_iter()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}
