use axum::Json;
use serde::Serialize;
use sysinfo::System;

#[derive(Serialize)]
pub struct SystemStats {
    pub cpu_usage_percent: f32,
    pub cpu_count: usize,
    pub cpu_per_core: Vec<f32>,
    pub memory_total_gb: f64,
    pub memory_used_gb: f64,
    pub memory_available_gb: f64,
    pub swap_total_gb: f64,
    pub swap_used_gb: f64,
}

#[derive(Serialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_percent: f32,
    pub memory_mb: u64,
    pub command: String,
}

#[derive(Serialize)]
pub struct ProcessesResponse {
    pub processes: Vec<ProcessInfo>,
}

pub async fn get_stats() -> Json<SystemStats> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let cpu_usage = if cpu_per_core.is_empty() {
        0.0
    } else {
        cpu_per_core.iter().sum::<f32>() / cpu_per_core.len() as f32
    };

    let gb = |bytes: u64| -> f64 { bytes as f64 / (1024.0 * 1024.0 * 1024.0) };

    Json(SystemStats {
        cpu_usage_percent: cpu_usage,
        cpu_count: sys.cpus().len(),
        cpu_per_core,
        memory_total_gb: gb(sys.total_memory()),
        memory_used_gb: gb(sys.used_memory()),
        memory_available_gb: gb(sys.available_memory()),
        swap_total_gb: gb(sys.total_swap()),
        swap_used_gb: gb(sys.used_swap()),
    })
}

pub async fn get_processes() -> Json<ProcessesResponse> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let mut procs: Vec<ProcessInfo> = sys
        .processes()
        .values()
        .map(|p| ProcessInfo {
            pid: p.pid().as_u32(),
            name: p.name().to_string_lossy().to_string(),
            cpu_percent: p.cpu_usage(),
            memory_mb: p.memory() / (1024 * 1024),
            command: p.cmd().iter().map(|s| s.to_string_lossy().to_string()).collect::<Vec<_>>().join(" "),
        })
        .collect();

    // Sort by CPU usage descending, take top 50
    procs.sort_by(|a, b| b.cpu_percent.partial_cmp(&a.cpu_percent).unwrap_or(std::cmp::Ordering::Equal));
    procs.truncate(50);

    Json(ProcessesResponse { processes: procs })
}
