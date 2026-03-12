use axum::Json;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use serde::Serialize;

#[derive(Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub utilization_gpu: u32,
    pub utilization_memory: u32,
    pub temperature: u32,
    pub fan_speed: u32,
    pub power_draw_w: f64,
    pub power_limit_w: f64,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub memory_free_mb: u64,
    pub processes: Vec<GpuProcess>,
}

#[derive(Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub used_gpu_memory_mb: u64,
}

#[derive(Serialize)]
pub struct GpuResponse {
    pub count: usize,
    pub gpus: Vec<GpuInfo>,
}

pub async fn get_gpus() -> Json<GpuResponse> {
    let gpus = match read_gpus() {
        Ok(g) => g,
        Err(_) => vec![],
    };
    Json(GpuResponse {
        count: gpus.len(),
        gpus,
    })
}

fn read_gpus() -> Result<Vec<GpuInfo>, Box<dyn std::error::Error>> {
    let nvml = Nvml::init()?;
    let count = nvml.device_count()?;
    let mut gpus = Vec::new();

    for i in 0..count {
        let device = nvml.device_by_index(i)?;
        let name = device.name()?;
        let util = device.utilization_rates()?;
        let temp = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)?;
        let fan = device.fan_speed(0).unwrap_or(0);
        let power = device.power_usage()? as f64 / 1000.0;
        let power_limit = device.power_management_limit().unwrap_or(0) as f64 / 1000.0;
        let mem = device.memory_info()?;

        let processes = device
            .running_compute_processes()
            .unwrap_or_default()
            .into_iter()
            .map(|p| GpuProcess {
                pid: p.pid,
                used_gpu_memory_mb: match p.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => bytes / (1024 * 1024),
                    UsedGpuMemory::Unavailable => 0,
                },
            })
            .collect();

        gpus.push(GpuInfo {
            index: i,
            name,
            utilization_gpu: util.gpu,
            utilization_memory: util.memory,
            temperature: temp,
            fan_speed: fan,
            power_draw_w: power,
            power_limit_w: power_limit,
            memory_used_mb: mem.used / (1024 * 1024),
            memory_total_mb: mem.total / (1024 * 1024),
            memory_free_mb: mem.free / (1024 * 1024),
            processes,
        });
    }

    Ok(gpus)
}
