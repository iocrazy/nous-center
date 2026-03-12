mod gpu;
mod system;
mod files;

use axum::{routing::get, Router};
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/sys/gpus", get(gpu::get_gpus))
        .route("/sys/stats", get(system::get_stats))
        .route("/sys/processes", get(system::get_processes))
        .route("/sys/models", get(files::list_models))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8001")
        .await
        .unwrap();

    println!("nous-core listening on http://127.0.0.1:8001");
    axum::serve(listener, app).await.unwrap();
}
