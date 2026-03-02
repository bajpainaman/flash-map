use thiserror::Error;

#[derive(Error, Debug)]
pub enum FlashMapError {
    #[error("CUDA initialization failed: {0}")]
    CudaInit(String),

    #[error("GPU memory allocation failed: {0}")]
    GpuAlloc(String),

    #[error("kernel launch failed: {0}")]
    KernelLaunch(String),

    #[error("host-device transfer failed: {0}")]
    Transfer(String),

    #[error(
        "table full: {occupied} occupied of {capacity} capacity (load factor {load_factor:.1}%)"
    )]
    TableFull {
        occupied: usize,
        capacity: usize,
        load_factor: f64,
    },

    #[error("capacity must be positive")]
    ZeroCapacity,

    #[error("no backend available: enable 'cuda' or 'cpu-fallback' feature")]
    NoBackend,
}
