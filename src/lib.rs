//! FlashMap — GPU-native concurrent hash map.
//!
//! Bulk-only API designed for maximum GPU throughput:
//! - `bulk_get` / `bulk_insert` / `bulk_remove` — host-facing (H↔D transfers)
//! - `bulk_get_device` / `bulk_insert_device` — device-resident (zero-copy)
//! - `bulk_get_values_device` — values-only lookup (no found mask)
//! - `bulk_insert_device_uncounted` — fire-and-forget insert (no readback)
//!
//! SoA (Struct of Arrays) memory layout on GPU for coalesced access.
//! Robin Hood hashing with probe distance early exit.
//! Warp-cooperative probing (32 slots per iteration).
//!
//! # Features
//!
//! - `cuda` — GPU backend via CUDA (requires NVIDIA GPU + CUDA toolkit)
//! - `rayon` — multi-threaded CPU backend (default)
//!
//! # Host API Example
//!
//! ```rust,no_run
//! use flash_map::{FlashMap, HashStrategy};
//!
//! let mut map: FlashMap<[u8; 32], [u8; 128]> =
//!     FlashMap::with_capacity(1_000_000).unwrap();
//!
//! let pairs: Vec<([u8; 32], [u8; 128])> = generate_pairs();
//! map.bulk_insert(&pairs).unwrap();
//!
//! let keys: Vec<[u8; 32]> = pairs.iter().map(|(k, _)| *k).collect();
//! let results: Vec<Option<[u8; 128]>> = map.bulk_get(&keys).unwrap();
//! # fn generate_pairs() -> Vec<([u8; 32], [u8; 128])> { vec![] }
//! ```
//!
//! # Device-Resident Pipeline Example
//!
//! ```rust,no_run,ignore
//! use flash_map::FlashMap;
//!
//! let mut map = FlashMap::<u64, [u8; 32]>::with_capacity(1_000_000).unwrap();
//!
//! // Upload keys once (H→D), then all operations stay on GPU
//! let d_keys = map.upload_keys(&[42u64]).unwrap();
//! let d_vals = map.bulk_get_values_device(&d_keys, 1).unwrap();
//! // d_vals is on GPU — pass to your CUDA kernel, then insert results back:
//! // map.bulk_insert_device_uncounted(&d_new_keys, &d_new_vals, n).unwrap();
//! ```

#[cfg(not(any(feature = "cuda", feature = "rayon")))]
compile_error!(
    "flash-map: enable at least one of 'cuda' or 'rayon' features"
);

mod error;
mod hash;

#[cfg(feature = "cuda")]
mod gpu;

#[cfg(feature = "rayon")]
mod rayon_cpu;

#[cfg(feature = "tokio")]
mod async_map;

pub use bytemuck::Pod;
pub use error::FlashMapError;
pub use hash::HashStrategy;

#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaDevice;

#[cfg(feature = "tokio")]
pub use async_map::AsyncFlashMap;

use bytemuck::Pod as PodBound;

// ---------------------------------------------------------------------------
// FlashMap — public API
// ---------------------------------------------------------------------------

/// GPU-native concurrent hash map with bulk-only operations.
///
/// Generic over fixed-size key `K` and value `V` types that implement
/// [`bytemuck::Pod`] (plain old data — `Copy`, fixed layout, any bit
/// pattern valid).
///
/// Common type combinations:
/// - `FlashMap<[u8; 32], [u8; 128]>` — blockchain state (pubkey → account)
/// - `FlashMap<u64, u64>` — numeric keys and values
/// - `FlashMap<[u8; 32], [u8; 32]>` — hash → hash mappings
pub struct FlashMap<K: PodBound, V: PodBound> {
    inner: FlashMapBackend<K, V>,
}

enum FlashMapBackend<K: PodBound, V: PodBound> {
    #[cfg(feature = "cuda")]
    Gpu(gpu::GpuFlashMap<K, V>),
    #[cfg(feature = "rayon")]
    Rayon(rayon_cpu::RayonFlashMap<K, V>),
}

impl<K: PodBound + Send + Sync, V: PodBound + Send + Sync> FlashMap<K, V> {
    /// Create a FlashMap with the given capacity using default settings.
    ///
    /// Tries GPU first (if `cuda` feature enabled), falls back to Rayon.
    /// Capacity is rounded up to the next power of 2.
    pub fn with_capacity(capacity: usize) -> Result<Self, FlashMapError> {
        FlashMapBuilder::new(capacity).build()
    }

    /// Create a builder for fine-grained configuration.
    pub fn builder(capacity: usize) -> FlashMapBuilder {
        FlashMapBuilder::new(capacity)
    }

    // =================================================================
    // Host-facing API (H↔D transfers per call)
    // =================================================================

    /// Look up multiple keys in parallel. Returns `None` for missing keys.
    pub fn bulk_get(&self, keys: &[K]) -> Result<Vec<Option<V>>, FlashMapError> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.bulk_get(keys),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.bulk_get(keys),
        }
    }

    /// Insert multiple key-value pairs in parallel.
    ///
    /// Returns the number of **new** insertions (updates don't count).
    /// If a key already exists, its value is updated in place.
    ///
    /// # Invariant
    ///
    /// No duplicate keys within a single batch. If the same key appears
    /// multiple times, behavior is undefined (one will win, but which
    /// one is non-deterministic on GPU).
    pub fn bulk_insert(&mut self, pairs: &[(K, V)]) -> Result<usize, FlashMapError> {
        match &mut self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.bulk_insert(pairs),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.bulk_insert(pairs),
        }
    }

    /// Remove multiple keys in parallel (tombstone-based).
    ///
    /// Returns the number of keys actually removed.
    pub fn bulk_remove(&mut self, keys: &[K]) -> Result<usize, FlashMapError> {
        match &mut self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.bulk_remove(keys),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.bulk_remove(keys),
        }
    }

    /// Number of occupied entries.
    pub fn len(&self) -> usize {
        match &self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.len(),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.len(),
        }
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total slot capacity (always a power of 2).
    pub fn capacity(&self) -> usize {
        match &self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.capacity(),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.capacity(),
        }
    }

    /// Current load factor (0.0 to 1.0).
    pub fn load_factor(&self) -> f64 {
        match &self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.load_factor(),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.load_factor(),
        }
    }

    /// Remove all entries, resetting to empty.
    pub fn clear(&mut self) -> Result<(), FlashMapError> {
        match &mut self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(m) => m.clear(),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(m) => m.clear(),
        }
    }

    // =================================================================
    // Device-resident API (zero-copy, GPU only)
    // =================================================================

    /// Reference to the CUDA device. Allows sharing device context
    /// with external CUDA kernels (e.g., SHA256 hashers).
    ///
    /// Returns `None` if using the Rayon backend.
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> Option<&std::sync::Arc<CudaDevice>> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => Some(m.device()),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => None,
        }
    }

    /// Transfer host keys to a device buffer (H→D).
    ///
    /// Returns a `CudaSlice<u8>` of `keys.len() * size_of::<K>()` bytes
    /// for use with `bulk_get_device` / `bulk_insert_device`.
    #[cfg(feature = "cuda")]
    pub fn upload_keys(&self, keys: &[K]) -> Result<CudaSlice<u8>, FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.upload_keys(keys),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Transfer host values to a device buffer (H→D).
    #[cfg(feature = "cuda")]
    pub fn upload_values(&self, values: &[V]) -> Result<CudaSlice<u8>, FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.upload_values(values),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Allocate a zeroed device buffer of `n` bytes.
    #[cfg(feature = "cuda")]
    pub fn alloc_device(&self, n: usize) -> Result<CudaSlice<u8>, FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.alloc_device(n),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Download a device buffer to host memory (D→H).
    #[cfg(feature = "cuda")]
    pub fn download(&self, d_buf: &CudaSlice<u8>) -> Result<Vec<u8>, FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.download(d_buf),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Device-to-device bulk lookup. No host memory touched.
    ///
    /// Returns `(d_values, d_found)` — both `CudaSlice<u8>` on GPU.
    /// `d_found` has 1 byte per query (1 = found, 0 = miss).
    #[cfg(feature = "cuda")]
    pub fn bulk_get_device(
        &self,
        d_query_keys: &CudaSlice<u8>,
        count: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<u8>), FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.bulk_get_device(d_query_keys, count),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Device-to-device values-only lookup. No found mask allocated.
    ///
    /// For pipelines where all keys are guaranteed to exist.
    /// Missing keys get zeroed values (from alloc_zeros).
    #[cfg(feature = "cuda")]
    pub fn bulk_get_values_device(
        &self,
        d_query_keys: &CudaSlice<u8>,
        count: usize,
    ) -> Result<CudaSlice<u8>, FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.bulk_get_values_device(d_query_keys, count),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Device-to-device bulk insert. Keys and values already on GPU.
    ///
    /// Returns the number of new insertions (4-byte D→H readback).
    #[cfg(feature = "cuda")]
    pub fn bulk_insert_device(
        &mut self,
        d_keys: &CudaSlice<u8>,
        d_values: &CudaSlice<u8>,
        count: usize,
    ) -> Result<usize, FlashMapError> {
        match &mut self.inner {
            FlashMapBackend::Gpu(m) => m.bulk_insert_device(d_keys, d_values, count),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Device-to-device insert without count readback. Fully async.
    ///
    /// No D→H sync. No load factor check (caller's responsibility).
    /// Assumes all insertions are new. Call `recount()` if exact len needed.
    #[cfg(feature = "cuda")]
    pub fn bulk_insert_device_uncounted(
        &mut self,
        d_keys: &CudaSlice<u8>,
        d_values: &CudaSlice<u8>,
        count: usize,
    ) -> Result<(), FlashMapError> {
        match &mut self.inner {
            FlashMapBackend::Gpu(m) => {
                m.bulk_insert_device_uncounted(d_keys, d_values, count)
            }
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }

    /// Recount occupied entries by scanning flags on GPU.
    ///
    /// Corrects internal `len` after `bulk_insert_device_uncounted`.
    #[cfg(feature = "cuda")]
    pub fn recount(&self) -> Result<usize, FlashMapError> {
        match &self.inner {
            FlashMapBackend::Gpu(m) => m.recount(),
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => Err(FlashMapError::GpuRequired),
        }
    }
}

impl<K: PodBound + Send + Sync, V: PodBound + Send + Sync> std::fmt::Debug
    for FlashMap<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend = match &self.inner {
            #[cfg(feature = "cuda")]
            FlashMapBackend::Gpu(_) => "GPU",
            #[cfg(feature = "rayon")]
            FlashMapBackend::Rayon(_) => "Rayon",
        };
        f.debug_struct("FlashMap")
            .field("backend", &backend)
            .field("len", &self.len())
            .field("capacity", &self.capacity())
            .field("load_factor", &format!("{:.1}%", self.load_factor() * 100.0))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for configuring a [`FlashMap`].
pub struct FlashMapBuilder {
    capacity: usize,
    hash_strategy: HashStrategy,
    device_id: usize,
    force_rayon: bool,
}

impl FlashMapBuilder {
    /// Create a builder targeting the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            hash_strategy: HashStrategy::Identity,
            device_id: 0,
            force_rayon: false,
        }
    }

    /// Set the hash strategy (default: Identity).
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }

    /// Set the CUDA device ordinal (default: 0).
    pub fn device_id(mut self, id: usize) -> Self {
        self.device_id = id;
        self
    }

    /// Force Rayon backend even if CUDA is available.
    pub fn force_cpu(mut self) -> Self {
        self.force_rayon = true;
        self
    }

    /// Build the FlashMap. Tries GPU first, falls back to Rayon.
    pub fn build<K: PodBound + Send + Sync, V: PodBound + Send + Sync>(
        self,
    ) -> Result<FlashMap<K, V>, FlashMapError> {
        let mut _gpu_err: Option<FlashMapError> = None;

        #[cfg(feature = "cuda")]
        if !self.force_rayon {
            match gpu::GpuFlashMap::<K, V>::new(
                self.capacity,
                self.hash_strategy,
                self.device_id,
            ) {
                Ok(m) => return Ok(FlashMap { inner: FlashMapBackend::Gpu(m) }),
                Err(e) => _gpu_err = Some(e),
            }
        }

        #[cfg(feature = "rayon")]
        {
            if let Some(ref e) = _gpu_err {
                eprintln!("[flash-map] GPU unavailable ({e}), using Rayon backend");
            }
            return Ok(FlashMap {
                inner: FlashMapBackend::Rayon(rayon_cpu::RayonFlashMap::new(
                    self.capacity,
                    self.hash_strategy,
                )),
            });
        }

        #[allow(unreachable_code)]
        match _gpu_err {
            Some(e) => Err(e),
            None => Err(FlashMapError::NoBackend),
        }
    }
}
