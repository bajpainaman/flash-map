//! FlashMap — GPU-native concurrent hash map.
//!
//! Bulk-only API designed for maximum GPU throughput:
//! - `bulk_get`: parallel key lookup
//! - `bulk_insert`: parallel key-value insertion (updates existing keys)
//! - `bulk_remove`: parallel key removal (tombstone-based)
//!
//! SoA (Struct of Arrays) memory layout on GPU for coalesced access.
//! Robin Hood hashing with probe distance early exit.
//!
//! # Features
//!
//! - `cuda` — GPU backend via CUDA (requires NVIDIA GPU + CUDA toolkit)
//! - `rayon` — multi-threaded CPU backend (default)
//!
//! # Example
//!
//! ```rust,no_run
//! use flash_map::{FlashMap, HashStrategy};
//!
//! let mut map: FlashMap<[u8; 32], [u8; 128]> =
//!     FlashMap::with_capacity(1_000_000).unwrap();
//!
//! // Insert 1M key-value pairs in one GPU kernel launch
//! let pairs: Vec<([u8; 32], [u8; 128])> = generate_pairs();
//! map.bulk_insert(&pairs).unwrap();
//!
//! // Lookup
//! let keys: Vec<[u8; 32]> = pairs.iter().map(|(k, _)| *k).collect();
//! let results: Vec<Option<[u8; 128]>> = map.bulk_get(&keys).unwrap();
//! # fn generate_pairs() -> Vec<([u8; 32], [u8; 128])> { vec![] }
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
