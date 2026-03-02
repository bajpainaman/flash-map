use std::sync::Arc;

use bytemuck::Pod;
use tokio::task;

use crate::error::FlashMapError;
use crate::FlashMap;

/// Async wrapper around [`FlashMap`] for use in tokio runtimes.
///
/// All bulk operations run on `spawn_blocking` to avoid blocking the
/// async executor. The inner map is wrapped in `Arc<parking_lot::Mutex>`
/// — if you don't have `parking_lot`, we use a simple `std::sync::Mutex`.
///
/// # Example
///
/// ```rust,no_run
/// use flash_map::{FlashMap, AsyncFlashMap};
///
/// # async fn example() {
/// let map: FlashMap<[u8; 32], [u8; 128]> =
///     FlashMap::with_capacity(1_000_000).unwrap();
/// let async_map = AsyncFlashMap::new(map);
///
/// let pairs = vec![([0u8; 32], [1u8; 128])];
/// async_map.bulk_insert(pairs).await.unwrap();
///
/// let keys = vec![[0u8; 32]];
/// let results = async_map.bulk_get(keys).await.unwrap();
/// # }
/// ```
pub struct AsyncFlashMap<K: Pod, V: Pod> {
    inner: Arc<std::sync::RwLock<FlashMap<K, V>>>,
}

impl<K: Pod + Send + Sync + 'static, V: Pod + Send + Sync + 'static>
    AsyncFlashMap<K, V>
{
    /// Wrap an existing [`FlashMap`] for async usage.
    pub fn new(map: FlashMap<K, V>) -> Self {
        Self {
            inner: Arc::new(std::sync::RwLock::new(map)),
        }
    }

    /// Async bulk lookup. Runs on a blocking thread to avoid stalling
    /// the tokio runtime during GPU kernel execution or CPU probing.
    pub async fn bulk_get(
        &self,
        keys: Vec<K>,
    ) -> Result<Vec<Option<V>>, FlashMapError> {
        let inner = Arc::clone(&self.inner);
        task::spawn_blocking(move || {
            let map = inner
                .read()
                .map_err(|_| FlashMapError::LockPoisoned)?;
            map.bulk_get(&keys)
        })
        .await
        .map_err(|e| FlashMapError::AsyncJoin(e.to_string()))?
    }

    /// Async bulk insert. Runs on a blocking thread.
    pub async fn bulk_insert(
        &self,
        pairs: Vec<(K, V)>,
    ) -> Result<usize, FlashMapError> {
        let inner = Arc::clone(&self.inner);
        task::spawn_blocking(move || {
            let mut map = inner
                .write()
                .map_err(|_| FlashMapError::LockPoisoned)?;
            map.bulk_insert(&pairs)
        })
        .await
        .map_err(|e| FlashMapError::AsyncJoin(e.to_string()))?
    }

    /// Async bulk remove. Runs on a blocking thread.
    pub async fn bulk_remove(
        &self,
        keys: Vec<K>,
    ) -> Result<usize, FlashMapError> {
        let inner = Arc::clone(&self.inner);
        task::spawn_blocking(move || {
            let mut map = inner
                .write()
                .map_err(|_| FlashMapError::LockPoisoned)?;
            map.bulk_remove(&keys)
        })
        .await
        .map_err(|e| FlashMapError::AsyncJoin(e.to_string()))?
    }

    /// Async clear. Runs on a blocking thread.
    pub async fn clear(&self) -> Result<(), FlashMapError> {
        let inner = Arc::clone(&self.inner);
        task::spawn_blocking(move || {
            let mut map = inner
                .write()
                .map_err(|_| FlashMapError::LockPoisoned)?;
            map.clear()
        })
        .await
        .map_err(|e| FlashMapError::AsyncJoin(e.to_string()))?
    }

    /// Current number of entries.
    pub fn len(&self) -> Result<usize, FlashMapError> {
        let map = self
            .inner
            .read()
            .map_err(|_| FlashMapError::LockPoisoned)?;
        Ok(map.len())
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> Result<bool, FlashMapError> {
        Ok(self.len()? == 0)
    }

    /// Total slot capacity.
    pub fn capacity(&self) -> Result<usize, FlashMapError> {
        let map = self
            .inner
            .read()
            .map_err(|_| FlashMapError::LockPoisoned)?;
        Ok(map.capacity())
    }

    /// Current load factor.
    pub fn load_factor(&self) -> Result<f64, FlashMapError> {
        let map = self
            .inner
            .read()
            .map_err(|_| FlashMapError::LockPoisoned)?;
        Ok(map.load_factor())
    }
}

impl<K: Pod, V: Pod> Clone for AsyncFlashMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
