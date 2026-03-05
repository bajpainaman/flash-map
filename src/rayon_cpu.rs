use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use bytemuck::Pod;
use rayon::prelude::*;

use crate::error::FlashMapError;
use crate::hash::{HashStrategy, hash_key};

/// Wrapper to send raw pointers across rayon threads.
/// SAFETY: The caller guarantees exclusive slot access via CAS on flags.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn ptr(self) -> *mut T {
        self.0
    }
}

const MAX_LOAD_FACTOR: f64 = 0.5;
const FLAG_EMPTY: u32 = 0;
const FLAG_OCCUPIED: u32 = 1;
const FLAG_TOMBSTONE: u32 = 2;
const FLAG_INSERTING: u32 = 3;

/// Rayon-parallelized CPU FlashMap — uses atomic flags for concurrent
/// insert/remove, mirroring the GPU kernel's CAS-based concurrency model.
pub struct RayonFlashMap<K: Pod, V: Pod> {
    keys: Vec<K>,
    values: Vec<V>,
    flags: Vec<AtomicU32>,
    capacity: usize,
    capacity_mask: usize,
    len: AtomicUsize,
    hash_strategy: HashStrategy,
}

// SAFETY: All fields are Send+Sync. Keys/values are Pod (Copy + no interior
// mutability). Flags use AtomicU32 for thread-safe concurrent access.
// Concurrent inserts use CAS on flags + fence before publishing, matching
// the GPU kernel's memory model.
unsafe impl<K: Pod, V: Pod> Send for RayonFlashMap<K, V> {}
unsafe impl<K: Pod, V: Pod> Sync for RayonFlashMap<K, V> {}

impl<K: Pod + Send + Sync, V: Pod + Send + Sync> RayonFlashMap<K, V> {
    pub fn new(capacity: usize, hash_strategy: HashStrategy) -> Self {
        let capacity = capacity.max(16).next_power_of_two();
        let flags: Vec<AtomicU32> = (0..capacity)
            .map(|_| AtomicU32::new(FLAG_EMPTY))
            .collect();
        Self {
            keys: vec![K::zeroed(); capacity],
            values: vec![V::zeroed(); capacity],
            flags,
            capacity,
            capacity_mask: capacity - 1,
            len: AtomicUsize::new(0),
            hash_strategy,
        }
    }

    /// Parallel bulk lookup using rayon. Each key lookup runs on a
    /// separate rayon worker — reads are lock-free against the atomic flags.
    pub fn bulk_get(&self, keys: &[K]) -> Result<Vec<Option<V>>, FlashMapError> {
        let results: Vec<Option<V>> = keys
            .par_iter()
            .map(|query_key| {
                let qk_bytes = bytemuck::bytes_of(query_key);
                let slot =
                    hash_key(qk_bytes, self.hash_strategy) as usize & self.capacity_mask;

                let mut p: usize = 0;
                while p <= self.capacity_mask {
                    let idx = (slot + p) & self.capacity_mask;
                    let flag = self.flags[idx].load(Ordering::Acquire);

                    if flag == FLAG_EMPTY {
                        return None;
                    }

                    if flag == FLAG_OCCUPIED {
                        let tk_bytes = bytemuck::bytes_of(&self.keys[idx]);
                        if tk_bytes == qk_bytes {
                            return Some(self.values[idx]);
                        }
                    }

                    if flag == FLAG_INSERTING {
                        // Another thread is mid-write — spin on this slot
                        std::hint::spin_loop();
                        continue;
                    }

                    // TOMBSTONE or different occupied key — advance
                    p += 1;
                }
                None
            })
            .collect();

        Ok(results)
    }

    /// Parallel bulk insert using rayon with atomic CAS on flags.
    ///
    /// Each thread claims a slot via CAS(EMPTY|TOMBSTONE → INSERTING),
    /// writes key+value, then publishes via store(OCCUPIED, Release).
    /// This mirrors the CUDA kernel's atomicCAS pattern.
    pub fn bulk_insert(&self, pairs: &[(K, V)]) -> Result<usize, FlashMapError> {
        let current_len = self.len.load(Ordering::Relaxed);
        let max_occupancy = (self.capacity as f64 * MAX_LOAD_FACTOR) as usize;
        if current_len + pairs.len() > max_occupancy {
            return Err(FlashMapError::TableFull {
                occupied: current_len,
                capacity: self.capacity,
                load_factor: current_len as f64 / self.capacity as f64 * 100.0,
            });
        }

        let num_new = AtomicUsize::new(0);

        // SAFETY: We use AtomicU32 flags with CAS to coordinate slot ownership.
        // Only the thread that wins the CAS writes to keys[idx]/values[idx].
        // The Acquire/Release ordering on the flag store ensures the key+value
        // writes are visible to subsequent readers.
        let kp = SendPtr(self.keys.as_ptr() as *mut K);
        let vp = SendPtr(self.values.as_ptr() as *mut V);

        pairs.par_iter().for_each(|(key, value)| {
            let keys_raw = kp.ptr();
            let vals_raw = vp.ptr();
            let kbytes = bytemuck::bytes_of(key);
            let slot =
                hash_key(kbytes, self.hash_strategy) as usize & self.capacity_mask;

            let mut p: usize = 0;
            while p <= self.capacity_mask {
                let idx = (slot + p) & self.capacity_mask;
                let flag = self.flags[idx].load(Ordering::Acquire);

                if flag == FLAG_OCCUPIED {
                    let tk = bytemuck::bytes_of(&self.keys[idx]);
                    if tk == kbytes {
                        // Update existing — we own the slot (key matches)
                        unsafe { vals_raw.add(idx).write(*value) };
                        return;
                    }
                    p += 1;
                    continue;
                }

                if flag == FLAG_EMPTY || flag == FLAG_TOMBSTONE {
                    // Try to claim this slot
                    if self.flags[idx]
                        .compare_exchange(
                            flag,
                            FLAG_INSERTING,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        // We own this slot — write key+value, then publish
                        unsafe {
                            keys_raw.add(idx).write(*key);
                            vals_raw.add(idx).write(*value);
                        }
                        self.flags[idx].store(FLAG_OCCUPIED, Ordering::Release);
                        num_new.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    // CAS failed — retry same slot (don't increment p)
                    std::hint::spin_loop();
                    continue;
                }

                if flag == FLAG_INSERTING {
                    // Another thread is mid-write — spin on this slot
                    std::hint::spin_loop();
                    continue;
                }

                p += 1;
            }
        });

        let added = num_new.load(Ordering::Relaxed);
        self.len.fetch_add(added, Ordering::Relaxed);
        Ok(added)
    }

    /// Parallel bulk remove using rayon. Uses CAS to atomically transition
    /// OCCUPIED → TOMBSTONE for matching keys.
    pub fn bulk_remove(&self, keys: &[K]) -> Result<usize, FlashMapError> {
        let num_removed = AtomicUsize::new(0);

        keys.par_iter().for_each(|key| {
            let kbytes = bytemuck::bytes_of(key);
            let slot =
                hash_key(kbytes, self.hash_strategy) as usize & self.capacity_mask;

            let mut p: usize = 0;
            while p <= self.capacity_mask {
                let idx = (slot + p) & self.capacity_mask;
                let flag = self.flags[idx].load(Ordering::Acquire);

                if flag == FLAG_EMPTY {
                    return;
                }

                if flag == FLAG_INSERTING {
                    // Another thread is mid-write — spin on this slot
                    std::hint::spin_loop();
                    continue;
                }

                if flag == FLAG_OCCUPIED {
                    let tk = bytemuck::bytes_of(&self.keys[idx]);
                    if tk == kbytes {
                        // CAS to ensure we don't double-remove
                        if self.flags[idx]
                            .compare_exchange(
                                FLAG_OCCUPIED,
                                FLAG_TOMBSTONE,
                                Ordering::AcqRel,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            num_removed.fetch_add(1, Ordering::Relaxed);
                        }
                        return;
                    }
                }

                p += 1;
            }
        });

        let removed = num_removed.load(Ordering::Relaxed);
        self.len.fetch_sub(removed, Ordering::Relaxed);
        Ok(removed)
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn load_factor(&self) -> f64 {
        self.len() as f64 / self.capacity as f64
    }

    pub fn clear(&self) -> Result<(), FlashMapError> {
        self.flags
            .par_iter()
            .for_each(|f| f.store(FLAG_EMPTY, Ordering::Relaxed));
        self.len.store(0, Ordering::Relaxed);
        Ok(())
    }
}
