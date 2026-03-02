use std::sync::atomic::{AtomicUsize, Ordering};

use bytemuck::Pod;

use crate::error::FlashMapError;
use crate::hash::{HashStrategy, hash_key};

const MAX_LOAD_FACTOR: f64 = 0.5;
const FLAG_EMPTY: u32 = 0;
const FLAG_OCCUPIED: u32 = 1;
const FLAG_TOMBSTONE: u32 = 2;

/// CPU fallback FlashMap — open-addressing with linear probing.
///
/// Single-threaded bulk operations. Intended for testing and development
/// on machines without an NVIDIA GPU. For production CPU usage, use DashMap.
pub struct CpuFlashMap<K: Pod, V: Pod> {
    keys: Vec<K>,
    values: Vec<V>,
    flags: Vec<u32>,
    capacity: usize,
    capacity_mask: usize,
    len: AtomicUsize,
    hash_strategy: HashStrategy,
}

impl<K: Pod, V: Pod> CpuFlashMap<K, V> {
    pub fn new(capacity: usize, hash_strategy: HashStrategy) -> Self {
        let capacity = capacity.max(16).next_power_of_two();
        Self {
            keys: vec![K::zeroed(); capacity],
            values: vec![V::zeroed(); capacity],
            flags: vec![FLAG_EMPTY; capacity],
            capacity,
            capacity_mask: capacity - 1,
            len: AtomicUsize::new(0),
            hash_strategy,
        }
    }

    pub fn bulk_get(&self, keys: &[K]) -> Result<Vec<Option<V>>, FlashMapError> {
        let mut results = Vec::with_capacity(keys.len());

        for query_key in keys {
            let qk_bytes = bytemuck::bytes_of(query_key);
            let slot = hash_key(qk_bytes, self.hash_strategy) as usize & self.capacity_mask;
            let mut found = false;

            for p in 0..self.capacity {
                let idx = (slot + p) & self.capacity_mask;
                let flag = self.flags[idx];

                if flag == FLAG_EMPTY {
                    break;
                }

                if flag == FLAG_OCCUPIED {
                    let tk_bytes = bytemuck::bytes_of(&self.keys[idx]);
                    if tk_bytes == qk_bytes {
                        results.push(Some(self.values[idx]));
                        found = true;
                        break;
                    }
                }
                // TOMBSTONE — keep probing
            }

            if !found {
                results.push(None);
            }
        }

        Ok(results)
    }

    pub fn bulk_insert(&mut self, pairs: &[(K, V)]) -> Result<usize, FlashMapError> {
        let current_len = self.len.load(Ordering::Relaxed);
        let max_occupancy = (self.capacity as f64 * MAX_LOAD_FACTOR) as usize;
        if current_len + pairs.len() > max_occupancy {
            return Err(FlashMapError::TableFull {
                occupied: current_len,
                capacity: self.capacity,
                load_factor: current_len as f64 / self.capacity as f64 * 100.0,
            });
        }

        let mut num_new: usize = 0;

        for (key, value) in pairs {
            let kbytes = bytemuck::bytes_of(key);
            let slot = hash_key(kbytes, self.hash_strategy) as usize & self.capacity_mask;

            let mut inserted = false;
            for p in 0..self.capacity {
                let idx = (slot + p) & self.capacity_mask;
                let flag = self.flags[idx];

                if flag == FLAG_OCCUPIED {
                    let tk = bytemuck::bytes_of(&self.keys[idx]);
                    if tk == kbytes {
                        // Update existing
                        self.values[idx] = *value;
                        inserted = true;
                        break;
                    }
                    continue;
                }

                if flag == FLAG_EMPTY || flag == FLAG_TOMBSTONE {
                    self.keys[idx] = *key;
                    self.values[idx] = *value;
                    self.flags[idx] = FLAG_OCCUPIED;
                    num_new += 1;
                    inserted = true;
                    break;
                }
            }

            if !inserted {
                return Err(FlashMapError::TableFull {
                    occupied: current_len + num_new,
                    capacity: self.capacity,
                    load_factor: (current_len + num_new) as f64 / self.capacity as f64 * 100.0,
                });
            }
        }

        self.len.fetch_add(num_new, Ordering::Relaxed);
        Ok(num_new)
    }

    pub fn bulk_remove(&mut self, keys: &[K]) -> Result<usize, FlashMapError> {
        let mut num_removed: usize = 0;

        for key in keys {
            let kbytes = bytemuck::bytes_of(key);
            let slot = hash_key(kbytes, self.hash_strategy) as usize & self.capacity_mask;

            for p in 0..self.capacity {
                let idx = (slot + p) & self.capacity_mask;
                let flag = self.flags[idx];

                if flag == FLAG_EMPTY {
                    break;
                }

                if flag == FLAG_OCCUPIED {
                    let tk = bytemuck::bytes_of(&self.keys[idx]);
                    if tk == kbytes {
                        self.flags[idx] = FLAG_TOMBSTONE;
                        num_removed += 1;
                        break;
                    }
                }
            }
        }

        self.len.fetch_sub(num_removed, Ordering::Relaxed);
        Ok(num_removed)
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

    pub fn clear(&mut self) -> Result<(), FlashMapError> {
        self.flags.fill(FLAG_EMPTY);
        self.len.store(0, Ordering::Relaxed);
        Ok(())
    }
}
