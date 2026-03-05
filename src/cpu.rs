use std::sync::atomic::{AtomicUsize, Ordering};

use bytemuck::Pod;

use crate::error::FlashMapError;
use crate::hash::{HashStrategy, hash_key};

const MAX_LOAD_FACTOR: f64 = 1.0;
const FLAG_EMPTY: u32 = 0;
const FLAG_OCCUPIED: u32 = 1;
const FLAG_TOMBSTONE: u32 = 2;

#[inline]
const fn get_flag(f: u32) -> u32 {
    f & 0xF
}

#[inline]
const fn get_dist(f: u32) -> usize {
    (f >> 4) as usize
}

#[inline]
const fn make_flag(flag: u32, dist: usize) -> u32 {
    ((dist as u32) << 4) | (flag & 0xF)
}

/// CPU fallback FlashMap — Robin Hood hashing with linear probing.
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

    /// Bulk lookup with Robin Hood early exit.
    ///
    /// Terminates miss probes when `probe_depth > stored_distance`,
    /// avoiding full-chain scans at high load factors.
    pub fn bulk_get(&self, keys: &[K]) -> Result<Vec<Option<V>>, FlashMapError> {
        let mut results = Vec::with_capacity(keys.len());

        for query_key in keys {
            let qk_bytes = bytemuck::bytes_of(query_key);
            let slot =
                hash_key(qk_bytes, self.hash_strategy) as usize & self.capacity_mask;
            let mut found = false;

            let mut p: usize = 0;
            while p <= self.capacity_mask {
                let idx = (slot + p) & self.capacity_mask;
                let f = self.flags[idx];
                let flag = get_flag(f);
                let dist = get_dist(f);

                if flag == FLAG_EMPTY {
                    break;
                }

                if flag == FLAG_OCCUPIED {
                    // Robin Hood early exit
                    if p > dist {
                        break;
                    }
                    let tk_bytes = bytemuck::bytes_of(&self.keys[idx]);
                    if tk_bytes == qk_bytes {
                        results.push(Some(self.values[idx]));
                        found = true;
                        break;
                    }
                }
                // TOMBSTONE — keep probing
                p += 1;
            }

            if !found {
                results.push(None);
            }
        }

        Ok(results)
    }

    /// Bulk insert with Robin Hood eviction.
    ///
    /// When a new key's probe distance exceeds a resident's, the resident
    /// is evicted and re-inserted further. This equalizes probe distances
    /// and keeps miss throughput flat at high load factors.
    pub fn bulk_insert(
        &mut self,
        pairs: &[(K, V)],
    ) -> Result<usize, FlashMapError> {
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

        for &(key, value) in pairs {
            let mut cur_key = key;
            let mut cur_val = value;
            let mut home = hash_key(
                bytemuck::bytes_of(&cur_key),
                self.hash_strategy,
            ) as usize
                & self.capacity_mask;
            let mut p: usize = 0;

            loop {
                if p > self.capacity_mask {
                    return Err(FlashMapError::TableFull {
                        occupied: current_len + num_new,
                        capacity: self.capacity,
                        load_factor: (current_len + num_new) as f64
                            / self.capacity as f64
                            * 100.0,
                    });
                }

                let idx = (home + p) & self.capacity_mask;
                let f = self.flags[idx];
                let flag = get_flag(f);
                let dist = get_dist(f);

                if flag == FLAG_OCCUPIED {
                    let tk = bytemuck::bytes_of(&self.keys[idx]);
                    if tk == bytemuck::bytes_of(&cur_key) {
                        // Update existing
                        self.values[idx] = cur_val;
                        break;
                    }

                    if p > dist {
                        // Robin Hood: evict resident, take slot
                        let evict_key = self.keys[idx];
                        let evict_val = self.values[idx];
                        self.keys[idx] = cur_key;
                        self.values[idx] = cur_val;
                        self.flags[idx] = make_flag(FLAG_OCCUPIED, p);

                        // Continue inserting evicted key
                        cur_key = evict_key;
                        cur_val = evict_val;
                        home = hash_key(
                            bytemuck::bytes_of(&cur_key),
                            self.hash_strategy,
                        ) as usize
                            & self.capacity_mask;
                        p = dist + 1;
                        continue;
                    }

                    p += 1;
                    continue;
                }

                if flag == FLAG_EMPTY || flag == FLAG_TOMBSTONE {
                    self.keys[idx] = cur_key;
                    self.values[idx] = cur_val;
                    self.flags[idx] = make_flag(FLAG_OCCUPIED, p);
                    num_new += 1;
                    break;
                }

                p += 1;
            }
        }

        self.len.fetch_add(num_new, Ordering::Relaxed);
        Ok(num_new)
    }

    /// Bulk remove with Robin Hood early exit.
    pub fn bulk_remove(
        &mut self,
        keys: &[K],
    ) -> Result<usize, FlashMapError> {
        let mut num_removed: usize = 0;

        for key in keys {
            let kbytes = bytemuck::bytes_of(key);
            let slot =
                hash_key(kbytes, self.hash_strategy) as usize & self.capacity_mask;

            let mut p: usize = 0;
            while p <= self.capacity_mask {
                let idx = (slot + p) & self.capacity_mask;
                let f = self.flags[idx];
                let flag = get_flag(f);
                let dist = get_dist(f);

                if flag == FLAG_EMPTY {
                    break;
                }

                if flag == FLAG_OCCUPIED {
                    // Robin Hood early exit
                    if p > dist {
                        break;
                    }
                    let tk = bytemuck::bytes_of(&self.keys[idx]);
                    if tk == kbytes {
                        self.flags[idx] = FLAG_TOMBSTONE;
                        num_removed += 1;
                        break;
                    }
                }
                p += 1;
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
