use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytemuck::Pod;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::error::FlashMapError;
use crate::hash::HashStrategy;

const CUDA_KERNEL_SOURCE: &str = include_str!("kernels.cu");
const MODULE_INSERT: &str = "flashmap_insert";
const MODULE_QUERY: &str = "flashmap_query";
const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const MAX_LOAD_FACTOR: f64 = 1.0;

const INSERT_KERNELS: &[&str] = &[
    "flashmap_bulk_insert",
    "flashmap_clear",
    "flashmap_count",
];

const QUERY_KERNELS: &[&str] = &[
    "flashmap_bulk_get",
    "flashmap_bulk_get_values_only",
    "flashmap_bulk_remove",
];

/// GPU-backed FlashMap using CUDA kernels for bulk operations.
pub struct GpuFlashMap<K: Pod, V: Pod> {
    device: Arc<CudaDevice>,
    d_keys: CudaSlice<u8>,
    d_flags: CudaSlice<u32>,
    d_values: CudaSlice<u8>,
    capacity: usize,
    capacity_mask: u64,
    len: AtomicUsize,
    hash_mode: u32,
    _marker: PhantomData<(K, V)>,
}

impl<K: Pod, V: Pod> GpuFlashMap<K, V> {
    pub fn new(
        capacity: usize,
        hash_strategy: HashStrategy,
        device_id: usize,
    ) -> Result<Self, FlashMapError> {
        if capacity == 0 {
            return Err(FlashMapError::ZeroCapacity);
        }

        let capacity = capacity.next_power_of_two();
        let capacity_mask = (capacity - 1) as u64;
        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();

        let device = CudaDevice::new(device_id)
            .map_err(|e| FlashMapError::CudaInit(e.to_string()))?;

        // Compile insert and query kernels as separate PTX modules.
        // This isolates register allocation so warp-coop intrinsics
        // in query kernels don't cause register spill in the insert
        // kernel's local arrays (cur_key/cur_val).
        let insert_src = format!("#define COMPILE_INSERT\n{CUDA_KERNEL_SOURCE}");
        let query_src = format!("#define COMPILE_QUERY\n{CUDA_KERNEL_SOURCE}");

        let ptx_insert = compile_ptx(&insert_src)
            .map_err(|e| FlashMapError::CudaInit(format!("PTX compile (insert): {e}")))?;
        let ptx_query = compile_ptx(&query_src)
            .map_err(|e| FlashMapError::CudaInit(format!("PTX compile (query): {e}")))?;

        device
            .load_ptx(ptx_insert, MODULE_INSERT, INSERT_KERNELS)
            .map_err(|e| FlashMapError::CudaInit(format!("module load (insert): {e}")))?;
        device
            .load_ptx(ptx_query, MODULE_QUERY, QUERY_KERNELS)
            .map_err(|e| FlashMapError::CudaInit(format!("module load (query): {e}")))?;

        // SoA device buffers — zeroed (all flags = EMPTY)
        let d_keys: CudaSlice<u8> = device
            .alloc_zeros(capacity * key_size)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let d_flags: CudaSlice<u32> = device
            .alloc_zeros(capacity)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let d_values: CudaSlice<u8> = device
            .alloc_zeros(capacity * value_size)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        Ok(Self {
            device,
            d_keys,
            d_flags,
            d_values,
            capacity,
            capacity_mask,
            len: AtomicUsize::new(0),
            hash_mode: hash_strategy.to_mode(),
            _marker: PhantomData,
        })
    }

    // =====================================================================
    // Host-facing API (H↔D transfers per call)
    // =====================================================================

    pub fn bulk_get(&self, keys: &[K]) -> Result<Vec<Option<V>>, FlashMapError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();
        let n = keys.len();

        let key_bytes: &[u8] = bytemuck::cast_slice(keys);
        let d_query = self
            .device
            .htod_copy(key_bytes.to_vec())
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;

        let mut d_out_vals: CudaSlice<u8> = self
            .device
            .alloc_zeros(n * value_size)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let mut d_out_found: CudaSlice<u8> = self
            .device
            .alloc_zeros(n)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_QUERY, "flashmap_bulk_get")
            .ok_or_else(|| FlashMapError::KernelLaunch("flashmap_bulk_get not found".into()))?;

        let total_threads = n as u32 * WARP_SIZE;
        let grid = ((total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &self.d_keys,
                    &self.d_flags,
                    &self.d_values,
                    &d_query,
                    &mut d_out_vals,
                    &mut d_out_found,
                    self.capacity_mask,
                    key_size as u32,
                    value_size as u32,
                    n as u32,
                    self.hash_mode,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        let out_bytes = self
            .device
            .dtoh_sync_copy(&d_out_vals)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;
        let out_found = self
            .device
            .dtoh_sync_copy(&d_out_found)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;

        let result = out_bytes
            .chunks_exact(value_size)
            .zip(out_found.iter())
            .map(|(chunk, &found)| {
                if found != 0 {
                    Some(bytemuck::pod_read_unaligned(chunk))
                } else {
                    None
                }
            })
            .collect();

        Ok(result)
    }

    pub fn bulk_insert(&mut self, pairs: &[(K, V)]) -> Result<usize, FlashMapError> {
        if pairs.is_empty() {
            return Ok(0);
        }

        let current_len = self.len.load(Ordering::Relaxed);
        let max_occupancy = (self.capacity as f64 * MAX_LOAD_FACTOR) as usize;
        if current_len + pairs.len() > max_occupancy {
            return Err(FlashMapError::TableFull {
                occupied: current_len,
                capacity: self.capacity,
                load_factor: current_len as f64 / self.capacity as f64 * 100.0,
            });
        }

        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();
        let n = pairs.len();

        let mut key_bytes = Vec::with_capacity(n * key_size);
        let mut val_bytes = Vec::with_capacity(n * value_size);
        for (k, v) in pairs {
            key_bytes.extend_from_slice(bytemuck::bytes_of(k));
            val_bytes.extend_from_slice(bytemuck::bytes_of(v));
        }

        let d_in_keys = self
            .device
            .htod_copy(key_bytes)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;
        let d_in_vals = self
            .device
            .htod_copy(val_bytes)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;

        let mut d_count: CudaSlice<u32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_INSERT, "flashmap_bulk_insert")
            .ok_or_else(|| FlashMapError::KernelLaunch("flashmap_bulk_insert not found".into()))?;

        let grid = ((n as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &mut self.d_keys,
                    &mut self.d_flags,
                    &mut self.d_values,
                    &d_in_keys,
                    &d_in_vals,
                    self.capacity_mask,
                    key_size as u32,
                    value_size as u32,
                    n as u32,
                    self.hash_mode,
                    &mut d_count,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        let count_vec = self
            .device
            .dtoh_sync_copy(&d_count)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;
        let num_new = count_vec[0] as usize;

        self.len.fetch_add(num_new, Ordering::Relaxed);
        Ok(num_new)
    }

    pub fn bulk_remove(&mut self, keys: &[K]) -> Result<usize, FlashMapError> {
        if keys.is_empty() {
            return Ok(0);
        }

        let key_size = std::mem::size_of::<K>();
        let n = keys.len();

        let key_bytes: &[u8] = bytemuck::cast_slice(keys);
        let d_query = self
            .device
            .htod_copy(key_bytes.to_vec())
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;

        let mut d_count: CudaSlice<u32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_QUERY, "flashmap_bulk_remove")
            .ok_or_else(|| FlashMapError::KernelLaunch("flashmap_bulk_remove not found".into()))?;

        let total_threads = n as u32 * WARP_SIZE;
        let grid = ((total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &self.d_keys,
                    &mut self.d_flags,
                    &d_query,
                    self.capacity_mask,
                    key_size as u32,
                    n as u32,
                    self.hash_mode,
                    &mut d_count,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        let count_vec = self
            .device
            .dtoh_sync_copy(&d_count)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;
        let num_removed = count_vec[0] as usize;

        self.len.fetch_sub(num_removed, Ordering::Relaxed);
        Ok(num_removed)
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

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
        let func = self
            .device
            .get_func(MODULE_INSERT, "flashmap_clear")
            .ok_or_else(|| FlashMapError::KernelLaunch("flashmap_clear not found".into()))?;

        let grid = (
            (self.capacity as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
            1,
            1,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (&mut self.d_flags, self.capacity as u64))
                .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        self.len.store(0, Ordering::Relaxed);
        Ok(())
    }

    // =====================================================================
    // Device-resident API — zero-copy operations for on-GPU pipelines
    // =====================================================================

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Transfer host keys to a device buffer (H→D).
    pub fn upload_keys(&self, keys: &[K]) -> Result<CudaSlice<u8>, FlashMapError> {
        let key_bytes: &[u8] = bytemuck::cast_slice(keys);
        self.device
            .htod_copy(key_bytes.to_vec())
            .map_err(|e| FlashMapError::Transfer(e.to_string()))
    }

    /// Transfer host values to a device buffer (H→D).
    pub fn upload_values(&self, values: &[V]) -> Result<CudaSlice<u8>, FlashMapError> {
        let val_bytes: &[u8] = bytemuck::cast_slice(values);
        self.device
            .htod_copy(val_bytes.to_vec())
            .map_err(|e| FlashMapError::Transfer(e.to_string()))
    }

    /// Allocate a zeroed device buffer of `n` bytes.
    pub fn alloc_device(&self, n: usize) -> Result<CudaSlice<u8>, FlashMapError> {
        self.device
            .alloc_zeros(n)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))
    }

    /// Download a device buffer to host memory (D→H).
    pub fn download(&self, d_buf: &CudaSlice<u8>) -> Result<Vec<u8>, FlashMapError> {
        self.device
            .dtoh_sync_copy(d_buf)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))
    }

    /// Device-to-device bulk lookup. Keys already on GPU, results stay on GPU.
    ///
    /// `d_query_keys`: device buffer of `count * size_of::<K>()` bytes.
    ///
    /// Returns `(d_values, d_found)`:
    /// - `d_values`: `count * size_of::<V>()` bytes
    /// - `d_found`: `count` bytes (1 = found, 0 = miss)
    pub fn bulk_get_device(
        &self,
        d_query_keys: &CudaSlice<u8>,
        count: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<u8>), FlashMapError> {
        if count == 0 {
            return Ok((self.alloc_device(0)?, self.alloc_device(0)?));
        }

        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();

        let mut d_out_vals: CudaSlice<u8> = self
            .device
            .alloc_zeros(count * value_size)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let mut d_out_found: CudaSlice<u8> = self
            .device
            .alloc_zeros(count)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_QUERY, "flashmap_bulk_get")
            .ok_or_else(|| FlashMapError::KernelLaunch("flashmap_bulk_get not found".into()))?;

        let total_threads = count as u32 * WARP_SIZE;
        let grid = ((total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &self.d_keys,
                    &self.d_flags,
                    &self.d_values,
                    d_query_keys,
                    &mut d_out_vals,
                    &mut d_out_found,
                    self.capacity_mask,
                    key_size as u32,
                    value_size as u32,
                    count as u32,
                    self.hash_mode,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        Ok((d_out_vals, d_out_found))
    }

    /// Device-to-device values-only lookup. No found mask allocated.
    ///
    /// For pipelines where all keys are guaranteed to exist (e.g., Merkle
    /// tree child lookups). Missing keys get zeroed values (from alloc_zeros).
    pub fn bulk_get_values_device(
        &self,
        d_query_keys: &CudaSlice<u8>,
        count: usize,
    ) -> Result<CudaSlice<u8>, FlashMapError> {
        if count == 0 {
            return self.alloc_device(0);
        }

        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();

        let mut d_out_vals: CudaSlice<u8> = self
            .device
            .alloc_zeros(count * value_size)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_QUERY, "flashmap_bulk_get_values_only")
            .ok_or_else(|| {
                FlashMapError::KernelLaunch("flashmap_bulk_get_values_only not found".into())
            })?;

        let total_threads = count as u32 * WARP_SIZE;
        let grid = ((total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &self.d_keys,
                    &self.d_flags,
                    &self.d_values,
                    d_query_keys,
                    &mut d_out_vals,
                    self.capacity_mask,
                    key_size as u32,
                    value_size as u32,
                    count as u32,
                    self.hash_mode,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        Ok(d_out_vals)
    }

    /// Device-to-device bulk insert. Keys and values already on GPU.
    ///
    /// Returns the number of new insertions (4-byte D→H readback).
    pub fn bulk_insert_device(
        &mut self,
        d_in_keys: &CudaSlice<u8>,
        d_in_vals: &CudaSlice<u8>,
        count: usize,
    ) -> Result<usize, FlashMapError> {
        if count == 0 {
            return Ok(0);
        }

        let current_len = self.len.load(Ordering::Relaxed);
        let max_occupancy = (self.capacity as f64 * MAX_LOAD_FACTOR) as usize;
        if current_len + count > max_occupancy {
            return Err(FlashMapError::TableFull {
                occupied: current_len,
                capacity: self.capacity,
                load_factor: current_len as f64 / self.capacity as f64 * 100.0,
            });
        }

        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();

        let mut d_count: CudaSlice<u32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_INSERT, "flashmap_bulk_insert")
            .ok_or_else(|| {
                FlashMapError::KernelLaunch("flashmap_bulk_insert not found".into())
            })?;

        let grid = ((count as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &mut self.d_keys,
                    &mut self.d_flags,
                    &mut self.d_values,
                    d_in_keys,
                    d_in_vals,
                    self.capacity_mask,
                    key_size as u32,
                    value_size as u32,
                    count as u32,
                    self.hash_mode,
                    &mut d_count,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        let count_vec = self
            .device
            .dtoh_sync_copy(&d_count)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;
        let num_new = count_vec[0] as usize;

        self.len.fetch_add(num_new, Ordering::Relaxed);
        Ok(num_new)
    }

    /// Device-to-device insert without count readback. Fully async.
    ///
    /// No D→H sync. No load factor check (caller's responsibility).
    /// Assumes all insertions are new — `len` is incremented by `count`.
    /// Call [`recount`] later if exact len is needed.
    pub fn bulk_insert_device_uncounted(
        &mut self,
        d_in_keys: &CudaSlice<u8>,
        d_in_vals: &CudaSlice<u8>,
        count: usize,
    ) -> Result<(), FlashMapError> {
        if count == 0 {
            return Ok(());
        }

        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();

        let mut d_count: CudaSlice<u32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_INSERT, "flashmap_bulk_insert")
            .ok_or_else(|| {
                FlashMapError::KernelLaunch("flashmap_bulk_insert not found".into())
            })?;

        let grid = ((count as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &mut self.d_keys,
                    &mut self.d_flags,
                    &mut self.d_values,
                    d_in_keys,
                    d_in_vals,
                    self.capacity_mask,
                    key_size as u32,
                    value_size as u32,
                    count as u32,
                    self.hash_mode,
                    &mut d_count,
                ),
            )
            .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        self.len.fetch_add(count, Ordering::Relaxed);
        Ok(())
    }

    /// Recount occupied entries by scanning flags on GPU.
    ///
    /// Corrects internal `len` after `bulk_insert_device_uncounted`.
    pub fn recount(&self) -> Result<usize, FlashMapError> {
        let mut d_count: CudaSlice<u32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| FlashMapError::GpuAlloc(e.to_string()))?;

        let func = self
            .device
            .get_func(MODULE_INSERT, "flashmap_count")
            .ok_or_else(|| FlashMapError::KernelLaunch("flashmap_count not found".into()))?;

        let grid = (
            (self.capacity as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
            1,
            1,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (&self.d_flags, self.capacity as u64, &mut d_count))
                .map_err(|e| FlashMapError::KernelLaunch(e.to_string()))?;
        }

        let count_vec = self
            .device
            .dtoh_sync_copy(&d_count)
            .map_err(|e| FlashMapError::Transfer(e.to_string()))?;
        let actual = count_vec[0] as usize;

        self.len.store(actual, Ordering::Relaxed);
        Ok(actual)
    }
}
