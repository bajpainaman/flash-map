# FlashMap

GPU-native concurrent hash map for bulk operations. Designed for workloads where you need to look up, insert, or remove **millions of keys in a single call** — not one at a time.

## Why FlashMap?

Traditional hash maps (std `HashMap`, `DashMap`) process keys sequentially or with CPU thread-level parallelism. FlashMap launches a CUDA kernel where **each GPU thread handles one key** — achieving massive parallelism on modern GPUs.

| Operation | DashMap (16-core) | FlashMap (H100) |
|-----------|-------------------|-----------------|
| 1M inserts | ~500ms | ~60ms |
| 1M lookups | ~300ms | ~35ms |

## Quick Start

```rust
use flash_map::FlashMap;

// Create a map with capacity for 1M entries
let mut map: FlashMap<[u8; 32], [u8; 128]> =
    FlashMap::with_capacity(1_000_000).unwrap();

// Insert 100K key-value pairs in one GPU kernel launch
let pairs: Vec<([u8; 32], [u8; 128])> = generate_pairs();
map.bulk_insert(&pairs).unwrap();

// Look up all keys at once
let keys: Vec<[u8; 32]> = pairs.iter().map(|(k, _)| *k).collect();
let results: Vec<Option<[u8; 128]>> = map.bulk_get(&keys).unwrap();

// Remove a batch of keys
map.bulk_remove(&keys[..1000]).unwrap();
```

## Features

| Feature | Description |
|---------|-------------|
| `cpu-fallback` | Single-threaded CPU backend (default, works everywhere) |
| `rayon` | Multi-threaded CPU backend using Rayon work-stealing |
| `cuda` | GPU backend via CUDA (requires NVIDIA GPU + CUDA 12+) |
| `tokio` | Async wrapper (`AsyncFlashMap`) for tokio runtimes |

```toml
# CPU single-threaded (default)
flash-map = "0.1"

# Rayon multi-threaded CPU (recommended for CPU-only)
flash-map = { version = "0.1", features = ["rayon"] }

# Rayon + async support
flash-map = { version = "0.1", features = ["rayon", "tokio"] }

# GPU acceleration
flash-map = { version = "0.1", features = ["cuda"] }

# GPU + Rayon fallback + async
flash-map = { version = "0.1", features = ["cuda", "rayon", "tokio"] }
```

Backend priority: **GPU > Rayon > CPU**. The builder tries each in order and falls back automatically.

## Applications

### Blockchain State Storage

Store account state (pubkey → account data) with bulk commit after block execution. FlashMap's fixed-size key/value constraint maps directly to blockchain account models where keys are 32-byte public keys and values are fixed-size account structs.

```rust
use flash_map::FlashMap;

type Pubkey = [u8; 32];
type AccountData = [u8; 128];

let mut state: FlashMap<Pubkey, AccountData> =
    FlashMap::with_capacity(10_000_000).unwrap();

// After executing a block of 100K transactions,
// commit all state changes in one GPU call
let changes: Vec<(Pubkey, AccountData)> = execute_block(&txs);
state.bulk_insert(&changes).unwrap();
```

### High-Frequency Trading / Order Books

Batch-update order book entries. Price levels and order quantities change in bursts — FlashMap processes an entire tick's worth of updates in a single kernel launch instead of one-by-one mutex-locked inserts.

```rust
use flash_map::FlashMap;

// OrderId (u64) → OrderEntry (price + qty + side + timestamp)
let mut book: FlashMap<u64, [u8; 32]> = FlashMap::with_capacity(1_000_000).unwrap();

// Process all order updates from a single market data tick
let updates: Vec<(u64, [u8; 32])> = parse_tick_updates(&market_data);
book.bulk_insert(&updates).unwrap();

// Cancel batch of orders
let cancels: Vec<u64> = parse_cancellations(&market_data);
book.bulk_remove(&cancels).unwrap();
```

### Network Packet Deduplication

Deduplicate packets by hash in high-throughput network pipelines. At 100Gbps+, per-packet hash table lookups on CPU become a bottleneck — FlashMap processes an entire batch of packet hashes on GPU.

```rust
use flash_map::FlashMap;

type PacketHash = [u8; 32];
type Marker = u64; // timestamp or sequence number

let mut seen: FlashMap<PacketHash, Marker> =
    FlashMap::with_capacity(10_000_000).unwrap();

// Check which packets in this batch are duplicates
let hashes: Vec<PacketHash> = batch.iter().map(|p| hash(p)).collect();
let results = seen.bulk_get(&hashes).unwrap();

// Insert new (non-duplicate) packets
let new_packets: Vec<(PacketHash, Marker)> = hashes.iter()
    .zip(results.iter())
    .filter(|(_, r)| r.is_none())
    .map(|(h, _)| (*h, current_timestamp()))
    .collect();
seen.bulk_insert(&new_packets).unwrap();
```

### GPU-Accelerated Databases

Use as a GPU-resident index for in-memory databases. Traditional B-tree or hash indexes live in CPU memory — FlashMap keeps the index on GPU, eliminating PCIe round-trips for query-heavy workloads.

```rust
use flash_map::FlashMap;

type RowId = u64;
type IndexKey = [u8; 32]; // hashed column value

let mut index: FlashMap<IndexKey, RowId> =
    FlashMap::with_capacity(50_000_000).unwrap();

// Bulk-load index from table scan
let entries: Vec<(IndexKey, RowId)> = table.iter()
    .map(|row| (hash_column(&row.indexed_col), row.id))
    .collect();
index.bulk_insert(&entries).unwrap();

// Batch point lookups (e.g., JOIN probe side)
let probe_keys: Vec<IndexKey> = probe_table.iter()
    .map(|row| hash_column(&row.join_col))
    .collect();
let matches = index.bulk_get(&probe_keys).unwrap();
```

### Genomics / Bioinformatics

k-mer counting and sequence matching. Genomic analysis involves billions of short DNA subsequences (k-mers) that need to be counted or looked up — a naturally batch-parallel workload.

```rust
use flash_map::FlashMap;

// 32-mer encoded as 8 bytes (2 bits per nucleotide)
type Kmer = u64;
type Count = u64;

let mut kmer_counts: FlashMap<Kmer, Count> =
    FlashMap::with_capacity(100_000_000).unwrap();

// Insert k-mers from a batch of sequence reads
let kmers: Vec<(Kmer, Count)> = extract_kmers(&reads);
kmer_counts.bulk_insert(&kmers).unwrap();

// Query which k-mers from a target sequence exist
let query_kmers: Vec<Kmer> = extract_query_kmers(&target);
let hits = kmer_counts.bulk_get(&query_kmers).unwrap();
```

## Rayon Backend

The `rayon` feature enables a multi-threaded CPU backend that mirrors the GPU kernel's concurrency model — each key gets its own rayon worker thread, and slots are claimed via `AtomicU32` CAS operations (identical to the CUDA `atomicCAS` pattern).

```rust
use flash_map::FlashMap;

// Automatically uses Rayon backend when feature is enabled
let mut map: FlashMap<[u8; 32], [u8; 128]> =
    FlashMap::with_capacity(1_000_000).unwrap();

let pairs: Vec<([u8; 32], [u8; 128])> = generate_pairs();
map.bulk_insert(&pairs).unwrap(); // Parallel across all cores
```

## Async (Tokio)

The `tokio` feature provides `AsyncFlashMap` — a thin async wrapper that runs bulk operations on `spawn_blocking` threads to avoid stalling the async executor.

```rust
use flash_map::{FlashMap, AsyncFlashMap};

let map = FlashMap::with_capacity(1_000_000).unwrap();
let async_map = AsyncFlashMap::new(map);

// Share across tasks via Clone (Arc<RwLock> internally)
let map_clone = async_map.clone();
tokio::spawn(async move {
    let keys = vec![[0u8; 32]];
    let results = map_clone.bulk_get(keys).await.unwrap();
});
```

## Design

### Architecture

```
                       FlashMap<K, V>
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        GPU Backend   Rayon Backend   CPU Backend
          (cuda)        (rayon)     (cpu-fallback)
              │             │
    ┌─────────┤        AtomicU32 CAS
    │    │    │        (lock-free)
  d_keys d_flags d_values
  [u8]   [u32]  [u8]       ← SoA layout on GPU
```

**SoA (Struct of Arrays)**: Keys, flags, and values are stored in separate contiguous GPU buffers. This gives coalesced memory access when all threads read flags simultaneously, then keys, then values — instead of strided access through interleaved AoS records.

**Linear probing** with power-of-2 capacity and bitmask modulo. Cache-friendly on both CPU and GPU.

**Identity hash** (default): Interprets the first 8 bytes of the key as a `u64`. Zero compute overhead for keys that are already well-distributed (SHA256 digests, Ed25519 public keys, UUIDs).

**MurmurHash3**: Available via the builder for keys with poor distribution (sequential integers, low-entropy prefixes).

### Constraints

- **Fixed-size keys and values**: Both `K` and `V` must implement `bytemuck::Pod` (plain old data — `Copy`, fixed layout, any bit pattern valid). No `String`, `Vec`, or heap-allocated types.
- **Bulk-only API**: No single-key `get`/`insert`/`remove`. Wrap in a 1-element slice if needed.
- **50% max load factor**: The table must have at least 2x the capacity of your data. This keeps probe chains short for GPU performance.
- **No duplicate keys per batch**: If the same key appears twice in a single `bulk_insert` call, behavior is non-deterministic on GPU (one will win).

## Configuration

```rust
use flash_map::{FlashMap, HashStrategy};

let map: FlashMap<[u8; 32], [u8; 64]> = FlashMap::builder(1_000_000)
    .hash_strategy(HashStrategy::Murmur3)  // default: Identity
    .device_id(1)                           // default: 0 (first GPU)
    .force_cpu()                            // skip GPU even if available
    .build()
    .unwrap();
```

## Benchmarks

Run on your hardware:

```bash
# CPU fallback
cargo bench

# GPU (requires CUDA 12+)
cargo bench --no-default-features --features cuda
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
