// FlashMap CUDA kernels — GPU-native concurrent hash map
//
// SoA memory layout (Struct of Arrays for coalesced GPU access):
//   keys:   capacity * key_size bytes   (contiguous key storage)
//   flags:  capacity * sizeof(u32)      (slot state per entry)
//   values: capacity * value_size bytes (contiguous value storage)
//
// Flag values:
//   0 = EMPTY      — slot has never been used
//   1 = OCCUPIED   — slot contains a valid key-value pair
//   2 = TOMBSTONE  — slot was occupied, then removed (probe-through)
//   3 = INSERTING  — transient: slot claimed, key/value being written
//
// All capacities are powers of 2. Modulo uses bitmask: slot & capacity_mask.

#define FLAG_EMPTY     0u
#define FLAG_OCCUPIED  1u
#define FLAG_TOMBSTONE 2u
#define FLAG_INSERTING 3u

// ============================================================================
// Hash functions
// ============================================================================

// Identity hash: first 8 bytes of key as little-endian u64.
// Zero compute — ideal for pre-hashed keys (SHA256 digests, ed25519 pubkeys).
__device__ __forceinline__ unsigned long long fm_identity_hash(
    const unsigned char* key, unsigned int key_size
) {
    unsigned long long h = 0;
    unsigned int n = key_size < 8 ? key_size : 8;
    for (unsigned int i = 0; i < n; i++)
        h |= ((unsigned long long)key[i]) << (i * 8);
    return h;
}

// MurmurHash3-inspired 64-bit hash over full key.
// Good distribution for sequential or low-entropy keys.
__device__ __forceinline__ unsigned long long fm_murmur3_hash(
    const unsigned char* key, unsigned int key_size
) {
    unsigned long long h = 0x9e3779b97f4a7c15ULL;
    unsigned int chunks = key_size / 8;

    for (unsigned int c = 0; c < chunks; c++) {
        unsigned long long k = 0;
        for (unsigned int i = 0; i < 8; i++)
            k |= ((unsigned long long)key[c * 8 + i]) << (i * 8);
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        h ^= k;
        h = h * 5 + 0x52dce729;
    }

    unsigned long long rem = 0;
    for (unsigned int i = chunks * 8; i < key_size; i++)
        rem |= ((unsigned long long)key[i]) << ((i - chunks * 8) * 8);
    if (rem != 0 || key_size % 8 != 0) {
        rem *= 0xff51afd7ed558ccdULL;
        rem ^= rem >> 33;
        h ^= rem;
    }

    h ^= (unsigned long long)key_size;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

__device__ __forceinline__ unsigned long long fm_hash(
    const unsigned char* key, unsigned int key_size, unsigned int mode
) {
    return mode == 0
        ? fm_identity_hash(key, key_size)
        : fm_murmur3_hash(key, key_size);
}

// ============================================================================
// Key comparison — uses 8-byte chunks when aligned, else byte-by-byte
// ============================================================================

__device__ __forceinline__ bool fm_keys_equal(
    const unsigned char* a, const unsigned char* b, unsigned int key_size
) {
    if (key_size >= 8 && key_size % 8 == 0) {
        const unsigned long long* a8 = (const unsigned long long*)a;
        const unsigned long long* b8 = (const unsigned long long*)b;
        for (unsigned int i = 0; i < key_size / 8; i++) {
            if (a8[i] != b8[i]) return false;
        }
        return true;
    }
    if (key_size >= 4 && key_size % 4 == 0) {
        const unsigned int* a4 = (const unsigned int*)a;
        const unsigned int* b4 = (const unsigned int*)b;
        for (unsigned int i = 0; i < key_size / 4; i++) {
            if (a4[i] != b4[i]) return false;
        }
        return true;
    }
    for (unsigned int i = 0; i < key_size; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// ============================================================================
// Bulk Lookup — thread-per-query, linear probing
// ============================================================================

extern "C" __global__ void flashmap_bulk_get(
    const unsigned char* __restrict__ keys,
    const unsigned int*  __restrict__ flags,
    const unsigned char* __restrict__ values,
    const unsigned char* __restrict__ query_keys,
    unsigned char*       __restrict__ out_values,
    unsigned char*       __restrict__ out_found,
    unsigned long long capacity_mask,
    unsigned int key_size,
    unsigned int value_size,
    unsigned int num_queries,
    unsigned int hash_mode
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    const unsigned char* qk = query_keys + (unsigned long long)tid * key_size;
    unsigned long long slot = fm_hash(qk, key_size, hash_mode) & capacity_mask;

    for (unsigned long long p = 0; p <= capacity_mask; p++) {
        unsigned long long idx = (slot + p) & capacity_mask;
        unsigned int f = flags[idx];

        if (f == FLAG_EMPTY) {
            out_found[tid] = 0;
            return;
        }

        if (f == FLAG_OCCUPIED) {
            if (fm_keys_equal(keys + idx * key_size, qk, key_size)) {
                const unsigned char* sv = values + idx * value_size;
                unsigned char* dv = out_values + (unsigned long long)tid * value_size;
                for (unsigned int i = 0; i < value_size; i++)
                    dv[i] = sv[i];
                out_found[tid] = 1;
                return;
            }
        }
        // TOMBSTONE or different occupied key — keep probing
    }

    out_found[tid] = 0;
}

// ============================================================================
// Bulk Insert — thread-per-op, atomicCAS for slot claiming
//
// Invariant: no duplicate keys within a single batch.
// Updates in place if key already exists in the table.
// ============================================================================

extern "C" __global__ void flashmap_bulk_insert(
    unsigned char*       __restrict__ keys,
    unsigned int*        __restrict__ flags,
    unsigned char*       __restrict__ values,
    const unsigned char* __restrict__ in_keys,
    const unsigned char* __restrict__ in_values,
    unsigned long long capacity_mask,
    unsigned int key_size,
    unsigned int value_size,
    unsigned int num_ops,
    unsigned int hash_mode,
    unsigned int* __restrict__ num_inserted
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ops) return;

    const unsigned char* ik = in_keys  + (unsigned long long)tid * key_size;
    const unsigned char* iv = in_values + (unsigned long long)tid * value_size;
    unsigned long long slot = fm_hash(ik, key_size, hash_mode) & capacity_mask;

    for (unsigned long long p = 0; p <= capacity_mask; p++) {
        unsigned long long idx = (slot + p) & capacity_mask;
        unsigned int f = flags[idx];

        // Occupied — check for same-key update
        if (f == FLAG_OCCUPIED) {
            if (fm_keys_equal(keys + idx * key_size, ik, key_size)) {
                unsigned char* tv = values + idx * value_size;
                for (unsigned int i = 0; i < value_size; i++)
                    tv[i] = iv[i];
                return;  // Updated existing, not a new insert
            }
            continue;
        }

        // Empty or tombstone — try to claim via atomicCAS
        if (f == FLAG_EMPTY || f == FLAG_TOMBSTONE) {
            unsigned int old = atomicCAS(&flags[idx], f, FLAG_INSERTING);
            if (old == f) {
                // Claimed — write key + value, then publish
                unsigned char* tk = keys   + idx * key_size;
                unsigned char* tv = values + idx * value_size;
                for (unsigned int i = 0; i < key_size; i++)
                    tk[i] = ik[i];
                for (unsigned int i = 0; i < value_size; i++)
                    tv[i] = iv[i];
                __threadfence();
                flags[idx] = FLAG_OCCUPIED;
                if (f == FLAG_EMPTY)
                    atomicAdd(num_inserted, 1u);
                return;
            }
            // CAS failed — another thread claimed this slot, retry same position
            p--;
            continue;
        }

        // FLAG_INSERTING — another thread is writing here, probe forward
    }
}

// ============================================================================
// Bulk Remove — thread-per-op, tombstone marking
// ============================================================================

extern "C" __global__ void flashmap_bulk_remove(
    const unsigned char* __restrict__ keys,
    unsigned int*        __restrict__ flags,
    const unsigned char* __restrict__ query_keys,
    unsigned long long capacity_mask,
    unsigned int key_size,
    unsigned int num_ops,
    unsigned int hash_mode,
    unsigned int* __restrict__ num_removed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ops) return;

    const unsigned char* qk = query_keys + (unsigned long long)tid * key_size;
    unsigned long long slot = fm_hash(qk, key_size, hash_mode) & capacity_mask;

    for (unsigned long long p = 0; p <= capacity_mask; p++) {
        unsigned long long idx = (slot + p) & capacity_mask;
        unsigned int f = flags[idx];

        if (f == FLAG_EMPTY) return;

        if (f == FLAG_OCCUPIED) {
            if (fm_keys_equal(keys + idx * key_size, qk, key_size)) {
                flags[idx] = FLAG_TOMBSTONE;
                atomicAdd(num_removed, 1u);
                return;
            }
        }
        // TOMBSTONE or different key — keep probing
    }
}

// ============================================================================
// Utility kernels
// ============================================================================

extern "C" __global__ void flashmap_clear(
    unsigned int* __restrict__ flags,
    unsigned long long capacity
) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    flags[tid] = FLAG_EMPTY;
}

extern "C" __global__ void flashmap_count(
    const unsigned int* __restrict__ flags,
    unsigned long long capacity,
    unsigned int* __restrict__ count
) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    if (flags[tid] == FLAG_OCCUPIED)
        atomicAdd(count, 1u);
}
