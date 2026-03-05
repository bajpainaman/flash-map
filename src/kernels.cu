// FlashMap CUDA kernels — GPU-native concurrent hash map
//
// Robin Hood hashing with probe distance early exit.
//
// SoA memory layout (Struct of Arrays for coalesced GPU access):
//   keys:   capacity * key_size bytes   (contiguous key storage)
//   flags:  capacity * sizeof(u32)      (packed: 28-bit probe distance | 4-bit flag)
//   values: capacity * value_size bytes (contiguous value storage)
//
// Packed flag format (u32):
//   bits [3:0]  — flag state (EMPTY, OCCUPIED, TOMBSTONE, INSERTING)
//   bits [31:4] — probe distance from home slot (max 268M)
//
// Robin Hood invariant: among consecutive OCCUPIED slots, probe distances
// are non-decreasing. This enables early exit on miss: if current probe
// depth exceeds a resident's stored distance, the key can't exist.
//
// All capacities are powers of 2. Modulo uses bitmask: slot & capacity_mask.

#define FLAG_EMPTY     0u
#define FLAG_OCCUPIED  1u
#define FLAG_TOMBSTONE 2u
#define FLAG_INSERTING 3u

#define GET_FLAG(f)          ((f) & 0xFu)
#define GET_DIST(f)          ((f) >> 4)
#define MAKE_FLAG(flag, dist) ((unsigned int)((unsigned int)(dist) << 4) | ((flag) & 0xFu))

#define FM_MAX_KV_SIZE 256u

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
// Key comparison — uses uint4 (16-byte) chunks when aligned, else byte-by-byte
// ============================================================================

__device__ __forceinline__ bool fm_keys_equal(
    const unsigned char* a, const unsigned char* b, unsigned int key_size
) {
    if (key_size % 16 == 0) {
        const uint4* a16 = (const uint4*)a;
        const uint4* b16 = (const uint4*)b;
        for (unsigned int i = 0; i < key_size / 16; i++) {
            uint4 va = a16[i], vb = b16[i];
            if (va.x != vb.x || va.y != vb.y || va.z != vb.z || va.w != vb.w)
                return false;
        }
        return true;
    }
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
// Copy / swap helpers — use uint4 for 16-byte aligned sizes
// ============================================================================

__device__ __forceinline__ void fm_copy(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    unsigned int n
) {
    if (n % 16 == 0) {
        for (unsigned int i = 0; i < n / 16; i++)
            ((uint4*)dst)[i] = ((const uint4*)src)[i];
    } else if (n % 4 == 0) {
        for (unsigned int i = 0; i < n / 4; i++)
            ((unsigned int*)dst)[i] = ((const unsigned int*)src)[i];
    } else {
        for (unsigned int i = 0; i < n; i++)
            dst[i] = src[i];
    }
}

__device__ __forceinline__ void fm_copy_ldg(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    unsigned int n
) {
    if (n % 16 == 0) {
        for (unsigned int i = 0; i < n / 16; i++)
            ((uint4*)dst)[i] = __ldg(&((const uint4*)src)[i]);
    } else if (n % 4 == 0) {
        for (unsigned int i = 0; i < n / 4; i++)
            ((unsigned int*)dst)[i] = __ldg(&((const unsigned int*)src)[i]);
    } else {
        for (unsigned int i = 0; i < n; i++)
            dst[i] = __ldg(&src[i]);
    }
}

__device__ __forceinline__ void fm_swap(
    unsigned char* __restrict__ a,
    unsigned char* __restrict__ b,
    unsigned int n
) {
    if (n % 16 == 0) {
        for (unsigned int i = 0; i < n / 16; i++) {
            uint4 tmp = ((uint4*)a)[i];
            ((uint4*)a)[i] = ((uint4*)b)[i];
            ((uint4*)b)[i] = tmp;
        }
    } else if (n % 4 == 0) {
        for (unsigned int i = 0; i < n / 4; i++) {
            unsigned int tmp = ((unsigned int*)a)[i];
            ((unsigned int*)a)[i] = ((unsigned int*)b)[i];
            ((unsigned int*)b)[i] = tmp;
        }
    } else {
        for (unsigned int i = 0; i < n; i++) {
            unsigned char tmp = a[i];
            a[i] = b[i];
            b[i] = tmp;
        }
    }
}

// ============================================================================
// Bulk Lookup — Robin Hood early exit on miss
//
// Exits immediately when probe depth > resident's stored distance.
// At 90% load, miss terminates in ~2-3 probes instead of scanning to EMPTY.
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

    for (unsigned long long p = 0; p <= capacity_mask; /* manual increment */) {
        unsigned long long idx = (slot + p) & capacity_mask;
        unsigned int f = __ldg(&flags[idx]);
        unsigned int flag = GET_FLAG(f);
        unsigned int dist = GET_DIST(f);

        if (flag == FLAG_EMPTY) {
            out_found[tid] = 0;
            return;
        }

        if (flag == FLAG_INSERTING) {
            // Another thread is mid-write — spin on this slot
            continue;
        }

        if (flag == FLAG_OCCUPIED) {
            // Robin Hood early exit: our probe depth exceeds resident's
            // distance, so the key can't be here or further
            if (p > dist) {
                out_found[tid] = 0;
                return;
            }
            if (fm_keys_equal(keys + idx * key_size, qk, key_size)) {
                fm_copy_ldg(
                    out_values + (unsigned long long)tid * value_size,
                    values + idx * value_size,
                    value_size
                );
                out_found[tid] = 1;
                return;
            }
        }
        // TOMBSTONE or different occupied key — advance
        p++;
    }

    out_found[tid] = 0;
}

// ============================================================================
// Bulk Insert — Robin Hood with eviction
//
// On insert: if probe_length(incoming) > probe_length(resident),
// evict resident, take slot, continue inserting evicted key.
// Keeps probe distances roughly equal → flat miss performance.
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

    // Local buffers for key/value (supports eviction chain)
    unsigned char cur_key[FM_MAX_KV_SIZE];
    unsigned char cur_val[FM_MAX_KV_SIZE];
    fm_copy(cur_key, in_keys + (unsigned long long)tid * key_size, key_size);
    fm_copy(cur_val, in_values + (unsigned long long)tid * value_size, value_size);

    unsigned long long home = fm_hash(cur_key, key_size, hash_mode) & capacity_mask;
    unsigned long long p = 0;

    for (;;) {
        if (p > capacity_mask) return; // safety: table full

        unsigned long long idx = (home + p) & capacity_mask;
        unsigned int f = flags[idx];
        unsigned int flag = GET_FLAG(f);
        unsigned int dist = GET_DIST(f);

        // Occupied — check for same-key update or Robin Hood eviction
        if (flag == FLAG_OCCUPIED) {
            if (fm_keys_equal(keys + idx * key_size, cur_key, key_size)) {
                // Update existing — not a new insert
                fm_copy(values + idx * value_size, cur_val, value_size);
                return;
            }

            if (p > dist) {
                // Robin Hood: evict resident (short probe dist), take slot
                unsigned int new_f = MAKE_FLAG(FLAG_INSERTING, p);
                unsigned int old = atomicCAS(&flags[idx], f, new_f);
                if (old == f) {
                    // Swap: our key/val into slot, evicted into cur buffers
                    fm_swap(keys + idx * key_size, cur_key, key_size);
                    fm_swap(values + idx * value_size, cur_val, value_size);
                    __threadfence();
                    flags[idx] = MAKE_FLAG(FLAG_OCCUPIED, p);

                    // Continue inserting evicted key from its next probe pos
                    home = fm_hash(cur_key, key_size, hash_mode) & capacity_mask;
                    p = dist + 1;
                    continue;
                }
                // CAS failed — another thread claimed this slot, retry
                continue;
            }

            p++;
            continue;
        }

        // Empty or tombstone — try to claim via atomicCAS
        if (flag == FLAG_EMPTY || flag == FLAG_TOMBSTONE) {
            unsigned int new_f = MAKE_FLAG(FLAG_INSERTING, p);
            unsigned int old = atomicCAS(&flags[idx], f, new_f);
            if (old == f) {
                // Claimed — write key + value, then publish
                fm_copy(keys + idx * key_size, cur_key, key_size);
                fm_copy(values + idx * value_size, cur_val, value_size);
                __threadfence();
                flags[idx] = MAKE_FLAG(FLAG_OCCUPIED, p);
                atomicAdd(num_inserted, 1u);
                return;
            }
            // CAS failed — another thread claimed this slot, retry
            continue;
        }

        // FLAG_INSERTING — another thread is writing here, spin
        if (flag == FLAG_INSERTING) {
            continue;
        }

        p++;
    }
}

// ============================================================================
// Bulk Remove — Robin Hood early exit + atomicCAS for tombstone marking
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

    for (unsigned long long p = 0; p <= capacity_mask; /* manual increment */) {
        unsigned long long idx = (slot + p) & capacity_mask;
        unsigned int f = flags[idx];
        unsigned int flag = GET_FLAG(f);
        unsigned int dist = GET_DIST(f);

        if (flag == FLAG_EMPTY) return;

        if (flag == FLAG_INSERTING) {
            // Another thread is mid-write — spin on this slot
            continue;
        }

        if (flag == FLAG_OCCUPIED) {
            // Robin Hood early exit
            if (p > dist) return;

            if (fm_keys_equal(keys + idx * key_size, qk, key_size)) {
                unsigned int old = atomicCAS(&flags[idx], f, FLAG_TOMBSTONE);
                if (old == f)
                    atomicAdd(num_removed, 1u);
                return;
            }
        }
        // TOMBSTONE or different key — advance
        p++;
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
    flags[tid] = FLAG_EMPTY; // MAKE_FLAG(EMPTY, 0) = 0
}

extern "C" __global__ void flashmap_count(
    const unsigned int* __restrict__ flags,
    unsigned long long capacity,
    unsigned int* __restrict__ count
) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    if (GET_FLAG(flags[tid]) == FLAG_OCCUPIED)
        atomicAdd(count, 1u);
}
