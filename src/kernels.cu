// FlashMap CUDA kernels — GPU-native concurrent hash map
//
// Robin Hood hashing with probe distance early exit.
// Warp-cooperative probing for bulk_get and bulk_remove:
//   32 threads (1 warp) handle 1 query, probing 32 slots in parallel.
//   __ballot_sync detects match/exit across all lanes simultaneously.
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
#define WARP_SIZE 32u
#define FULL_WARP_MASK 0xFFFFFFFFu

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
// Warp-Cooperative Bulk Lookup
//
// 1 warp (32 threads) handles 1 query. Each iteration, 32 lanes probe 32
// consecutive slots in parallel. __ballot_sync detects:
//   - EMPTY slot → key definitely absent (Robin Hood or not)
//   - OCCUPIED with matching key → found
//   - Robin Hood early exit: probe_depth > stored distance
//
// Lane 0 broadcasts the query's home slot to all lanes via __shfl_sync.
// Each lane probes slot (home + base + lane) & capacity_mask.
//
// Exit logic:
//   1. If any lane sees EMPTY → key can't be further. Done.
//   2. If any lane finds a match → copy value, mark found. Done.
//   3. Robin Hood exit: find earliest lane where probe_dist > resident_dist
//      AND slot is not TOMBSTONE. Any match must be before that lane.
//      Filter match_mask to only include lanes before first_exit.
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
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = global_tid / WARP_SIZE;
    unsigned int lane = global_tid % WARP_SIZE;

    if (warp_id >= num_queries) return;

    // Lane 0 computes hash, broadcasts to all lanes
    const unsigned char* qk = query_keys + (unsigned long long)warp_id * key_size;
    unsigned long long home = 0;
    if (lane == 0) {
        home = fm_hash(qk, key_size, hash_mode) & capacity_mask;
    }
    home = __shfl_sync(FULL_WARP_MASK, home, 0);

    unsigned long long base = 0;

    while (base <= capacity_mask) {
        unsigned long long probe_dist = base + lane;
        unsigned long long idx = (home + probe_dist) & capacity_mask;

        // Each lane loads its slot's flag
        unsigned int f = __ldg(&flags[idx]);
        unsigned int flag = GET_FLAG(f);
        unsigned int dist = GET_DIST(f);

        // Detect EMPTY slots across warp
        unsigned int empty_mask = __ballot_sync(FULL_WARP_MASK, flag == FLAG_EMPTY);

        // Detect Robin Hood early exit: probe deeper than resident
        // Only for OCCUPIED slots (tombstones don't count for RH exit)
        unsigned int rh_exit = (flag == FLAG_OCCUPIED && probe_dist > dist) ? 1u : 0u;
        unsigned int exit_mask = __ballot_sync(FULL_WARP_MASK, rh_exit);

        // Combined termination mask: either EMPTY or RH early exit
        unsigned int term_mask = empty_mask | exit_mask;

        // Detect key matches (only in OCCUPIED, non-INSERTING slots)
        unsigned int is_match = 0u;
        if (flag == FLAG_OCCUPIED && probe_dist <= capacity_mask) {
            is_match = fm_keys_equal(keys + idx * key_size, qk, key_size) ? 1u : 0u;
        }
        unsigned int match_mask = __ballot_sync(FULL_WARP_MASK, is_match);

        // If there's a termination point, only matches BEFORE it are valid
        if (term_mask != 0u) {
            unsigned int first_term = __ffs(term_mask) - 1; // 0-indexed lane
            unsigned int valid_mask = (first_term < 31u) ? ((1u << (first_term + 1)) - 1) : FULL_WARP_MASK;
            match_mask &= valid_mask;
        }

        if (match_mask != 0u) {
            // Found! Lane with the match copies the value
            unsigned int winner = __ffs(match_mask) - 1;
            if (lane == winner) {
                fm_copy_ldg(
                    out_values + (unsigned long long)warp_id * value_size,
                    values + idx * value_size,
                    value_size
                );
                out_found[warp_id] = 1;
            }
            return;
        }

        // If any termination lane was hit and no match found → miss
        if (term_mask != 0u) {
            if (lane == 0) {
                out_found[warp_id] = 0;
            }
            return;
        }

        // Check if any lane is spinning on INSERTING — if so, don't advance
        // past those slots. But we can advance past completed slots.
        unsigned int inserting_mask = __ballot_sync(FULL_WARP_MASK, flag == FLAG_INSERTING);
        if (inserting_mask != 0u) {
            // Retry from the first INSERTING lane's position
            unsigned int first_inserting = __ffs(inserting_mask) - 1;
            base += first_inserting;
            continue;
        }

        // All 32 lanes saw OCCUPIED or TOMBSTONE with no match — advance 32
        base += WARP_SIZE;
    }

    // Wrapped entire table — not found
    if (lane == 0) {
        out_found[warp_id] = 0;
    }
}

// ============================================================================
// Bulk Insert — Robin Hood with eviction (1 thread per key)
//
// Insert stays thread-per-key because Robin Hood eviction chains are
// inherently sequential: each eviction depends on the previous one.
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
// Warp-Cooperative Bulk Remove — Robin Hood early exit + atomicCAS tombstone
//
// Same warp-cooperative pattern as bulk_get: 1 warp per query, 32 slots
// probed in parallel. On match, the winning lane atomicCAS to tombstone.
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
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = global_tid / WARP_SIZE;
    unsigned int lane = global_tid % WARP_SIZE;

    if (warp_id >= num_ops) return;

    const unsigned char* qk = query_keys + (unsigned long long)warp_id * key_size;
    unsigned long long home = 0;
    if (lane == 0) {
        home = fm_hash(qk, key_size, hash_mode) & capacity_mask;
    }
    home = __shfl_sync(FULL_WARP_MASK, home, 0);

    unsigned long long base = 0;

    while (base <= capacity_mask) {
        unsigned long long probe_dist = base + lane;
        unsigned long long idx = (home + probe_dist) & capacity_mask;

        unsigned int f = flags[idx];
        unsigned int flag = GET_FLAG(f);
        unsigned int dist = GET_DIST(f);

        // Detect EMPTY and Robin Hood early exit
        unsigned int empty_mask = __ballot_sync(FULL_WARP_MASK, flag == FLAG_EMPTY);
        unsigned int rh_exit = (flag == FLAG_OCCUPIED && probe_dist > dist) ? 1u : 0u;
        unsigned int exit_mask = __ballot_sync(FULL_WARP_MASK, rh_exit);
        unsigned int term_mask = empty_mask | exit_mask;

        // Detect key matches
        unsigned int is_match = 0u;
        if (flag == FLAG_OCCUPIED && probe_dist <= capacity_mask) {
            is_match = fm_keys_equal(keys + idx * key_size, qk, key_size) ? 1u : 0u;
        }
        unsigned int match_mask = __ballot_sync(FULL_WARP_MASK, is_match);

        // Filter matches to only those before termination point
        if (term_mask != 0u) {
            unsigned int first_term = __ffs(term_mask) - 1;
            unsigned int valid_mask = (first_term < 31u) ? ((1u << (first_term + 1)) - 1) : FULL_WARP_MASK;
            match_mask &= valid_mask;
        }

        if (match_mask != 0u) {
            // Found — winning lane does atomicCAS to tombstone
            unsigned int winner = __ffs(match_mask) - 1;
            if (lane == winner) {
                unsigned int old = atomicCAS(&flags[idx], f, FLAG_TOMBSTONE);
                if (old == f) {
                    atomicAdd(num_removed, 1u);
                }
            }
            return;
        }

        if (term_mask != 0u) {
            return; // not found
        }

        // Handle INSERTING slots — don't skip past them
        unsigned int inserting_mask = __ballot_sync(FULL_WARP_MASK, flag == FLAG_INSERTING);
        if (inserting_mask != 0u) {
            unsigned int first_inserting = __ffs(inserting_mask) - 1;
            base += first_inserting;
            continue;
        }

        base += WARP_SIZE;
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
    if (GET_FLAG(flags[tid]) == FLAG_OCCUPIED)
        atomicAdd(count, 1u);
}
