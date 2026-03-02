/// Hash strategy for key distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashStrategy {
    /// First 8 bytes of key as little-endian u64 (zero compute).
    /// Best for pre-hashed keys: SHA256 digests, ed25519 pubkeys.
    Identity,

    /// MurmurHash3 64-bit finalizer.
    /// Use for keys with low-entropy prefixes (sequential IDs, etc.).
    Murmur3,
}

impl Default for HashStrategy {
    fn default() -> Self {
        Self::Identity
    }
}

impl HashStrategy {
    /// Convert to u32 mode flag for the CUDA kernel.
    #[allow(dead_code)]
    pub(crate) fn to_mode(&self) -> u32 {
        match self {
            Self::Identity => 0,
            Self::Murmur3 => 1,
        }
    }
}

/// Identity hash: first 8 bytes as little-endian u64.
#[inline]
pub(crate) fn identity_hash(key_bytes: &[u8]) -> u64 {
    let mut h: u64 = 0;
    let n = key_bytes.len().min(8);
    for (i, &b) in key_bytes[..n].iter().enumerate() {
        h |= (b as u64) << (i * 8);
    }
    h
}

/// MurmurHash3-inspired 64-bit hash over full key.
#[inline]
pub(crate) fn murmur3_hash(key_bytes: &[u8]) -> u64 {
    let mut h: u64 = 0x9e3779b97f4a7c15; // golden ratio
    let chunks = key_bytes.len() / 8;

    for c in 0..chunks {
        let mut k: u64 = 0;
        for i in 0..8 {
            k |= (key_bytes[c * 8 + i] as u64) << (i * 8);
        }
        k = k.wrapping_mul(0xff51afd7ed558ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
        k ^= k >> 33;
        h ^= k;
        h = h.wrapping_mul(5).wrapping_add(0x52dce729);
    }

    let mut rem: u64 = 0;
    for i in (chunks * 8)..key_bytes.len() {
        rem |= (key_bytes[i] as u64) << ((i - chunks * 8) * 8);
    }
    if rem != 0 || key_bytes.len() % 8 != 0 {
        rem = rem.wrapping_mul(0xff51afd7ed558ccd);
        rem ^= rem >> 33;
        h ^= rem;
    }

    h ^= key_bytes.len() as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Hash key bytes using the specified strategy.
#[inline]
pub(crate) fn hash_key(key_bytes: &[u8], strategy: HashStrategy) -> u64 {
    match strategy {
        HashStrategy::Identity => identity_hash(key_bytes),
        HashStrategy::Murmur3 => murmur3_hash(key_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_hash_reads_first_8_bytes() {
        let key = [1u8, 2, 3, 4, 5, 6, 7, 8, 99, 99];
        let h = identity_hash(&key);
        assert_eq!(h, u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn identity_hash_short_key() {
        let key = [0xAB, 0xCD];
        let h = identity_hash(&key);
        assert_eq!(h, 0xCDAB);
    }

    #[test]
    fn murmur3_different_keys_differ() {
        let a = murmur3_hash(&[1, 2, 3, 4]);
        let b = murmur3_hash(&[5, 6, 7, 8]);
        assert_ne!(a, b);
    }

    #[test]
    fn murmur3_deterministic() {
        let key = [10u8; 32];
        assert_eq!(murmur3_hash(&key), murmur3_hash(&key));
    }
}
