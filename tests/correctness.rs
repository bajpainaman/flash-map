use flash_map::{FlashMap, FlashMapError, HashStrategy};

#[test]
fn insert_and_get() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(1024).unwrap();

    let pairs: Vec<([u8; 8], [u8; 8])> = (0u64..100)
        .map(|i| (i.to_le_bytes(), (i * 100).to_le_bytes()))
        .collect();

    let inserted = map.bulk_insert(&pairs).unwrap();
    assert_eq!(inserted, 100);
    assert_eq!(map.len(), 100);

    let keys: Vec<[u8; 8]> = (0u64..100).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&keys).unwrap();

    for (i, result) in results.iter().enumerate() {
        let expected = (i as u64 * 100).to_le_bytes();
        assert_eq!(result, &Some(expected), "mismatch at index {i}");
    }
}

#[test]
fn get_missing_keys() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(1024).unwrap();

    let pairs: Vec<_> = (0u64..10)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    let missing: Vec<[u8; 8]> = (100u64..110).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&missing).unwrap();

    for r in &results {
        assert!(r.is_none());
    }
}

#[test]
fn update_existing_keys() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(1024).unwrap();

    let pairs: Vec<_> = (0u64..10)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    let inserted = map.bulk_insert(&pairs).unwrap();
    assert_eq!(inserted, 10);

    // Update with new values
    let updates: Vec<_> = (0u64..10)
        .map(|i| (i.to_le_bytes(), (i + 1000).to_le_bytes()))
        .collect();
    let updated = map.bulk_insert(&updates).unwrap();
    assert_eq!(updated, 0); // No new keys, only updates
    assert_eq!(map.len(), 10);

    let keys: Vec<_> = (0u64..10).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        let expected = (i as u64 + 1000).to_le_bytes();
        assert_eq!(r, &Some(expected));
    }
}

#[test]
fn remove_keys() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(1024).unwrap();

    let pairs: Vec<_> = (0u64..100)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();
    assert_eq!(map.len(), 100);

    // Remove first 50
    let remove_keys: Vec<_> = (0u64..50).map(|i| i.to_le_bytes()).collect();
    let removed = map.bulk_remove(&remove_keys).unwrap();
    assert_eq!(removed, 50);
    assert_eq!(map.len(), 50);

    // Removed keys return None
    let results = map.bulk_get(&remove_keys).unwrap();
    for r in &results {
        assert!(r.is_none());
    }

    // Remaining keys still exist
    let remaining: Vec<_> = (50u64..100).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&remaining).unwrap();
    for r in &results {
        assert!(r.is_some());
    }
}

#[test]
fn insert_after_remove_reuses_tombstones() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(64).unwrap();

    let pairs: Vec<_> = (0u64..10)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    // Remove all
    let keys: Vec<_> = (0u64..10).map(|i| i.to_le_bytes()).collect();
    map.bulk_remove(&keys).unwrap();
    assert_eq!(map.len(), 0);

    // Re-insert different keys
    let new_pairs: Vec<_> = (100u64..110)
        .map(|i| (i.to_le_bytes(), (i * 2).to_le_bytes()))
        .collect();
    let inserted = map.bulk_insert(&new_pairs).unwrap();
    assert_eq!(inserted, 10);

    let new_keys: Vec<_> = (100u64..110).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&new_keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        let expected = ((100 + i as u64) * 2).to_le_bytes();
        assert_eq!(r, &Some(expected));
    }
}

#[test]
fn clear_resets_map() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(1024).unwrap();

    let pairs: Vec<_> = (0u64..100)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();
    assert_eq!(map.len(), 100);

    map.clear().unwrap();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());

    let keys: Vec<_> = (0u64..100).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&keys).unwrap();
    for r in &results {
        assert!(r.is_none());
    }
}

#[test]
fn murmur3_hash_strategy() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::<[u8; 8], [u8; 8]>::builder(1024)
        .hash_strategy(HashStrategy::Murmur3)
        .build()
        .unwrap();

    let pairs: Vec<_> = (0u64..100)
        .map(|i| (i.to_le_bytes(), (i * 7).to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    let keys: Vec<_> = (0u64..100).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r, &Some((i as u64 * 7).to_le_bytes()));
    }
}

#[test]
fn large_keys_and_values() {
    let mut map: FlashMap<[u8; 32], [u8; 64]> = FlashMap::with_capacity(4096).unwrap();

    let pairs: Vec<_> = (0u64..500)
        .map(|i| {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&i.to_le_bytes());
            let mut val = [0u8; 64];
            val[..8].copy_from_slice(&(i * 42).to_le_bytes());
            (key, val)
        })
        .collect();

    map.bulk_insert(&pairs).unwrap();
    assert_eq!(map.len(), 500);

    let keys: Vec<_> = pairs.iter().map(|(k, _)| *k).collect();
    let results = map.bulk_get(&keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_some(), "missing key at index {i}");
        let mut expected = [0u8; 64];
        expected[..8].copy_from_slice(&(i as u64 * 42).to_le_bytes());
        assert_eq!(r.unwrap(), expected);
    }
}

#[test]
fn load_factor_and_capacity() {
    let map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(1024).unwrap();
    assert_eq!(map.load_factor(), 0.0);
    assert!(map.capacity() >= 1024);
    assert!(map.capacity().is_power_of_two());
}

#[test]
fn empty_operations_are_no_ops() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(64).unwrap();

    let empty_pairs: Vec<([u8; 8], [u8; 8])> = vec![];
    assert_eq!(map.bulk_insert(&empty_pairs).unwrap(), 0);

    let empty_keys: Vec<[u8; 8]> = vec![];
    assert!(map.bulk_get(&empty_keys).unwrap().is_empty());
    assert_eq!(map.bulk_remove(&empty_keys).unwrap(), 0);
}

#[test]
fn table_full_error() {
    let mut map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(16).unwrap();

    // Capacity rounds to 16, max load 100% = 16 entries
    let pairs: Vec<_> = (0u64..16)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    // Inserting one more should fail
    let overflow: Vec<_> = vec![(99u64.to_le_bytes(), 99u64.to_le_bytes())];
    let result = map.bulk_insert(&overflow);
    assert!(matches!(result, Err(FlashMapError::TableFull { .. })));
}

#[test]
fn debug_formatting() {
    let map: FlashMap<[u8; 8], [u8; 8]> = FlashMap::with_capacity(256).unwrap();
    let debug_str = format!("{:?}", map);
    assert!(debug_str.contains("FlashMap"));
    assert!(debug_str.contains("len"));
    assert!(debug_str.contains("capacity"));
}

#[test]
fn u64_key_value_types() {
    let mut map: FlashMap<u64, u64> = FlashMap::with_capacity(1024).unwrap();

    let pairs: Vec<_> = (0u64..50).map(|i| (i, i * i)).collect();
    map.bulk_insert(&pairs).unwrap();

    let keys: Vec<_> = (0u64..50).collect();
    let results = map.bulk_get(&keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r, &Some((i as u64) * (i as u64)));
    }
}

// =========================================================================
// Robin Hood hashing tests
// =========================================================================

#[test]
fn robin_hood_high_load_correctness() {
    // Test at 80%, 90%, 95% load — Robin Hood must not lose data
    for load_pct in [80, 90, 95] {
        let capacity = 1024_usize;
        let n = capacity * load_pct / 100;

        let mut map: FlashMap<[u8; 8], [u8; 8]> =
            FlashMap::with_capacity(capacity).unwrap();

        let pairs: Vec<_> = (0..n as u64)
            .map(|i| (i.to_le_bytes(), (i * 7).to_le_bytes()))
            .collect();
        let inserted = map.bulk_insert(&pairs).unwrap();
        assert_eq!(inserted, n, "wrong insert count at {load_pct}% load");
        assert_eq!(map.len(), n);

        // Every inserted key must be retrievable
        let keys: Vec<_> = (0..n as u64).map(|i| i.to_le_bytes()).collect();
        let results = map.bulk_get(&keys).unwrap();
        for (i, r) in results.iter().enumerate() {
            let expected = (i as u64 * 7).to_le_bytes();
            assert_eq!(
                r,
                &Some(expected),
                "missing key {i} at {load_pct}% load"
            );
        }
    }
}

#[test]
fn robin_hood_miss_exits_early() {
    // At high load, missing keys must still return None (not hang or panic)
    let capacity = 1024_usize;
    let n = capacity * 90 / 100; // 90% load

    let mut map: FlashMap<[u8; 8], [u8; 8]> =
        FlashMap::with_capacity(capacity).unwrap();

    let pairs: Vec<_> = (0..n as u64)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    // Keys that were never inserted — should all be None
    let missing: Vec<_> = (10_000u64..10_500)
        .map(|i| i.to_le_bytes())
        .collect();
    let results = map.bulk_get(&missing).unwrap();
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_none(), "false positive at index {i}");
    }
}

#[test]
fn robin_hood_insert_remove_reinsert() {
    let capacity = 512_usize;
    let n = capacity * 80 / 100;

    let mut map: FlashMap<[u8; 8], [u8; 8]> =
        FlashMap::with_capacity(capacity).unwrap();

    // Fill to 80%
    let pairs: Vec<_> = (0..n as u64)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    // Remove half
    let remove_keys: Vec<_> = (0..n as u64 / 2)
        .map(|i| i.to_le_bytes())
        .collect();
    let removed = map.bulk_remove(&remove_keys).unwrap();
    assert_eq!(removed, n / 2);

    // Reinsert different keys into tombstone slots
    let new_pairs: Vec<_> = (10_000u64..10_000 + n as u64 / 2)
        .map(|i| (i.to_le_bytes(), (i * 3).to_le_bytes()))
        .collect();
    let reinserted = map.bulk_insert(&new_pairs).unwrap();
    assert_eq!(reinserted, n / 2);

    // Verify surviving original keys
    let surviving: Vec<_> = (n as u64 / 2..n as u64)
        .map(|i| i.to_le_bytes())
        .collect();
    let results = map.bulk_get(&surviving).unwrap();
    for (i, r) in results.iter().enumerate() {
        let key_val = i as u64 + n as u64 / 2;
        assert_eq!(
            r,
            &Some(key_val.to_le_bytes()),
            "lost surviving key {key_val}"
        );
    }

    // Verify new keys
    let new_keys: Vec<_> = (10_000u64..10_000 + n as u64 / 2)
        .map(|i| i.to_le_bytes())
        .collect();
    let results = map.bulk_get(&new_keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        let expected = ((10_000 + i as u64) * 3).to_le_bytes();
        assert_eq!(r, &Some(expected), "missing new key at index {i}");
    }
}

#[test]
fn robin_hood_update_at_high_load() {
    let capacity = 1024_usize;
    let n = capacity * 45 / 100;

    let mut map: FlashMap<[u8; 8], [u8; 8]> =
        FlashMap::with_capacity(capacity).unwrap();

    let pairs: Vec<_> = (0..n as u64)
        .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
        .collect();
    map.bulk_insert(&pairs).unwrap();

    // Update all values
    let updates: Vec<_> = (0..n as u64)
        .map(|i| (i.to_le_bytes(), (i + 999).to_le_bytes()))
        .collect();
    let updated = map.bulk_insert(&updates).unwrap();
    assert_eq!(updated, 0, "updates should not count as new inserts");
    assert_eq!(map.len(), n);

    // Verify updated values
    let keys: Vec<_> = (0..n as u64).map(|i| i.to_le_bytes()).collect();
    let results = map.bulk_get(&keys).unwrap();
    for (i, r) in results.iter().enumerate() {
        let expected = (i as u64 + 999).to_le_bytes();
        assert_eq!(r, &Some(expected), "wrong value at index {i}");
    }
}
