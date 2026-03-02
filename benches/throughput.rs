use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flash_map::FlashMap;

fn bench_bulk_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_insert");
    group.sample_size(10);

    for size in [1_000, 10_000, 100_000] {
        let pairs: Vec<([u8; 32], [u8; 128])> = (0..size)
            .map(|i| {
                let mut key = [0u8; 32];
                key[..8].copy_from_slice(&(i as u64).to_le_bytes());
                let mut val = [0u8; 128];
                val[..8].copy_from_slice(&(i as u64).to_le_bytes());
                (key, val)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &pairs,
            |b, pairs| {
                b.iter(|| {
                    let mut map: FlashMap<[u8; 32], [u8; 128]> =
                        FlashMap::with_capacity(size * 4).unwrap();
                    map.bulk_insert(pairs).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_bulk_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_get");
    group.sample_size(10);

    for size in [1_000, 10_000, 100_000] {
        let pairs: Vec<([u8; 32], [u8; 128])> = (0..size)
            .map(|i| {
                let mut key = [0u8; 32];
                key[..8].copy_from_slice(&(i as u64).to_le_bytes());
                let val = [0u8; 128];
                (key, val)
            })
            .collect();

        let mut map: FlashMap<[u8; 32], [u8; 128]> =
            FlashMap::with_capacity(size * 4).unwrap();
        map.bulk_insert(&pairs).unwrap();

        let keys: Vec<[u8; 32]> = pairs.iter().map(|(k, _)| *k).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    map.bulk_get(keys).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed");
    group.sample_size(10);

    let size = 10_000usize;
    let pairs: Vec<([u8; 32], [u8; 128])> = (0..size)
        .map(|i| {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let mut val = [0u8; 128];
            val[..8].copy_from_slice(&(i as u64).to_le_bytes());
            (key, val)
        })
        .collect();

    group.bench_function("insert_get_remove_10k", |b| {
        b.iter(|| {
            let mut map: FlashMap<[u8; 32], [u8; 128]> =
                FlashMap::with_capacity(size * 4).unwrap();

            // Insert
            map.bulk_insert(&pairs).unwrap();

            // Get
            let keys: Vec<[u8; 32]> = pairs.iter().map(|(k, _)| *k).collect();
            let results = map.bulk_get(&keys).unwrap();
            assert!(results.iter().all(|r| r.is_some()));

            // Remove half
            let remove_keys: Vec<[u8; 32]> = keys[..size / 2].to_vec();
            map.bulk_remove(&remove_keys).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_bulk_insert, bench_bulk_get, bench_mixed_workload);
criterion_main!(benches);
