//! Packet deduplication — batch-check hashes against a seen set.
//!
//! Run: `cargo run --example dedup`

use flash_map::FlashMap;

type PacketHash = [u8; 32];
type SeqNum = u64;

fn main() {
    let capacity: usize = 1_000_000;
    let mut seen: FlashMap<PacketHash, SeqNum> =
        FlashMap::with_capacity(capacity).unwrap();

    // Simulate incoming packet batch (50K packets)
    let batch_size: usize = 50_000;
    let packets: Vec<(PacketHash, SeqNum)> = (0..batch_size)
        .map(|i| {
            let mut hash = [0u8; 32];
            hash[..8].copy_from_slice(&(i as u64).to_le_bytes());
            (hash, i as u64)
        })
        .collect();

    // Insert first batch
    seen.bulk_insert(&packets).unwrap();
    println!("Inserted {} packet hashes", seen.len());

    // Second batch — half duplicates, half new
    let mixed_batch: Vec<PacketHash> = (25_000u64..75_000)
        .map(|i| {
            let mut hash = [0u8; 32];
            hash[..8].copy_from_slice(&i.to_le_bytes());
            hash
        })
        .collect();

    let start = std::time::Instant::now();
    let results = seen.bulk_get(&mixed_batch).unwrap();
    let elapsed = start.elapsed();

    let duplicates = results.iter().filter(|r| r.is_some()).count();
    let unique = results.iter().filter(|r| r.is_none()).count();

    println!(
        "Checked {} hashes in {:.2?}: {} duplicates, {} new",
        mixed_batch.len(),
        elapsed,
        duplicates,
        unique
    );

    // Insert only the new ones
    let new_entries: Vec<(PacketHash, SeqNum)> = mixed_batch
        .iter()
        .zip(results.iter())
        .filter(|(_, r)| r.is_none())
        .enumerate()
        .map(|(seq, (hash, _))| (*hash, (batch_size + seq) as u64))
        .collect();

    let added = seen.bulk_insert(&new_entries).unwrap();
    println!("Added {} new hashes, total seen: {}", added, seen.len());
}
