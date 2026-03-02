//! Blockchain state storage — bulk commit account changes after block execution.
//!
//! Run: `cargo run --example blockchain_state`

use flash_map::FlashMap;

type Pubkey = [u8; 32];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Account {
    lamports: u64,
    data: [u8; 128],
}

fn main() {
    let num_accounts: usize = 100_000;

    let mut state: FlashMap<Pubkey, Account> =
        FlashMap::with_capacity(num_accounts * 4).unwrap();

    println!("FlashMap created: {:?}", state);

    // Simulate block execution — generate account changes
    let changes: Vec<(Pubkey, Account)> = (0..num_accounts)
        .map(|i| {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let account = Account {
                lamports: (i as u64 + 1) * 1_000_000,
                data: [0u8; 128],
            };
            (key, account)
        })
        .collect();

    // Bulk commit all state changes in one call
    let start = std::time::Instant::now();
    let inserted = state.bulk_insert(&changes).unwrap();
    let elapsed = start.elapsed();

    println!(
        "Committed {} accounts in {:.2?} ({:.0} accounts/sec)",
        inserted,
        elapsed,
        inserted as f64 / elapsed.as_secs_f64()
    );
    println!("Load factor: {:.1}%", state.load_factor() * 100.0);

    // Bulk lookup — verify all accounts exist
    let keys: Vec<Pubkey> = changes.iter().map(|(k, _)| *k).collect();
    let start = std::time::Instant::now();
    let results = state.bulk_get(&keys).unwrap();
    let elapsed = start.elapsed();

    let found = results.iter().filter(|r| r.is_some()).count();
    println!(
        "Looked up {} keys in {:.2?} — {}/{} found",
        keys.len(),
        elapsed,
        found,
        keys.len()
    );

    // Verify balances
    for (i, result) in results.iter().enumerate() {
        let account = result.expect("account missing");
        assert_eq!(
            account.lamports,
            (i as u64 + 1) * 1_000_000,
            "balance mismatch at index {i}"
        );
    }

    println!("All balances verified.");
}
