//! GPU-vs-reference test: the search shader's output equals the CPU oracle on
//! real hardware. Combined with `vectors.rs` (reference == canonical vectors),
//! this establishes GPU correctness transitively.
//!
//! Requires an RDNA3 GPU + glslc, and builds a real (epoch-0) DAG, so it is
//! `#[ignore]`d by default. Run it explicitly:
//!
//!   cargo test --release --test gpu_search -- --ignored --nocapture

use rdna3_kawpow::keccak::Variant;
use rdna3_kawpow::miner::{VulkanMiner, SEARCH_LOCAL_SIZE};
use rdna3_kawpow::reference;

#[test]
#[ignore = "needs an RDNA3 GPU; run with --ignored"]
fn gpu_search_matches_reference() {
    // Epoch 0, KawPow (512 parents). Builds the light cache + generates the full
    // DAG on the GPU.
    let mut m = VulkanMiner::new(None, Variant::Kawpow, 7500, SEARCH_LOCAL_SIZE, false, None)
        .expect("create miner");
    m.ensure_epoch(0, None).expect("ensure_epoch");
    m.ensure_period(0).expect("ensure_period");

    let header: [u8; 32] = std::array::from_fn(|i| i as u8);
    let start_nonce = 0x1234_5678_0000_0000u64;

    // target = MAX -> every hashed nonce "passes", so the kernel fills its output
    // slots; each slot records its own gid, so we compare per returned gid.
    let (sols, hashes) = m
        .search(&header, u64::MAX, start_nonce, m.local_size as u64)
        .expect("search");
    assert!(hashes > 0, "no hashes reported");
    assert!(!sols.is_empty(), "no solutions returned with MAX target");

    let light = m.light().expect("light cache present");
    for s in &sols {
        let (digest, _result) =
            reference::hash_one(&header, s.nonce, light, 0, Variant::Kawpow, 11, 18, None);
        assert_eq!(
            s.mix_words, digest,
            "nonce {:#018x}: GPU mix != CPU reference",
            s.nonce
        );
    }
    eprintln!(
        "GPU == reference for {} nonces (of {} hashed this batch)",
        sols.len(),
        hashes
    );
}
