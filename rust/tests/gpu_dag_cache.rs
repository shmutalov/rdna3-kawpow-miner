//! GPU test for the on-disk DAG cache staging path: generate+save a DAG, reload it
//! into a fresh miner via the staging window, and verify the disk-loaded DAG still
//! produces hashes that match the CPU reference. Exercises `save_dag_to_disk` /
//! `load_dag_from_disk` (the device<->host staging copies), which the plain
//! gpu_search test does not.
//!
//!   cargo test --release --test gpu_dag_cache -- --ignored --nocapture

use rdna3_kawpow::keccak::Variant;
use rdna3_kawpow::miner::{VulkanMiner, SEARCH_LOCAL_SIZE};
use rdna3_kawpow::reference;

#[test]
#[ignore = "needs an RDNA3 GPU; run with --ignored"]
fn gpu_dag_cache_roundtrip_and_correct() {
    let dir = std::env::temp_dir().join(format!("rk_dagcache_smoke_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // 1) Fresh miner generates the DAG on the GPU and streams it to disk.
    {
        let mut m = VulkanMiner::new(None, Variant::Kawpow, 7500, SEARCH_LOCAL_SIZE, true, Some(dir.clone()))
            .expect("miner 1");
        m.ensure_epoch(0, None).expect("generate+save epoch");
    }

    // 2) A second miner must reload the light cache + DAG from disk (no GPU gen).
    let mut m = VulkanMiner::new(None, Variant::Kawpow, 7500, SEARCH_LOCAL_SIZE, true, Some(dir.clone()))
        .expect("miner 2");
    m.ensure_epoch(0, None).expect("load epoch from disk");
    m.ensure_period(0).expect("ensure_period");

    // 3) The disk-loaded DAG must produce hashes identical to the CPU reference.
    let header: [u8; 32] = std::array::from_fn(|i| i as u8);
    let (sols, hashes) = m
        .search(&header, u64::MAX, 0xABCD_0000_0000, m.local_size as u64)
        .expect("search");
    assert!(hashes > 0 && !sols.is_empty());
    let light = m.light().expect("light");
    for s in &sols {
        let (digest, _) =
            reference::hash_one(&header, s.nonce, light, 0, Variant::Kawpow, 11, 18, None);
        assert_eq!(
            s.mix_words, digest,
            "disk-loaded DAG produced wrong hash for nonce {:#018x}",
            s.nonce
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "DAG disk-cache round-trip OK; reloaded DAG matches reference for {} nonces",
        sols.len()
    );
}
