//! Offline unit tests for the on-disk light-cache + DAG cache (port of the Python
//! `test_dagcache.py`). No GPU.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU32, Ordering};

use rdna3_kawpow::dagcache::{Cache, KEEP_DAG};
use rdna3_kawpow::ethash;

const VARIANT: &str = "kawpow";
const EPOCH: u32 = 123;
const PARENTS: u32 = 512;

fn seed() -> [u8; 32] {
    std::array::from_fn(|i| i as u8)
}

static N: AtomicU32 = AtomicU32::new(0);

fn tmp_cache() -> Cache {
    let n = N.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("dagcache_test_{}_{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    Cache::new(Some(dir)).unwrap()
}

fn rand_bytes(len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    getrandom::getrandom(&mut v).unwrap();
    v
}

#[test]
fn light_round_trip() {
    let c = tmp_cache();
    let light = rand_bytes(64 * 1000);
    assert!(c.load_light(VARIANT, EPOCH, PARENTS, &seed()).is_none()); // miss before write
    c.save_light(VARIANT, EPOCH, PARENTS, &seed(), &light).unwrap();
    assert_eq!(c.load_light(VARIANT, EPOCH, PARENTS, &seed()).as_deref(), Some(&light[..]));
}

#[test]
fn light_key_mismatches_miss() {
    let c = tmp_cache();
    let light = rand_bytes(64 * 16);
    c.save_light(VARIANT, EPOCH, PARENTS, &seed(), &light).unwrap();
    assert!(c.load_light(VARIANT, EPOCH, PARENTS, &rand_bytes(32)).is_none()); // seed
    assert!(c.load_light(VARIANT, EPOCH + 1, PARENTS, &seed()).is_none()); // epoch
    assert!(c.load_light(VARIANT, EPOCH, 256, &seed()).is_none()); // parents
    assert!(c.load_light("vanilla", EPOCH, PARENTS, &seed()).is_none()); // variant
}

#[test]
fn dag_round_trip_chunked() {
    let c = tmp_cache();
    let full = 64 * 4096u64;
    let payload = rand_bytes(full as usize);
    assert!(!c.has_dag(VARIANT, EPOCH, PARENTS, &seed(), full));

    let mut w = c.open_dag_write(VARIANT, EPOCH, PARENTS, &seed(), full).unwrap();
    let step = 7777; // deliberately not a divisor of `full`
    let mut i = 0;
    while i < payload.len() {
        let end = (i + step).min(payload.len());
        w.write_all(&payload[i..end]).unwrap();
        i = end;
    }
    w.commit().unwrap();

    assert!(c.has_dag(VARIANT, EPOCH, PARENTS, &seed(), full));
    let mut r = c.open_dag_read(VARIANT, EPOCH, PARENTS).unwrap();
    let mut out = Vec::new();
    r.read_to_end(&mut out).unwrap();
    assert_eq!(out, payload);
}

#[test]
fn dag_size_and_seed_mismatch_miss() {
    let c = tmp_cache();
    let full = 64 * 256u64;
    let mut w = c.open_dag_write(VARIANT, EPOCH, PARENTS, &seed(), full).unwrap();
    w.write_all(&rand_bytes(full as usize)).unwrap();
    w.commit().unwrap();
    assert!(!c.has_dag(VARIANT, EPOCH, PARENTS, &seed(), full + 64)); // size
    assert!(!c.has_dag(VARIANT, EPOCH, PARENTS, &rand_bytes(32), full)); // seed
}

#[test]
fn corrupt_header_is_a_miss() {
    let c = tmp_cache();
    let full = 64 * 64u64;
    let mut w = c.open_dag_write(VARIANT, EPOCH, PARENTS, &seed(), full).unwrap();
    w.write_all(&rand_bytes(full as usize)).unwrap();
    w.commit().unwrap();

    let path = fs::read_dir(&c.dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("dag-"))
        .unwrap();
    let mut f = fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    f.write_all(b"XXXX").unwrap(); // clobber the magic
    drop(f);
    assert!(!c.has_dag(VARIANT, EPOCH, PARENTS, &seed(), full));
}

#[test]
fn prune_keeps_newest() {
    let c = tmp_cache();
    let full = 64 * 16u64;
    for e in EPOCH..EPOCH + KEEP_DAG as u32 + 3 {
        let mut w = c.open_dag_write(VARIANT, e, PARENTS, &seed(), full).unwrap();
        w.write_all(&rand_bytes(full as usize)).unwrap();
        w.commit().unwrap();
        c.prune_dag(VARIANT);
    }
    let remaining = fs::read_dir(&c.dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("dag-{VARIANT}-e"))
        })
        .count();
    assert_eq!(remaining, KEEP_DAG);
}

#[test]
fn precomputed_lightcache_matches_built() {
    let seed = [0x11u8; 32];
    let cache = ethash::mkcache(64 * 211, &seed); // small reference cache
    let flat: Vec<u8> = cache.iter().flatten().copied().collect();
    let split: Vec<[u8; 64]> = flat.chunks_exact(64).map(|c| c.try_into().unwrap()).collect();
    assert_eq!(split, cache);
    for i in [0u32, 1, 7, 100, 210] {
        assert_eq!(
            ethash::calc_dataset_item(&cache, i, 512),
            ethash::calc_dataset_item(&split, i, 512)
        );
    }
}

#[test]
fn precomputed_wrong_size_rejected() {
    let size = ethash::get_cache_size(0);
    let bad = vec![0u8; (size - 64) as usize];
    let r = ethash::LightCache::from_precomputed(0, ethash::KAWPOW_EPOCH_LENGTH, [0; 32], 512, &bad);
    assert!(r.is_err(), "wrong-size precomputed cache was accepted");
}
