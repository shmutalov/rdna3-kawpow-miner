//! Canonical vector regression tests (mirror of the Python test suite).
//!
//! These pin the entire algorithm engine -- keccak_f800, the program generator's
//! draw order, ethash cache+DAG, fill_mix, the 64-loop execution, digest reduction
//! and the final hash -- against the authoritative vectors, with NO dependency on
//! the Python implementation. GPU correctness is then established transitively
//! (Phase 2: GPU == this reference).
//!
//! Cache builds are ~16 MiB of keccak; run with `cargo test --release`.

use rdna3_kawpow::ethash::{LightCache, ETHEREUM_EPOCH_LENGTH};
use rdna3_kawpow::keccak::Variant;
use rdna3_kawpow::reference::{self, Collected};

fn hex32(s: &str) -> [u8; 32] {
    let bytes = hex_decode(s);
    assert_eq!(bytes.len(), 32, "expected 32 bytes from {s}");
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// ProgPoW 0.9.2 canonical vector (block 30000, upstream kernel.cu + result.log).
// Vanilla path: 256 DAG parents, big-endian seed, 12/20 op counts.
// ---------------------------------------------------------------------------
#[test]
fn vanilla_progpow_092_block_30000() {
    const HEADER: &str =
        "ffeeddccbbaa9988776655443322110000112233445566778899aabbccddeeff";
    const NONCE: u64 = 0x123456789ABCDEF0;
    const EXP_DIGEST: &str =
        "11f19805c58ab46610ff9c719dcf0a5f18fa2f1605798eef770c47219274767d";
    const EXP_RESULT: u64 = 0x5B7CCD472DBEFDD9;
    const EXP_DIGEST_LANES: [u32; 16] = [
        0x5883883E, 0x2FB0FD2E, 0xEADB7563, 0x4A171075, 0xAC2758F5, 0xAA5B06CF,
        0x52156E93, 0x4F7A7FFF, 0xFE91E36A, 0x9964C8B6, 0x6A3D93E2, 0x3C6D641F,
        0xE90DA618, 0x80CD8AB9, 0xCE72386F, 0x95517D28,
    ];

    let lc = LightCache::new(30000, ETHEREUM_EPOCH_LENGTH, None, 256);
    assert_eq!(lc.cache_size, 16_907_456);
    assert_eq!(lc.full_size, 1_082_130_304);
    assert_eq!(lc.dag_elements, 4_227_071);

    let mut col = Collected::default();
    let (digest, result) = reference::hash_one(
        &hex32(HEADER),
        NONCE,
        &lc,
        600,
        Variant::Vanilla,
        12,
        20,
        Some(&mut col),
    );

    assert_eq!(col.fill_mix[0][0], 0x10C02F0D);
    assert_eq!(col.fill_mix[15][31], 0x20201012);
    assert_eq!(&col.loop_entry[..4], &[2043727, 1878577, 1972818, 4192557]);
    assert_eq!(col.loop_entry[63], 574671);
    assert_eq!(col.digest_lane, EXP_DIGEST_LANES);
    assert_eq!(hex_encode(&reference::mix_hash_bytes(&digest)), EXP_DIGEST);
    assert_eq!(result, EXP_RESULT);
}

// ---------------------------------------------------------------------------
// Standard Ravencoin KawPow vectors (cpp-kawpow progpow_test_vectors.hpp).
// 512 DAG parents, Ravencoin keccak tail, direct seed, 11/18. All epoch 0.
// ---------------------------------------------------------------------------
#[test]
fn kawpow_cpp_vectors_epoch0() {
    // (block, header_hex, nonce, mix_hex(LE), final_hex(BE))
    let vectors: &[(u64, &str, u64, &str, &str)] = &[
        (
            0,
            "0000000000000000000000000000000000000000000000000000000000000000",
            0x0000000000000000,
            "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
            "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1",
        ),
        (
            49,
            "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b",
            0x0000000007073c07,
            "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
            "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463",
        ),
        (
            50,
            "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922",
            0x00000000076e482e,
            "d6dc634ae837e2785b347648ea515e25e5d8821ae0b95e1c2a9c2d497e0dcfbd",
            "ab0ad7ef8d8ee317dd12d10310aceed7321d34fb263791c2de5776a6658d177e",
        ),
        (
            99,
            "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d",
            0x000000003917afab,
            "fa706860e5e0e830d5d1d7157e5bea7f5f8a350c7c8612ac1d1fcf2974d64244",
            "aa85340690f2e907054324a5021937910e15edfd1ef1577231843e7d32ec3a61",
        ),
    ];

    // epoch 0, 512 parents (default), seed = all zeros.
    let lc = LightCache::new(0, 7500, None, 512);
    for &(block, hdr_hex, nonce, exp_mix, exp_final) in vectors {
        let prog_seed = block / 3; // PROGPOW_PERIOD = 3
        let (digest, result) = reference::hash_one(
            &hex32(hdr_hex),
            nonce,
            &lc,
            prog_seed,
            Variant::Kawpow,
            11,
            18,
            None,
        );
        // mix hex is the LITTLE-endian byte layout of the 8 digest words.
        let mut mix_le = Vec::with_capacity(32);
        for w in digest.iter() {
            mix_le.extend_from_slice(&w.to_le_bytes());
        }
        assert_eq!(hex_encode(&mix_le), exp_mix, "block {block} mix");

        let exp_bytes = hex_decode(exp_final);
        let exp_result64 = u64::from_be_bytes(exp_bytes[..8].try_into().unwrap());
        assert_eq!(result, exp_result64, "block {block} result");
    }
}
