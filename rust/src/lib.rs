//! RDNA3-optimized KawPow (ProgPoW 0.9.3 / Ravencoin) Vulkan miner -- Rust host.
//!
//! A focused port of the Python host in `rdna3_kawpow/`. All hashing runs in the
//! GLSL compute shaders (carried over unchanged); this crate only sets up Vulkan,
//! generates/compiles the per-period shader, manages buffers, and dispatches.
//!
//! Migration status: Phase 0 (scaffold) -- `constants`, device enumeration, and
//! the shader-compile path are in place. Algorithm core (keccak/ethash/progpow IR/
//! reference), the full Vulkan host, stratum, and the HiveOS stats API follow.

pub mod constants;
pub mod dagcache;
pub mod ethash;
pub mod keccak;
pub mod miner;
pub mod progpow;
pub mod reference;
pub mod shader_compiler;
pub mod stats;
pub mod stratum;
pub mod vkhost;
