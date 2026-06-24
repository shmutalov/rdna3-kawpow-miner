"""rdna3-kawpow: an RDNA3-optimized KawPow (ProgPoW 0.9.3) GPU miner.

Host orchestration in Python; all hashing runs on the GPU as Vulkan compute
(SPIR-V) shaders. Targeted and tuned for AMD RDNA3 (gfx11, e.g. RX 7900 XT):
native wave32 forced via VK_EXT_subgroup_size_control, cross-lane data exchange
via subgroupShuffle (replacing the GCN-era LDS+barrier pattern), and compute-unit
sizing from VK_AMD_shader_core_properties2.

The host language has no effect on hashrate: the GPU executes identical SPIR-V
regardless of whether the host is C++, Rust, or Python.
"""

__version__ = "0.1.0"
