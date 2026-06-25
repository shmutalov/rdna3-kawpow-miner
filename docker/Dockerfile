# syntax=docker/dockerfile:1
#
# Linux build for the Rust KawPow miner host.
#
# Phase 0 goal: prove the crate compiles for x86_64-unknown-linux-gnu (ash links,
# the glslc-driven shader path builds). Build from the REPO ROOT as context:
#
#   docker build -f docker/Dockerfile --target export -o type=local,dest=dist .
#
# That writes dist/rdna3-kawpow (a Linux binary). To get a shell with the binary:
#
#   docker build -f docker/Dockerfile --target runtime -t rdna3-kawpow:linux .
#
# HiveOS (latest stable) is Ubuntu 22.04 -> glibc 2.35. A binary must require a
# glibc <= the rig's, so we build on Debian bullseye (glibc 2.31): forward-compatible
# with Ubuntu 22.04 and any older rig. (Do NOT build on bookworm/glibc 2.36 -- newer
# than the target, so it would fail on the rig with "GLIBC_2.36 not found".)
# ash's runtime loader needs dlopen, so a fully static musl build is not an option;
# building against an older-or-equal glibc is the path. The build prints the max
# GLIBC_ version the binary actually requires so this stays honest.

ARG BUILDER_IMAGE=rust:1-bullseye

FROM ${BUILDER_IMAGE} AS build
WORKDIR /src
# include_str! reaches from rust/src into the shared shader tree, so both paths
# must be present with their original relative layout.
COPY rust/ rust/
COPY rdna3_kawpow/shaders/ rdna3_kawpow/shaders/
WORKDIR /src/rust
# Only the crate registry is cached. The build target dir is deliberately NOT
# cached: that cache is not keyed by base image, so reusing it across glibc bases
# would serve a wrong-glibc binary. Release correctness > a ~15s recompile.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release \
 && install -D target/release/rdna3-kawpow /out/rdna3-kawpow \
 && /out/rdna3-kawpow --version \
 && echo "max glibc required:" \
 && { objdump -T /out/rdna3-kawpow | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1 || echo none; }

# Minimal artifact stage: `-o type=local` exports just the binary to the host.
FROM scratch AS export
COPY --from=build /out/rdna3-kawpow /rdna3-kawpow

# Runnable image: Vulkan loader + AMD (RADV) ICD. Mirrors the HiveOS rig (Ubuntu
# 22.04 + AMD Vulkan stack); this stage is for local Linux testing only.
FROM ubuntu:22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/rdna3-kawpow /usr/local/bin/rdna3-kawpow
ENTRYPOINT ["rdna3-kawpow"]
