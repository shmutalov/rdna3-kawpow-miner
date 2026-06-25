//! rdna3-kawpow CLI / main mining loop (port of `rdna3_kawpow/__main__.py`).
//!
//!   rdna3-kawpow stratum+tcp://WALLET.WORKER:x@pool:port
//!   rdna3-kawpow --benchmark
//!   rdna3-kawpow --list-devices
//!
//! The mining loop is watchdog-safe: it searches in auto-calibrated, time-bounded
//! batches and checks for a new job between batches, so no GPU dispatch runs long
//! enough to trip the OS watchdog.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use clap::Parser;

use rdna3_kawpow::constants::{PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH, PROGPOW_PERIOD};
use rdna3_kawpow::keccak::Variant;
use rdna3_kawpow::miner::{Solution, VulkanMiner, SEARCH_LOCAL_SIZE};
use rdna3_kawpow::stats::Stats;
use rdna3_kawpow::stratum::{parse_host_port, parse_pool_url, Job, Pool, StratumClient};
use rdna3_kawpow::{reference, shader_compiler, vkhost::VulkanDevice};

#[derive(Parser, Debug)]
#[command(name = "rdna3-kawpow", version, about = "RDNA3-optimized KawPow Vulkan miner (Rust)")]
struct Args {
    /// Pool URL: stratum+tcp://WALLET.WORKER:x@host:port
    pool: Option<String>,

    /// Algorithm (T-Rex/kawpowminer style; overrides --variant).
    #[arg(short = 'a', long, value_parser = ["kawpow", "vanilla"])]
    algo: Option<String>,
    /// Pool URL without userinfo (stratum+tcp://host:port).
    #[arg(short = 'o', long)]
    url: Option<String>,
    /// WALLET.WORKER.
    #[arg(short = 'u', long)]
    user: Option<String>,
    /// Pool password.
    #[arg(short = 'p', long = "pass")]
    password: Option<String>,

    /// GPU index to use (default: first discrete device).
    #[arg(long)]
    device: Option<usize>,

    #[arg(long)]
    list_devices: bool,
    /// Print the number of discrete GPUs and exit (for per-GPU launchers).
    #[arg(long)]
    device_count: bool,
    #[arg(long)]
    selftest: bool,
    /// Measure hashrate and exit (no pool, uninitialized DAG).
    #[arg(long)]
    benchmark: bool,

    #[arg(long, value_parser = ["kawpow", "vanilla"], default_value = "kawpow")]
    variant: String,
    #[arg(long, default_value_t = 7500)]
    epoch_length: u64,
    /// Search workgroup size (wave32 multiple, e.g. 64/128/256); tunes occupancy.
    #[arg(long)]
    local_size: Option<u32>,
    /// Submit GPU solutions without host re-verification.
    #[arg(long)]
    no_recheck: bool,
    /// Disable on-disk caching of the light cache + DAG.
    #[arg(long)]
    no_dag_cache: bool,
    /// Directory for the light cache + DAG disk cache.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Serve a JSON stats endpoint (for HiveOS), e.g. 127.0.0.1:4068.
    #[arg(long)]
    api_bind: Option<String>,

    #[arg(long, default_value_t = 30000)]
    bench_block: u64,
    #[arg(long, default_value_t = 8.0)]
    bench_seconds: f64,
}

fn variant_of(s: &str) -> Variant {
    if s == "vanilla" {
        Variant::Vanilla
    } else {
        Variant::Kawpow
    }
}

fn resolve_pool(args: &Args) -> Result<Pool> {
    if let Some(url) = &args.url {
        let (host, port) = parse_host_port(url).map_err(|e| anyhow!(e))?;
        let user = args.user.clone().unwrap_or_default();
        let (wallet, worker) = user.split_once('.').unwrap_or((user.as_str(), ""));
        return Ok(Pool {
            host,
            port,
            wallet: wallet.to_string(),
            worker: if worker.is_empty() { "rdna3" } else { worker }.to_string(),
            password: args.password.clone().unwrap_or_else(|| "x".to_string()),
        });
    }
    parse_pool_url(args.pool.as_deref().unwrap_or_default()).map_err(|e| anyhow!(e))
}

fn rand_u64() -> u64 {
    let mut b = [0u8; 8];
    getrandom::getrandom(&mut b).expect("system RNG");
    u64::from_le_bytes(b)
}

/// Recompute the hash on the host and verify it really meets the target.
fn self_recheck(miner: &VulkanMiner, job: &Job, sol: &Solution, variant: Variant) -> bool {
    let Some(light) = miner.light() else {
        return true; // if recheck can't run, trust the GPU
    };
    let period = job.height / PROGPOW_PERIOD;
    let (digest, result) = reference::hash_one(
        &job.header,
        sol.nonce,
        light,
        period,
        variant,
        PROGPOW_CNT_CACHE,
        PROGPOW_CNT_MATH,
        None,
    );
    digest == sol.mix_words && result <= job.boundary64()
}

fn connect_with_retry(client: &StratumClient, pool: &Pool) {
    loop {
        println!(
            "Connecting to {}:{} as {}.{}",
            pool.host, pool.port, pool.wallet, pool.worker
        );
        match client.connect() {
            Ok(()) => {
                for _ in 0..300 {
                    if client.current_job().is_some() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                println!("no job received; retrying");
            }
            Err(e) => println!("connect failed: {e}; retrying in 5s"),
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn mine(args: &Args, variant: Variant) -> Result<()> {
    let local_size = args.local_size.unwrap_or(SEARCH_LOCAL_SIZE);
    let mut miner = VulkanMiner::new(
        args.device,
        variant,
        args.epoch_length,
        local_size,
        !args.no_dag_cache,
        args.cache_dir.clone(),
    )?;
    println!("{}", miner.dev.summary());

    let stats = Stats::new(if variant == Variant::Kawpow { "kawpow" } else { "vanilla" });
    stats.set_device(&miner.dev.name);
    if let Some(addr) = &args.api_bind {
        match stats.serve(addr) {
            Ok(bound) => println!("stats API on http://{bound}"),
            Err(e) => println!("stats API bind failed ({e}); continuing without it"),
        }
    }

    let pool = resolve_pool(args)?;
    let client = StratumClient::new(&pool.host, pool.port, &pool.wallet, &pool.worker, &pool.password);
    connect_with_retry(&client, &pool);

    let mut calibrated = false;
    loop {
        if !client.alive() {
            println!("disconnected; reconnecting...");
            connect_with_retry(&client, &pool);
        }
        let job = match client.current_job() {
            Some(j) => j,
            None => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let job_gen = client.job_gen();

        if let Err(e) = miner
            .ensure_epoch(job.height, Some(job.seed))
            .and_then(|()| miner.ensure_period(job.height))
        {
            println!("epoch/period setup failed: {e}");
            std::thread::sleep(Duration::from_secs(2));
            continue;
        }
        if !calibrated {
            println!("Calibrating safe dispatch size...");
            miner.calibrate(&job.header)?;
            calibrated = true;
        }

        // Keep the search strictly inside the miner's nonce region so every nonce
        // still starts with the extranonce.
        let batch = miner.safe_batch() as u128;
        let miner_bits = 64 - job.extranonce_bits;
        let span: u128 = 1u128 << miner_bits;
        let fresh_salt = || -> u128 {
            if span <= batch {
                0
            } else {
                (rand_u64() as u128) % (span - batch)
            }
        };
        let mut salt = fresh_salt();
        let mut acc_hashes = 0u64;
        let mut t_acc = Instant::now();

        while client.job_gen() == job_gen && client.alive() {
            let start_nonce = job.start_nonce(salt as u64);
            let (sols, hashes) =
                miner.search(&job.header, job.boundary64(), start_nonce, batch as u64)?;
            salt += batch;
            if salt + batch > span {
                salt = fresh_salt();
            }
            acc_hashes += hashes;
            for s in &sols {
                if args.no_recheck || self_recheck(&miner, &job, s, variant) {
                    client.submit(&job.job_id, s.nonce, &s.mix_hash());
                    println!("  submit nonce=0x{:016x}", s.nonce);
                } else {
                    println!("  dropped false-positive nonce=0x{:016x}", s.nonce);
                    stats.add_invalid();
                }
            }
            let dt = t_acc.elapsed().as_secs_f64();
            if dt >= 5.0 {
                let hr = acc_hashes as f64 / dt;
                println!("  {:.2} MH/s", hr / 1e6);
                client.set_hashrate(hr);
                client.report_hashrate(hr);
                stats.set_hashrate(hr);
                stats.set_shares(client.accepted(), client.rejected());
                acc_hashes = 0;
                t_acc = Instant::now();
            }
        }
        println!("new job (gen {} -> {})", job_gen, client.job_gen());
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.device_count {
        let n = rdna3_kawpow::vkhost::enumerate_devices()?
            .iter()
            .filter(|d| d.discrete)
            .count();
        println!("{n}");
        return Ok(());
    }

    if args.list_devices {
        for d in rdna3_kawpow::vkhost::enumerate_devices()? {
            println!(
                "device {}: {}{}",
                d.index,
                d.name,
                if d.discrete { " [discrete]" } else { "" }
            );
        }
        // Detailed feature summary for the selected/first device.
        println!("\n{}", VulkanDevice::new(args.device)?.summary());
        return Ok(());
    }

    if args.selftest {
        match VulkanDevice::new(args.device) {
            Ok(dev) => println!("[selftest] device OK:\n{}", dev.summary()),
            Err(e) => println!("[selftest] device probe failed (no GPU here?): {e:#}"),
        }
        let bytes = shader_compiler::selftest()?;
        println!("[selftest] shader compiler OK: {bytes} bytes of SPIR-V");
        println!("[selftest] PASS");
        return Ok(());
    }

    if args.benchmark {
        let local_size = args.local_size.unwrap_or(SEARCH_LOCAL_SIZE);
        let mut m = VulkanMiner::new(
            args.device,
            variant_of(&args.variant),
            args.epoch_length,
            local_size,
            false,
            None,
        )?;
        println!("{}", m.dev.summary());
        m.benchmark(args.bench_block, args.bench_seconds, false)?;
        return Ok(());
    }

    if args.pool.is_none() && args.url.is_none() {
        eprintln!("a pool is required: positional URL, or -o/-u/-p (or --benchmark/--list-devices)");
        std::process::exit(2);
    }

    let variant = variant_of(args.algo.as_deref().unwrap_or(&args.variant));
    if let Err(e) = mine(&args, variant) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
    Ok(())
}
