"""rdna3-kawpow CLI / main mining loop.

  python -m rdna3_kawpow stratum+tcp://WALLET.WORKER:x@pool:port
  python -m rdna3_kawpow --benchmark
  python -m rdna3_kawpow --list-devices

The mining loop is watchdog-safe: it searches in auto-calibrated, time-bounded
batches and checks for a new job between batches, so no GPU dispatch runs long
enough to trip the OS watchdog.
"""

import argparse
import os
import sys
import threading
import time

from . import keccak, reference
from .constants import PROGPOW_PERIOD, PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH
from .ethash import KAWPOW_EPOCH_LENGTH
from .miner import VulkanMiner
from .stratum import StratumClient


def parse_pool_url(url):
    """scheme://wallet.worker:password@host:port -> dict."""
    if "://" in url:
        url = url.split("://", 1)[1]
    userinfo, _, hostport = url.rpartition("@")
    if not userinfo:
        raise ValueError("pool URL needs WALLET[.WORKER][:PASSWORD]@HOST:PORT")
    user, _, password = userinfo.partition(":")
    wallet, _, worker = user.partition(".")
    host, _, port = hostport.partition(":")
    return dict(host=host, port=int(port or 0), wallet=wallet,
                worker=worker or "rdna3", password=password or "x")


def parse_host_port(url):
    """stratum+tcp://host:port (no userinfo) -> (host, port)."""
    if "://" in url:
        url = url.split("://", 1)[1]
    url = url.rstrip("/")
    host, _, port = url.rpartition(":")
    return host, int(port)


def resolve_pool(args):
    """Accept either a full positional URL or -o/-u/-p style arguments."""
    if args.url:
        host, port = parse_host_port(args.url)
        user = args.user or ""
        wallet, _, worker = user.partition(".")
        return dict(host=host, port=port, wallet=wallet,
                    worker=worker or "rdna3", password=args.password or "x")
    return parse_pool_url(args.pool)


def mine(args, variant):
    miner = VulkanMiner(device_index=args.device, variant=variant,
                        epoch_length=args.epoch_length,
                        dag_cache=not args.no_dag_cache, cache_dir=args.cache_dir)
    print(miner.dev.summary())

    pool = resolve_pool(args)
    client = StratumClient(pool["host"], pool["port"], pool["wallet"],
                           pool["worker"], pool["password"])
    job_changed = threading.Event()
    client.on_new_job = lambda job: (
        print(f"new job {job.job_id} height={job.height} "
              f"target=0x{job.boundary64:016x}"), job_changed.set())
    def connect_with_retry():
        while True:
            try:
                print(f"Connecting to {pool['host']}:{pool['port']} as "
                      f"{pool['wallet']}.{pool['worker']}")
                client.connect()
                for _ in range(300):
                    if client.current_job is not None:
                        return
                    time.sleep(0.1)
                print("no job received; retrying")
            except OSError as e:
                print(f"connect failed: {e}; retrying in 5s")
            time.sleep(5)

    connect_with_retry()

    calibrated = False
    while True:
        if not client.alive:
            print("disconnected; reconnecting...")
            connect_with_retry()
        job = client.current_job
        job_changed.clear()
        try:
            miner.ensure_epoch(job.height, seed=job.seed)
            miner.ensure_period(job.height)
        except Exception as e:
            print("epoch/period setup failed:", e)
            time.sleep(2)
            continue
        if not calibrated:
            print("Calibrating safe dispatch size...")
            miner.calibrate(job.header)
            calibrated = True

        # Keep the search strictly inside the miner's nonce region so every nonce
        # still starts with the extranonce (overflowing into the extranonce bits
        # yields "invalid nonce prefix" rejects).
        batch = miner._safe_batch
        miner_bits = 64 - (job.extranonce_bits or 0)
        span = 1 << miner_bits
        def fresh_salt():
            return 0 if span <= batch else int.from_bytes(os.urandom(6), "little") % (span - batch)
        salt = fresh_salt()
        acc_hashes, t_acc = 0, time.time()
        while client.current_job is job and not job_changed.is_set() and client.alive:
            start_nonce = job.start_nonce(salt)
            sols, hashes = miner.search(job.header, job.boundary64, start_nonce, batch)
            salt += batch
            if salt + batch > span:
                salt = fresh_salt()
            acc_hashes += hashes
            for s in sols:
                if args.no_recheck or self_recheck(miner, job, s, variant):
                    client.submit(job.job_id, s.nonce, s.mix_hash())
                    print(f"  submit nonce=0x{s.nonce:016x}")
                else:
                    print(f"  dropped false-positive nonce=0x{s.nonce:016x}")
            dt = time.time() - t_acc
            if dt >= 5.0:
                hr = acc_hashes / dt
                print(f"  {hr/1e6:.2f} MH/s")
                client.hashrate = hr
                client.report_hashrate(hr)
                acc_hashes, t_acc = 0, time.time()


def self_recheck(miner, job, sol, variant):
    """Recompute the hash on the host and verify it really meets the target."""
    try:
        period = job.height // PROGPOW_PERIOD
        digest, result = reference.hash_one(
            job.header, sol.nonce, miner._light, period, variant,
            cnt_cache=PROGPOW_CNT_CACHE, cnt_math=PROGPOW_CNT_MATH)
        return tuple(digest) == tuple(sol.mix_words) and result <= job.boundary64
    except Exception:
        return True  # if recheck can't run, trust the GPU


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rdna3-kawpow",
                                 description="RDNA3-optimized KawPow Vulkan miner")
    ap.add_argument("pool", nargs="?",
                    help="stratum+tcp://WALLET.WORKER:x@host:port")
    # T-Rex / kawpowminer-style alternatives:
    ap.add_argument("-a", "--algo", choices=["kawpow", "vanilla"], default=None)
    ap.add_argument("-o", "--url", help="pool url without userinfo (stratum+tcp://host:port)")
    ap.add_argument("-u", "--user", help="WALLET.WORKER")
    ap.add_argument("-p", "--pass", dest="password", help="pool password")
    ap.add_argument("--device", type=int, default=None, help="GPU index")
    ap.add_argument("--variant", choices=["kawpow", "vanilla"], default="kawpow")
    ap.add_argument("--epoch-length", type=int, default=KAWPOW_EPOCH_LENGTH)
    ap.add_argument("--no-recheck", action="store_true",
                    help="submit GPU solutions without host re-verification")
    ap.add_argument("--no-dag-cache", action="store_true",
                    help="disable on-disk caching of the light cache + DAG")
    ap.add_argument("--cache-dir", default=None,
                    help="directory for the light cache + DAG disk cache "
                         "(default: %%LOCALAPPDATA%%/rdna3_kawpow/dagcache or "
                         "~/.cache/rdna3_kawpow/dagcache; "
                         "env RDNA3_KAWPOW_CACHE_DIR also overrides)")
    ap.add_argument("--benchmark", action="store_true",
                    help="measure hashrate and exit (no pool, uninitialized DAG)")
    ap.add_argument("--bench-block", type=int, default=30000)
    ap.add_argument("--bench-seconds", type=float, default=8.0)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args(argv)

    if args.list_devices:
        from .vkhost import VulkanDevice
        print(VulkanDevice(args.device).summary())
        return 0

    if args.benchmark:
        variant = keccak.KAWPOW if args.variant == "kawpow" else keccak.VANILLA
        m = VulkanMiner(device_index=args.device, variant=variant,
                        epoch_length=args.epoch_length, dag_cache=False)
        print(m.dev.summary())
        m.benchmark(args.bench_block, args.bench_seconds, use_real_dag=False)
        return 0

    if not args.pool and not args.url:
        ap.error("a pool is required: positional URL, or -o/-u/-p (or --benchmark)")
    variant = keccak.KAWPOW if (args.algo or args.variant) == "kawpow" else keccak.VANILLA
    try:
        mine(args, variant)
    except KeyboardInterrupt:
        print("\nstopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
