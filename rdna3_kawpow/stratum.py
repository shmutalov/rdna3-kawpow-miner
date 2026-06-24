"""KawPow stratum client (the common Ravencoin "kawpow"/ethproxy-style protocol).

Flow:
  -> mining.subscribe  [agent, null]
  -> mining.authorize  [wallet.worker, password]
  <- mining.set_target / mining.set_difficulty
  <- mining.notify  [job_id, headerhash, seedhash, target, clean, height?, bits?]
  -> mining.submit  [wallet.worker, job_id, nonce_hex, headerhash_hex, mixhash_hex]

Pure helpers (difficulty/target/nonce/mixhash formatting) are unit-tested offline;
the socket I/O is exercised against a live pool. Pools vary slightly (extranonce
size, target vs difficulty, header byte order) -- see notes inline.
"""

import json
import socket
import threading
import time

TWO256 = 1 << 256
U64 = (1 << 64) - 1


# --- pure helpers -------------------------------------------------------------

def difficulty_to_target(diff):
    """KawPow/ethash share difficulty -> 256-bit target integer (capped at 2^256-1)."""
    if diff <= 0:
        return TWO256 - 1
    return min(TWO256 // int(diff), TWO256 - 1)


def target_hex_to_int(target_hex):
    """A 256-bit target sent as hex (big-endian) -> integer."""
    return int(target_hex.replace("0x", ""), 16)


def boundary64(target_int):
    """Top 64 bits of the 256-bit boundary (the GPU's fast compare value)."""
    return (target_int >> 192) & U64


def nonce_hex(nonce):
    """64-bit nonce -> 16-hex-char big-endian string (kawpowminer convention)."""
    return "%016x" % (nonce & U64)


def parse_int(v):
    """Parse a stratum integer field: int, '0x..' hex, decimal, or bare hex."""
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    if s.isdigit():
        return int(s, 10)
    return int(s, 16)  # bare hex (e.g. block height as hex without 0x)


def mixhash_hex(mix_bytes):
    return "0x" + mix_bytes.hex()


class Job:
    def __init__(self, job_id, header, seed, target_int, height=0, clean=True,
                 extranonce=0, extranonce_bits=0):
        self.job_id = job_id
        self.header = header              # 32 bytes
        self.seed = seed                  # 32 bytes
        self.target_int = target_int
        self.height = height
        self.clean = clean
        self.extranonce = extranonce
        self.extranonce_bits = extranonce_bits

    @property
    def boundary64(self):
        return boundary64(self.target_int)

    def start_nonce(self, salt):
        """Place `salt` in the low bits, the pool extranonce in the high bits."""
        if self.extranonce_bits:
            hi = (self.extranonce << (64 - self.extranonce_bits)) & U64
            return hi | (salt & ((1 << (64 - self.extranonce_bits)) - 1))
        return salt & U64


class StratumClient:
    def __init__(self, host, port, wallet, worker="rdna3", password="x", log=print):
        self.host = host
        self.port = int(port)
        self.wallet = wallet
        self.worker = worker
        self.password = password
        self.log = log
        self.sock = None
        self._id = 0
        self._buf = b""
        self._lock = threading.Lock()
        self.extranonce = 0
        self.extranonce_bits = 0
        self.hashrate = 0
        self.target_int = difficulty_to_target(1)
        self.current_job = None
        self.on_new_job = None            # callback(Job)
        self._running = False
        self._pending = {}                # id -> method (to interpret responses)

    # --- io ---
    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=30)
        self._running = True
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()
        self._send("mining.subscribe", ["rdna3-kawpow/0.1.0", None])
        self._send("mining.authorize", [self._login(), self.password])

    def _keepalive(self):
        # Keep the connection alive during the (idle) DAG build so the pool does
        # not drop us, which would invalidate the extranonce.
        while self._running:
            time.sleep(15)
            if self._running:
                self.report_hashrate(self.hashrate)

    def _login(self):
        return f"{self.wallet}.{self.worker}" if self.worker else self.wallet

    def _send(self, method, params):
        try:
            with self._lock:
                self._id += 1
                mid = self._id
                self._pending[mid] = method
                msg = json.dumps({"id": mid, "method": method, "params": params}) + "\n"
                self.sock.sendall(msg.encode())
            return mid
        except OSError as e:
            self.log(f"send failed ({method}): {e}")
            self._running = False
            return None

    @property
    def alive(self):
        return self._running

    def _recv_loop(self):
        while self._running:
            try:
                data = self.sock.recv(4096)
            except OSError:
                break
            if not data:
                break
            self._buf += data
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                line = line.strip()
                if line:
                    try:
                        self._handle(json.loads(line))
                    except Exception as e:  # never let a bad line kill the loop
                        self.log(f"stratum parse error: {e}: {line[:200]}")
        self._running = False

    # --- protocol ---
    def _handle(self, msg):
        method = msg.get("method")
        if method == "mining.notify":
            self._on_notify(msg["params"])
        elif method in ("mining.set_target", "mining.set_difficulty"):
            self._on_target(method, msg["params"])
        elif method == "mining.set_extranonce":
            self._on_extranonce(msg["params"])
        elif "result" in msg and msg.get("id") is not None:
            self._on_response(msg)

    def _on_response(self, msg):
        method = self._pending.pop(msg.get("id"), None)
        if method == "mining.subscribe":
            # Common shape: [[...subscriptions...], extranonce_hex, extranonce2_size]
            r = msg.get("result")
            self.log(f"subscribe result: {r}")
            if isinstance(r, list) and len(r) >= 2 and isinstance(r[1], str):
                self._set_extranonce(r[1])
        elif method == "mining.submit":
            ok = msg.get("result") is True
            self.log("share " + ("ACCEPTED" if ok else f"REJECTED {msg.get('error')}"))

    def _on_extranonce(self, params):
        if params:
            self._set_extranonce(params[0])

    def _set_extranonce(self, hexstr):
        hexstr = hexstr.replace("0x", "")
        if hexstr:
            self.extranonce = int(hexstr, 16)
            self.extranonce_bits = len(hexstr) * 4
            self.log(f"extranonce={hexstr} ({self.extranonce_bits} bits)")

    def _on_target(self, method, params):
        if method == "mining.set_target":
            self.target_int = target_hex_to_int(params[0])
        else:
            self.target_int = difficulty_to_target(float(params[0]))

    def _on_notify(self, p):
        # [job_id, headerhash, seedhash, target, clean_jobs, height?, bits?]
        if not getattr(self, "_logged_notify", False):
            self.log(f"raw mining.notify params ({len(p)}): {p}")
            self._logged_notify = True
        job_id = p[0]
        header = bytes.fromhex(p[1].replace("0x", ""))
        seed = bytes.fromhex(p[2].replace("0x", ""))
        target_int = self.target_int
        if len(p) > 3 and isinstance(p[3], str) and len(p[3]) >= 16:
            try:
                target_int = target_hex_to_int(p[3])
            except ValueError:
                pass
        clean = bool(p[4]) if len(p) > 4 else True
        height = parse_int(p[5]) if len(p) > 5 and p[5] is not None else 0
        job = Job(job_id, header, seed, target_int, height, clean,
                  self.extranonce, self.extranonce_bits)
        self.current_job = job
        if self.on_new_job:
            self.on_new_job(job)

    def submit(self, job_id, nonce, mix_bytes):
        # The pool wants the FULL 16-hex nonce (which already carries the
        # extranonce in its high bits, so it starts with extraNonce1), plus the
        # header hash and mix hash. No "0x" prefixes (notify fields are bare).
        self._send("mining.submit",
                   [self._login(), job_id, nonce_hex(nonce),
                    self.current_job.header.hex(), mix_bytes.hex()])

    def report_hashrate(self, hr):
        try:
            self._send("eth_submitHashrate", ["0x%x" % int(hr), "0x0"])
        except Exception:
            pass

    def close(self):
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
