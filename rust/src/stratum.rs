//! KawPow stratum client (port of `rdna3_kawpow/stratum.py`).
//!
//!   -> mining.subscribe / mining.authorize
//!   <- mining.set_target / mining.set_difficulty / mining.set_extranonce
//!   <- mining.notify  [job_id, headerhash, seedhash, target, clean, height?]
//!   -> mining.submit  [login, job_id, nonce_hex, headerhash_hex, mixhash_hex]
//!
//! Pure helpers (difficulty/target/nonce) are unit-tested offline; socket I/O is a
//! background reader thread. The 256-bit target is kept as 32 big-endian bytes; the
//! GPU only ever needs its top 64 bits (`boundary64`).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

// --- pure helpers -------------------------------------------------------------

/// KawPow/ethash share difficulty -> 256-bit target (big-endian bytes), capped.
pub fn difficulty_to_target(diff: f64) -> [u8; 32] {
    if diff <= 0.0 {
        return [0xFF; 32];
    }
    let d = diff as u128; // truncate toward zero, like int(diff)
    if d <= 1 {
        return [0xFF; 32]; // 2^256 // 1 capped to 2^256-1
    }
    div_2pow256_by(d)
}

/// floor(2^256 / d) as 32 big-endian bytes (d >= 2).
fn div_2pow256_by(d: u128) -> [u8; 32] {
    let mut q = [0u8; 32];
    let mut r: u128 = 0;
    // Dividend 2^256: bit 256 set, bits 255..0 zero.
    r = (r << 1) | 1; // bit 256
    if r >= d {
        r -= d; // (only when d==1, excluded by caller)
    }
    for i in (0..256).rev() {
        r <<= 1; // shift in a 0 bit
        if r >= d {
            r -= d;
            let byte = 31 - (i / 8);
            q[byte] |= 1 << (i % 8);
        }
    }
    q
}

/// A 256-bit target sent as big-endian hex -> 32 bytes (right-aligned).
pub fn target_hex_to_bytes(hex: &str) -> [u8; 32] {
    let v = hex_to_vec(hex.trim_start_matches("0x")).unwrap_or_default();
    let mut out = [0u8; 32];
    if v.len() >= 32 {
        out.copy_from_slice(&v[v.len() - 32..]);
    } else {
        out[32 - v.len()..].copy_from_slice(&v);
    }
    out
}

/// Top 64 bits of the 256-bit boundary (the GPU's fast compare value).
pub fn boundary64(target: &[u8; 32]) -> u64 {
    u64::from_be_bytes(target[0..8].try_into().unwrap())
}

/// 64-bit nonce -> 16-hex-char string (kawpowminer convention).
pub fn nonce_hex(nonce: u64) -> String {
    format!("{nonce:016x}")
}

fn hex_to_vec(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    let s = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_string()
    };
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn hex32(s: &str) -> [u8; 32] {
    let v = hex_to_vec(s.trim_start_matches("0x")).unwrap_or_default();
    let mut out = [0u8; 32];
    let n = v.len().min(32);
    out[..n].copy_from_slice(&v[..n]);
    out
}

/// Parse a stratum integer field: int, "0x.." hex, decimal, or bare hex.
fn parse_int(v: &Value) -> u64 {
    if let Some(n) = v.as_u64() {
        return n;
    }
    let s = v.as_str().unwrap_or("").trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).unwrap_or(0);
    }
    if let Ok(d) = s.parse::<u64>() {
        return d;
    }
    u64::from_str_radix(s, 16).unwrap_or(0)
}

#[derive(Clone)]
pub struct Job {
    pub job_id: String,
    pub header: [u8; 32],
    pub seed: [u8; 32],
    pub target: [u8; 32],
    pub height: u64,
    pub clean: bool,
    pub extranonce: u64,
    pub extranonce_bits: u32,
}

impl Job {
    pub fn boundary64(&self) -> u64 {
        boundary64(&self.target)
    }

    /// Place `salt` in the low bits, the pool extranonce in the high bits.
    pub fn start_nonce(&self, salt: u64) -> u64 {
        if self.extranonce_bits > 0 {
            let lo_bits = 64 - self.extranonce_bits;
            let hi = self.extranonce.wrapping_shl(lo_bits);
            let mask = if lo_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << lo_bits) - 1
            };
            hi | (salt & mask)
        } else {
            salt
        }
    }
}

struct State {
    current_job: Option<Job>,
    target: [u8; 32],
    extranonce: u64,
    extranonce_bits: u32,
    hashrate: f64,
    logged_notify: bool,
}

struct Inner {
    host: String,
    port: u16,
    wallet: String,
    worker: String,
    password: String,
    state: Mutex<State>,
    write_sock: Mutex<Option<TcpStream>>,
    running: AtomicBool,
    id_ctr: AtomicU64,
    job_gen: AtomicU64,
    pending: Mutex<HashMap<u64, String>>,
    accepted: AtomicU64,
    rejected: AtomicU64,
}

pub struct StratumClient {
    inner: Arc<Inner>,
}

impl StratumClient {
    pub fn new(host: &str, port: u16, wallet: &str, worker: &str, password: &str) -> Self {
        StratumClient {
            inner: Arc::new(Inner {
                host: host.to_string(),
                port,
                wallet: wallet.to_string(),
                worker: worker.to_string(),
                password: password.to_string(),
                state: Mutex::new(State {
                    current_job: None,
                    target: difficulty_to_target(1.0),
                    extranonce: 0,
                    extranonce_bits: 0,
                    hashrate: 0.0,
                    logged_notify: false,
                }),
                write_sock: Mutex::new(None),
                running: AtomicBool::new(false),
                id_ctr: AtomicU64::new(0),
                job_gen: AtomicU64::new(0),
                pending: Mutex::new(HashMap::new()),
                accepted: AtomicU64::new(0),
                rejected: AtomicU64::new(0),
            }),
        }
    }

    pub fn connect(&self) -> std::io::Result<()> {
        let stream = TcpStream::connect((self.inner.host.as_str(), self.inner.port))?;
        let read_stream = stream.try_clone()?;
        // Drop any job from a previous session: a reconnect gets a NEW extranonce,
        // so we must wait for a fresh notify (created after that extranonce is set)
        // before mining. Reusing the stale job submits the old prefix and the pool
        // rejects every share with "Invalid nonce prefix".
        self.inner.state.lock().unwrap().current_job = None;
        *self.inner.write_sock.lock().unwrap() = Some(stream);
        self.inner.running.store(true, Ordering::SeqCst);

        let r = self.inner.clone();
        thread::spawn(move || recv_loop(r, read_stream));
        let k = self.inner.clone();
        thread::spawn(move || keepalive(k));

        self.inner
            .send("mining.subscribe", vec![json!("rdna3-kawpow/0.1.0"), Value::Null]);
        self.inner.send(
            "mining.authorize",
            vec![json!(self.inner.login()), json!(self.inner.password)],
        );
        Ok(())
    }

    pub fn alive(&self) -> bool {
        self.inner.running.load(Ordering::SeqCst)
    }
    pub fn job_gen(&self) -> u64 {
        self.inner.job_gen.load(Ordering::SeqCst)
    }
    pub fn current_job(&self) -> Option<Job> {
        self.inner.state.lock().unwrap().current_job.clone()
    }
    pub fn accepted(&self) -> u64 {
        self.inner.accepted.load(Ordering::SeqCst)
    }
    pub fn rejected(&self) -> u64 {
        self.inner.rejected.load(Ordering::SeqCst)
    }
    pub fn set_hashrate(&self, hr: f64) {
        self.inner.state.lock().unwrap().hashrate = hr;
    }

    pub fn submit(&self, job_id: &str, nonce: u64, mix_bytes: &[u8]) {
        let header_hex = self
            .inner
            .state
            .lock()
            .unwrap()
            .current_job
            .as_ref()
            .map(|j| hex_encode(&j.header))
            .unwrap_or_default();
        self.inner.send(
            "mining.submit",
            vec![
                json!(self.inner.login()),
                json!(job_id),
                json!(nonce_hex(nonce)),
                json!(header_hex),
                json!(hex_encode(mix_bytes)),
            ],
        );
    }

    pub fn report_hashrate(&self, hr: f64) {
        self.inner
            .send("eth_submitHashrate", vec![json!(format!("0x{:x}", hr as u64)), json!("0x0")]);
    }

    pub fn close(&self) {
        self.inner.running.store(false, Ordering::SeqCst);
        if let Some(s) = self.inner.write_sock.lock().unwrap().as_ref() {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
    }
}

/// A resolved pool endpoint + credentials.
pub struct Pool {
    pub host: String,
    pub port: u16,
    pub wallet: String,
    pub worker: String,
    pub password: String,
}

/// scheme://wallet.worker:password@host:port -> Pool.
pub fn parse_pool_url(url: &str) -> Result<Pool, String> {
    let url = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let (userinfo, hostport) = url
        .rsplit_once('@')
        .ok_or_else(|| "pool URL needs WALLET[.WORKER][:PASSWORD]@HOST:PORT".to_string())?;
    let (user, password) = userinfo.split_once(':').unwrap_or((userinfo, ""));
    let (wallet, worker) = user.split_once('.').unwrap_or((user, ""));
    let (host, port) = hostport.split_once(':').unwrap_or((hostport, "0"));
    Ok(Pool {
        host: host.to_string(),
        port: port.parse().unwrap_or(0),
        wallet: wallet.to_string(),
        worker: if worker.is_empty() { "rdna3" } else { worker }.to_string(),
        password: if password.is_empty() { "x" } else { password }.to_string(),
    })
}

/// stratum+tcp://host:port (no userinfo) -> (host, port).
pub fn parse_host_port(url: &str) -> Result<(String, u16), String> {
    let url = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let url = url.trim_end_matches('/');
    let (host, port) = url
        .rsplit_once(':')
        .ok_or_else(|| format!("expected host:port in {url}"))?;
    Ok((host.to_string(), port.parse().map_err(|_| "bad port".to_string())?))
}

impl Inner {
    fn login(&self) -> String {
        if self.worker.is_empty() {
            self.wallet.clone()
        } else {
            format!("{}.{}", self.wallet, self.worker)
        }
    }

    fn send(&self, method: &str, params: Vec<Value>) -> Option<u64> {
        let id = self.id_ctr.fetch_add(1, Ordering::SeqCst) + 1;
        self.pending.lock().unwrap().insert(id, method.to_string());
        let msg = json!({"id": id, "method": method, "params": params}).to_string() + "\n";
        let mut g = self.write_sock.lock().unwrap();
        match g.as_mut() {
            Some(s) => {
                if s.write_all(msg.as_bytes()).is_err() {
                    self.running.store(false, Ordering::SeqCst);
                    None
                } else {
                    Some(id)
                }
            }
            None => None,
        }
    }
}

fn keepalive(inner: Arc<Inner>) {
    // Keep the connection alive during the (idle) DAG build so the pool does not
    // drop us (which would invalidate the extranonce).
    while inner.running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_secs(15));
        if inner.running.load(Ordering::SeqCst) {
            let hr = inner.state.lock().unwrap().hashrate;
            inner.send("eth_submitHashrate", vec![json!(format!("0x{:x}", hr as u64)), json!("0x0")]);
        }
    }
}

fn recv_loop(inner: Arc<Inner>, stream: TcpStream) {
    // Guard: whatever ends this thread (EOF, error, or a panic while handling a
    // message) MUST mark the connection dead so the main loop reconnects.
    struct DeadOnExit(Arc<Inner>);
    impl Drop for DeadOnExit {
        fn drop(&mut self) {
            self.0.running.store(false, Ordering::SeqCst);
        }
    }
    let _guard = DeadOnExit(inner.clone());

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    while inner.running.load(Ordering::SeqCst) {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let t = line.trim();
                if !t.is_empty() {
                    if let Ok(v) = serde_json::from_str::<Value>(t) {
                        handle(&inner, v); // never let a bad line kill the loop
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn handle(inner: &Arc<Inner>, msg: Value) {
    match msg.get("method").and_then(|m| m.as_str()) {
        Some("mining.notify") => on_notify(inner, &msg["params"]),
        Some("mining.set_target") => {
            if let Some(p0) = msg["params"].get(0).and_then(|v| v.as_str()) {
                inner.state.lock().unwrap().target = target_hex_to_bytes(p0);
            }
        }
        Some("mining.set_difficulty") => {
            if let Some(p0) = msg["params"].get(0) {
                let d = p0.as_f64().unwrap_or_else(|| parse_int(p0) as f64);
                inner.state.lock().unwrap().target = difficulty_to_target(d);
            }
        }
        Some("mining.set_extranonce") => {
            if let Some(p0) = msg["params"].get(0).and_then(|v| v.as_str()) {
                set_extranonce(inner, p0);
            }
        }
        _ => {
            if msg.get("result").is_some()
                && msg.get("id").map(|i| !i.is_null()).unwrap_or(false)
            {
                on_response(inner, &msg);
            }
        }
    }
}

fn on_response(inner: &Arc<Inner>, msg: &Value) {
    let id = msg["id"].as_u64().unwrap_or(0);
    let method = inner.pending.lock().unwrap().remove(&id);
    match method.as_deref() {
        Some("mining.subscribe") => {
            // Common shape: [[...subscriptions...], extranonce_hex, extranonce2_size]
            if let Some(arr) = msg.get("result").and_then(|r| r.as_array()) {
                if arr.len() >= 2 {
                    if let Some(en) = arr[1].as_str() {
                        set_extranonce(inner, en);
                    }
                }
            }
        }
        Some("mining.submit") => {
            if msg.get("result") == Some(&Value::Bool(true)) {
                inner.accepted.fetch_add(1, Ordering::SeqCst);
                eprintln!("share ACCEPTED");
            } else {
                inner.rejected.fetch_add(1, Ordering::SeqCst);
                eprintln!("share REJECTED {:?}", msg.get("error"));
            }
        }
        _ => {}
    }
}

fn set_extranonce(inner: &Arc<Inner>, hexstr: &str) {
    let hexstr = hexstr.trim_start_matches("0x");
    if hexstr.is_empty() {
        return;
    }
    if let Ok(en) = u64::from_str_radix(hexstr, 16) {
        let mut st = inner.state.lock().unwrap();
        st.extranonce = en;
        st.extranonce_bits = (hexstr.len() * 4) as u32;
        eprintln!("extranonce={hexstr} ({} bits)", st.extranonce_bits);
    }
}

fn on_notify(inner: &Arc<Inner>, p: &Value) {
    let arr = match p.as_array() {
        Some(a) if a.len() >= 3 => a,
        _ => return,
    };
    {
        let mut st = inner.state.lock().unwrap();
        if !st.logged_notify {
            eprintln!("raw mining.notify params ({}): {}", arr.len(), p);
            st.logged_notify = true;
        }
    }
    let job_id = arr[0].as_str().unwrap_or("").to_string();
    let header = hex32(arr[1].as_str().unwrap_or(""));
    let seed = hex32(arr[2].as_str().unwrap_or(""));

    let mut st = inner.state.lock().unwrap();
    let mut target = st.target;
    if let Some(s) = arr.get(3).and_then(|v| v.as_str()) {
        if s.trim_start_matches("0x").len() >= 16 {
            target = target_hex_to_bytes(s);
        }
    }
    let clean = arr.get(4).map(|v| v.as_bool().unwrap_or(true)).unwrap_or(true);
    let height = arr
        .get(5)
        .filter(|v| !v.is_null())
        .map(parse_int)
        .unwrap_or(0);
    st.current_job = Some(Job {
        job_id,
        header,
        seed,
        target,
        height,
        clean,
        extranonce: st.extranonce,
        extranonce_bits: st.extranonce_bits,
    });
    inner.job_gen.fetch_add(1, Ordering::SeqCst);
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_u256(bytes: &[u8; 32]) -> u128 {
        // top 128 bits only (enough for the assertions we make)
        u128::from_be_bytes(bytes[0..16].try_into().unwrap())
    }

    #[test]
    fn difficulty_target() {
        assert_eq!(difficulty_to_target(1.0), [0xFF; 32]); // capped, not zero
        // diff 2 halves it: 2^255 -> top byte 0x80, rest zero.
        let half = difficulty_to_target(2.0);
        assert_eq!(half[0], 0x80);
        assert!(half[1..].iter().all(|&b| b == 0));
        assert_eq!(boundary64(&difficulty_to_target(1.0)), u64::MAX);
        let t = difficulty_to_target(256.0);
        // boundary64 is the top 64 bits of the 256-bit target.
        assert_eq!(boundary64(&t), (to_u256(&t) >> 64) as u64);
    }

    #[test]
    fn target_hex() {
        let h = "00000000ffff0000000000000000000000000000000000000000000000000000";
        assert_eq!(boundary64(&target_hex_to_bytes(h)), 0x00000000ffff0000);
    }

    #[test]
    fn nonce_hex_fmt() {
        assert_eq!(nonce_hex(0x123456789abcdef0), "123456789abcdef0");
        assert_eq!(nonce_hex(0), "0000000000000000");
    }

    #[test]
    fn extranonce_start_nonce() {
        let mut target = [0u8; 32];
        target[0] = 0x80; // 1<<255
        let j = Job {
            job_id: "j".into(),
            header: [0; 32],
            seed: [0; 32],
            target,
            height: 0,
            clean: true,
            extranonce: 0xABCD,
            extranonce_bits: 16,
        };
        let n = j.start_nonce(0x1234);
        assert_eq!(n >> 48, 0xABCD); // extranonce in the top 16 bits
        assert_eq!(n & 0xFFFF_FFFF_FFFF, 0x1234);
    }

    #[test]
    fn pool_url_parsing() {
        let p = parse_pool_url("stratum+tcp://RAVENADDR.rig1:x@pool.example:4444").unwrap();
        assert_eq!(p.host, "pool.example");
        assert_eq!(p.port, 4444);
        assert_eq!(p.wallet, "RAVENADDR");
        assert_eq!(p.worker, "rig1");
        assert_eq!(p.password, "x");
        let p2 = parse_pool_url("RAVENADDR@host:1234").unwrap();
        assert_eq!(p2.wallet, "RAVENADDR");
        assert_eq!(p2.worker, "rdna3");
        assert_eq!(p2.port, 1234);
    }

    #[test]
    fn connect_clears_stale_session_job() {
        // Regression: after a reconnect the pool issues a NEW extranonce, so a job
        // from the previous session must NOT be reused (it submits the old prefix ->
        // "Invalid nonce prefix"). connect() must clear current_job so the loop
        // waits for a fresh notify.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept the connect() socket and hold it open (send nothing).
        let h = std::thread::spawn(move || {
            let _c = listener.accept();
            std::thread::sleep(std::time::Duration::from_millis(300));
        });

        let c = StratumClient::new("127.0.0.1", addr.port(), "w", "rig", "x");
        // Inject a job as if from a prior session.
        let tgt = "00000000ffff0000000000000000000000000000000000000000000000000000";
        handle(
            &c.inner,
            json!({"method": "mining.notify",
                   "params": ["old", "ab".repeat(32), "cd".repeat(32), tgt, true, 7050]}),
        );
        assert!(c.current_job().is_some());

        c.connect().unwrap(); // a fresh session must drop the stale job
        assert!(
            c.current_job().is_none(),
            "connect() must clear the previous-session job"
        );
        c.close();
        let _ = h.join();
    }

    #[test]
    fn notify_parsing() {
        let c = StratumClient::new("h", 1, "w", "rdna3", "x");
        handle(
            &c.inner,
            json!({"method": "mining.set_difficulty", "params": [2]}),
        );
        let hdr = "ab".repeat(32);
        let seed = "cd".repeat(32);
        let tgt = "00000000ffff0000000000000000000000000000000000000000000000000000";
        handle(
            &c.inner,
            json!({"method": "mining.notify",
                   "params": ["job1", hdr, seed, tgt, true, "0x7530"]}),
        );
        let job = c.current_job().expect("a job");
        assert_eq!(job.job_id, "job1");
        assert_eq!(job.height, 0x7530);
        assert_eq!(job.boundary64(), 0x00000000ffff0000);
        assert_eq!(c.job_gen(), 1);
    }
}
