#!/usr/bin/env bash
# Launch one miner process per GPU. The bundled glslangValidator (next to the
# binary) is found automatically; nothing to add to PATH. Output from every GPU
# is prefixed and streamed to stdout (the HiveOS screen) and the per-GPU log.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf
[[ -f "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"

BIN="$(pwd)/rdna3-kawpow"
chmod +x "$BIN" ./glslangValidator 2>/dev/null
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

N=$("$BIN" --device-count 2>/dev/null)
[[ -z "$N" || "$N" -lt 1 ]] && N=1
echo "rdna3kawpow: launching $N GPU process(es)  algo=$ALGO  pool=$POOL"

pids=()
for ((i=0; i<N; i++)); do
  port=$(( ${CUSTOM_API_PORT_BASE:-4068} + i ))
  log="${CUSTOM_LOG_BASENAME}_gpu${i}.log"
  ( "$BIN" -a "$ALGO" -o "$POOL" -u "$WALLET" -p "$PASS" \
        --device "$i" --api-bind "127.0.0.1:${port}" $EXTRA 2>&1 \
      | tee -a "$log" | sed -u "s/^/[gpu${i}] /" ) &
  pids+=("$!")
done

# Return as soon as ANY GPU process exits, then stop the rest so HiveOS restarts
# the whole set cleanly (avoids a half-dead rig).
wait -n 2>/dev/null || wait
kill "${pids[@]}" 2>/dev/null
