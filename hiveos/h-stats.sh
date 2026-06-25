#!/usr/bin/env bash
# Report stats to HiveOS. Sourced by the agent, which reads $khs (total kH/s) and
# $stats (JSON). Aggregates each per-GPU miner's JSON API. temp/fan/bus_numbers are
# filled by the HiveOS agent from gpu-stats, so we only report hs + shares + algo.
cd "$(dirname "$0")" 2>/dev/null
. h-manifest.conf 2>/dev/null

BASE=${CUSTOM_API_PORT_BASE:-4068}
N=$(./rdna3-kawpow --device-count 2>/dev/null)
[[ -z "$N" || "$N" -lt 1 ]] && N=1

declare -a hs bus
total=0; acc=0; rej=0; inv=0; uptime=0; algo="kawpow"; have_bus=1
for ((i=0; i<N; i++)); do
  resp=$(curl -s --max-time 3 "http://127.0.0.1:$((BASE + i))" 2>/dev/null)
  if [[ -n "$resp" ]]; then
    h=$(jq -r '.hashrate // 0'  <<< "$resp" 2>/dev/null); h=${h:-0}
    a=$(jq -r '.accepted // 0'  <<< "$resp" 2>/dev/null); a=${a:-0}
    r=$(jq -r '.rejected // 0'  <<< "$resp" 2>/dev/null); r=${r:-0}
    v=$(jq -r '.invalid  // 0'  <<< "$resp" 2>/dev/null); v=${v:-0}
    u=$(jq -r '.uptime   // 0'  <<< "$resp" 2>/dev/null); u=${u:-0}
    b=$(jq -r '.bus_id  // -1'  <<< "$resp" 2>/dev/null); b=${b:--1}
    algo=$(jq -r '.algo // "kawpow"' <<< "$resp" 2>/dev/null)
    hs[$i]=$h
    total=$(echo "$total + $h" | bc -l)
    acc=$((acc + a)); rej=$((rej + r)); inv=$((inv + v))
    (( u > uptime )) && uptime=$u
  else
    hs[$i]=0; b=-1
  fi
  bus[$i]=$b
  [[ "$b" == "-1" ]] && have_bus=0
done

khs=$(echo "scale=3; $total / 1000" | bc -l)
[[ -z "$khs" ]] && khs=0

hs_json=$(printf '%s\n' "${hs[@]}" | jq -cs '.')
# Only send bus_numbers if every GPU reported one, so HiveOS can map each
# hashrate to the correct physical card (PCI bus order, not enumeration order).
if [[ $have_bus -eq 1 ]]; then
  bus_json=$(printf '%s\n' "${bus[@]}" | jq -cs '.')
  stats=$(jq -nc \
    --argjson hs "$hs_json" --argjson bus "$bus_json" \
    --arg algo "$algo" \
    --argjson a "$acc" --argjson r "$rej" --argjson v "$inv" \
    --argjson up "$uptime" \
    '{hs: $hs, hs_units: "hs", bus_numbers: $bus, algo: $algo, uptime: $up, ar: [$a, $r, $v]}')
else
  stats=$(jq -nc \
    --argjson hs "$hs_json" \
    --arg algo "$algo" \
    --argjson a "$acc" --argjson r "$rej" --argjson v "$inv" \
    --argjson up "$uptime" \
    '{hs: $hs, hs_units: "hs", algo: $algo, uptime: $up, ar: [$a, $r, $v]}')
fi
