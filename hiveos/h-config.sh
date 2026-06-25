#!/usr/bin/env bash
# Render the runtime config from the HiveOS flight sheet. HiveOS exports:
#   CUSTOM_TEMPLATE     wallet[.worker]   (the "Wallet and worker template" field)
#   CUSTOM_URL          pool URL          (stratum+tcp://host:port or host:port)
#   CUSTOM_PASS         pool password
#   CUSTOM_USER_CONFIG  extra CLI args    (the "Extra config arguments" box)
#   CUSTOM_ALGO         algorithm
cd "$(dirname "$0")" || exit 1
. h-manifest.conf

mkdir -p "$(dirname "$CUSTOM_CONFIG_FILENAME")"
cat > "$CUSTOM_CONFIG_FILENAME" <<EOF
ALGO="${CUSTOM_ALGO:-kawpow}"
POOL="$CUSTOM_URL"
WALLET="$CUSTOM_TEMPLATE"
PASS="${CUSTOM_PASS:-x}"
EXTRA="$CUSTOM_USER_CONFIG"
EOF

echo "rdna3kawpow: wrote $CUSTOM_CONFIG_FILENAME"
