#!/usr/bin/env bash
# Download Rockchip's NPU runtime (librknnrt.so and its two headers) from the official
# rknn-toolkit2 repository. This project does not redistribute these files: they are
# Rockchip's, under Rockchip's licence, and are fetched straight from their GitHub.
#
# usage: scripts/get-runtime.sh [--version 2.3.2] [--headers-only] DIR
#
# Result:
#   DIR/lib/librknnrt.so
#   DIR/include/rknn_api.h
#   DIR/include/rknn_matmul_api.h

set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/versions.env"

version=$RKNN_RUNTIME_VERSION
headers_only=0
dir=""

while [ $# -gt 0 ]; do
    case "$1" in
        --version)      version=$2; shift 2 ;;
        --headers-only) headers_only=1; shift ;;
        -h|--help)      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)             echo "get-runtime: unknown option $1" >&2; exit 2 ;;
        *)              dir=$1; shift ;;
    esac
done

if [ -z "$dir" ]; then
    echo "usage: $0 [--version X.Y.Z] [--headers-only] DIR" >&2
    exit 2
fi

base="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v${version}/rknpu2/runtime/Linux/librknn_api"

fetch() {
    # retry a few times, GitHub raw downloads fail now and then on slow board networks
    curl -fsSL --retry 3 --retry-delay 2 -o "$2.part" "$1" && mv "$2.part" "$2"
}

mkdir -p "$dir/include"
for h in rknn_api.h rknn_matmul_api.h; do
    fetch "$base/include/$h" "$dir/include/$h"
done

if [ "$headers_only" -eq 0 ]; then
    mkdir -p "$dir/lib"
    fetch "$base/aarch64/librknnrt.so" "$dir/lib/librknnrt.so"
    chmod 0644 "$dir/lib/librknnrt.so"
fi

[ "$headers_only" -eq 1 ] && what="headers" || what="runtime"
echo "Rockchip $what $version -> $dir"
