#!/usr/bin/env bash
# Off-device correctness tests: builds the backend against a strict CPU stand-in for
# librknnrt (tests/rknnrt_stub.cpp) and checks every supported chip's rules.
# Runs on any Linux machine, x86 or arm, no board needed.
#
# usage: tests/run.sh [--work DIR] [soc ...]     default: all five chips

set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
work="$here/.cache"
socs=()

while [ $# -gt 0 ]; do
    case "$1" in
        --work) work=$2; shift 2 ;;
        *)      socs+=("$1"); shift ;;
    esac
done
[ ${#socs[@]} -eq 0 ] && socs=(rk3588 rk3576 rk3566 rk3568 rk3562)

mkdir -p "$work"
work=$(cd "$work" && pwd)
stub="$work/stub"

# headers only: the stub replaces librknnrt.so, so nothing of Rockchip's is linked in
[ -e "$stub/include/rknn_matmul_api.h" ] || "$here/scripts/get-runtime.sh" --headers-only "$stub"

mkdir -p "$stub/lib"
c++ -O2 -std=c++17 -I"$stub/include" -c "$here/tests/rknnrt_stub.cpp" -o "$stub/rknnrt_stub.o"
rm -f "$stub/lib/librknnrt.a"
ar rcs "$stub/lib/librknnrt.a" "$stub/rknnrt_stub.o"

src=$("$here/scripts/build.sh" --work "$work" --prepare-only | tail -1)
build="$src/build-stub"

cmake -S "$src" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RKNPU2=ON \
    -DGGML_RKNPU2_LIB_DIR="$stub/lib" \
    -DGGML_RKNPU2_INCLUDE_DIR="$stub/include" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF >/dev/null
cmake --build "$build" --target ggml-base ggml-rknpu2 -j "$(nproc)"

c++ -O2 -std=c++17 "$here/tests/test-rknpu2.cpp" \
    -I"$src/ggml/include" -I"$stub/include" \
    "$build/ggml/src/ggml-rknpu2/libggml-rknpu2.a" "$build/ggml/src/libggml-base.a" \
    "$stub/lib/librknnrt.a" -fopenmp -lpthread -lm -o "$work/test-rknpu2"

failed=0
for soc in "${socs[@]}"; do
    "$work/test-rknpu2" "$soc" || failed=$((failed + 1))
done

if [ "$failed" -ne 0 ]; then
    echo "$failed chip(s) failed"
    exit 1
fi
echo "all chips passed"
