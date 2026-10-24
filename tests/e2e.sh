#!/usr/bin/env bash
# End-to-end test without a board: builds llama.cpp the way releases are built (backends
# loaded at run time), with the CPU stub standing in for librknnrt.so, then runs a small
# real model twice, with the NPU path on and off. Passes when both give the same text
# and the first run really sent its matmuls through the NPU path.
#
# usage: tests/e2e.sh [--work DIR]

set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
work="$here/.cache"
[ "${1:-}" = "--work" ] && work=$2

mkdir -p "$work"
work=$(cd "$work" && pwd)
rt="$work/stub-shared"

[ -e "$rt/include/rknn_matmul_api.h" ] || "$here/scripts/get-runtime.sh" --headers-only "$rt"
mkdir -p "$rt/lib"
c++ -O2 -std=c++17 -shared -fPIC -I"$rt/include" "$here/tests/rknnrt_stub.cpp" -o "$rt/lib/librknnrt.so"

src=$("$here/scripts/build.sh" --work "$work" --prepare-only | tail -1)
build="$src/build-e2e"

# shellcheck disable=SC2016  # $ORIGIN is for the linker
cmake -S "$src" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
    -DGGML_RKNPU2=ON \
    -DGGML_RKNPU2_LIB_DIR="$rt/lib" \
    -DGGML_RKNPU2_INCLUDE_DIR="$rt/include" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF >/dev/null
# backends are run-time modules, not link dependencies, so they are named explicitly
cmake --build "$build" --target llama-completion ggml-cpu ggml-rknpu2 -j "$(nproc)"
cp "$rt/lib/librknnrt.so" "$build/bin/"

model="$work/models/stories15M-q4_0.gguf"
if [ ! -e "$model" ]; then
    mkdir -p "$work/models"
    curl -fsSL --retry 3 -o "$model.part" \
        https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
    mv "$model.part" "$model"
fi

# well over 128 tokens, so the prompt batch is big enough for the NPU path
prompt="Once upon a time there was a little girl named Lily who lived in a small house near a big forest. Every morning she walked to the river with her dog Max to look at the fish. One day she found a shiny red stone on the path. She picked it up and showed it to her mother, who smiled and said it was a magic stone. Lily did not believe her, but she kept the stone in her pocket anyway. That night she put the stone under her pillow and fell asleep. When she woke up, the room was full of light and her dog was talking to her. Max said that the stone could grant one wish, but only if Lily was kind to everyone she met that day. Lily thought about it for a long time and then she said"

run() {
    "$build/bin/llama-completion" -m "$model" --no-repack -p "$prompt" -n 24 -c 512 --temp 0 \
        -no-cnv -t 2 --no-warmup
}

export GGML_RKNPU2_SOC=rk3588 STUB_REPORT=1
run > "$work/e2e-npu.txt" 2> "$work/e2e-npu.log"
GGML_RKNPU2_MIN_M=1000000 run > "$work/e2e-cpu.txt" 2> "$work/e2e-cpu.log"

# the backend runs one tiny M=1 matmul per core at startup as a self test, only M > 1 is real work
npu_runs=$(sed -n 's/^stub: .*, \([0-9]*\) with M > 1/\1/p' "$work/e2e-npu.log")
cpu_runs=$(sed -n 's/^stub: .*, \([0-9]*\) with M > 1/\1/p' "$work/e2e-cpu.log")
devices=$("$build/bin/llama-completion" --list-devices 2>&1)
# a board without the driver: no NPU listed, and the model still runs on the CPU
no_npu_devices=$(STUB_NO_NPU=1 "$build/bin/llama-completion" --list-devices 2>&1)
STUB_NO_NPU=1 run > "$work/e2e-nonpu.txt" 2> "$work/e2e-nonpu.log" || true

echo "$devices"
echo "NPU path: ${npu_runs:-0} NPU matmul runs"
echo "CPU path: ${cpu_runs:-0} NPU matmul runs"
echo "--- with the NPU path:"; tail -c 200 "$work/e2e-npu.txt"; echo
echo "--- CPU only:";          tail -c 200 "$work/e2e-cpu.txt"; echo

fail=0
grep -q "RKNPU2: Rockchip NPU (rk3588)" <<<"$devices" || { echo "FAIL: --list-devices does not show the NPU"; fail=1; }
[ "${npu_runs:-0}" -gt 0 ] || { echo "FAIL: nothing ran on the NPU path"; fail=1; }
[ "${cpu_runs:-0}" -eq 0 ] || { echo "FAIL: GGML_RKNPU2_MIN_M did not keep work off the NPU"; fail=1; }
cmp -s "$work/e2e-npu.txt" "$work/e2e-cpu.txt" || { echo "FAIL: the NPU path changed the answer"; fail=1; }
grep -q "RKNPU2" <<<"$no_npu_devices" && { echo "FAIL: NPU listed on a board without one"; fail=1; }
cmp -s "$work/e2e-nonpu.txt" "$work/e2e-cpu.txt" || { echo "FAIL: without an NPU the model did not run the same on the CPU"; fail=1; }

[ "$fail" -eq 0 ] && echo "e2e passed"
exit "$fail"
