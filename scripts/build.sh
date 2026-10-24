#!/usr/bin/env bash
# Build llama.cpp with the RKNPU2 backend.
#
# Fetches the llama.cpp commit pinned in versions.env, adds the backend from backend/,
# applies patches/llama.cpp.patch and builds. The result is one folder you can move
# anywhere: PREFIX/bin holds the llama.cpp programs, their libraries and librknnrt.so.
#
# usage: scripts/build.sh [options]
#
#   --prefix DIR         install into DIR (default: ./dist/ggml-rknpu2)
#   --work DIR           where llama.cpp is downloaded and built (default: ./.cache)
#   --runtime DIR        use the Rockchip runtime in DIR (DIR/lib, DIR/include)
#                        instead of downloading it
#   --system-runtime     use librknnrt.so from /usr/lib and /usr/include
#   --runtime-version V  Rockchip runtime to download (default: from versions.env)
#   --no-bundle-runtime  do not copy librknnrt.so into PREFIX/bin (release packages)
#   --portable           build every CPU variant instead of tuning for this machine,
#                        so one build runs on RK3588, RK3576, RK356x and RK3562
#   --jobs N             parallel compile jobs (default: picked from CPU count and RAM)
#   --prepare-only       only download and patch llama.cpp, print its path
#
# Environment: CMAKE_ARGS is passed to cmake as extra arguments.

set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/versions.env"

prefix="$here/dist/ggml-rknpu2"
work="$here/.cache"
runtime=""
system_runtime=0
runtime_version=$RKNN_RUNTIME_VERSION
bundle_runtime=1
portable=0
jobs=""
prepare_only=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)            prefix=$2; shift 2 ;;
        --work)              work=$2; shift 2 ;;
        --runtime)           runtime=$2; shift 2 ;;
        --system-runtime)    system_runtime=1; shift ;;
        --runtime-version)   runtime_version=$2; shift 2 ;;
        --no-bundle-runtime) bundle_runtime=0; shift ;;
        --portable)          portable=1; shift ;;
        --jobs|-j)           jobs=$2; shift 2 ;;
        --prepare-only)      prepare_only=1; shift ;;
        -h|--help)           sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "build: unknown option $1 (try --help)" >&2; exit 2 ;;
    esac
done

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'build: %s\n' "$*" >&2; exit 1; }

for t in git cmake c++ curl; do
    command -v "$t" >/dev/null 2>&1 || die "$t is not installed. On Debian/Ubuntu: sudo apt install git cmake build-essential curl"
done

mkdir -p "$work"
work=$(cd "$work" && pwd)
src="$work/llama.cpp-${LLAMA_CPP_COMMIT:0:12}"

# 1. llama.cpp at the pinned commit. A shallow fetch of one commit keeps it to ~30 MB.
if [ ! -f "$src/.fetched" ]; then
    say "Downloading llama.cpp ${LLAMA_CPP_COMMIT:0:12}"
    rm -rf "$src"
    git init -q "$src"
    git -C "$src" remote add origin "$LLAMA_CPP_REPO"
    git -C "$src" fetch -q --depth 1 origin "$LLAMA_CPP_COMMIT"
    git -C "$src" -c advice.detachedHead=false checkout -q FETCH_HEAD
    touch "$src/.fetched"
fi

# 2. the backend. Copied every time, so edits in backend/ are always picked up.
mkdir -p "$src/ggml/src/ggml-rknpu2"
cp "$here/backend/ggml-rknpu2.cpp" "$here/backend/CMakeLists.txt" "$src/ggml/src/ggml-rknpu2/"
cp "$here/backend/ggml-rknpu2.h" "$src/ggml/include/"

if git -C "$src" apply --reverse --check "$here/patches/llama.cpp.patch" 2>/dev/null; then
    : # already applied
else
    git -C "$src" apply "$here/patches/llama.cpp.patch" || die "patches/llama.cpp.patch does not apply to llama.cpp ${LLAMA_CPP_COMMIT:0:12}"
fi

if [ "$prepare_only" -eq 1 ]; then
    echo "$src"
    exit 0
fi

# 3. the Rockchip runtime to build against
if [ "$system_runtime" -eq 1 ]; then
    rt_lib=/usr/lib
    rt_inc=/usr/include
    for d in /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib; do
        [ -e "$d/librknnrt.so" ] && rt_lib=$d && break
    done
    for d in /usr/include /usr/local/include; do
        [ -e "$d/rknn_matmul_api.h" ] && rt_inc=$d && break
    done
    [ -e "$rt_lib/librknnrt.so" ] || die "--system-runtime: librknnrt.so is not installed"
    [ -e "$rt_inc/rknn_matmul_api.h" ] || die "--system-runtime: rknn_matmul_api.h is not installed"
else
    if [ -z "$runtime" ]; then
        runtime="$work/rknn-runtime-$runtime_version"
        if [ ! -e "$runtime/lib/librknnrt.so" ]; then
            say "Downloading the Rockchip NPU runtime $runtime_version"
            "$here/scripts/get-runtime.sh" --version "$runtime_version" "$runtime"
        fi
    fi
    rt_lib=$(cd "$runtime/lib" && pwd)
    rt_inc=$(cd "$runtime/include" && pwd)
fi

# 4. configure and build
if [ -z "$jobs" ]; then
    # compiling llama.cpp needs up to ~1 GB per job, a 4 GB board runs out with -j8
    cpus=$(nproc 2>/dev/null || echo 2)
    mem_gb=$(awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo 2>/dev/null || echo 4)
    jobs=$(( mem_gb > 1 ? mem_gb - 1 : 1 ))
    [ "$jobs" -gt "$cpus" ] && jobs=$cpus
fi

if [ "$portable" -eq 1 ]; then
    build="$src/build-portable"
    cpu_args=(-DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON)
    # The armv9.2 CPU variants need a compiler that knows SME (GCC 14+). Older ones, like
    # Ubuntu 22.04's GCC 11, stop the build. No Rockchip chip is armv9, so drop them there.
    if [ "$(uname -m)" = "aarch64" ] && ! echo 'int main(void){return 0;}' \
            | c++ -x c++ -march=armv9.2-a+dotprod+fp16+sve+i8mm+sme -o /dev/null - 2>/dev/null; then
        sed -i '/ggml_add_cpu_backend_variant(armv9\.2_/d' "$src/ggml/src/CMakeLists.txt"
        echo "build: compiler has no armv9.2 SME support, skipping those CPU variants (not used by Rockchip chips)"
    fi
else
    build="$src/build-native"
    cpu_args=(-DGGML_NATIVE=ON)
fi

say "Configuring ($([ "$portable" -eq 1 ] && echo portable || echo tuned for this board))"
# $ORIGIN is for the linker, not the shell; CMAKE_ARGS is split into words on purpose
# shellcheck disable=SC2016,SC2086
cmake -S "$src" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_RKNPU2=ON \
    -DGGML_RKNPU2_LIB_DIR="$rt_lib" \
    -DGGML_RKNPU2_INCLUDE_DIR="$rt_inc" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    "${cpu_args[@]}" \
    ${CMAKE_ARGS:-} >"$work/cmake-configure.log" 2>&1 \
    || { tail -30 "$work/cmake-configure.log"; die "cmake configure failed, full log: $work/cmake-configure.log"; }

if grep -q "OpenSSL not found" "$work/cmake-configure.log"; then
    echo "build: warning: OpenSSL headers not found, so downloading models by name (-hf) will"
    echo "       not work. Fix: sudo apt install libssl-dev, then run this again."
fi

say "Building with $jobs jobs (10 to 40 minutes on a board)"
cmake --build "$build" --config Release -j "$jobs"

# 5. install: one self-contained folder, programs find their libraries next to them
say "Installing into $prefix"
mkdir -p "$prefix"
rm -rf "${prefix:?}/bin"
mkdir -p "$prefix/bin"
cp -a "$build/bin/." "$prefix/bin/"
if [ "$bundle_runtime" -eq 1 ] && [ "$system_runtime" -eq 0 ]; then
    cp "$rt_lib/librknnrt.so" "$prefix/bin/"
fi

cp "$here/scripts/llama-npu" "$prefix/llama-npu"
cp "$here/scripts/check.sh" "$prefix/check.sh"
chmod +x "$prefix/llama-npu" "$prefix/check.sh"
cp "$here/LICENSE" "$prefix/LICENSE"
cp "$src/LICENSE" "$prefix/LICENSE-llama.cpp"
cp "$here/THIRD_PARTY.md" "$prefix/THIRD_PARTY.md"
cp "$here/versions.env" "$prefix/versions.env"
cat "$here/VERSION" > "$prefix/VERSION"

say "Done"
echo "Programs are in $prefix/bin"
echo "Try:  $prefix/llama-npu check"
