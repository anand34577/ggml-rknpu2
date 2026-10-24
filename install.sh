#!/usr/bin/env bash
# Install llama.cpp with the Rockchip NPU backend (ggml-rknpu2) on this board.
#
#   curl -fsSL https://raw.githubusercontent.com/anand34577/ggml-rknpu2/main/install.sh | bash
#
# Options (after "bash -s --" when piping, e.g. ... | bash -s -- --from-source):
#   --prefix DIR           install here (default: ~/.local/share/ggml-rknpu2)
#   --version vX.Y.Z       install this release instead of the latest
#   --from-source          build on the board instead of downloading a ready build
#   --runtime-version V    Rockchip runtime to download (default: the tested one)
#   --system-runtime       use the librknnrt.so already installed in /usr/lib
#   --no-path              do not add ~/.local/bin to PATH in ~/.bashrc
#
# Nothing here needs root, except installing build tools for --from-source.

set -euo pipefail

repo="anand34577/ggml-rknpu2"
prefix="$HOME/.local/share/ggml-rknpu2"
version=""
from_source=0
runtime_version=""
system_runtime=0
edit_path=1

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)          prefix=$2; shift 2 ;;
        --version)         version=$2; shift 2 ;;
        --from-source)     from_source=1; shift ;;
        --runtime-version) runtime_version=$2; shift 2 ;;
        --system-runtime)  system_runtime=1; shift ;;
        --no-path)         edit_path=0; shift ;;
        -h|--help)         sed -n '2,15p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' || true; exit 0 ;;
        *) echo "install: unknown option $1" >&2; exit 2 ;;
    esac
done

bold=""; reset=""
if [ -t 1 ]; then bold=$(printf '\033[1m'); reset=$(printf '\033[0m'); fi
step() { printf '\n%s==> %s%s\n' "$bold" "$*" "$reset"; }
die()  { printf '\ninstall: %s\n' "$*" >&2; exit 1; }

# 1. this has to run on the board itself
arch=$(uname -m)
[ "$arch" = "aarch64" ] || die "this installs on the Rockchip board itself (64-bit ARM). This machine is $arch.
Copy the command to the board (over SSH, for example) and run it there."
[ "$(uname -s)" = "Linux" ] || die "Linux is required."

for t in curl tar; do
    command -v "$t" >/dev/null 2>&1 || die "$t is missing. Install it with: sudo apt install $t"
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch() { curl -fSL --retry 3 --retry-delay 2 --progress-bar -o "$2" "$1"; }

# Ready builds are made on Ubuntu 22.04 and need glibc 2.35 or newer (Ubuntu 22.04+,
# Debian 12+, recent Armbian). Older systems build from source automatically.
glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{ print $2 }')
if [ "$from_source" -eq 0 ] && [ -n "$glibc" ]; then
    if [ "$(printf '%s\n2.35\n' "$glibc" | sort -V | head -1)" != "2.35" ]; then
        echo "This system has glibc $glibc, the ready build needs 2.35 or newer. Building from source instead."
        from_source=1
    fi
fi

mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

if [ "$from_source" -eq 0 ]; then
    # 2a. download the ready build and check it
    if [ -n "$version" ]; then
        base="https://github.com/$repo/releases/download/$version"
    else
        base="https://github.com/$repo/releases/latest/download"
    fi
    asset="ggml-rknpu2-linux-arm64.tar.gz"

    step "Downloading ggml-rknpu2 ${version:-(latest release)}"
    fetch "$base/$asset" "$tmp/$asset"
    fetch "$base/$asset.sha256" "$tmp/$asset.sha256"
    (cd "$tmp" && sha256sum -c "$asset.sha256" >/dev/null) || die "the download is damaged (checksum mismatch), please run the installer again"

    tar -xzf "$tmp/$asset" -C "$tmp"
    rm -rf "${prefix:?}/bin"
    cp -a "$tmp/ggml-rknpu2/." "$prefix/"
else
    # 2b. build on the board
    missing=""
    for t in git cmake c++ make; do
        command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
    done
    # OpenSSL headers: without them the build works but model downloads (-hf) do not
    if command -v dpkg >/dev/null 2>&1 && ! dpkg -s libssl-dev >/dev/null 2>&1; then
        missing="$missing libssl-dev"
    fi
    if [ -n "$missing" ]; then
        command -v apt-get >/dev/null 2>&1 || die "missing build tools:$missing. Install git, cmake and a C++ compiler, then run this again."
        step "Installing build tools (git, cmake, compiler), this asks for your password"
        sudo apt-get update
        sudo apt-get install -y git cmake build-essential libssl-dev
    fi

    ref=${version:-}
    if [ -z "$ref" ]; then
        ref=$(curl -fsSL "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null \
              | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
        ref=${ref:-main}
    fi

    step "Downloading the ggml-rknpu2 source ($ref)"
    case "$ref" in
        v*) url="https://github.com/$repo/archive/refs/tags/$ref.tar.gz" ;;
        *)  url="https://github.com/$repo/archive/refs/heads/$ref.tar.gz" ;;
    esac
    fetch "$url" "$tmp/src.tar.gz"
    mkdir "$tmp/src"
    tar -xzf "$tmp/src.tar.gz" -C "$tmp/src" --strip-components 1

    step "Building (10 to 40 minutes, the board stays usable)"
    build_args=(--prefix "$prefix" --work "$HOME/.cache/ggml-rknpu2" --no-bundle-runtime)
    [ "$system_runtime" -eq 1 ] && build_args+=(--system-runtime)
    [ -n "$runtime_version" ] && build_args+=(--runtime-version "$runtime_version")
    bash "$tmp/src/scripts/build.sh" "${build_args[@]}"
fi

# 3. Rockchip's runtime, straight from Rockchip. Kept next to the programs, which find
# it there first, so an older copy in /usr/lib does not get in the way.
if [ "$system_runtime" -eq 0 ]; then
    # shellcheck disable=SC1091
    . "$prefix/versions.env"
    rv=${runtime_version:-$RKNN_RUNTIME_VERSION}
    step "Downloading the Rockchip NPU runtime $rv from github.com/airockchip/rknn-toolkit2"
    echo "This file is Rockchip's and comes under Rockchip's licence, see THIRD_PARTY.md."
    fetch "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v$rv/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" "$tmp/librknnrt.so"
    install -m 0644 "$tmp/librknnrt.so" "$prefix/bin/librknnrt.so"
else
    rm -f "$prefix/bin/librknnrt.so"
fi

# 4. every program and library has to load. Minimal images sometimes lack the OpenMP
# or OpenSSL runtime, say exactly what to install instead of failing later.
missing_libs=$(for f in "$prefix"/bin/llama-cli "$prefix"/bin/*.so; do ldd "$f" 2>/dev/null; done \
               | awk '/not found/ { print $1 }' | sort -u | tr '\n' ' ')
if [ -n "$missing_libs" ]; then
    pkgs=""
    for l in $missing_libs; do
        case "$l" in
            libgomp*)           pkg=libgomp1 ;;
            libssl*|libcrypto*) pkg=libssl3 ;;
            libstdc++*)         pkg=libstdc++6 ;;
            *)                  pkg="" ;;
        esac
        [ -n "$pkg" ] && case " $pkgs " in *" $pkg "*) ;; *) pkgs="$pkgs $pkg" ;; esac
    done
    echo
    echo "Some system libraries are missing: $missing_libs"
    [ -n "$pkgs" ] && echo "Install them with:  sudo apt install$pkgs"
fi

# 5. the llama-npu command
mkdir -p "$HOME/.local/bin"
ln -sf "$prefix/llama-npu" "$HOME/.local/bin/llama-npu"

case ":$PATH:" in
    *":$HOME/.local/bin:"*) on_path=1 ;;
    *) on_path=0 ;;
esac
if [ "$on_path" -eq 0 ] && [ "$edit_path" -eq 1 ]; then
    # shellcheck disable=SC2016  # written to .bashrc unexpanded on purpose
    line='export PATH="$HOME/.local/bin:$PATH"  # added by ggml-rknpu2'
    grep -qF "$line" "$HOME/.bashrc" 2>/dev/null || printf '\n%s\n' "$line" >> "$HOME/.bashrc"
fi

# 6. tell the user where they stand
step "Checking the board"
sh "$prefix/check.sh" || true

step "Installed ggml-rknpu2 $(cat "$prefix/VERSION") in $prefix"
if [ "$on_path" -eq 0 ]; then
    echo "Open a new terminal (or run: source ~/.bashrc), then:"
else
    echo "Next:"
fi
cat <<EOF

  llama-npu chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0     chat in the terminal
  llama-npu serve Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0    chat in your browser
  llama-npu help                                           everything else

The first run downloads the model (about 700 MB). Guide: https://github.com/$repo/wiki
EOF
