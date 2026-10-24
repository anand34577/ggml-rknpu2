#!/bin/sh
# Check whether this board can run the RKNPU2 backend, and say how to fix what is missing.
# Run on the board:  sh check.sh   (or: llama-npu check)

ok=0
fail=0
here=$(cd "$(dirname "$0")" && pwd)

say()  { printf '%-22s %s\n' "$1" "$2"; }
good() { say "$1" "OK    $2"; ok=$((ok+1)); }
bad()  { say "$1" "FIX   $2"; fail=$((fail+1)); }
info() { say "$1" "      $2"; }

echo "ggml-rknpu2 board check"
echo

# 1. which chip
compat=$(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' ')
model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
soc=""
for c in rk3588 rk3576 rk3566 rk3568 rk3562; do
    case "$compat" in *"$c"*) soc=$c ;; esac
done
if [ -n "$soc" ]; then
    good "Chip" "$soc${model:+  ($model)}"
else
    bad "Chip" "not a supported Rockchip chip (RK3588, RK3576, RK3566, RK3568, RK3562)"
    [ -n "$compat" ] && echo "                       device tree says: $compat"
fi

info "Kernel" "$(uname -r)"
info "Memory" "$(awk '/MemTotal/ { printf "%.1f GB", $2 / 1048576 }' /proc/meminfo)"

# 2. the NPU driver. Vendor kernels expose it as /dev/rknpu or as a DRM device under
# /dev/dri; librknnrt.so handles either.
node=""
if [ -e /dev/rknpu ]; then
    node=/dev/rknpu
elif [ -d /dev/dri/by-path ]; then
    node=$(find /dev/dri/by-path -iname '*npu*render*' 2>/dev/null | head -1)
    [ -z "$node" ] && node=$(find /dev/dri/by-path -iname '*npu*card*' 2>/dev/null | head -1)
    [ -n "$node" ] && node=$(readlink -f "$node")
fi

if [ -n "$node" ]; then
    good "NPU driver" "$node"
    if [ -r "$node" ] && [ -w "$node" ]; then
        good "NPU access" "allowed for $(id -un)"
    else
        bad "NPU access" "$(id -un) cannot open $node"
        grp=$(stat -c %G "$node" 2>/dev/null)
        if [ -n "$grp" ] && [ "$grp" != "root" ]; then
            echo "                       Fix: sudo usermod -aG $grp $(id -un)   then log out and back in"
        else
            echo "                       Fix: echo 'KERNEL==\"$(basename "$node")\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/99-rknpu.rules"
            echo "                            sudo udevadm control --reload && sudo udevadm trigger"
            echo "                       (or run llama-npu with sudo)"
        fi
    fi
else
    bad "NPU driver" "no /dev/rknpu and no NPU under /dev/dri"
    echo "                       Rockchip's runtime needs the rknpu driver from Rockchip's vendor"
    echo "                       kernel (the mainline 'rocket' driver does not work with it)."
    echo "                       On Armbian: sudo armbian-config -> System -> Kernel, pick a"
    echo "                       'vendor' or 'legacy' kernel, reboot, run this again."
fi

drv=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null)
[ -n "$drv" ] && info "Driver version" "$drv"

# 3. the Rockchip runtime: the copy installed with llama-npu wins over the system one
lib=""
for f in "$here/bin/librknnrt.so" "$here/lib/librknnrt.so"; do
    [ -e "$f" ] && lib=$f && break
done
[ -z "$lib" ] && lib=$(ldconfig -p 2>/dev/null | grep -m1 'librknnrt\.so ' | awk '{ print $NF }')

if [ -n "$lib" ]; then
    ver=$(strings "$lib" 2>/dev/null | grep -m1 -oE 'librknnrt version: [0-9][^ ]*' | sed 's/librknnrt version: //')
    good "NPU runtime" "$lib${ver:+  (version $ver)}"
    case "$ver" in
        1.*) echo "                       Version 1.x is old. RK3576 needs 2.0 or newer; reinstall llama-npu." ;;
    esac
else
    bad "NPU runtime" "librknnrt.so not found. Reinstall llama-npu, it downloads it for you."
fi

# 4. NPU clock. Not a blocker: the default governor ramps up slowly on short bursts,
# which costs some speed. Only reported.
for d in /sys/class/devfreq/*npu*; do
    [ -d "$d" ] || continue
    gov=$(cat "$d/governor" 2>/dev/null)
    cur=$(cat "$d/cur_freq" 2>/dev/null)
    max=$(cat "$d/max_freq" 2>/dev/null)
    if [ -n "$cur" ] && [ -n "$max" ]; then
        info "NPU clock" "$((cur / 1000000)) MHz now, $((max / 1000000)) MHz max, governor $gov"
    fi
    if [ "$gov" != "performance" ]; then
        echo "                       Tip, for steady benchmark numbers:"
        echo "                       echo performance | sudo tee $d/governor"
    fi
    break
done

# 5. the final word: does llama.cpp itself see the NPU
if [ -x "$here/bin/llama-cli" ]; then
    dev=$("$here/bin/llama-cli" --list-devices 2>/dev/null | grep -m1 'RKNPU2')
    if [ -n "$dev" ]; then
        good "llama.cpp" "can use $(echo "$dev" | sed 's/^ *//; s/ (0 MiB.*//')"
    else
        bad "llama.cpp" "does not see the NPU yet, fix the lines above first"
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "All good. Try:  llama-npu chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0"
else
    echo "$fail thing(s) to fix above. llama-npu still works without the NPU, just slower."
    echo "Help: https://github.com/anand34577/ggml-rknpu2/wiki/Troubleshooting"
fi
