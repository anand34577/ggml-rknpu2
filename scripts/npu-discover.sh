#!/bin/sh
# Find whatever NPU access path already works on this board, regardless of which
# Rockchip runtime it belongs to. Run on the board: sh npu-discover.sh

echo "== device nodes =="
find /dev -iname '*npu*' -o -iname '*rknn*' -o -iname '*rkllm*' -o -iname '*mali*' 2>/dev/null
ls -l /dev/dri/ 2>/dev/null

echo
echo "== kernel modules =="
lsmod | grep -iE 'rknpu|rkllm|npu'
find /lib/modules/"$(uname -r)" -iname '*npu*' 2>/dev/null
find /lib/modules/"$(uname -r)" -iname '*rknn*' 2>/dev/null

echo
echo "== dmesg =="
dmesg 2>/dev/null | grep -iE 'rknpu|rkllm|npu' | tail -20

echo
echo "== device tree NPU node =="
find /proc/device-tree -iname '*npu*' 2>/dev/null
find /sys/firmware/devicetree/base -iname '*npu*' 2>/dev/null

echo
echo "== installed Rockchip NPU libraries =="
ldconfig -p | grep -iE 'rknn|rkllm'
find / -xdev \( -iname 'librknnrt.so*' -o -iname 'librkllmrt.so*' \) 2>/dev/null

echo
echo "== debugfs =="
ls /sys/kernel/debug/rknpu 2>/dev/null
cat /sys/kernel/debug/rknpu/version 2>/dev/null

echo
echo "== was the NPU used before, from shell history or known apps =="
which rkllm_server 2>/dev/null
find / -xdev -iname '*.rkllm' 2>/dev/null | head -5
find / -xdev -iname 'rkllama*' -not -path '*/node_modules/*' 2>/dev/null | head -10

echo
echo "== paste all of the above back =="
