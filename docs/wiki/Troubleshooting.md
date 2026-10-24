# Troubleshooting

Start with `llama-npu check`, then find the matching problem below.

## `llama-npu: command not found`

Open a new terminal, or run `source ~/.bashrc`. Still missing? Use the full path:
`~/.local/share/ggml-rknpu2/llama-npu`.

## Chip: not a supported Rockchip chip

The board isn't an RK3588, RK3576, RK3566, RK3568 or RK3562, or it runs a 32-bit system.
llama-npu still works, on the CPU only.

## NPU driver: no /dev/rknpu and no NPU under /dev/dri

The kernel has no Rockchip NPU driver. Most board makers' own images include it. On Armbian:
`sudo armbian-config` → System → Kernel → pick a **vendor** (or legacy) kernel, then reboot.
Mainline kernels, including their new `rocket` NPU driver, don't work with Rockchip's runtime.

## NPU access: cannot open ...

Your user isn't allowed to use the NPU. The check prints the exact fix; usually it's:

```sh
sudo usermod -aG render $USER     # then log out and back in
```

## llama.cpp: does not see the NPU yet

Fix the lines above it first. If they're all `OK`, run `llama-npu devices -v`, look for lines
mentioning `rknpu` or `rknn`, and open an issue with that output.

## NPU runtime: version 1.x is old

Run `llama-npu update`. It downloads runtime 2.3.2 next to the program.

## The model download fails

Check the board's internet: `curl -I https://huggingface.co`. Some models ask you to accept their
licence on Hugging Face first; they need a token: `export HF_TOKEN=your-token`.

## "Illegal instruction"

A build from one board was copied to a different chip. Use the installer, or build on the board
you run it on.

## "Killed", or the board freezes while loading

Out of memory. Choose a smaller model or shrink the NPU copy:
`GGML_RKNPU2_MEM_MB=1024 llama-npu chat MODEL`. See [Models and memory](Models-and-memory).

## It's not faster

- Short prompts mostly measure answer-writing, which runs on the CPU either way.
- Only prompt chunks of 128 tokens or more go to the NPU.
- Check the NPU clock: see [Speed and settings](Speed-and-settings).

## Still stuck?

[Open an issue](https://github.com/anand34577/ggml-rknpu2/issues/new/choose) with the output of
`llama-npu check` and `llama-npu version`.
