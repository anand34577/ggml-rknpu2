# Speed and settings

## Measure your board

```sh
llama-npu bench Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
```

You get a table. The columns that matter:

- `rpk 0` rows use the NPU, `rpk 1` rows are the CPU alone at its best.
- `pp512`: reading a 512-token prompt, in tokens per second. **This is what the NPU speeds up.**
- `tg64`: writing 64 tokens. The CPU does this either way, so expect similar numbers.

For steady numbers, run the NPU at full clock first (this resets at reboot):

```sh
echo performance | sudo tee /sys/class/devfreq/*npu*/governor
```

Please share your table in
[Discussions](https://github.com/anand34577/ggml-rknpu2/discussions).

## Why llama-npu adds `--no-repack`

On chips with fast CPU instructions (RK3588), llama.cpp normally rearranges ("repacks") the
weights into a layout only the CPU can read, and the NPU then gets almost nothing. `llama-npu`
turns that off. The trade-off: repacking makes answer-writing a little faster on the CPU. If you
only ever ask short questions, compare both with `llama-npu bench`, and if repacking wins, use
plain llama.cpp: `llama-npu run llama-cli -m MODEL`.

## Settings

Put them before the command, for example `GGML_RKNPU2_CORES=1 llama-npu chat MODEL`. To keep a
setting, add a line like `export GGML_RKNPU2_MEM_MB=2048` to `~/.bashrc`.

| Setting | Default | What it changes |
|---|---|---|
| `GGML_RKNPU2_MEM_MB` | `3072` | Most memory for the NPU copy of the weights, in MB (about 4000 at most) |
| `GGML_RKNPU2_CORES` | all | NPU cores used (3 on RK3588, 2 on RK3576) |
| `GGML_RKNPU2_MIN_M` | `128` | Shortest prompt chunk sent to the NPU, in tokens |
| `GGML_RKNPU2_MAX_M` | `512` | Longest chunk the NPU takes at once |
| `GGML_RKNPU2_SOC` | detected | Force the chip, e.g. `rk3576` |
| `GGML_RKNPU2_F32_WEIGHTS` | off | Also send 32-bit weights to the NPU (computed in 16-bit) |

## See what the NPU is doing

Add `-v` to any command for a detailed log. Lines starting with `ggml_rknpu2_probe` show the
chip, the runtime version and how many NPU cores are used. As root,
`watch -n 0.5 cat /sys/kernel/debug/rknpu/load` shows each core's load live.
