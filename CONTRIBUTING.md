# Contributing to ggml-rknpu2

Thanks for helping. The goal is simple: llama.cpp on a Rockchip board should use the NPU without
the user having to know anything about NPUs. Please keep changes in that spirit.

## The most useful things you can do

- **Test on your board** and report what you see: chip, board, OS image, kernel, and the
  output of `llama-npu check` and `llama-npu bench`. Post it in
  [Discussions](https://github.com/anand34577/ggml-rknpu2/discussions). Results from RK3576,
  RK3566/RK3568 and RK3562 boards are especially welcome.
- **Report bugs** with the [bug template](https://github.com/anand34577/ggml-rknpu2/issues/new/choose).
- For anything bigger than a bug fix, open an issue first so we can agree on the approach.
- Security problems: see [SECURITY.md](SECURITY.md), not the issue tracker.

## Layout

| Path | What |
|---|---|
| `backend/` | The ggml backend: `ggml-rknpu2.cpp`, its header and CMake file |
| `patches/llama.cpp.patch` | Registers the backend with llama.cpp (keep it small) |
| `versions.env` | Pinned llama.cpp commit and Rockchip runtime version |
| `scripts/build.sh` | Fetches llama.cpp, adds the backend, builds |
| `scripts/llama-npu`, `scripts/check.sh` | The user-facing command and board check |
| `install.sh` | The one-line installer |
| `tests/` | Stand-in NPU runtime, correctness tests, end-to-end test |
| `tools/` | On-board benchmarks: `bench-rknpu2.cpp`, `probe-cores.cpp` |
| `docs/wiki/` | The wiki. It's published from here, don't edit it on GitHub |

## Before you send a pull request

Run the tests. They work on any Linux PC or WSL, x86 or arm, without a board:

```sh
tests/run.sh      # backend against a strict stand-in for the NPU, all five chips
tests/e2e.sh      # a small model with and without the NPU path, answers must match
shellcheck -x install.sh scripts/*.sh scripts/llama-npu tests/*.sh
```

The stand-in (`tests/rknnrt_stub.cpp`) behaves like real hardware where it matters: memory is
not coherent until synced, weights really are stored in the NPU layout, and each chip's shape
limits are enforced. If you change how the backend talks to the runtime, add a case there.

If you have a board, also run `llama-npu bench` before and after your change and put both
results in the PR.

## Updating llama.cpp

1. Change `LLAMA_CPP_COMMIT` in `versions.env`.
2. Run `scripts/build.sh --prepare-only`. If the patch no longer applies, fix it in the
   downloaded copy and regenerate it with `git diff` there.
3. Run both test scripts and note the new commit in `CHANGELOG.md`.

## Style

- Match the code around you. The backend follows llama.cpp's style: 4 spaces, `snake_case`,
  short comments that say why, not what.
- Shell scripts must pass `shellcheck`.
- Messages users see should say what happened and what to do next.
- Commit messages: a short imperative subject (`Split K for wide weights on RK3576`), and a
  body that explains why when it isn't obvious.

## Licensing

By contributing you agree that your work is released under the MIT licence of this project.
Never add Rockchip's files (`librknnrt.so`, `rknn_api.h`, `rknn_matmul_api.h`) to the
repository: they are not ours to redistribute. The scripts download them from Rockchip.
