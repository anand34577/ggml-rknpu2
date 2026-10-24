# Changelog

All notable changes to ggml-rknpu2. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-10-24

First public release.

### Added
- **NPU backend for llama.cpp** on RK3588, RK3576, RK3566, RK3568 and RK3562. Prompt
  processing (batches of 128 tokens or more) runs on the NPU, answer generation stays on the
  CPU. Large matrix multiplications are split across all NPU cores (3 on RK3588, 2 on RK3576).
- **Works with any GGUF model.** Quantized weights (Q4_0, Q4_K, Q8_0 and the rest) are unpacked
  to 16-bit for the NPU once, when the model loads.
- **Never changes the answer.** Shapes the NPU can't take, weights that don't fit and runtime
  errors all fall back to the CPU.
- **One-line installer** for the board. Downloads a ready build (Ubuntu 22.04, Debian 12 or
  newer) or builds from source on older systems, and fetches Rockchip's NPU runtime from
  Rockchip. No root needed.
- **`llama-npu` command**: `chat`, `serve`, `bench`, `check`, `devices`, `update` and
  `uninstall`. Takes a `.gguf` file or a Hugging Face model name, and sets the llama.cpp
  options the NPU needs.
- **`llama-npu check`** finds the chip, NPU driver, permissions and runtime, and says how to fix
  what's missing.
- Settings through environment variables: NPU memory cap, cores, batch sizes, chip override.
- Built on llama.cpp `d89651a` and Rockchip runtime 2.3.2.

[Unreleased]: https://github.com/anand34577/ggml-rknpu2/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anand34577/ggml-rknpu2/releases/tag/v0.1.0
