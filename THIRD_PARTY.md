# Third-party software

ggml-rknpu2 itself (the code in this repository) is MIT licensed, see [LICENSE](LICENSE).
It builds on, and is used together with, the software below. None of it is copied into
this repository.

## llama.cpp and ggml

- Project: https://github.com/ggml-org/llama.cpp
- Licence: MIT, Copyright (c) 2023-2026 The ggml authors
- How it is used: the build script downloads one pinned commit (see `versions.env`), adds
  the backend and applies `patches/llama.cpp.patch`. Release packages contain programs built
  from that source; its licence is included as `LICENSE-llama.cpp`.

## Rockchip RKNN runtime (librknnrt.so, rknn_api.h, rknn_matmul_api.h)

- Project: https://github.com/airockchip/rknn-toolkit2
- Licence: Rockchip's own terms, see the `LICENSE` file in that repository.
- How it is used: **not redistributed.** The installer and build script download these
  files directly from Rockchip's repository onto your machine, and release packages do not
  contain them. By installing, you obtain them from Rockchip under Rockchip's terms.

## Models

Models are not part of this project. When you name a model on Hugging Face, llama.cpp
downloads it from there. Each model has its own licence; check it on the model's page
before you use it.

## Trademarks

Rockchip, RKNN and RKNPU are trademarks of Rockchip Electronics Co., Ltd. Llama is a
trademark of Meta Platforms, Inc. This project is independent and is not affiliated with,
endorsed by or sponsored by Rockchip, Meta, or the ggml / llama.cpp maintainers. The
names are used only to say what this software works with.
