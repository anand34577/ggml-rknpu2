## Install or update

On your Rockchip board:

```sh
curl -fsSL https://raw.githubusercontent.com/anand34577/ggml-rknpu2/main/install.sh | bash
```

Already installed? Run `llama-npu update`.

| File | What it is |
|---|---|
| `ggml-rknpu2-linux-arm64.tar.gz` | The ready build the installer uses: llama.cpp with the NPU backend, for RK3588, RK3576, RK3566, RK3568 and RK3562. Needs Ubuntu 22.04, Debian 12 or newer. |
| `ggml-rknpu2-linux-arm64.tar.gz.sha256` | Checksum of the package |
| `install.sh` | The installer, the same as the command above |

Rockchip's NPU runtime is not in the package. The installer downloads it from
[Rockchip's repository](https://github.com/airockchip/rknn-toolkit2).

Step-by-step guides: **[wiki](https://github.com/anand34577/ggml-rknpu2/wiki)** ·
changes: [CHANGELOG.md](https://github.com/anand34577/ggml-rknpu2/blob/main/CHANGELOG.md)
