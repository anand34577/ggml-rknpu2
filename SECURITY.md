# Security

## Reporting a problem

Please don't open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/anand34577/ggml-rknpu2/security/advisories/new)
instead. You'll get a reply within a week. Once a fix is out, the advisory is published with
credit to you, unless you'd rather stay anonymous.

Problems in llama.cpp itself belong with the
[llama.cpp project](https://github.com/ggml-org/llama.cpp/security), and problems in
Rockchip's runtime with Rockchip.

## Supported versions

Only the latest release gets fixes. `llama-npu update` installs it.

## Things to know

- **`llama-npu serve` has no password by default.** It listens only on the board itself
  (`127.0.0.1`) unless you add `--host 0.0.0.0`. If you open it to your network, anyone on that
  network can use it. Add `--api-key YOUR-KEY` to require a key, and don't expose it to the
  internet without a reverse proxy that handles authentication.
- **Downloads are checked.** The installer verifies the release package against its SHA-256
  checksum. Rockchip's runtime comes from Rockchip's official GitHub repository over HTTPS.
- **Models are code-adjacent.** GGUF files are parsed by llama.cpp. Use models from sources you
  trust.
- **NPU permissions.** `llama-npu check` may suggest a udev rule that lets every user on the
  board open the NPU device. On a shared machine, prefer adding just your user to the device's
  group.
