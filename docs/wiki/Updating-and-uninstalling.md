# Updating and uninstalling

## Update

```sh
llama-npu update
```

This installs the latest release over the old one. Your models stay where they are.

## Uninstall

```sh
llama-npu uninstall
```

It asks before removing anything. Downloaded models stay in `~/.cache/llama.cpp`; delete that
folder too if you want the space back: `rm -rf ~/.cache/llama.cpp`.
