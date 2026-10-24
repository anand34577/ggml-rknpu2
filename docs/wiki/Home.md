<p align="center"><img src="https://raw.githubusercontent.com/anand34577/ggml-rknpu2/main/docs/images/banner.png" alt="ggml-rknpu2" width="720"></p>

# ggml-rknpu2

ggml-rknpu2 lets llama.cpp, a popular program for running AI chat models on your own computer,
use the **NPU** in Rockchip chips. The NPU is a small part of the chip built for the kind of maths
these models do. With ggml-rknpu2 it reads your prompts while the CPU writes the answers.

You don't need to know anything about NPUs to use it. If you can copy a command into a terminal,
you can follow these guides.

## Start here

1. **[Install](Install)**: one command on your board.
2. **[Your first chat](Your-first-chat)**: talk to a model in the terminal.
3. **[Chat in your browser](Chat-in-your-browser)**: a chat page, and an API for other apps.

## When you need more

- [Models and memory](Models-and-memory): which model to pick for your board
- [Speed and settings](Speed-and-settings): measuring speed, changing the defaults
- [Troubleshooting](Troubleshooting): when something doesn't work
- [Updating and uninstalling](Updating-and-uninstalling)
- [Building from source](Building-from-source)

## Supported chips

RK3588 and RK3588S, RK3576, RK3566, RK3568 and RK3562, on 64-bit Linux with Rockchip's vendor
kernel (the kernel most board makers ship).
