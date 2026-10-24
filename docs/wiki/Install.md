# Install

This takes about two minutes. You need the board switched on and connected to the internet, and
a terminal on it: either a keyboard and screen plugged in, or SSH from your computer.

## 1. Open a terminal on the board

Over SSH from your computer (Windows, macOS and Linux all have `ssh` built in):

```sh
ssh your-user@board-ip
```

Don't know the board's address? Your router's device list shows it, or run `hostname -I` on the
board itself.

## 2. Run the installer

Copy this line, paste it into the terminal and press Enter:

```sh
curl -fsSL https://raw.githubusercontent.com/anand34577/ggml-rknpu2/main/install.sh | bash
```

What it does, in order:

1. Downloads the ready-made program package and checks it isn't damaged.
2. Downloads Rockchip's NPU runtime from Rockchip's own GitHub page.
3. Adds the `llama-npu` command.
4. Checks your board and prints what it found.

Everything goes into `~/.local/share/ggml-rknpu2` in your home folder. It doesn't need your
password and doesn't touch the rest of the system.

> **Older system?** On Ubuntu older than 22.04 or Debian older than 12, the installer builds the
> program on the board instead. That takes 10 to 40 minutes and asks for your password once, to
> install the build tools. You can keep using the board meanwhile.

## 3. Read the check at the end

The installer ends with a check like this:

```
Chip                   OK    rk3588  (Radxa ROCK 5B)
NPU driver             OK    /dev/dri/renderD129
NPU access             OK    allowed for radxa
NPU runtime            OK    /home/radxa/.local/share/ggml-rknpu2/bin/librknnrt.so  (version 2.3.2)
llama.cpp              OK    can use RKNPU2: Rockchip NPU (rk3588)
```

All `OK`? You're done. A line saying `FIX` tells you what to do, and
[Troubleshooting](Troubleshooting) explains each one. Even with a `FIX` line everything works,
just on the CPU.

## 4. Open a new terminal

Close the terminal and open a new one (or run `source ~/.bashrc`) so it finds the new
`llama-npu` command. Then go to **[Your first chat](Your-first-chat)**.

## Installer options

Add them after `bash -s --`, for example
`curl -fsSL .../install.sh | bash -s -- --from-source`.

| Option | What it does |
|---|---|
| `--from-source` | Build on the board instead of downloading the ready build |
| `--version v0.1.0` | Install a specific release |
| `--prefix DIR` | Install somewhere else |
| `--system-runtime` | Use the Rockchip runtime already installed in `/usr/lib` |
| `--runtime-version 2.5.0` | Download a different Rockchip runtime version |
| `--no-path` | Don't add `~/.local/bin` to your PATH |
