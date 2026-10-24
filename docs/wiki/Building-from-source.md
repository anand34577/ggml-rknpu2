# Building from source

You only need this to change the code. The [installer](Install) builds from source by itself on
older systems.

## On the board

```sh
sudo apt install git cmake build-essential libssl-dev curl
git clone https://github.com/anand34577/ggml-rknpu2.git
cd ggml-rknpu2
scripts/build.sh
dist/ggml-rknpu2/llama-npu check
```

`scripts/build.sh` downloads the llama.cpp version pinned in `versions.env`, adds the backend,
downloads Rockchip's runtime and builds, tuned for the board's CPU. It takes 10 to 40 minutes.
`scripts/build.sh --help` lists the options, for example `--portable` for one build that runs on
every supported chip, or `--system-runtime` to use `/usr/lib/librknnrt.so`.

## Tests, on any Linux PC

No board needed. The tests replace Rockchip's runtime with a strict stand-in:

```sh
tests/run.sh      # backend correctness on all five chips
tests/e2e.sh      # a small model with and without the NPU path; the answers must match
```

## Board tools

`tools/` has two programs that measure the NPU directly. Build them on the board after
`scripts/build.sh`; the comment at the top of each file says how.
