# Your first chat

Type this and press Enter:

```sh
llama-npu chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0
```

The first time, it downloads the model (675 MB). That happens once: it's kept in
`~/.cache/llama.cpp` for next time. When you see a `>` prompt, type a question and press Enter.

Inside the chat:

| Type | To |
|---|---|
| `/clear` | Start a fresh conversation |
| `/regen` | Write the last answer again |
| `/exit` or `Ctrl+C` | Quit |

## What the name means

`Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0` is a model on [Hugging Face](https://huggingface.co):
the account (`Qwen`), the model page (`Qwen2.5-0.5B-Instruct-GGUF`), and after the colon the
version of the file (`Q8_0`). Any GGUF model on Hugging Face works this way, and so does a
`.gguf` file you already have:

```sh
llama-npu chat ~/models/my-model.gguf
```

[Models and memory](Models-and-memory) helps you pick a bigger, smarter model for your board.

## Is the NPU working?

```sh
llama-npu devices
```

It should list `RKNPU2: Rockchip NPU (rk3588)` (or your chip). If it doesn't, run
`llama-npu check`.

The NPU helps most with **long prompts**: a page of text you paste in, a long conversation, a
document to summarise. Short questions are mostly answer-writing, which the CPU does either way.

## Useful extras

Anything after the model name goes straight to llama.cpp:

```sh
llama-npu chat MODEL -sys "You are a helpful assistant who answers briefly."   # system prompt
llama-npu chat MODEL -c 8192                                                   # longer memory
llama-npu chat MODEL -v                                                        # detailed log
```
