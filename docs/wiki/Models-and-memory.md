# Models and memory

## Good first models

All of these are free to use (Apache 2.0):

| Model name to type | Download | Board RAM | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0` | 675 MB | 4 GB+ | Fast, fine for simple tasks |
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` | 1.1 GB | 8 GB+ | Noticeably smarter |
| `Qwen/Qwen3-1.7B-GGUF:Q8_0` | 1.8 GB | 8 GB+ | Thinks before it answers |
| `Qwen/Qwen3-4B-GGUF:Q4_K_M` | 2.5 GB | 8 GB+ | Best answers of these, slower |

Any other GGUF model works too. Check its licence on its Hugging Face page.

## Why the NPU needs extra memory

The NPU computes with 16-bit numbers. Most model files are squeezed to 4 or 8 bits, so the
backend unpacks the weights once, when the model loads, and keeps that copy for the NPU. The
copy is about **3.5 times** the size of a 4-bit file, or **2 times** an 8-bit one.

So a model needs roughly *file size + NPU copy + about 0.5 GB*. The NPU copy is capped at
**3 GB**, because the NPU can't address more than 4 GB. Whatever doesn't fit stays on the CPU,
so big models still run; only part of them gets the NPU.

| Model | File | NPU copy | Total, roughly |
|---|---|---|---|
| Qwen2.5 0.5B Q8_0 | 0.7 GB | 1 GB | 2.2 GB |
| Qwen2.5 1.5B Q4_K_M | 1.1 GB | 3 GB (capped) | 4.6 GB |
| Qwen3 4B Q4_K_M | 2.5 GB | 3 GB (capped) | 6 GB |

On a 4 GB board, stay with the 0.5B model, or lower the cap:
`GGML_RKNPU2_MEM_MB=1024 llama-npu chat MODEL`.

## Where models are kept

Downloaded models live in `~/.cache/llama.cpp`. `du -sh ~/.cache/llama.cpp` shows how much
space they take; delete a model by deleting its files there.

## Using your own files

Give the path instead of a name: `llama-npu chat ~/models/model.gguf`.
