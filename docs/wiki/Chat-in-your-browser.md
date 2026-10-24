# Chat in your browser

`llama-npu serve` starts a chat web page and an API on the board.

## On the board only

```sh
llama-npu serve Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0
```

Open `http://localhost:8080` in a browser on the board.

## From your computer or phone

```sh
llama-npu serve Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0 --host 0.0.0.0
```

Then open `http://board-ip:8080` on any device on the same network (`hostname -I` on the board
shows the address).

> Anyone on your network can now use it. To require a key, add `--api-key choose-a-secret`.
> Don't open port 8080 to the internet.

Stop it with `Ctrl+C`.

## Use it from other apps

The server speaks the same API as OpenAI, so most AI apps can use it. Point them at:

- **Base URL:** `http://board-ip:8080/v1`
- **API key:** anything (or the one you set with `--api-key`)
- **Model:** anything

Quick test from any computer:

```sh
curl http://board-ip:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## Start it automatically at boot

1. Create the file `~/.config/systemd/user/llama-npu.service` with this content (change the
   model if you like):

   ```ini
   [Unit]
   Description=llama-npu chat server
   After=network-online.target

   [Service]
   ExecStart=%h/.local/bin/llama-npu serve Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0 --host 0.0.0.0
   Restart=on-failure

   [Install]
   WantedBy=default.target
   ```

2. Turn it on:

   ```sh
   systemctl --user daemon-reload
   systemctl --user enable --now llama-npu
   sudo loginctl enable-linger "$USER"    # keeps it running when you log out
   ```

Check it with `systemctl --user status llama-npu`, follow its log with
`journalctl --user -u llama-npu -f`, and turn it off with
`systemctl --user disable --now llama-npu`.
