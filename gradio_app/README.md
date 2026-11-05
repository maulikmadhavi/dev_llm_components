# gradio_app

Backend for the React Flow canvas (`../reactflow_app`) and a Gradio pipeline builder, served together.

```
pixi run dev     # http://127.0.0.1:7860  (Gradio)  and  /flow/  (canvas, once built)
pixi run test    # unit tests for the chat assistant (no model server needed)
```

The nodes call an OpenAI-compatible model server, LM Studio by default (`lms server start --cors`).

## Chat assistant (`POST /api/chat`)

The canvas has a chat panel where an agent picks nodes, connects, configures and arranges them, and reviews
the flow. The model only proposes edits; the server validates each one against the node specs before the
canvas applies it.

Configure it with environment variables on the process that runs `pixi run dev`. All are optional.

| Variable | Default | What it does |
| --- | --- | --- |
| `AGENT_BASE_URL` | `http://localhost:1234/v1` | The model server the assistant talks to. This is the **only** URL `/api/chat` accepts, unless `AGENT_ALLOW_CUSTOM_BASE_URL` is set. |
| `AGENT_MODEL` | `qwen/qwen3-4b` | Model name sent to that server. Clients may override it per request. |
| `AGENT_API_KEY` | none | Bearer token for `AGENT_BASE_URL`. It is only ever sent to that URL, never to a client-chosen one. |
| `AGENT_ALLOW_CUSTOM_BASE_URL` | off | Set to `1`, `true` or `yes` to let clients pick any other `http(s)` model server from the panel's "Assistant model" box. See the warning below. |
| `AGENT_TIMEOUT` | `120` | Seconds to wait for one model reply. Clamped to 5-600; a non-numeric or non-finite value falls back to 120. |

Setting them:

```bash
# bash
AGENT_BASE_URL=http://llm.internal:8000/v1 AGENT_API_KEY=... AGENT_TIMEOUT=60 pixi run dev
```
```powershell
# PowerShell
$env:AGENT_BASE_URL = "http://llm.internal:8000/v1"; $env:AGENT_TIMEOUT = "60"; pixi run dev
```

### `AGENT_ALLOW_CUSTOM_BASE_URL`

Leave it **off** on any server that others can reach. With it on, anyone who can call `/api/chat` can make the
server send requests to an arbitrary `http(s)` host, including services on the server's own network
(for example `http://169.254.169.254/` or `http://10.x.x.x`). Turn it on only for a single-user machine you
trust, such as a laptop bound to `127.0.0.1`.

When it is off and a client asks for another URL, the request fails with HTTP 400 before any outbound call.
`AGENT_API_KEY` is still withheld from custom URLs when it is on; a client must send its own key.

### `AGENT_TIMEOUT`

Each chat turn makes one blocking model call in a server worker thread. The browser cancels its own request
on **New chat** or when the panel is closed, but the server call is not interrupted: it runs until the model
replies or `AGENT_TIMEOUT` passes. Keep it short enough that abandoned chats free their threads quickly, and
long enough for your model to finish about 3000 tokens (small CPU-only models can need more than 120).
On timeout the user sees "The model server took too long to reply".

### Errors and logs

To avoid turning the endpoint into a network probe, `/api/chat` returns generic model-server errors
("Could not get a reply from the model server…"). The real cause (connection refused, HTTP status, bad
response) is written to the server's log at WARNING level, so check there when the panel reports a failure.

### Not covered by these settings

The node endpoints (`POST /api/run`, used by the LLM/VLM/detector nodes) read `base_url` from each node's own
settings and return detailed errors. They are meant for a trusted local setup and are not restricted by
`AGENT_BASE_URL`; do not expose this server to untrusted clients without putting access control in front of it.
