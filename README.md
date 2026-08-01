# plap

plap is an OpenAI Responses-compatible server for building plugin-driven model workflows.

It accepts normal Responses API requests, runs them through a configurable model loop, and lets Python plugins add new
capabilities or modify response execution through hooks.

## Start plap

You need [Pixi](https://pixi.sh/) and Docker.

Create your local environment file:

```sh
cp .env.example .env
```

Add an OpenRouter key to `.env`:

```dotenv
OPENROUTER_API_KEY=your-key
```

Start the development server:

```sh
pixi run dev
```

This starts temporary PostgreSQL and telemetry containers, applies migrations, creates a development API key, and runs
the server. The command prints the active model, URLs, and log path.

Keep it running. In another terminal, load the generated client settings:

```sh
source .dev/.env
```

## Send a response

The OpenAI Python client is already installed in the Pixi environment:

```sh
pixi run python - <<'PY'
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["PLAP_DEV_BASE_URL"],
    api_key=os.environ["PLAP_DEV_API_KEY"],
)

response = client.responses.create(
    model=os.environ["PLAP_DEV_MODEL"],
    input="Say hello in one sentence.",
)

print(response.output_text)
PY
```

You now have a working local plap server.

## What happens to a request?

1. plap reads the Responses request into response state.
2. The model loop runs completions, plugins, and server tools.
3. Commit saves the final state and publishes Responses output.

## Add something

[Write your first plugin](docs/first-plugin.md) to add a `server_time` tool that the model can call.

The [documentation index](docs/README.md) covers the event bus, server tools, hooks, reasoning summaries, state, separate
model contexts, and the lower-level LLM library.

## Development commands

```sh
pixi run setup
pixi run pytest tests/unit
pixi run ruff check src tests scripts
pixi run ruff format --check src tests scripts
```

Tests marked `money` or `expensive` may call live providers and use credentials from root `.env`.
