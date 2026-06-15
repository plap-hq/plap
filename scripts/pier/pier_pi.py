from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from pier.agents.installed.base import BaseInstalledAgent, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pier.models.trial.paths import EnvironmentPaths
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json

_PI_MODELS_CONFIG_PATH = Path(__file__).with_name("models.json")


def _load_plap_models_config(base_url: str) -> str:
    config = json.loads(_PI_MODELS_CONFIG_PATH.read_text())
    config["providers"]["plap"]["baseUrl"] = base_url
    return json.dumps(config, indent=2)


class PiAgent(BaseInstalledAgent):
    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "pi-output.jsonl"
    _DEFAULT_PROVIDER_DOMAINS: ClassVar[dict[str, list[str]]] = {
        "anthropic": ["api.anthropic.com"],
        "deepseek": ["api.deepseek.com"],
        "google": [".googleapis.com"],
        "groq": ["api.groq.com"],
        "mistral": ["api.mistral.ai"],
        "openai": ["api.openai.com"],
        "openrouter": ["openrouter.ai"],
        "xai": ["api.x.ai"],
    }

    @staticmethod
    def name() -> str:
        return "pi"

    def get_version_command(self) -> str | None:
        return '. "$HOME/.nvm/nvm.sh"; pi --version'

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip()

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"@{self._version}" if self._version else "@latest"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run="apt-get update && apt-get install -y curl fd-find ripgrep",
                ),
                InstallStep(
                    user="agent",
                    run=(
                        "set -euo pipefail; "
                        "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                        'export NVM_DIR="$HOME/.nvm" && '
                        '\\. "$NVM_DIR/nvm.sh" || true && '
                        "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                        "nvm install 22 && npm -v && "
                        f"npm install -g --ignore-scripts @earendil-works/pi-coding-agent{version_spec} && "
                        "pi --version"
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        urls = [
            self._get_env("OPENAI_BASE_URL"),
            self._get_env("OPENAI_API_BASE"),
            self._get_env("ANTHROPIC_BASE_URL"),
            self._get_env("GEMINI_API_BASE"),
            self._get_env("OPENROUTER_API_BASE"),
        ]
        provider = None
        if self.model_name and "/" in self.model_name:
            provider = self.model_name.split("/", 1)[0]
        return allowlist_from_urls(
            [url for url in urls if url],
            default_domains=self._DEFAULT_PROVIDER_DOMAINS.get(provider or "", []),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._last_instruction = instruction
        env = self.build_process_env(
            {
                "PI_OFFLINE": "1",
                "PI_SKIP_VERSION_CHECK": "1",
            }
        )
        for key in self._provider_env_keys():
            if value := self._get_env(key):
                env[key] = value

        if not self.model_name:
            raise ValueError("Model name is required for Pi agent")

        pi_model_name = self.model_name
        pi_agent_dir = PurePosixPath("/tmp/pi-agent")
        env["PI_CODING_AGENT_DIR"] = pi_agent_dir.as_posix()

        if self._needs_plap_models_config():
            env["PLAP_PI_BASE_URL"] = self._plap_base_url(env)
            pi_model_name = self._plap_pi_model_name()
            models_json = _load_plap_models_config(env["PLAP_PI_BASE_URL"])
            write_models_command = (
                f"mkdir -p {shlex.quote(pi_agent_dir.as_posix())} && "
                f"cat > {shlex.quote((pi_agent_dir / 'models.json').as_posix())} <<'EOF'\n"
                f"{models_json}\nEOF"
            )
            await self.exec_as_agent(environment, command=write_models_command, env=env)

        output_path = EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME
        command = (
            '. "$HOME/.nvm/nvm.sh"; '
            "mkdir -p /logs/agent && "
            "pi --mode json --no-session "
            "--tools read,write,edit,bash,grep,find,ls "
            f"--model {shlex.quote(pi_model_name)} {shlex.quote(instruction)} "
            f"2>&1 | stdbuf -oL tee {shlex.quote(str(output_path))}"
        )
        await self.exec_as_agent(environment, command=command, env=env)

    def _needs_plap_models_config(self) -> bool:
        if not self.model_name or "/" not in self.model_name:
            return False
        provider, model_id = self.model_name.split("/", 1)
        return provider == "openai" and model_id.startswith("plap-ai/")

    def _plap_base_url(self, env: dict[str, str]) -> str:
        base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL or OPENAI_API_BASE is required for Pi plap models")
        return base_url

    def _plap_pi_model_name(self) -> str:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("plap Pi model requires provider/model format")
        _, model_id = self.model_name.split("/", 1)
        return f"plap/{model_id}"

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_stdout()
        if not events:
            return
        trajectory = self._convert_events_to_trajectory(events)
        if trajectory is None:
            return
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(format_trajectory_json(trajectory.to_json_dict()))
        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)
        context.n_agent_steps = sum(1 for step in trajectory.steps if step.source == "agent")

    def _provider_env_keys(self) -> list[str]:
        provider = self.model_name.split("/", 1)[0] if self.model_name and "/" in self.model_name else "anthropic"
        return {
            "amazon-bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN"],
            "azure": ["AZURE_API_KEY", "AZURE_RESOURCE_NAME"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "google": [
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "GOOGLE_GENERATIVE_AI_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_GENAI_USE_VERTEXAI",
            ],
            "groq": ["GROQ_API_KEY"],
            "openai": ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "xai": ["XAI_API_KEY"],
        }.get(provider, [])

    def _parse_stdout(self) -> list[dict[str, Any]]:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for raw_line in output_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _convert_events_to_trajectory(self, events: list[dict[str, Any]]) -> Trajectory | None:
        if not events:
            return None

        session_id = None
        steps: list[Step] = []
        step_id = 1
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_cost = 0.0
        compaction_count = 0
        pending_assistant_messages: list[tuple[int, dict[str, Any]]] = []
        turn_end_timestamps: set[int] = set()
        saw_user_message = False

        for index, event in enumerate(events):
            event_type = event.get("type")
            if event_type == "session" and session_id is None:
                raw_session_id = event.get("id")
                if isinstance(raw_session_id, str) and raw_session_id:
                    session_id = raw_session_id
                continue

            if event_type == "compaction_end" and not event.get("aborted"):
                compaction_count += 1
                continue

            if event_type == "message_end":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                if role == "user":
                    saw_user_message = True
                    steps.append(
                        Step(
                            step_id=step_id,
                            timestamp=self._iso_timestamp(message.get("timestamp")),
                            source="user",
                            message=self._content_to_atif(message.get("content")),
                        )
                    )
                    step_id += 1
                elif role == "assistant":
                    pending_assistant_messages.append((index, message))
                continue

            if event_type == "turn_end":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                timestamp_value = message.get("timestamp")
                if isinstance(timestamp_value, int | float):
                    turn_end_timestamps.add(int(timestamp_value))
                step = self._agent_step_from_message(
                    step_id=step_id,
                    message=message,
                    tool_results=event.get("toolResults"),
                )
                steps.append(step)
                step_id += 1
                usage = self._usage_from_message(message)
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_cached_tokens += usage["cached_tokens"]
                total_cost += usage["cost_usd"]
                continue

        for _, message in pending_assistant_messages:
            timestamp_value = message.get("timestamp")
            if isinstance(timestamp_value, int | float) and int(timestamp_value) in turn_end_timestamps:
                continue
            step = self._agent_step_from_message(step_id=step_id, message=message, tool_results=None)
            steps.append(step)
            step_id += 1
            usage = self._usage_from_message(message)
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_cached_tokens += usage["cached_tokens"]
            total_cost += usage["cost_usd"]

        if not saw_user_message and getattr(self, "_last_instruction", None):
            steps.insert(
                0,
                Step(
                    step_id=1,
                    timestamp=None,
                    source="user",
                    message=self._last_instruction,
                ),
            )
            for index, step in enumerate(steps, start=1):
                step.step_id = index

        if not steps:
            return None

        final_metrics = FinalMetrics(
            total_prompt_tokens=total_prompt_tokens or None,
            total_completion_tokens=total_completion_tokens or None,
            total_cached_tokens=total_cached_tokens or None,
            total_cost_usd=total_cost if total_cost > 0 else None,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=compaction_count or None,
            ),
        )

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(name="pi", version=self.version() or "unknown", model_name=self.model_name),
            steps=steps,
            final_metrics=final_metrics,
            notes="Converted from Pi JSON mode output.",
        )

    def _agent_step_from_message(
        self,
        *,
        step_id: int,
        message: dict[str, Any],
        tool_results: Any,
    ) -> Step:
        text_parts = self._text_blocks_from_assistant(message)
        message_text = "\n\n".join(text_parts) if text_parts else (message.get("errorMessage") or "")
        reasoning_parts = self._thinking_blocks_from_assistant(message)
        tool_calls = self._tool_calls_from_assistant(message)
        observation = self._observation_from_tool_results(tool_results)
        usage = self._usage_from_message(message)

        metrics: Metrics | None = None
        if usage["prompt_tokens"] or usage["completion_tokens"] or usage["cached_tokens"]:
            extra = {"cache_write_tokens": usage["cache_write_tokens"]} if usage["cache_write_tokens"] else None
            metrics = Metrics(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                cached_tokens=usage["cached_tokens"] or None,
                cost_usd=usage["cost_usd"] or None,
                extra=extra,
            )

        extra: dict[str, Any] = {}
        for key in ("api", "provider", "responseId", "responseModel", "stopReason", "diagnostics"):
            value = message.get(key)
            if value is not None:
                extra[key] = value

        return Step(
            step_id=step_id,
            timestamp=self._iso_timestamp(message.get("timestamp")),
            source="agent",
            model_name=self.model_name,
            message=message_text,
            reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls or None,
            observation=observation,
            metrics=metrics,
            llm_call_count=1,
            extra=extra or None,
        )

    def _usage_from_message(self, message: dict[str, Any]) -> dict[str, int | float]:
        usage = message.get("usage")
        if not isinstance(usage, dict):
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.0,
            }
        input_tokens = self._as_int(usage.get("input"))
        output_tokens = self._as_int(usage.get("output"))
        cache_read_tokens = self._as_int(usage.get("cacheRead"))
        cache_write_tokens = self._as_int(usage.get("cacheWrite"))
        cost = usage.get("cost")
        total_cost = float(cost.get("total", 0.0)) if isinstance(cost, dict) else 0.0
        return {
            "prompt_tokens": input_tokens + cache_read_tokens,
            "completion_tokens": output_tokens,
            "cached_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": total_cost,
        }

    def _text_blocks_from_assistant(self, message: dict[str, Any]) -> list[str]:
        return [
            block["text"]
            for block in self._content_blocks(message)
            if block.get("type") == "text" and isinstance(block.get("text"), str) and block.get("text")
        ]

    def _thinking_blocks_from_assistant(self, message: dict[str, Any]) -> list[str]:
        return [
            block["thinking"]
            for block in self._content_blocks(message)
            if block.get("type") == "thinking" and isinstance(block.get("thinking"), str) and block.get("thinking")
        ]

    def _tool_calls_from_assistant(self, message: dict[str, Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for block in self._content_blocks(message):
            if block.get("type") != "toolCall":
                continue
            tool_call_id = block.get("id")
            tool_name = block.get("name")
            arguments = block.get("arguments")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            if not isinstance(tool_name, str) or not tool_name:
                continue
            if not isinstance(arguments, dict):
                arguments = {"value": arguments} if arguments is not None else {}
            tool_extra = {}
            thought_signature = block.get("thoughtSignature")
            if thought_signature is not None:
                tool_extra["thoughtSignature"] = thought_signature
            tool_calls.append(
                ToolCall(
                    tool_call_id=tool_call_id,
                    function_name=tool_name,
                    arguments=arguments,
                    extra=tool_extra or None,
                )
            )
        return tool_calls

    def _observation_from_tool_results(self, tool_results: Any) -> Observation | None:
        if not isinstance(tool_results, list):
            return None
        results: list[ObservationResult] = []
        for result in tool_results:
            if not isinstance(result, dict):
                continue
            content = self._content_to_atif(result.get("content"))
            if content in (None, ""):
                content = None
            extra: dict[str, Any] = {}
            if result.get("toolName") is not None:
                extra["tool_name"] = result.get("toolName")
            if result.get("details") is not None:
                extra["details"] = result.get("details")
            if result.get("isError") is not None:
                extra["is_error"] = bool(result.get("isError"))
            results.append(
                ObservationResult(
                    source_call_id=result.get("toolCallId") if isinstance(result.get("toolCallId"), str) else None,
                    content=content,
                    extra=extra or None,
                )
            )
        return Observation(results=results) if results else None

    def _content_blocks(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        content = message.get("content")
        if not isinstance(content, list):
            return []
        return [block for block in content if isinstance(block, dict)]

    def _content_to_atif(self, content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif block.get("type") == "image":
                mime = block.get("mimeType")
                parts.append(f"[image:{mime}]" if isinstance(mime, str) and mime else "[image]")
        return "\n".join(parts)

    def _iso_timestamp(self, timestamp_ms: Any) -> str | None:
        if not isinstance(timestamp_ms, int | float):
            return None
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
        except (OSError, ValueError, OverflowError):
            return None

    def _as_int(self, value: Any) -> int:
        return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0
