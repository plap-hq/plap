from __future__ import annotations

from collections.abc import Sequence

from plap.responses.contracts import FunctionTool, SupportedTool, WebSearchTool
from plap.responses.tools.classifier import CachedToolClassifier
from plap.responses.tools.registry import get_server_tool_policy
from plap.responses.tools.signatures import function_tool_signature
from plap.responses.tools.types import ToolPolicy


class ToolPolicyError(ValueError):
    pass


class CachedToolPolicyResolver:
    def __init__(self, classifier: CachedToolClassifier) -> None:
        self._classifier = classifier

    async def resolve(self, tools: Sequence[SupportedTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        client_signatures_by_name = {}
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                policy = get_server_tool_policy(tool.type)
                if policy is None:
                    raise ToolPolicyError(f"unknown server tool: {tool.type}")
                policies[tool.type] = policy
                continue

            if isinstance(tool, FunctionTool):
                signature = function_tool_signature(tool)
                previous_hash = signatures_by_name.get(tool.name)
                if (
                    previous_hash is not None
                    and previous_hash != signature.signature_hash
                ):
                    raise ToolPolicyError(
                        "duplicate function tool name with different signature: "
                        f"{tool.name}"
                    )
                signatures_by_name[tool.name] = signature.signature_hash
                if (
                    tool.name not in policies
                    and tool.name not in client_signatures_by_name
                ):
                    client_signatures_by_name[tool.name] = signature
        classifications = await self._classifier.classify_many(
            list(client_signatures_by_name.values())
        )
        for tool_name, signature in client_signatures_by_name.items():
            classification = classifications[signature.signature_hash]
            policies[tool_name] = ToolPolicy(
                name=tool_name,
                source="client",
                effect_class=classification.effect_class,
                classification=classification,
            )
        return policies


class StaticToolPolicyResolver:
    async def resolve(self, tools: Sequence[SupportedTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                policy = get_server_tool_policy(tool.type)
                if policy is not None:
                    policies[tool.type] = policy
                continue
            if isinstance(tool, FunctionTool):
                signature = function_tool_signature(tool)
                previous_hash = signatures_by_name.get(tool.name)
                if (
                    previous_hash is not None
                    and previous_hash != signature.signature_hash
                ):
                    raise ToolPolicyError(
                        "duplicate function tool name with different signature: "
                        f"{tool.name}"
                    )
                signatures_by_name[tool.name] = signature.signature_hash
                policies.setdefault(
                    tool.name,
                    ToolPolicy(
                        name=tool.name,
                        source="client",
                        effect_class="unknown",
                    ),
                )
        return policies
