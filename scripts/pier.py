from __future__ import annotations

import json
from pathlib import Path

from pier.environments.agent_setup import EGRESS_PROXY_SERVICE
from pier.environments.docker.docker import DockerEnvironment


def _write_host_gateway_compose_file(path: Path, *, services: tuple[str, ...]) -> Path:
    compose = {
        "services": {service: {"extra_hosts": ["host.docker.internal:host-gateway"]} for service in services}
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def _normalize_safe_port(value: int | str | None) -> int | None:
    if value is None:
        return None
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid Pier proxy port: {port}")
    return port


def _patch_proxy_bootstrap_script(path: Path, *, safe_port: int | None) -> None:
    if safe_port is None or safe_port in {80, 443}:
        return

    original = path.read_text()
    needle = "acl Safe_ports port 80 443"
    replacement = f"acl Safe_ports port 80 443 {safe_port}"
    if replacement in original:
        return
    if needle not in original:
        raise ValueError(f"could not find Safe_ports ACL in {path}")
    path.write_text(original.replace(needle, replacement, 1))


class PlapPierDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, plap_safe_port: int | str | None = None, **kwargs) -> None:
        self._plap_main_host_gateway_compose_path: Path | None = None
        self._plap_proxy_host_gateway_compose_path: Path | None = None
        self._plap_safe_port = _normalize_safe_port(plap_safe_port)
        super().__init__(*args, **kwargs)

    def _prepare_host_gateway_compose(self) -> None:
        self._plap_main_host_gateway_compose_path = _write_host_gateway_compose_file(
            self.trial_paths.trial_dir / "docker-compose-host-gateway-main.json",
            services=("main",),
        )

    def _prepare_egress_proxy_compose(self) -> None:
        super()._prepare_egress_proxy_compose()
        if self._egress_proxy_compose_path is None:
            self._plap_proxy_host_gateway_compose_path = None
            return
        _patch_proxy_bootstrap_script(
            self.trial_paths.trial_dir / "egress-proxy" / "start-squid.sh",
            safe_port=self._plap_safe_port,
        )
        self._plap_proxy_host_gateway_compose_path = _write_host_gateway_compose_file(
            self.trial_paths.trial_dir / "docker-compose-host-gateway-proxy.json",
            services=(EGRESS_PROXY_SERVICE,),
        )

    async def start(self, force_build: bool) -> None:
        self._prepare_host_gateway_compose()
        await super().start(force_build)

    @property
    def _docker_compose_paths(self) -> list[Path]:
        paths = super()._docker_compose_paths
        if self._plap_main_host_gateway_compose_path is not None:
            paths = [*paths, self._plap_main_host_gateway_compose_path]
        if self._plap_proxy_host_gateway_compose_path is not None:
            paths = [*paths, self._plap_proxy_host_gateway_compose_path]
        return paths
