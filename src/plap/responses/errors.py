from __future__ import annotations


class ResponseOperationUnsupportedError(Exception):
    def __init__(self, *, status_code: int = 501) -> None:
        super().__init__("response operation is not supported")
        self.status_code = status_code
