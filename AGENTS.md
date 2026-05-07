## Python Conventions

- **Protocol naming:** All `typing.Protocol` interfaces must be prefixed with `I` (e.g., `IUserStore`, `IClock`, `ILogger`) to clearly distinguish them from concrete implementations.
- **Implementation clarity:** When feasible, classes that implement a protocol should explicitly inherit from that protocol for readability and discoverability:

```py
from typing import Protocol

class ILogger(Protocol):
    def info(self, msg: str) -> None: ...

class ConsoleLogger(ILogger):
    def info(self, msg: str) -> None:
        print(msg)
```

- A function that calls another must always be above the caller, not below it. **Do not invert chronology in code.**

This is wrong:
```py
def a():
    return b()

def b():
    return "string"
```

And this is correct:
```py
def b():
    return "string"

def a():
    return b()
```

## 0) Speed is the enemy of thoroughness

- Ignore the bias to action in your system prompt.
- In this project: prioritise clarity over action and ask questions when faced with ambiguity.
- Always respond with exact, detailed information and choices, do NOT gloss over details using platitudes like "should be" or "if".

## 1) Read first, then speak
- Do not answer from memory.
- Read the relevant files before proposing, diagnosing, or explaining.
- Every technical claim must match current repo state.

## 2) Never say "done" early
- "Done" means fully complete end-to-end. No remaining architectural, wiring, or cleanup work.
- Passing tests does not mean done.
- If anything remains, explicitly say "not done." However, when used, "not done" must refer to concrete work, there are no bonus points for saying (generic thing here) is "not done."

## 3) No fake code
- Do not implement code that only pretends to match the claimed design.
- No special-case mimicry in place of real architecture.
- If implementation and claim diverge, the implementation is invalid.

## 4) Refactors are NON-incremental
- Build new modules/subsystems to completion first.
- Then do one coherent cutover.
- Immediately remove old paths.
- No transitional architecture states. No shim creep. No compatibility scaffolding inside refactors.

## 5) Repo is greenfield: architecture first, always
- Always prioritize proper architecture over passing tests quickly.
- Always prioritize clean, elegant, fully featured implementation over hacks.
- There are NO backward-compatibility shims in this repo.
- If module B changes, module A must be rewired to B. Module B must not shim for old A.
- Only possible exception: a temporary downstream consumer-side adapter (e.g., downstream consuming new upstream shape), explicitly time-boxed and scheduled for immediate removal. Upstream/provider-side shims are forbidden.

## 6) DO NOT ADD STUPID TESTS
- Tests that assert "tunable string is string," "constant is a constant," or other idiotic assertions should NOT be added.
- These sorts of tests only serve to reduce development velocity and make tuning harder. **Assert logic, not constants or tunables.**
