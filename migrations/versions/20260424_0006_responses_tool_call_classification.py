"""Add responses tool call classification cache.

Revision ID: 20260424_0006
Revises: 20260424_0005
Create Date: 2026-04-24 00:06:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260424_0006"
down_revision: str | None = "20260424_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    _execute_sql_script(UPGRADE_SQL)


def downgrade() -> None:
    _execute_sql_script(DOWNGRADE_SQL)


def _execute_sql_script(script: str) -> None:
    for statement in _split_sql_statements(script):
        op.execute(statement)


def _split_sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_dollar_quote = False
    index = 0

    while index < len(script):
        if script.startswith("$$", index):
            in_dollar_quote = not in_dollar_quote
            current.append("$$")
            index += 2
            continue

        char = script[index]
        if char == ";" and not in_dollar_quote:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)
        index += 1

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)

    return statements


UPGRADE_SQL = r"""
set search_path = responses, public;

create table tool_call_classifications (
  scope_id uuid not null,
  signature_hash bytea not null
    references tool_signatures (signature_hash)
    on delete cascade,
  arguments_hash bytea not null,
  classifier text not null,
  classifier_model text not null,
  prompt_hash bytea not null,
  effect_class text not null,
  confidence numeric not null,
  rationale text not null,
  raw_output jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  primary key (
    scope_id,
    signature_hash,
    arguments_hash,
    classifier,
    classifier_model,
    prompt_hash
  ),

  check (octet_length(signature_hash) = 32),
  check (octet_length(arguments_hash) = 32),
  check (octet_length(prompt_hash) = 32),
  check (classifier <> ''),
  check (classifier_model <> ''),
  check (effect_class in ('safe', 'mutation', 'unknown')),
  check (confidence >= 0 and confidence <= 1),
  check (rationale <> ''),
  check (jsonb_typeof(raw_output) = 'object')
);

create trigger trg_tool_call_classifications_immutable
before update on tool_call_classifications
for each row execute function forbid_any_update();
"""


DOWNGRADE_SQL = r"""
set search_path = responses, public;

drop trigger if exists trg_tool_call_classifications_immutable
on tool_call_classifications;

drop table if exists tool_call_classifications;
"""
