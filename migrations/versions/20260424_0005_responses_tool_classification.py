"""Add responses tool classification cache.

Revision ID: 20260424_0005
Revises: 20260424_0004
Create Date: 2026-04-24 00:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260424_0005"
down_revision: str | None = "20260424_0004"
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

create table tool_signatures (
  signature_hash bytea primary key,
  signature_json jsonb not null,
  created_at timestamptz not null default now(),

  check (octet_length(signature_hash) = 32),
  check (jsonb_typeof(signature_json) = 'object')
);

create table tool_classifications (
  signature_hash bytea not null
    references tool_signatures (signature_hash)
    on delete cascade,
  classifier text not null,
  classifier_model text not null,
  prompt_hash bytea not null,
  effect_class text not null,
  confidence numeric not null,
  rationale text not null,
  raw_output jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  primary key (
    signature_hash,
    classifier,
    classifier_model,
    prompt_hash
  ),

  check (octet_length(prompt_hash) = 32),
  check (classifier <> ''),
  check (classifier_model <> ''),
  check (effect_class in ('safe', 'visible', 'mutation', 'contextual')),
  check (confidence >= 0 and confidence <= 1),
  check (rationale <> ''),
  check (jsonb_typeof(raw_output) = 'object')
);

create trigger trg_tool_signatures_immutable
before update on tool_signatures
for each row execute function forbid_any_update();

create trigger trg_tool_classifications_immutable
before update on tool_classifications
for each row execute function forbid_any_update();
"""


DOWNGRADE_SQL = r"""
set search_path = responses, public;

drop trigger if exists trg_tool_classifications_immutable on tool_classifications;
drop trigger if exists trg_tool_signatures_immutable on tool_signatures;

drop table if exists tool_classifications;
drop table if exists tool_signatures;
"""
