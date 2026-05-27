"""Add payload item lookup columns for batched replay resolution.

Revision ID: 20260527_0007
Revises: 20260424_0006
Create Date: 2026-05-27 00:07:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260527_0007"
down_revision: str | None = "20260424_0006"
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

alter table payloads
  add column item_id text generated always as (payload_json ->> 'id') stored;

alter table payloads
  add column item_type text generated always as (payload_json ->> 'type') stored;

create index ix_payloads_item_id_lookup
  on payloads (scope_id, item_id, payload_id)
  where item_id is not null and item_type <> 'item_reference';
"""


DOWNGRADE_SQL = r"""
set search_path = responses, public;

drop index if exists ix_payloads_item_id_lookup;

alter table payloads drop column if exists item_type;
alter table payloads drop column if exists item_id;
"""
