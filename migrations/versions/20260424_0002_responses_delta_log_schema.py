"""Add responses delta-log lineage schema.

Revision ID: 20260424_0002
Revises: 20260423_0001
Create Date: 2026-04-24 00:02:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260424_0002"
down_revision: str | None = "20260423_0001"
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
create schema if not exists responses;

create extension if not exists pgcrypto with schema public;

set search_path = responses, public;

create table payloads (
  scope_id uuid not null,
  payload_id uuid not null default gen_random_uuid(),
  payload_hash bytea not null,
  payload_json jsonb not null,
  refcount bigint not null default 0,
  created_at timestamptz not null default now(),

  primary key (scope_id, payload_id),
  unique (scope_id, payload_hash),

  check (octet_length(payload_hash) = 32),
  check (refcount >= 0)
);

create table response_records (
  scope_id uuid not null,
  response_id text not null,
  prev_response_id text,
  replay_base_response_id text not null,
  child_refcount bigint not null default 0,
  lease_refcount bigint not null default 0,
  status text not null default 'completed',
  completed_at timestamptz,
  fields jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  primary key (scope_id, response_id),

  foreign key (scope_id, prev_response_id)
    references response_records (scope_id, response_id),

  foreign key (scope_id, replay_base_response_id)
    references response_records (scope_id, response_id)
    deferrable initially deferred,

  check (response_id <> ''),
  check (replay_base_response_id <> ''),
  check (prev_response_id is null or prev_response_id <> response_id),
  check (child_refcount >= 0),
  check (lease_refcount >= 0),
  check (status in (
    'queued',
    'in_progress',
    'completed',
    'failed',
    'cancelled',
    'incomplete'
  )),
  check (jsonb_typeof(fields) = 'object')
);

create table response_input_items (
  scope_id uuid not null,
  response_id text not null,
  input_index integer not null,
  payload_id uuid not null,

  primary key (scope_id, response_id, input_index),

  foreign key (scope_id, response_id)
    references response_records (scope_id, response_id)
    on delete cascade,

  foreign key (scope_id, payload_id)
    references payloads (scope_id, payload_id),

  check (response_id <> ''),
  check (input_index >= 0)
);

create table response_output_items (
  scope_id uuid not null,
  response_id text not null,
  output_index integer not null,
  payload_id uuid not null,

  primary key (scope_id, response_id, output_index),

  foreign key (scope_id, response_id)
    references response_records (scope_id, response_id)
    on delete cascade,

  foreign key (scope_id, payload_id)
    references payloads (scope_id, payload_id),

  check (response_id <> ''),
  check (output_index >= 0)
);

create table response_leases (
  scope_id uuid not null,
  lease_id uuid not null default gen_random_uuid(),
  response_id text not null,
  owner_type text not null,
  owner_id text not null,
  status text not null default 'live',
  expires_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  primary key (scope_id, lease_id),

  foreign key (scope_id, response_id)
    references response_records (scope_id, response_id),

  check (response_id <> ''),
  check (owner_type <> ''),
  check (owner_id <> ''),
  check (status in ('live', 'expiring', 'expired'))
);

create unique index uq_response_leases_live_owner
  on response_leases (scope_id, owner_type, owner_id)
  where status = 'live';

create table conversations (
  scope_id uuid not null,
  conversation_id text not null,
  current_response_id text not null,
  created_at timestamptz not null default now(),
  last_used_at timestamptz not null default now(),
  retention_expires_at timestamptz,

  primary key (scope_id, conversation_id),

  foreign key (scope_id, current_response_id)
    references response_records (scope_id, response_id),

  check (conversation_id <> ''),
  check (current_response_id <> '')
);

create index ix_payloads_gc
  on payloads (created_at, scope_id, payload_id)
  where refcount = 0;

create index ix_response_records_created_at
  on response_records (scope_id, created_at);

create index ix_response_records_prev
  on response_records (scope_id, prev_response_id)
  where prev_response_id is not null;

create index ix_response_records_replay_base
  on response_records (scope_id, replay_base_response_id, response_id);

create index ix_response_records_gc
  on response_records (created_at, scope_id, response_id)
  where child_refcount = 0 and lease_refcount = 0;

create index ix_response_input_items_payload_lookup
  on response_input_items (scope_id, payload_id, response_id, input_index);

create index ix_response_output_items_payload_lookup
  on response_output_items (scope_id, payload_id, response_id, output_index);

create index ix_response_leases_expiration
  on response_leases (expires_at, scope_id, lease_id)
  where status = 'live' and expires_at is not null;

create index ix_response_leases_response
  on response_leases (scope_id, response_id, lease_id);

create index ix_conversations_last_used_at
  on conversations (last_used_at, scope_id, conversation_id);

create index ix_conversations_retention_expiration
  on conversations (retention_expires_at, scope_id, conversation_id)
  where retention_expires_at is not null;

create index ix_conversations_current_response
  on conversations (scope_id, current_response_id, conversation_id);

comment on table response_records is
  'Immutable response lineage. Replay walks prev_response_id backward until replay_base_response_id.';

comment on column response_records.prev_response_id is
  'Immediate parent response in append lineage.';

comment on column response_records.replay_base_response_id is
  'Self or ancestor response that starts replay after compaction reset.';

create function forbid_payloads_structural_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.payload_id is distinct from new.payload_id
    or old.payload_hash is distinct from new.payload_hash
    or old.payload_json is distinct from new.payload_json
    or old.created_at is distinct from new.created_at then
    raise exception 'payloads rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_payloads_immutable
before update on payloads
for each row execute function forbid_payloads_structural_update();

create function forbid_any_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  raise exception '% rows are immutable', tg_table_name;
end;
$$;

create trigger trg_response_input_items_immutable
before update on response_input_items
for each row execute function forbid_any_update();

create trigger trg_response_output_items_immutable
before update on response_output_items
for each row execute function forbid_any_update();

create function forbid_response_records_structural_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.response_id is distinct from new.response_id
    or old.prev_response_id is distinct from new.prev_response_id
    or old.replay_base_response_id is distinct from new.replay_base_response_id
    or old.fields is distinct from new.fields
    or old.created_at is distinct from new.created_at then
    raise exception 'response_records rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_response_records_immutable
before update on response_records
for each row execute function forbid_response_records_structural_update();

create function normalize_response_lease_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.lease_id is distinct from new.lease_id
    or old.owner_type is distinct from new.owner_type
    or old.owner_id is distinct from new.owner_id
    or old.created_at is distinct from new.created_at then
    raise exception 'response_leases owner identity is immutable';
  end if;
  new.updated_at = now();
  return new;
end;
$$;

create trigger trg_response_leases_normalize_update
before update on response_leases
for each row execute function normalize_response_lease_update();

create function forbid_conversations_structural_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.conversation_id is distinct from new.conversation_id
    or old.created_at is distinct from new.created_at then
    raise exception 'conversations rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_conversations_immutable
before update on conversations
for each row execute function forbid_conversations_structural_update();

create function apply_payload_refcount_from_response_item()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    update payloads
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and payload_id = new.payload_id;
    return new;
  elsif tg_op = 'DELETE' then
    update payloads
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and payload_id = old.payload_id;
    return old;
  end if;
end;
$$;

create trigger trg_response_input_items_payload_refcount
after insert or delete on response_input_items
for each row execute function apply_payload_refcount_from_response_item();

create trigger trg_response_output_items_payload_refcount
after insert or delete on response_output_items
for each row execute function apply_payload_refcount_from_response_item();

create function apply_response_refcounts()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    if new.prev_response_id is not null then
      update response_records
         set child_refcount = child_refcount + 1
       where scope_id = new.scope_id
         and response_id = new.prev_response_id;
    end if;
    return new;
  elsif tg_op = 'DELETE' then
    if old.prev_response_id is not null then
      update response_records
         set child_refcount = child_refcount - 1
       where scope_id = old.scope_id
         and response_id = old.prev_response_id;
    end if;
    return old;
  end if;
end;
$$;

create trigger trg_response_records_refcounts
after insert or delete on response_records
for each row execute function apply_response_refcounts();

create function apply_response_lease_refcounts()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    if new.status = 'live' then
      update response_records
         set lease_refcount = lease_refcount + 1
       where scope_id = new.scope_id
         and response_id = new.response_id;
    end if;
    return new;
  elsif tg_op = 'DELETE' then
    if old.status = 'live' then
      update response_records
         set lease_refcount = lease_refcount - 1
       where scope_id = old.scope_id
         and response_id = old.response_id;
    end if;
    return old;
  end if;

  if old.status = 'live'
    and (
      new.status is distinct from 'live'
      or old.response_id is distinct from new.response_id
    ) then
    update response_records
       set lease_refcount = lease_refcount - 1
     where scope_id = old.scope_id
       and response_id = old.response_id;
  end if;

  if new.status = 'live'
    and (
      old.status is distinct from 'live'
      or old.response_id is distinct from new.response_id
    ) then
    update response_records
       set lease_refcount = lease_refcount + 1
     where scope_id = new.scope_id
       and response_id = new.response_id;
  end if;

  return new;
end;
$$;

create trigger trg_response_leases_refcounts
after insert or update or delete on response_leases
for each row execute function apply_response_lease_refcounts();

create function maintain_conversation_response_lease()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    insert into response_leases (
      scope_id,
      response_id,
      owner_type,
      owner_id,
      status,
      expires_at
    ) values (
      new.scope_id,
      new.current_response_id,
      'conversation',
      new.conversation_id,
      'live',
      null
    );
    return new;
  elsif tg_op = 'DELETE' then
    delete from response_leases
     where scope_id = old.scope_id
       and owner_type = 'conversation'
       and owner_id = old.conversation_id
       and status = 'live';
    return old;
  end if;

  update response_leases
     set response_id = new.current_response_id,
         status = 'live',
         expires_at = null
   where scope_id = new.scope_id
     and owner_type = 'conversation'
     and owner_id = new.conversation_id
     and status = 'live';

  if not found then
    insert into response_leases (
      scope_id,
      response_id,
      owner_type,
      owner_id,
      status,
      expires_at
    ) values (
      new.scope_id,
      new.current_response_id,
      'conversation',
      new.conversation_id,
      'live',
      null
    );
  end if;

  return new;
end;
$$;

create trigger trg_conversations_response_lease
after insert or update or delete on conversations
for each row execute function maintain_conversation_response_lease();
"""


DOWNGRADE_SQL = r"""
drop trigger if exists trg_conversations_response_lease on conversations;
drop trigger if exists trg_response_leases_refcounts on response_leases;
drop trigger if exists trg_response_records_refcounts on response_records;
drop trigger if exists trg_response_output_items_payload_refcount on response_output_items;
drop trigger if exists trg_response_input_items_payload_refcount on response_input_items;
drop trigger if exists trg_conversations_immutable on conversations;
drop trigger if exists trg_response_leases_normalize_update on response_leases;
drop trigger if exists trg_response_records_immutable on response_records;
drop trigger if exists trg_response_output_items_immutable on response_output_items;
drop trigger if exists trg_response_input_items_immutable on response_input_items;
drop trigger if exists trg_payloads_immutable on payloads;

drop function if exists maintain_conversation_response_lease();
drop function if exists apply_response_lease_refcounts();
drop function if exists apply_response_refcounts();
drop function if exists apply_payload_refcount_from_response_item();
drop function if exists forbid_conversations_structural_update();
drop function if exists normalize_response_lease_update();
drop function if exists forbid_response_records_structural_update();
drop function if exists forbid_any_update();
drop function if exists forbid_payloads_structural_update();

drop index if exists ix_conversations_current_response;
drop index if exists ix_conversations_retention_expiration;
drop index if exists ix_conversations_last_used_at;
drop index if exists ix_response_leases_response;
drop index if exists ix_response_leases_expiration;
drop index if exists ix_response_output_items_payload_lookup;
drop index if exists ix_response_input_items_payload_lookup;
drop index if exists ix_response_records_gc;
drop index if exists ix_response_records_replay_base;
drop index if exists ix_response_records_prev;
drop index if exists ix_response_records_created_at;
drop index if exists ix_payloads_gc;
drop index if exists uq_response_leases_live_owner;

drop table if exists conversations;
drop table if exists response_leases;
drop table if exists response_output_items;
drop table if exists response_input_items;
drop table if exists response_records;
drop table if exists payloads;
"""
