"""Add responses state tree schema.

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

create table item_namespaces (
  namespace_id smallint generated always as identity primary key,
  namespace_name text not null unique,
  is_required boolean not null default false,

  check (namespace_name <> ''),
  check (namespace_name = lower(namespace_name))
);

insert into item_namespaces (namespace_name, is_required)
values ('m', true), ('r', true), ('s', true)
on conflict (namespace_name) do nothing;

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

create table state_nodes (
  scope_id uuid not null,
  node_id bigint generated always as identity,
  kind text not null,
  height smallint not null,
  item_count bigint not null,
  child_count smallint not null,
  refcount bigint not null default 0,
  created_at timestamptz not null default now(),

  primary key (scope_id, node_id),

  check (kind in ('leaf', 'internal')),
  check (height >= 0),
  check (item_count > 0),
  check (child_count >= 0),
  check (refcount >= 0),

  check (
    (kind = 'leaf' and height = 0 and child_count = 0)
    or
    (kind = 'internal' and height > 0 and child_count between 2 and 64)
  )
);

create table state_node_children (
  scope_id uuid not null,
  parent_node_id bigint not null,
  child_index smallint not null,
  child_node_id bigint not null,
  child_item_count bigint not null,

  primary key (scope_id, parent_node_id, child_index),
  unique (scope_id, parent_node_id, child_node_id),

  foreign key (scope_id, parent_node_id)
    references state_nodes (scope_id, node_id)
    on delete cascade,

  foreign key (scope_id, child_node_id)
    references state_nodes (scope_id, node_id),

  check (child_index >= 0),
  check (parent_node_id <> child_node_id),
  check (child_item_count > 0)
);

create table state_leaves (
  scope_id uuid not null,
  node_id bigint not null,
  entry_count integer not null,

  primary key (scope_id, node_id),

  foreign key (scope_id, node_id)
    references state_nodes (scope_id, node_id)
    on delete cascade,

  check (entry_count > 0),
  check (entry_count <= 128)
);

create table state_leaf_entries (
  scope_id uuid not null,
  node_id bigint not null,
  item_index integer not null,
  namespace_id smallint not null,
  ordinal bigint not null,
  payload_id uuid not null,

  primary key (scope_id, node_id, item_index),
  unique (scope_id, node_id, namespace_id, ordinal),

  foreign key (scope_id, node_id)
    references state_leaves (scope_id, node_id)
    on delete cascade,

  foreign key (namespace_id)
    references item_namespaces (namespace_id),

  foreign key (scope_id, payload_id)
    references payloads (scope_id, payload_id),

  check (item_index >= 0),
  check (ordinal >= 0)
);

create table response_records (
  scope_id uuid not null,
  response_id text not null,
  prev_response_id text,
  state_root_id bigint not null,
  output_state_root_id bigint not null,
  child_refcount bigint not null default 0,
  lease_refcount bigint not null default 0,
  status text not null default 'completed',
  completed_at timestamptz,
  fields jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  primary key (scope_id, response_id),

  foreign key (scope_id, prev_response_id)
    references response_records (scope_id, response_id),

  foreign key (scope_id, state_root_id)
    references state_nodes (scope_id, node_id),

  foreign key (scope_id, output_state_root_id)
    references state_nodes (scope_id, node_id),

  check (response_id <> ''),
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

create table response_namespace_cursors (
  scope_id uuid not null,
  response_id text not null,
  namespace_id smallint not null,
  next_ordinal bigint not null,

  primary key (scope_id, response_id, namespace_id),

  foreign key (scope_id, response_id)
    references response_records (scope_id, response_id)
    on delete cascade,

  foreign key (namespace_id)
    references item_namespaces (namespace_id),

  check (response_id <> ''),
  check (next_ordinal >= 0)
);

create table response_checkpoints (
  scope_id uuid not null,
  response_id text not null,
  checkpoint_id bigint generated always as identity,
  state_root_id bigint not null,
  created_at timestamptz not null default now(),

  primary key (scope_id, response_id, checkpoint_id),

  foreign key (scope_id, response_id)
    references response_records (scope_id, response_id)
    on delete cascade,

  foreign key (scope_id, state_root_id)
    references state_nodes (scope_id, node_id),

  check (response_id <> '')
);

create table checkpoint_namespace_cursors (
  scope_id uuid not null,
  response_id text not null,
  checkpoint_id bigint not null,
  namespace_id smallint not null,
  next_ordinal bigint not null,

  primary key (scope_id, response_id, checkpoint_id, namespace_id),

  foreign key (scope_id, response_id, checkpoint_id)
    references response_checkpoints (scope_id, response_id, checkpoint_id)
    on delete cascade,

  foreign key (namespace_id)
    references item_namespaces (namespace_id),

  check (response_id <> ''),
  check (next_ordinal >= 0)
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

create index ix_state_nodes_gc
  on state_nodes (created_at, scope_id, node_id)
  where refcount = 0;

create index ix_state_node_children_child_lookup
  on state_node_children (scope_id, child_node_id, parent_node_id, child_index);

create index ix_state_leaf_entries_payload_lookup
  on state_leaf_entries (scope_id, payload_id, node_id, item_index);

create index ix_state_leaf_entries_namespace_ordinal
  on state_leaf_entries (scope_id, namespace_id, ordinal, node_id, item_index);

create index ix_response_records_created_at
  on response_records (scope_id, created_at);

create index ix_response_records_prev
  on response_records (scope_id, prev_response_id)
  where prev_response_id is not null;

create index ix_response_records_state_root
  on response_records (scope_id, state_root_id, response_id);

create index ix_response_records_output_state_root
  on response_records (scope_id, output_state_root_id, response_id);

create index ix_response_records_gc
  on response_records (created_at, scope_id, response_id)
  where child_refcount = 0 and lease_refcount = 0;

create index ix_response_checkpoints_state_root
  on response_checkpoints (scope_id, state_root_id, response_id, checkpoint_id);

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

create function forbid_state_nodes_structural_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.node_id is distinct from new.node_id
    or old.kind is distinct from new.kind
    or old.height is distinct from new.height
    or old.item_count is distinct from new.item_count
    or old.child_count is distinct from new.child_count
    or old.created_at is distinct from new.created_at then
    raise exception 'state_nodes rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_state_nodes_immutable
before update on state_nodes
for each row execute function forbid_state_nodes_structural_update();

create function forbid_any_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  raise exception '% rows are immutable', tg_table_name;
end;
$$;

create trigger trg_state_node_children_immutable
before update on state_node_children
for each row execute function forbid_any_update();

create trigger trg_state_leaves_immutable
before update on state_leaves
for each row execute function forbid_any_update();

create trigger trg_state_leaf_entries_immutable
before update on state_leaf_entries
for each row execute function forbid_any_update();

create trigger trg_response_namespace_cursors_immutable
before update on response_namespace_cursors
for each row execute function forbid_any_update();

create trigger trg_checkpoint_namespace_cursors_immutable
before update on checkpoint_namespace_cursors
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
    or old.state_root_id is distinct from new.state_root_id
    or old.output_state_root_id is distinct from new.output_state_root_id
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

create function forbid_response_checkpoints_structural_update()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.response_id is distinct from new.response_id
    or old.checkpoint_id is distinct from new.checkpoint_id
    or old.state_root_id is distinct from new.state_root_id
    or old.created_at is distinct from new.created_at then
    raise exception 'response_checkpoints rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_response_checkpoints_immutable
before update on response_checkpoints
for each row execute function forbid_response_checkpoints_structural_update();

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

create function apply_payload_refcount_from_leaf_entry()
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

create trigger trg_state_leaf_entries_payload_refcount
after insert or delete on state_leaf_entries
for each row execute function apply_payload_refcount_from_leaf_entry();

create function apply_child_refcounts()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    update state_nodes
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and node_id = new.child_node_id;
    return new;
  elsif tg_op = 'DELETE' then
    update state_nodes
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and node_id = old.child_node_id;
    return old;
  end if;
end;
$$;

create trigger trg_state_node_children_refcounts
after insert or delete on state_node_children
for each row execute function apply_child_refcounts();

create function apply_response_refcounts()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    update state_nodes
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and node_id = new.state_root_id;

    update state_nodes
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and node_id = new.output_state_root_id;

    if new.prev_response_id is not null then
      update response_records
         set child_refcount = child_refcount + 1
       where scope_id = new.scope_id
         and response_id = new.prev_response_id;
    end if;
    return new;
  elsif tg_op = 'DELETE' then
    update state_nodes
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and node_id = old.state_root_id;

    update state_nodes
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and node_id = old.output_state_root_id;

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

create function apply_checkpoint_root_refcounts()
returns trigger
language plpgsql
set search_path = responses, public
as $$
begin
  if tg_op = 'INSERT' then
    update state_nodes
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and node_id = new.state_root_id;
    return new;
  elsif tg_op = 'DELETE' then
    update state_nodes
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and node_id = old.state_root_id;
    return old;
  end if;
end;
$$;

create trigger trg_response_checkpoints_refcount
after insert or delete on response_checkpoints
for each row execute function apply_checkpoint_root_refcounts();

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

create function validate_state_node_consistency()
returns trigger
language plpgsql
set search_path = responses, public
as $$
declare
  actual_child_count integer;
  total_child_count bigint;
  dense_slots boolean;
  invalid_child boolean;
  leaf_exists boolean;
begin
  if new.kind = 'leaf' then
    select exists (
      select 1
        from state_leaves
       where scope_id = new.scope_id
         and node_id = new.node_id
    ) into leaf_exists;

    if not leaf_exists then
      raise exception 'leaf state_node %/% has no state_leaves row',
        new.scope_id,
        new.node_id;
    end if;

    return new;
  end if;

  select count(*), coalesce(sum(child_item_count), 0)
    into actual_child_count, total_child_count
    from state_node_children
   where scope_id = new.scope_id
     and parent_node_id = new.node_id;

  if actual_child_count <> new.child_count then
    raise exception 'internal state_node %/% has % children, expected %',
      new.scope_id,
      new.node_id,
      actual_child_count,
      new.child_count;
  end if;

  if total_child_count <> new.item_count then
    raise exception 'internal state_node %/% has item_count %, expected %',
      new.scope_id,
      new.node_id,
      new.item_count,
      total_child_count;
  end if;

  select count(*) = new.child_count
    into dense_slots
    from generate_series(0, new.child_count - 1) expected(slot_index)
    join state_node_children child
      on child.scope_id = new.scope_id
     and child.parent_node_id = new.node_id
     and child.child_index = expected.slot_index;

  if not dense_slots then
    raise exception 'internal state_node %/% child slots are not dense',
      new.scope_id,
      new.node_id;
  end if;

  select exists (
    select 1
      from state_node_children edge
      join state_nodes child
        on child.scope_id = edge.scope_id
       and child.node_id = edge.child_node_id
     where edge.scope_id = new.scope_id
       and edge.parent_node_id = new.node_id
       and (
         edge.child_item_count <> child.item_count
         or child.height <> new.height - 1
       )
  ) into invalid_child;

  if invalid_child then
    raise exception 'internal state_node %/% has invalid child metadata',
      new.scope_id,
      new.node_id;
  end if;

  return new;
end;
$$;

create constraint trigger ct_state_nodes_consistency
after insert on state_nodes
deferrable initially deferred
for each row execute function validate_state_node_consistency();

create function validate_state_node_child_consistency()
returns trigger
language plpgsql
set search_path = responses, public
as $$
declare
  check_scope uuid;
  check_parent bigint;
  parent_node state_nodes%rowtype;
  actual_child_count integer;
  total_child_count bigint;
  dense_slots boolean;
  invalid_child boolean;
begin
  check_scope = coalesce(new.scope_id, old.scope_id);
  check_parent = coalesce(new.parent_node_id, old.parent_node_id);

  select *
    into parent_node
    from state_nodes
   where scope_id = check_scope
     and node_id = check_parent;

  if not found then
    return coalesce(new, old);
  end if;

  if parent_node.kind <> 'internal' then
    return coalesce(new, old);
  end if;

  select count(*), coalesce(sum(child_item_count), 0)
    into actual_child_count, total_child_count
    from state_node_children
   where scope_id = parent_node.scope_id
     and parent_node_id = parent_node.node_id;

  if actual_child_count <> parent_node.child_count then
    raise exception 'internal state_node %/% has % children, expected %',
      parent_node.scope_id,
      parent_node.node_id,
      actual_child_count,
      parent_node.child_count;
  end if;

  if total_child_count <> parent_node.item_count then
    raise exception 'internal state_node %/% has item_count %, expected %',
      parent_node.scope_id,
      parent_node.node_id,
      parent_node.item_count,
      total_child_count;
  end if;

  select count(*) = parent_node.child_count
    into dense_slots
    from generate_series(0, parent_node.child_count - 1) expected(slot_index)
    join state_node_children child
      on child.scope_id = parent_node.scope_id
     and child.parent_node_id = parent_node.node_id
     and child.child_index = expected.slot_index;

  if not dense_slots then
    raise exception 'internal state_node %/% child slots are not dense',
      parent_node.scope_id,
      parent_node.node_id;
  end if;

  select exists (
    select 1
      from state_node_children edge
      join state_nodes child
        on child.scope_id = edge.scope_id
       and child.node_id = edge.child_node_id
     where edge.scope_id = parent_node.scope_id
       and edge.parent_node_id = parent_node.node_id
       and (
         edge.child_item_count <> child.item_count
         or child.height <> parent_node.height - 1
       )
  ) into invalid_child;

  if invalid_child then
    raise exception 'internal state_node %/% has invalid child metadata',
      parent_node.scope_id,
      parent_node.node_id;
  end if;

  return coalesce(new, old);
end;
$$;

create constraint trigger ct_state_node_children_consistency
after insert or delete on state_node_children
deferrable initially deferred
for each row execute function validate_state_node_child_consistency();

create function validate_state_leaf_consistency()
returns trigger
language plpgsql
set search_path = responses, public
as $$
declare
  check_scope uuid;
  check_node bigint;
  node_kind text;
  node_count bigint;
  expected_count integer;
  actual_count bigint;
  min_pos integer;
  max_pos integer;
begin
  check_scope = coalesce(new.scope_id, old.scope_id);
  check_node = coalesce(new.node_id, old.node_id);

  select n.kind, n.item_count, l.entry_count
    into node_kind, node_count, expected_count
    from state_nodes n
    left join state_leaves l
      on l.scope_id = n.scope_id
     and l.node_id = n.node_id
   where n.scope_id = check_scope
     and n.node_id = check_node;

  if not found then
    return coalesce(new, old);
  end if;

  if node_kind <> 'leaf' then
    raise exception 'state_leaves row %/% points to non-leaf state_node',
      check_scope,
      check_node;
  end if;

  if expected_count is null then
    raise exception 'leaf state_node %/% has no state_leaves row',
      check_scope,
      check_node;
  end if;

  if expected_count::bigint <> node_count then
    raise exception 'state_leaf %/% entry_count %, expected item_count %',
      check_scope,
      check_node,
      expected_count,
      node_count;
  end if;

  select count(*), min(item_index), max(item_index)
    into actual_count, min_pos, max_pos
    from state_leaf_entries
   where scope_id = check_scope
     and node_id = check_node;

  if actual_count <> expected_count then
    raise exception 'state_leaf %/% has % entries, expected %',
      check_scope,
      check_node,
      actual_count,
      expected_count;
  end if;

  if min_pos <> 0 or max_pos <> expected_count - 1 then
    raise exception 'state_leaf %/% positions are not dense',
      check_scope,
      check_node;
  end if;

  return coalesce(new, old);
end;
$$;

create constraint trigger ct_state_leaves_consistency
after insert or delete on state_leaves
deferrable initially deferred
for each row execute function validate_state_leaf_consistency();

create constraint trigger ct_state_leaf_entries_consistency
after insert or delete on state_leaf_entries
deferrable initially deferred
for each row execute function validate_state_leaf_consistency();
"""


DOWNGRADE_SQL = r"""
set search_path = responses, public;

drop trigger if exists ct_state_leaf_entries_consistency
  on state_leaf_entries;
drop trigger if exists ct_state_leaves_consistency on state_leaves;
drop trigger if exists ct_state_node_children_consistency
  on state_node_children;
drop trigger if exists ct_state_nodes_consistency on state_nodes;
drop trigger if exists trg_conversations_response_lease on conversations;
drop trigger if exists trg_response_leases_refcounts on response_leases;
drop trigger if exists trg_response_checkpoints_refcount
  on response_checkpoints;
drop trigger if exists trg_response_records_refcounts on response_records;
drop trigger if exists trg_state_node_children_refcounts
  on state_node_children;
drop trigger if exists trg_state_leaf_entries_payload_refcount
  on state_leaf_entries;
drop trigger if exists trg_response_leases_normalize_update
  on response_leases;
drop trigger if exists trg_conversations_immutable on conversations;
drop trigger if exists trg_response_checkpoints_immutable
  on response_checkpoints;
drop trigger if exists trg_response_records_immutable on response_records;
drop trigger if exists trg_state_leaf_entries_immutable
  on state_leaf_entries;
drop trigger if exists trg_state_leaves_immutable on state_leaves;
drop trigger if exists trg_state_node_children_immutable
  on state_node_children;
drop trigger if exists trg_response_namespace_cursors_immutable
  on response_namespace_cursors;
drop trigger if exists trg_checkpoint_namespace_cursors_immutable
  on checkpoint_namespace_cursors;
drop trigger if exists trg_state_nodes_immutable on state_nodes;
drop trigger if exists trg_payloads_immutable on payloads;

drop function if exists validate_state_leaf_consistency();
drop function if exists validate_state_node_child_consistency();
drop function if exists validate_state_node_consistency();
drop function if exists maintain_conversation_response_lease();
drop function if exists apply_response_lease_refcounts();
drop function if exists apply_checkpoint_root_refcounts();
drop function if exists apply_response_refcounts();
drop function if exists apply_child_refcounts();
drop function if exists apply_payload_refcount_from_leaf_entry();
drop function if exists normalize_response_lease_update();
drop function if exists forbid_conversations_structural_update();
drop function if exists forbid_response_checkpoints_structural_update();
drop function if exists forbid_response_records_structural_update();
drop function if exists forbid_any_update();
drop function if exists forbid_state_nodes_structural_update();
drop function if exists forbid_payloads_structural_update();

drop table if exists conversations;
drop table if exists response_leases;
drop table if exists checkpoint_namespace_cursors;
drop table if exists response_checkpoints;
drop table if exists response_namespace_cursors;
drop table if exists response_records;
drop table if exists state_leaf_entries;
drop table if exists state_leaves;
drop table if exists state_node_children;
drop table if exists state_nodes;
drop table if exists payloads;
drop table if exists item_namespaces;
drop schema if exists responses;
"""
