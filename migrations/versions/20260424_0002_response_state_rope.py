"""Add response state rope schema.

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
create extension if not exists pgcrypto;

create table ordinal_namespaces (
  namespace_id smallint generated always as identity primary key,
  namespace_name text not null unique,

  check (namespace_name <> ''),
  check (namespace_name = lower(namespace_name))
);

create table payload_objects (
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
  left_id bigint,
  right_id bigint,
  item_count bigint not null,
  refcount bigint not null default 0,
  created_at timestamptz not null default now(),

  primary key (scope_id, node_id),

  foreign key (scope_id, left_id)
    references state_nodes (scope_id, node_id),

  foreign key (scope_id, right_id)
    references state_nodes (scope_id, node_id),

  check (kind in ('leaf', 'concat')),
  check (item_count > 0),
  check (refcount >= 0),

  check (
    (kind = 'leaf' and left_id is null and right_id is null)
    or
    (kind = 'concat' and left_id is not null and right_id is not null)
  ),

  check (left_id is null or left_id <> node_id),
  check (right_id is null or right_id <> node_id)
);

create table state_leaves (
  scope_id uuid not null,
  node_id bigint not null,
  entry_count integer not null,

  primary key (scope_id, node_id),

  foreign key (scope_id, node_id)
    references state_nodes (scope_id, node_id)
    on delete cascade,

  check (entry_count > 0)
);

create table state_leaf_entries (
  scope_id uuid not null,
  node_id bigint not null,
  pos integer not null,
  namespace_id smallint not null,
  ord bigint not null,
  payload_id uuid not null,

  primary key (scope_id, node_id, pos),

  foreign key (scope_id, node_id)
    references state_leaves (scope_id, node_id)
    on delete cascade,

  foreign key (namespace_id)
    references ordinal_namespaces (namespace_id),

  foreign key (scope_id, payload_id)
    references payload_objects (scope_id, payload_id),

  check (pos >= 0),
  check (ord >= 0)
);

create table responses (
  scope_id uuid not null,
  response_id text not null,
  prev_response_id text,
  full_state_root_id bigint not null,
  child_refcount bigint not null default 0,
  lease_refcount bigint not null default 0,
  created_at timestamptz not null default now(),

  primary key (scope_id, response_id),

  foreign key (scope_id, prev_response_id)
    references responses (scope_id, response_id),

  foreign key (scope_id, full_state_root_id)
    references state_nodes (scope_id, node_id),

  check (response_id <> ''),
  check (child_refcount >= 0),
  check (lease_refcount >= 0)
);

create table response_namespace_counters (
  scope_id uuid not null,
  response_id text not null,
  namespace_id smallint not null,
  next_ord bigint not null,

  primary key (scope_id, response_id, namespace_id),

  foreign key (scope_id, response_id)
    references responses (scope_id, response_id)
    on delete cascade,

  foreign key (namespace_id)
    references ordinal_namespaces (namespace_id),

  check (response_id <> ''),
  check (next_ord >= 0)
);

create table response_checkpoints (
  scope_id uuid not null,
  response_id text not null,
  checkpoint_id bigint generated always as identity,
  root_id bigint not null,
  created_at timestamptz not null default now(),

  primary key (scope_id, response_id, checkpoint_id),

  foreign key (scope_id, response_id)
    references responses (scope_id, response_id)
    on delete cascade,

  foreign key (scope_id, root_id)
    references state_nodes (scope_id, node_id),

  check (response_id <> '')
);

create table checkpoint_namespace_counters (
  scope_id uuid not null,
  response_id text not null,
  checkpoint_id bigint not null,
  namespace_id smallint not null,
  next_ord bigint not null,

  primary key (scope_id, response_id, checkpoint_id, namespace_id),

  foreign key (scope_id, response_id, checkpoint_id)
    references response_checkpoints (scope_id, response_id, checkpoint_id)
    on delete cascade,

  foreign key (namespace_id)
    references ordinal_namespaces (namespace_id),

  check (response_id <> ''),
  check (next_ord >= 0)
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
    references responses (scope_id, response_id),

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

  primary key (scope_id, conversation_id),

  foreign key (scope_id, current_response_id)
    references responses (scope_id, response_id),

  check (conversation_id <> ''),
  check (current_response_id <> '')
);

create index ix_payload_objects_gc
  on payload_objects (scope_id, created_at)
  where refcount = 0;

create index ix_state_nodes_gc
  on state_nodes (scope_id, created_at)
  where refcount = 0;

create index ix_responses_created_at
  on responses (scope_id, created_at);

create index ix_responses_prev
  on responses (scope_id, prev_response_id)
  where prev_response_id is not null;

create index ix_responses_gc
  on responses (scope_id, created_at)
  where child_refcount = 0 and lease_refcount = 0;

create index ix_response_leases_expiration
  on response_leases (scope_id, expires_at)
  where status = 'live' and expires_at is not null;

create index ix_conversations_last_used_at
  on conversations (scope_id, last_used_at);

create index ix_state_leaf_entries_namespace_ord
  on state_leaf_entries (scope_id, namespace_id, ord);

create function forbid_payload_objects_structural_update()
returns trigger
language plpgsql
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.payload_id is distinct from new.payload_id
    or old.payload_hash is distinct from new.payload_hash
    or old.payload_json is distinct from new.payload_json
    or old.created_at is distinct from new.created_at then
    raise exception 'payload_objects rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_payload_objects_immutable
before update on payload_objects
for each row execute function forbid_payload_objects_structural_update();

create function forbid_state_nodes_structural_update()
returns trigger
language plpgsql
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.node_id is distinct from new.node_id
    or old.kind is distinct from new.kind
    or old.left_id is distinct from new.left_id
    or old.right_id is distinct from new.right_id
    or old.item_count is distinct from new.item_count
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
as $$
begin
  raise exception '% rows are immutable', tg_table_name;
end;
$$;

create trigger trg_state_leaves_immutable
before update on state_leaves
for each row execute function forbid_any_update();

create trigger trg_state_leaf_entries_immutable
before update on state_leaf_entries
for each row execute function forbid_any_update();

create function forbid_responses_structural_update()
returns trigger
language plpgsql
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.response_id is distinct from new.response_id
    or old.prev_response_id is distinct from new.prev_response_id
    or old.full_state_root_id is distinct from new.full_state_root_id
    or old.created_at is distinct from new.created_at then
    raise exception 'responses rows are structurally immutable';
  end if;
  return new;
end;
$$;

create trigger trg_responses_immutable
before update on responses
for each row execute function forbid_responses_structural_update();

create function forbid_response_checkpoints_structural_update()
returns trigger
language plpgsql
as $$
begin
  if old.scope_id is distinct from new.scope_id
    or old.response_id is distinct from new.response_id
    or old.checkpoint_id is distinct from new.checkpoint_id
    or old.root_id is distinct from new.root_id
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

create function apply_payload_refcount_from_leaf_entry()
returns trigger
language plpgsql
as $$
begin
  if tg_op = 'INSERT' then
    update payload_objects
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and payload_id = new.payload_id;
    return new;
  elsif tg_op = 'DELETE' then
    update payload_objects
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and payload_id = old.payload_id;
    return old;
  end if;

  if old.scope_id is distinct from new.scope_id
    or old.payload_id is distinct from new.payload_id then
    update payload_objects
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and payload_id = old.payload_id;

    update payload_objects
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and payload_id = new.payload_id;
  end if;
  return new;
end;
$$;

create trigger trg_state_leaf_entries_payload_refcount
after insert or update or delete on state_leaf_entries
for each row execute function apply_payload_refcount_from_leaf_entry();

create function apply_concat_child_refcounts()
returns trigger
language plpgsql
as $$
begin
  if tg_op = 'INSERT' then
    if new.kind = 'concat' then
      update state_nodes
         set refcount = refcount + 1
       where scope_id = new.scope_id
         and node_id in (new.left_id, new.right_id);
    end if;
    return new;
  elsif tg_op = 'DELETE' then
    if old.kind = 'concat' then
      update state_nodes
         set refcount = refcount - 1
       where scope_id = old.scope_id
         and node_id in (old.left_id, old.right_id);
    end if;
    return old;
  end if;

  return new;
end;
$$;

create trigger trg_state_nodes_concat_refcount
after insert or update or delete on state_nodes
for each row execute function apply_concat_child_refcounts();

create function apply_response_refcounts()
returns trigger
language plpgsql
as $$
begin
  if tg_op = 'INSERT' then
    update state_nodes
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and node_id = new.full_state_root_id;

    if new.prev_response_id is not null then
      update responses
         set child_refcount = child_refcount + 1
       where scope_id = new.scope_id
         and response_id = new.prev_response_id;
    end if;
    return new;
  elsif tg_op = 'DELETE' then
    update state_nodes
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and node_id = old.full_state_root_id;

    if old.prev_response_id is not null then
      update responses
         set child_refcount = child_refcount - 1
       where scope_id = old.scope_id
         and response_id = old.prev_response_id;
    end if;
    return old;
  end if;

  return new;
end;
$$;

create trigger trg_responses_refcounts
after insert or update or delete on responses
for each row execute function apply_response_refcounts();

create function apply_checkpoint_root_refcounts()
returns trigger
language plpgsql
as $$
begin
  if tg_op = 'INSERT' then
    update state_nodes
       set refcount = refcount + 1
     where scope_id = new.scope_id
       and node_id = new.root_id;
    return new;
  elsif tg_op = 'DELETE' then
    update state_nodes
       set refcount = refcount - 1
     where scope_id = old.scope_id
       and node_id = old.root_id;
    return old;
  end if;

  return new;
end;
$$;

create trigger trg_response_checkpoints_refcount
after insert or update or delete on response_checkpoints
for each row execute function apply_checkpoint_root_refcounts();

create function apply_response_lease_refcounts()
returns trigger
language plpgsql
as $$
begin
  if tg_op = 'INSERT' then
    if new.status = 'live' then
      update responses
         set lease_refcount = lease_refcount + 1
       where scope_id = new.scope_id
         and response_id = new.response_id;
    end if;
    return new;
  elsif tg_op = 'DELETE' then
    if old.status = 'live' then
      update responses
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
    update responses
       set lease_refcount = lease_refcount - 1
     where scope_id = old.scope_id
       and response_id = old.response_id;
  end if;

  if new.status = 'live'
    and (
      old.status is distinct from 'live'
      or old.response_id is distinct from new.response_id
    ) then
    update responses
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
as $$
declare
  left_count bigint;
  right_count bigint;
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

  select item_count
    into left_count
    from state_nodes
   where scope_id = new.scope_id
     and node_id = new.left_id;

  select item_count
    into right_count
    from state_nodes
   where scope_id = new.scope_id
     and node_id = new.right_id;

  if left_count is null or right_count is null then
    raise exception 'concat state_node %/% references missing children',
      new.scope_id,
      new.node_id;
  end if;

  if new.item_count <> left_count + right_count then
    raise exception 'concat state_node %/% has item_count %, expected %',
      new.scope_id,
      new.node_id,
      new.item_count,
      left_count + right_count;
  end if;

  return new;
end;
$$;

create constraint trigger ct_state_nodes_consistency
after insert or update on state_nodes
deferrable initially deferred
for each row execute function validate_state_node_consistency();

create function validate_state_leaf_consistency()
returns trigger
language plpgsql
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

  select count(*), min(pos), max(pos)
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
after insert or update or delete on state_leaves
deferrable initially deferred
for each row execute function validate_state_leaf_consistency();

create constraint trigger ct_state_leaf_entries_consistency
after insert or update or delete on state_leaf_entries
deferrable initially deferred
for each row execute function validate_state_leaf_consistency();
"""


DOWNGRADE_SQL = r"""
drop trigger if exists ct_state_leaf_entries_consistency
  on state_leaf_entries;
drop trigger if exists ct_state_leaves_consistency on state_leaves;
drop trigger if exists ct_state_nodes_consistency on state_nodes;
drop trigger if exists trg_conversations_response_lease on conversations;
drop trigger if exists trg_response_leases_refcounts on response_leases;
drop trigger if exists trg_response_checkpoints_refcount
  on response_checkpoints;
drop trigger if exists trg_responses_refcounts on responses;
drop trigger if exists trg_state_nodes_concat_refcount on state_nodes;
drop trigger if exists trg_state_leaf_entries_payload_refcount
  on state_leaf_entries;
drop trigger if exists trg_response_leases_normalize_update
  on response_leases;
drop trigger if exists trg_response_checkpoints_immutable
  on response_checkpoints;
drop trigger if exists trg_responses_immutable on responses;
drop trigger if exists trg_state_leaf_entries_immutable
  on state_leaf_entries;
drop trigger if exists trg_state_leaves_immutable on state_leaves;
drop trigger if exists trg_state_nodes_immutable on state_nodes;
drop trigger if exists trg_payload_objects_immutable on payload_objects;

drop function if exists validate_state_leaf_consistency();
drop function if exists validate_state_node_consistency();
drop function if exists maintain_conversation_response_lease();
drop function if exists apply_response_lease_refcounts();
drop function if exists apply_checkpoint_root_refcounts();
drop function if exists apply_response_refcounts();
drop function if exists apply_concat_child_refcounts();
drop function if exists apply_payload_refcount_from_leaf_entry();
drop function if exists normalize_response_lease_update();
drop function if exists forbid_response_checkpoints_structural_update();
drop function if exists forbid_responses_structural_update();
drop function if exists forbid_any_update();
drop function if exists forbid_state_nodes_structural_update();
drop function if exists forbid_payload_objects_structural_update();

drop table if exists conversations;
drop table if exists response_leases;
drop table if exists checkpoint_namespace_counters;
drop table if exists response_checkpoints;
drop table if exists response_namespace_counters;
drop table if exists responses;
drop table if exists state_leaf_entries;
drop table if exists state_leaves;
drop table if exists state_nodes;
drop table if exists payload_objects;
drop table if exists ordinal_namespaces;
"""
