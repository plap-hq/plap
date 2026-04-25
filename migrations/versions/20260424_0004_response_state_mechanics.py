"""Add response state mechanics functions.

Revision ID: 20260424_0004
Revises: 20260424_0003
Create Date: 2026-04-24 00:04:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260424_0004"
down_revision: str | None = "20260424_0003"
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

set search_path = responses, public;

create function responses.validate_namespace_counters(
  p_counters jsonb,
  p_context text
)
returns void
language plpgsql
set search_path = responses, public
as $$
declare
  v_bad_position bigint;
  v_namespace_name text;
begin
  if jsonb_typeof(p_counters) is distinct from 'array' then
    raise exception '% namespace counters must be a JSON array', p_context;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_counters) with ordinality
   where jsonb_typeof(value) is distinct from 'object'
   limit 1;
  if v_bad_position is not null then
    raise exception '% namespace counter at position % must be an object',
      p_context,
      v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_counters) with ordinality
   where coalesce(value ->> 'namespace', '') = ''
   limit 1;
  if v_bad_position is not null then
    raise exception '% namespace counter at position % is missing namespace',
      p_context,
      v_bad_position;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_counters) counter(value)
    left join ordinal_namespaces namespace
      on namespace.namespace_name = counter.value ->> 'namespace'
   where namespace.namespace_id is null
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace counter references unknown namespace %',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_counters) counter(value)
   group by value ->> 'namespace'
  having count(*) > 1
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace counter duplicates namespace %',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_counters) counter(value)
   where not (value ? 'next_ord')
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace counter for % is missing next_ord',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_counters) counter(value)
   where (value ->> 'next_ord') !~ '^-?\d+$'
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace counter for % has invalid next_ord',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_counters) counter(value)
   where (value ->> 'next_ord')::bigint < 0
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace counter for % has negative next_ord',
      p_context,
      v_namespace_name;
  end if;

  select namespace.namespace_name
    into v_namespace_name
    from ordinal_namespaces namespace
    left join jsonb_array_elements(p_counters) counter(value)
      on counter.value ->> 'namespace' = namespace.namespace_name
   where namespace.is_required
     and counter.value is null
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace counters missing required namespace %',
      p_context,
      v_namespace_name;
  end if;
end;
$$;

create function responses.get_or_create_payload(
  p_scope_id uuid,
  p_payload_hash bytea,
  p_payload_json jsonb
)
returns uuid
language plpgsql
set search_path = responses, public
as $$
declare
  v_payload_id uuid;
  v_payload_json jsonb;
begin
  if p_payload_hash is null or octet_length(p_payload_hash) <> 32 then
    raise exception 'payload_hash must be a 32-byte digest';
  end if;

  if p_payload_json is null then
    raise exception 'payload_json is required';
  end if;

  insert into payload_objects (
    scope_id,
    payload_hash,
    payload_json
  ) values (
    p_scope_id,
    p_payload_hash,
    p_payload_json
  )
  on conflict (scope_id, payload_hash) do nothing
  returning payload_id into v_payload_id;

  if v_payload_id is not null then
    return v_payload_id;
  end if;

  select payload_id, payload_json
    into v_payload_id, v_payload_json
    from payload_objects
   where scope_id = p_scope_id
     and payload_hash = p_payload_hash;

  if v_payload_id is null then
    raise exception 'payload dedupe lookup failed for scope %', p_scope_id;
  end if;

  if v_payload_json <> p_payload_json then
    raise exception 'payload hash collision or non-canonical payload hash';
  end if;

  return v_payload_id;
end;
$$;

create function responses.create_leaf(
  p_scope_id uuid,
  p_entries jsonb
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_bad_position bigint;
  v_entry_count integer;
  v_namespace_name text;
  v_ord bigint;
  v_payload_hash_hex text;
  v_payload_hash bytea;
  v_node_id bigint;
begin
  if jsonb_typeof(p_entries) is distinct from 'array' then
    raise exception 'entries must be a JSON array';
  end if;

  v_entry_count = jsonb_array_length(p_entries);
  if v_entry_count <= 0 then
    raise exception 'entries must not be empty';
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where jsonb_typeof(value) is distinct from 'object'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % must be an object', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where coalesce(value ->> 'namespace', '') = ''
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is missing namespace', v_bad_position;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_entries) entry(value)
    left join ordinal_namespaces namespace
      on namespace.namespace_name = entry.value ->> 'namespace'
   where namespace.namespace_id is null
   limit 1;
  if v_namespace_name is not null then
    raise exception 'unknown ordinal namespace %', v_namespace_name;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where not (value ? 'ord')
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is missing ord', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where (value ->> 'ord') !~ '^-?\d+$'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % has invalid ord', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where (value ->> 'ord')::bigint < 0
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % has negative ord', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where value ->> 'payload_hash' is null
      or value ->> 'payload_hash' !~ '^[0-9a-fA-F]{64}$'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % has invalid payload_hash', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where not (value ? 'payload')
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is missing payload', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where jsonb_typeof(value -> 'payload') is distinct from 'object'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % payload must be an object', v_bad_position;
  end if;

  create temporary table if not exists pg_temp.response_create_leaf_entries (
    pos integer not null,
    namespace_id smallint not null,
    ord bigint not null,
    payload_hash bytea not null,
    payload_json jsonb not null,
    payload_id uuid
  ) on commit drop;
  truncate pg_temp.response_create_leaf_entries;

  insert into pg_temp.response_create_leaf_entries (
    pos,
    namespace_id,
    ord,
    payload_hash,
    payload_json
  )
  select
    (entry.ordinality - 1)::integer,
    namespace.namespace_id,
    (entry.value ->> 'ord')::bigint,
    decode(entry.value ->> 'payload_hash', 'hex'),
    entry.value -> 'payload'
    from jsonb_array_elements(p_entries) with ordinality as entry(value, ordinality)
    join ordinal_namespaces namespace
      on namespace.namespace_name = entry.value ->> 'namespace';

  select namespace.namespace_name, entry.ord
    into v_namespace_name, v_ord
    from pg_temp.response_create_leaf_entries entry
    join ordinal_namespaces namespace
      on namespace.namespace_id = entry.namespace_id
   group by namespace.namespace_name, entry.ord
  having count(*) > 1
   limit 1;
  if v_namespace_name is not null then
    raise exception 'duplicate ordinal %.%', v_namespace_name, v_ord;
  end if;

  select payload_hash
    into v_payload_hash
    from pg_temp.response_create_leaf_entries
   group by payload_hash
  having count(distinct payload_json) > 1
   limit 1;
  if v_payload_hash is not null then
    raise exception 'payload hash collision or non-canonical payload hash';
  end if;

  insert into payload_objects (
    scope_id,
    payload_hash,
    payload_json
  )
  select distinct
    p_scope_id,
    payload_hash,
    payload_json
    from pg_temp.response_create_leaf_entries
  on conflict (scope_id, payload_hash) do nothing;

  select entry.payload_hash
    into v_payload_hash
    from pg_temp.response_create_leaf_entries entry
    join payload_objects payload
      on payload.scope_id = p_scope_id
     and payload.payload_hash = entry.payload_hash
   where payload.payload_json <> entry.payload_json
   limit 1;
  if v_payload_hash is not null then
    raise exception 'payload hash collision or non-canonical payload hash';
  end if;

  update pg_temp.response_create_leaf_entries entry
     set payload_id = payload.payload_id
    from payload_objects payload
   where payload.scope_id = p_scope_id
     and payload.payload_hash = entry.payload_hash;

  insert into state_nodes (scope_id, kind, item_count)
  values (p_scope_id, 'leaf', v_entry_count)
  returning node_id into v_node_id;

  insert into state_leaves (scope_id, node_id, entry_count)
  values (p_scope_id, v_node_id, v_entry_count);

  insert into state_leaf_entries (
    scope_id,
    node_id,
    pos,
    namespace_id,
    ord,
    payload_id
  )
  select
    p_scope_id,
    v_node_id,
    pos,
    namespace_id,
    ord,
    payload_id
    from pg_temp.response_create_leaf_entries
   order by pos;

  return v_node_id;
end;
$$;

create function responses.create_concat(
  p_scope_id uuid,
  p_left_id bigint,
  p_right_id bigint
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_left_count bigint;
  v_right_count bigint;
  v_node_id bigint;
begin
  if p_left_id = p_right_id then
    raise exception 'concat children must be distinct';
  end if;

  select item_count
    into v_left_count
    from state_nodes
   where scope_id = p_scope_id
     and node_id = p_left_id;

  if v_left_count is null then
    raise exception 'left state node %/% does not exist', p_scope_id, p_left_id;
  end if;

  select item_count
    into v_right_count
    from state_nodes
   where scope_id = p_scope_id
     and node_id = p_right_id;

  if v_right_count is null then
    raise exception 'right state node %/% does not exist', p_scope_id, p_right_id;
  end if;

  insert into state_nodes (
    scope_id,
    kind,
    left_id,
    right_id,
    item_count
  ) values (
    p_scope_id,
    'concat',
    p_left_id,
    p_right_id,
    v_left_count + v_right_count
  )
  returning node_id into v_node_id;

  return v_node_id;
end;
$$;

create function responses.create_response(
  p_scope_id uuid,
  p_response_id text,
  p_prev_response_id text,
  p_full_state_root_id bigint,
  p_namespace_counters jsonb,
  p_checkpoints jsonb default '[]'::jsonb
)
returns text
language plpgsql
set search_path = responses, public
as $$
declare
  v_counter record;
  v_checkpoint record;
  v_checkpoint_counter record;
  v_namespace_name text;
  v_namespace_id smallint;
  v_checkpoint_id bigint;
begin
  perform responses.validate_namespace_counters(
    p_namespace_counters,
    'response'
  );

  if jsonb_typeof(p_checkpoints) is distinct from 'array' then
    raise exception 'checkpoints must be a JSON array';
  end if;

  insert into responses (
    scope_id,
    response_id,
    prev_response_id,
    full_state_root_id
  ) values (
    p_scope_id,
    p_response_id,
    p_prev_response_id,
    p_full_state_root_id
  );

  for v_counter in
    select value
      from jsonb_array_elements(p_namespace_counters)
  loop
    v_namespace_name = v_counter.value ->> 'namespace';
    select namespace_id
      into v_namespace_id
      from ordinal_namespaces
     where namespace_name = v_namespace_name;

    if v_namespace_id is null then
      raise exception 'unknown ordinal namespace %', v_namespace_name;
    end if;

    insert into response_namespace_counters (
      scope_id,
      response_id,
      namespace_id,
      next_ord
    ) values (
      p_scope_id,
      p_response_id,
      v_namespace_id,
      (v_counter.value ->> 'next_ord')::bigint
    );
  end loop;

  for v_checkpoint in
    select value
      from jsonb_array_elements(p_checkpoints)
  loop
    insert into response_checkpoints (
      scope_id,
      response_id,
      root_id
    ) values (
      p_scope_id,
      p_response_id,
      (v_checkpoint.value ->> 'root_id')::bigint
    )
    returning checkpoint_id into v_checkpoint_id;

    if not (v_checkpoint.value ? 'namespace_counters') then
      raise exception 'checkpoint namespace_counters are required';
    end if;

    perform responses.validate_namespace_counters(
      v_checkpoint.value -> 'namespace_counters',
      'checkpoint'
    );

    for v_checkpoint_counter in
      select value
        from jsonb_array_elements(v_checkpoint.value -> 'namespace_counters')
    loop
      v_namespace_name = v_checkpoint_counter.value ->> 'namespace';
      select namespace_id
        into v_namespace_id
        from ordinal_namespaces
       where namespace_name = v_namespace_name;

      insert into checkpoint_namespace_counters (
        scope_id,
        response_id,
        checkpoint_id,
        namespace_id,
        next_ord
      ) values (
        p_scope_id,
        p_response_id,
        v_checkpoint_id,
        v_namespace_id,
        (v_checkpoint_counter.value ->> 'next_ord')::bigint
      );
    end loop;
  end loop;

  return p_response_id;
end;
$$;

create function responses.append_items(
  p_scope_id uuid,
  p_response_id text,
  p_prev_response_id text,
  p_items jsonb,
  p_namespace_counters jsonb,
  p_checkpoints jsonb default '[]'::jsonb
)
returns table (
  created_response_id text,
  root_node_id bigint
)
language plpgsql
set search_path = responses, public
as $$
declare
  v_leaf_id bigint;
  v_prev_root_id bigint;
  v_root_id bigint;
begin
  v_leaf_id = responses.create_leaf(p_scope_id, p_items);

  if p_prev_response_id is null then
    v_root_id = v_leaf_id;
  else
    select full_state_root_id
      into v_prev_root_id
      from responses
     where scope_id = p_scope_id
       and response_id = p_prev_response_id;

    if v_prev_root_id is null then
      raise exception 'previous response %/% does not exist',
        p_scope_id,
        p_prev_response_id;
    end if;

    v_root_id = responses.create_concat(
      p_scope_id,
      v_prev_root_id,
      v_leaf_id
    );
  end if;

  perform responses.create_response(
    p_scope_id,
    p_response_id,
    p_prev_response_id,
    v_root_id,
    p_namespace_counters,
    p_checkpoints
  );

  created_response_id = p_response_id;
  root_node_id = v_root_id;
  return next;
end;
$$;

create function responses.create_or_refresh_lease(
  p_scope_id uuid,
  p_response_id text,
  p_owner_type text,
  p_owner_id text,
  p_expires_at timestamptz default null
)
returns uuid
language plpgsql
set search_path = responses, public
as $$
declare
  v_lease_id uuid;
begin
  update response_leases
     set response_id = p_response_id,
         status = 'live',
         expires_at = p_expires_at
   where scope_id = p_scope_id
     and owner_type = p_owner_type
     and owner_id = p_owner_id
     and status = 'live'
  returning lease_id into v_lease_id;

  if v_lease_id is not null then
    return v_lease_id;
  end if;

  begin
    insert into response_leases (
      scope_id,
      response_id,
      owner_type,
      owner_id,
      status,
      expires_at
    ) values (
      p_scope_id,
      p_response_id,
      p_owner_type,
      p_owner_id,
      'live',
      p_expires_at
    )
    returning lease_id into v_lease_id;
  exception when unique_violation then
    update response_leases
       set response_id = p_response_id,
           status = 'live',
           expires_at = p_expires_at
     where scope_id = p_scope_id
       and owner_type = p_owner_type
       and owner_id = p_owner_id
       and status = 'live'
    returning lease_id into v_lease_id;
  end;

  return v_lease_id;
end;
$$;

create function responses.release_lease(
  p_scope_id uuid,
  p_owner_type text,
  p_owner_id text
)
returns void
language plpgsql
set search_path = responses, public
as $$
begin
  delete from response_leases
   where scope_id = p_scope_id
     and owner_type = p_owner_type
     and owner_id = p_owner_id
     and status = 'live';
end;
$$;

create function responses.move_conversation(
  p_scope_id uuid,
  p_conversation_id text,
  p_response_id text
)
returns void
language plpgsql
set search_path = responses, public
as $$
begin
  insert into conversations (
    scope_id,
    conversation_id,
    current_response_id
  ) values (
    p_scope_id,
    p_conversation_id,
    p_response_id
  )
  on conflict (scope_id, conversation_id) do update
     set current_response_id = excluded.current_response_id,
         last_used_at = now();
end;
$$;
"""


DOWNGRADE_SQL = r"""
drop function if exists responses.move_conversation(uuid, text, text);
drop function if exists responses.release_lease(uuid, text, text);
drop function if exists responses.create_or_refresh_lease(
  uuid,
  text,
  text,
  text,
  timestamptz
);
drop function if exists responses.append_items(
  uuid,
  text,
  text,
  jsonb,
  jsonb,
  jsonb
);
drop function if exists responses.create_response(
  uuid,
  text,
  text,
  bigint,
  jsonb,
  jsonb
);
drop function if exists responses.create_concat(uuid, bigint, bigint);
drop function if exists responses.create_leaf(uuid, jsonb);
drop function if exists responses.get_or_create_payload(uuid, bytea, jsonb);
drop function if exists responses.validate_namespace_counters(jsonb, text);
"""
