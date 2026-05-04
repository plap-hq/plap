"""Add responses delta-log lineage functions.

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

create function responses.validate_response_items(
  p_items jsonb,
  p_context text
)
returns void
language plpgsql
set search_path = responses, public
as $$
declare
  v_bad_position bigint;
begin
  if jsonb_typeof(p_items) is distinct from 'array' then
    raise exception '% items must be a JSON array', p_context;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_items) with ordinality
   where jsonb_typeof(value) is distinct from 'object'
   limit 1;
  if v_bad_position is not null then
    raise exception '% item at index % must be an object', p_context, v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_items) with ordinality
   where coalesce(value ->> 'type', '') = ''
   limit 1;
  if v_bad_position is not null then
    raise exception '% item at index % is missing type', p_context, v_bad_position;
  end if;
end;
$$;

create function responses.payload_hash(
  p_payload_json jsonb
)
returns bytea
language sql
immutable
set search_path = responses, public
as $$
  select digest(convert_to(p_payload_json::text, 'UTF8'), 'sha256')
$$;

create function responses.get_or_create_payload(
  p_scope_id uuid,
  p_payload_json jsonb
)
returns uuid
language plpgsql
set search_path = responses, public
as $$
declare
  v_payload_hash bytea;
  v_payload_id uuid;
  v_existing jsonb;
begin
  if p_payload_json is null or jsonb_typeof(p_payload_json) is distinct from 'object' then
    raise exception 'payload_json must be a JSON object';
  end if;

  v_payload_hash = responses.payload_hash(p_payload_json);

  insert into payloads (scope_id, payload_hash, payload_json)
  values (p_scope_id, v_payload_hash, p_payload_json)
  on conflict (scope_id, payload_hash) do nothing
  returning payload_id into v_payload_id;

  if v_payload_id is not null then
    return v_payload_id;
  end if;

  select payload_id, payload_json
    into v_payload_id, v_existing
    from payloads
   where scope_id = p_scope_id
     and payload_hash = v_payload_hash;

  if v_payload_id is null then
    raise exception 'payload dedupe lookup failed for scope %', p_scope_id;
  end if;

  if v_existing <> p_payload_json then
    raise exception 'payload hash collision or non-canonical payload hash';
  end if;

  return v_payload_id;
end;
$$;

create function responses.insert_response_input_items(
  p_scope_id uuid,
  p_response_id text,
  p_input_items jsonb
)
returns void
language plpgsql
set search_path = responses, public
as $$
declare
  v_item record;
  v_payload_id uuid;
begin
  perform responses.validate_response_items(p_input_items, 'response input');

  for v_item in
    select value, ordinality - 1 as input_index
      from jsonb_array_elements(p_input_items) with ordinality
  loop
    v_payload_id = responses.get_or_create_payload(p_scope_id, v_item.value);

    insert into response_input_items (
      scope_id,
      response_id,
      input_index,
      payload_id
    ) values (
      p_scope_id,
      p_response_id,
      v_item.input_index,
      v_payload_id
    );
  end loop;
end;
$$;

create function responses.insert_response_output_items(
  p_scope_id uuid,
  p_response_id text,
  p_output_items jsonb
)
returns void
language plpgsql
set search_path = responses, public
as $$
declare
  v_item record;
  v_payload_id uuid;
begin
  perform responses.validate_response_items(p_output_items, 'response output');

  for v_item in
    select value, ordinality - 1 as output_index
      from jsonb_array_elements(p_output_items) with ordinality
  loop
    v_payload_id = responses.get_or_create_payload(p_scope_id, v_item.value);

    insert into response_output_items (
      scope_id,
      response_id,
      output_index,
      payload_id
    ) values (
      p_scope_id,
      p_response_id,
      v_item.output_index,
      v_payload_id
    );
  end loop;
end;
$$;

create function responses.response_items_contain_compaction(
  p_items jsonb
)
returns boolean
language sql
immutable
set search_path = responses, public
as $$
  select exists(
    select 1
      from jsonb_array_elements(p_items) item(value)
     where item.value ->> 'type' = 'compaction'
  )
$$;

create function responses.create_response_record(
  p_scope_id uuid,
  p_response_id text,
  p_prev_response_id text default null,
  p_input_items jsonb default '[]'::jsonb,
  p_output_items jsonb default '[]'::jsonb,
  p_retention interval default interval '30 days',
  p_status text default 'completed',
  p_completed_at timestamptz default null,
  p_fields jsonb default '{}'::jsonb
)
returns text
language plpgsql
set search_path = responses, public
as $$
declare
  v_prev_base text;
  v_replay_base text;
begin
  perform responses.validate_response_items(p_input_items, 'response input');
  perform responses.validate_response_items(p_output_items, 'response output');

  if p_status not in (
    'queued',
    'in_progress',
    'completed',
    'failed',
    'cancelled',
    'incomplete'
  ) then
    raise exception 'invalid response status %', p_status;
  end if;

  if jsonb_typeof(p_fields) is distinct from 'object' then
    raise exception 'response fields must be a JSON object';
  end if;

  if p_prev_response_id = p_response_id then
    raise exception 'response cannot reference itself as previous response';
  end if;

  if p_prev_response_id is not null then
    select replay_base_response_id
      into v_prev_base
      from response_records
     where scope_id = p_scope_id
       and response_id = p_prev_response_id;

    if v_prev_base is null then
      raise exception 'previous response %/% does not exist', p_scope_id, p_prev_response_id;
    end if;
  end if;

  if p_prev_response_id is null or responses.response_items_contain_compaction(p_output_items) then
    v_replay_base = p_response_id;
  else
    v_replay_base = v_prev_base;
  end if;

  insert into response_records (
    scope_id,
    response_id,
    prev_response_id,
    replay_base_response_id,
    status,
    completed_at,
    fields
  ) values (
    p_scope_id,
    p_response_id,
    p_prev_response_id,
    v_replay_base,
    p_status,
    coalesce(p_completed_at, case when p_status = 'completed' then now() end),
    p_fields
  );

  if p_retention is not null then
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
      'response',
      p_response_id,
      'live',
      now() + p_retention
    );
  end if;

  perform responses.insert_response_input_items(p_scope_id, p_response_id, p_input_items);
  perform responses.insert_response_output_items(p_scope_id, p_response_id, p_output_items);

  return p_response_id;
end;
$$;

create function responses.append_response(
  p_scope_id uuid,
  p_response_id text,
  p_prev_response_id text,
  p_input_items jsonb default '[]'::jsonb,
  p_output_items jsonb default '[]'::jsonb,
  p_retention interval default interval '30 days',
  p_status text default 'completed',
  p_completed_at timestamptz default null,
  p_fields jsonb default '{}'::jsonb
)
returns table (
  response_id text,
  replay_base_response_id text
)
language plpgsql
set search_path = responses, public
as $$
begin
  perform responses.create_response_record(
    p_scope_id,
    p_response_id,
    p_prev_response_id,
    p_input_items,
    p_output_items,
    p_retention,
    p_status,
    p_completed_at,
    p_fields
  );

  return query
  select record.response_id, record.replay_base_response_id
    from response_records record
   where record.scope_id = p_scope_id
     and record.response_id = p_response_id;
end;
$$;

create function responses.list_response_replay(
  p_scope_id uuid,
  p_response_id text
)
returns table (
  replay_response_id text,
  direction text,
  item_index integer,
  payload_hash text,
  payload jsonb
)
language sql
stable
set search_path = responses, public
as $$
  with recursive lineage(response_id, prev_response_id, replay_base_response_id, depth) as (
    select record.response_id, record.prev_response_id, record.replay_base_response_id, 0::bigint
      from response_records record
     where record.scope_id = p_scope_id
       and record.response_id = p_response_id
    union all
    select prev.response_id, prev.prev_response_id, lineage.replay_base_response_id, lineage.depth + 1
      from lineage
      join response_records prev
        on prev.scope_id = p_scope_id
       and prev.response_id = lineage.prev_response_id
     where lineage.response_id <> lineage.replay_base_response_id
  ), ordered_lineage as (
    select response_id,
           row_number() over (order by depth desc) - 1 as response_position
      from lineage
  ), replay_items as (
    select ordered_lineage.response_position,
           0 as direction_order,
           ordered_lineage.response_id,
           'input'::text as direction,
           item.input_index as item_index,
           payload.payload_hash,
           payload.payload_json
      from ordered_lineage
      join response_input_items item
        on item.scope_id = p_scope_id
       and item.response_id = ordered_lineage.response_id
      join payloads payload
        on payload.scope_id = item.scope_id
       and payload.payload_id = item.payload_id
    union all
    select ordered_lineage.response_position,
           1 as direction_order,
           ordered_lineage.response_id,
           'output'::text as direction,
           item.output_index as item_index,
           payload.payload_hash,
           payload.payload_json
      from ordered_lineage
      join response_output_items item
        on item.scope_id = p_scope_id
       and item.response_id = ordered_lineage.response_id
      join payloads payload
        on payload.scope_id = item.scope_id
       and payload.payload_id = item.payload_id
  )
  select response_id,
         direction,
         item_index,
         encode(payload_hash, 'hex'),
         payload_json
    from replay_items
   order by response_position, direction_order, item_index
$$;

create function responses.create_or_refresh_response_lease(
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
    ) returning lease_id into v_lease_id;
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

create function responses.release_response_lease(
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

create function responses.move_conversation_head(
  p_scope_id uuid,
  p_conversation_id text,
  p_response_id text,
  p_retention interval default interval '30 days'
)
returns void
language plpgsql
set search_path = responses, public
as $$
begin
  insert into conversations (
    scope_id,
    conversation_id,
    current_response_id,
    retention_expires_at
  ) values (
    p_scope_id,
    p_conversation_id,
    p_response_id,
    case when p_retention is null then null else now() + p_retention end
  )
  on conflict (scope_id, conversation_id) do update
     set current_response_id = excluded.current_response_id,
         last_used_at = now(),
         retention_expires_at = excluded.retention_expires_at;
end;
$$;
"""


DOWNGRADE_SQL = r"""
drop function if exists responses.move_conversation_head(
  uuid,
  text,
  text,
  interval
);
drop function if exists responses.release_response_lease(uuid, text, text);
drop function if exists responses.create_or_refresh_response_lease(
  uuid,
  text,
  text,
  text,
  timestamptz
);
drop function if exists responses.list_response_replay(uuid, text);
drop function if exists responses.append_response(
  uuid,
  text,
  text,
  jsonb,
  jsonb,
  interval,
  text,
  timestamptz,
  jsonb
);
drop function if exists responses.create_response_record(
  uuid,
  text,
  text,
  jsonb,
  jsonb,
  interval,
  text,
  timestamptz,
  jsonb
);
drop function if exists responses.response_items_contain_compaction(jsonb);
drop function if exists responses.insert_response_output_items(uuid, text, jsonb);
drop function if exists responses.insert_response_input_items(uuid, text, jsonb);
drop function if exists responses.get_or_create_payload(uuid, jsonb);
drop function if exists responses.payload_hash(jsonb);
drop function if exists responses.validate_response_items(jsonb, text);
"""
