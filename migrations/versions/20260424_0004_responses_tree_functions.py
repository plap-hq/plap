"""Add responses tree functions.

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

create function responses.validate_namespace_cursors(
  p_cursors jsonb,
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
  if jsonb_typeof(p_cursors) is distinct from 'array' then
    raise exception '% namespace cursors must be a JSON array', p_context;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_cursors) with ordinality
   where jsonb_typeof(value) is distinct from 'object'
   limit 1;
  if v_bad_position is not null then
    raise exception '% namespace cursor at position % must be an object',
      p_context,
      v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_cursors) with ordinality
   where coalesce(value ->> 'namespace', '') = ''
   limit 1;
  if v_bad_position is not null then
    raise exception '% namespace cursor at position % is missing namespace',
      p_context,
      v_bad_position;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_cursors) counter(value)
    left join item_namespaces namespace
      on namespace.namespace_name = counter.value ->> 'namespace'
   where namespace.namespace_id is null
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace cursor references unknown namespace %',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_cursors) counter(value)
   group by value ->> 'namespace'
  having count(*) > 1
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace cursor duplicates namespace %',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_cursors) counter(value)
   where not (value ? 'next_ordinal')
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace cursor for % is missing next_ordinal',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_cursors) counter(value)
   where (value ->> 'next_ordinal') !~ '^-?\d+$'
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace cursor for % has invalid next_ordinal',
      p_context,
      v_namespace_name;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_cursors) counter(value)
   where (value ->> 'next_ordinal')::numeric < 0
      or (value ->> 'next_ordinal')::numeric > 9223372036854775807
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace cursor for % has out-of-range next_ordinal',
      p_context,
      v_namespace_name;
  end if;

  select namespace.namespace_name
    into v_namespace_name
    from item_namespaces namespace
    left join jsonb_array_elements(p_cursors) counter(value)
      on counter.value ->> 'namespace' = namespace.namespace_name
   where namespace.is_required
     and counter.value is null
   limit 1;
  if v_namespace_name is not null then
    raise exception '% namespace cursors missing required namespace %',
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

  insert into payloads (scope_id, payload_hash, payload_json)
  values (p_scope_id, p_payload_hash, p_payload_json)
  on conflict (scope_id, payload_hash) do nothing
  returning payload_id into v_payload_id;

  if v_payload_id is not null then
    return v_payload_id;
  end if;

  select payload_id, payload_json
    into v_payload_id, v_payload_json
    from payloads
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

create function responses.create_state_leaf(
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
  if v_entry_count > 128 then
    raise exception 'leaf entries must not exceed 128';
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
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where not (value ? 'ordinal')
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where (value ->> 'ordinal') !~ '^-?\d+$'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where (value ->> 'ordinal')::numeric < 0
      or (value ->> 'ordinal')::numeric > 9223372036854775807
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where value ->> 'payload_hash' is null
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where value ->> 'payload_hash' !~ '^[0-9a-fA-F]{64}$'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select ordinality - 1
    into v_bad_position
    from jsonb_array_elements(p_entries) with ordinality
   where not (value ? 'payload')
      or jsonb_typeof(value -> 'payload') is distinct from 'object'
   limit 1;
  if v_bad_position is not null then
    raise exception 'entry at position % is malformed', v_bad_position;
  end if;

  select value ->> 'namespace'
    into v_namespace_name
    from jsonb_array_elements(p_entries) entry(value)
    left join item_namespaces namespace
      on namespace.namespace_name = entry.value ->> 'namespace'
   where namespace.namespace_id is null
   limit 1;
  if v_namespace_name is not null then
    raise exception 'unknown ordinal namespace %', v_namespace_name;
  end if;

  create temporary table if not exists pg_temp.response_create_leaf_entries (
    item_index integer not null,
    namespace_id smallint not null,
    ordinal bigint not null,
    payload_hash bytea not null,
    payload_json jsonb not null,
    payload_id uuid
  ) on commit drop;
  truncate pg_temp.response_create_leaf_entries;

  insert into pg_temp.response_create_leaf_entries (
    item_index,
    namespace_id,
    ordinal,
    payload_hash,
    payload_json
  )
  select
    (entry.ordinality - 1)::integer,
    namespace.namespace_id,
    (entry.value ->> 'ordinal')::bigint,
    decode(entry.value ->> 'payload_hash', 'hex'),
    entry.value -> 'payload'
    from jsonb_array_elements(p_entries) with ordinality as entry(value, ordinality)
    join item_namespaces namespace
      on namespace.namespace_name = entry.value ->> 'namespace';

  select namespace.namespace_name, entry.ordinal
    into v_namespace_name, v_ord
    from pg_temp.response_create_leaf_entries entry
    join item_namespaces namespace
      on namespace.namespace_id = entry.namespace_id
   group by namespace.namespace_name, entry.ordinal
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

  insert into payloads (scope_id, payload_hash, payload_json)
  select distinct p_scope_id, payload_hash, payload_json
    from pg_temp.response_create_leaf_entries
  on conflict (scope_id, payload_hash) do nothing;

  select entry.payload_hash
    into v_payload_hash
    from pg_temp.response_create_leaf_entries entry
    join payloads payload
      on payload.scope_id = p_scope_id
     and payload.payload_hash = entry.payload_hash
   where payload.payload_json <> entry.payload_json
   limit 1;
  if v_payload_hash is not null then
    raise exception 'payload hash collision or non-canonical payload hash';
  end if;

  update pg_temp.response_create_leaf_entries entry
     set payload_id = payload.payload_id
    from payloads payload
   where payload.scope_id = p_scope_id
     and payload.payload_hash = entry.payload_hash;

  insert into state_nodes (
    scope_id,
    kind,
    height,
    item_count,
    child_count
  ) values (
    p_scope_id,
    'leaf',
    0,
    v_entry_count,
    0
  ) returning node_id into v_node_id;

  insert into state_leaves (scope_id, node_id, entry_count)
  values (p_scope_id, v_node_id, v_entry_count);

  insert into state_leaf_entries (
    scope_id,
    node_id,
    item_index,
    namespace_id,
    ordinal,
    payload_id
  )
  select p_scope_id, v_node_id, item_index, namespace_id, ordinal, payload_id
    from pg_temp.response_create_leaf_entries
   order by item_index;

  return v_node_id;
end;
$$;

create function responses.create_state_internal_node(
  p_scope_id uuid,
  p_child_ids bigint[]
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_child_count integer;
  v_duplicate_child_id bigint;
  v_existing_count integer;
  v_child_height smallint;
  v_height_count integer;
  v_item_count bigint;
  v_node_id bigint;
begin
  v_child_count = coalesce(array_length(p_child_ids, 1), 0);
  if v_child_count = 0 then
    raise exception 'internal nodes require at least one child';
  end if;
  if v_child_count = 1 then
    return p_child_ids[1];
  end if;
  if v_child_count > 64 then
    raise exception 'internal nodes cannot have more than 64 children';
  end if;

  select child_node_id
    into v_duplicate_child_id
    from unnest(p_child_ids) child(child_node_id)
   group by child_node_id
  having count(*) > 1
   limit 1;
  if v_duplicate_child_id is not null then
    raise exception 'internal node cannot reference child % more than once',
      v_duplicate_child_id;
  end if;

  select count(*), count(distinct node.height), min(node.height), sum(node.item_count)
    into v_existing_count, v_height_count, v_child_height, v_item_count
    from unnest(p_child_ids) with ordinality child(child_node_id, ordinality)
    join state_nodes node
      on node.scope_id = p_scope_id
     and node.node_id = child.child_node_id;

  if v_existing_count <> v_child_count then
    raise exception 'one or more child state nodes do not exist';
  end if;
  if v_height_count <> 1 then
    raise exception 'internal node children must have the same height';
  end if;

  insert into state_nodes (
    scope_id,
    kind,
    height,
    item_count,
    child_count
  ) values (
    p_scope_id,
    'internal',
    v_child_height + 1,
    v_item_count,
    v_child_count
  ) returning node_id into v_node_id;

  insert into state_node_children (
    scope_id,
    parent_node_id,
    child_index,
    child_node_id,
    child_item_count
  )
  select
    p_scope_id,
    v_node_id,
    (child.ordinality - 1)::smallint,
    child.child_node_id,
    node.item_count
    from unnest(p_child_ids) with ordinality child(child_node_id, ordinality)
    join state_nodes node
      on node.scope_id = p_scope_id
     and node.node_id = child.child_node_id
   order by child.ordinality;

  return v_node_id;
end;
$$;

create function responses.build_state_tree(
  p_scope_id uuid,
  p_entries jsonb
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_entry_count integer;
  v_start integer = 0;
  v_chunk jsonb;
  v_leaf_id bigint;
  v_nodes bigint[];
  v_namespace_name text;
  v_ord bigint;
begin
  if jsonb_typeof(p_entries) is distinct from 'array' then
    raise exception 'entries must be a JSON array';
  end if;

  v_entry_count = jsonb_array_length(p_entries);
  if v_entry_count <= 0 then
    raise exception 'entries must not be empty';
  end if;

  with valid_ord_text as materialized (
    select entry.value ->> 'namespace' as namespace_name,
           entry.value ->> 'ordinal' as ord_text
      from jsonb_array_elements(p_entries) entry(value)
     where jsonb_typeof(entry.value) = 'object'
       and coalesce(entry.value ->> 'namespace', '') <> ''
       and entry.value ? 'ordinal'
       and (entry.value ->> 'ordinal') ~ '^-?\d+$'
  ), valid_ord as materialized (
    select namespace_name, ord_text::bigint as ordinal
      from valid_ord_text
     where ord_text::numeric between 0 and 9223372036854775807
  )
  select namespace_name, ordinal
    into v_namespace_name, v_ord
    from valid_ord
   group by namespace_name, ordinal
  having count(*) > 1
   limit 1;
  if v_namespace_name is not null then
    raise exception 'duplicate ordinal %.%', v_namespace_name, v_ord;
  end if;

  create temporary table if not exists pg_temp.response_tree_level (
    item_index integer not null primary key,
    node_id bigint not null
  ) on commit drop;
  truncate pg_temp.response_tree_level;

  while v_start < v_entry_count loop
    select jsonb_agg(entry.value order by entry.ordinality)
      into v_chunk
      from jsonb_array_elements(p_entries) with ordinality as entry(value, ordinality)
     where entry.ordinality > v_start
       and entry.ordinality <= v_start + 128;

    v_leaf_id = responses.create_state_leaf(p_scope_id, v_chunk);
    insert into pg_temp.response_tree_level (item_index, node_id)
    values (v_start / 128, v_leaf_id);
    v_start = v_start + 128;
  end loop;

  select array_agg(node_id order by item_index)
    into v_nodes
    from pg_temp.response_tree_level;

  return responses.build_state_tree_from_roots(p_scope_id, v_nodes);
end;
$$;

create function responses.build_state_tree_from_roots(
  p_scope_id uuid,
  p_child_ids bigint[]
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_nodes bigint[];
  v_child_count integer;
  v_group_count integer;
  v_groups_left integer;
  v_group_size integer;
  v_group_start integer;
  v_group_end integer;
  v_pos integer;
  v_parent_id bigint;
begin
  v_child_count = coalesce(array_length(p_child_ids, 1), 0);
  if v_child_count = 0 then
    return null;
  end if;
  if v_child_count = 1 then
    return p_child_ids[1];
  end if;

  create temporary table if not exists pg_temp.response_child_tree_level (
    item_index integer not null primary key,
    node_id bigint not null
  ) on commit drop;
  truncate pg_temp.response_child_tree_level;

  insert into pg_temp.response_child_tree_level (item_index, node_id)
  select (child.ordinality - 1)::integer, child.node_id
    from unnest(p_child_ids) with ordinality child(node_id, ordinality);

  loop
    select array_agg(node_id order by item_index)
      into v_nodes
      from pg_temp.response_child_tree_level;
    exit when array_length(v_nodes, 1) = 1;

    truncate pg_temp.response_child_tree_level;
    v_child_count = array_length(v_nodes, 1);
    v_group_count = ceiling(v_child_count::numeric / 64)::integer;
    v_group_start = 1;
    v_pos = 0;

    while v_group_start <= v_child_count loop
      v_groups_left = v_group_count - v_pos;
      v_group_size = ceiling(
        (v_child_count - v_group_start + 1)::numeric / v_groups_left
      )::integer;
      v_group_end = v_group_start + v_group_size - 1;
      v_parent_id = responses.create_state_internal_node(
        p_scope_id,
        v_nodes[v_group_start:v_group_end]
      );
      insert into pg_temp.response_child_tree_level (item_index, node_id)
      values (v_pos, v_parent_id);
      v_group_start = v_group_end + 1;
      v_pos = v_pos + 1;
    end loop;
  end loop;

  return v_nodes[1];
end;
$$;

create function responses.list_state_items(
  p_scope_id uuid,
  p_root_id bigint,
  p_start_index bigint default 0,
  p_limit bigint default null
)
returns table (
  item_position bigint,
  namespace text,
  ordinal bigint,
  payload_hash text,
  payload jsonb
)
language sql
stable
set search_path = responses, public
as $$
  with recursive walk(node_id, offset_items, item_count, kind) as (
    select node_id, 0::bigint, item_count, kind
      from state_nodes
     where scope_id = p_scope_id
       and node_id = p_root_id
       and (
         p_limit is null
         or p_limit > 0
       )
    union all
    select
      edge.child_node_id,
      child_offset.offset_items,
      child.item_count,
      child.kind
      from walk
      join state_node_children edge
        on edge.scope_id = p_scope_id
       and edge.parent_node_id = walk.node_id
      join state_nodes child
        on child.scope_id = edge.scope_id
       and child.node_id = edge.child_node_id
      left join lateral (
        select sum(prev.child_item_count)::bigint as item_count
          from state_node_children prev
          where prev.scope_id = edge.scope_id
            and prev.parent_node_id = edge.parent_node_id
            and prev.child_index < edge.child_index
      ) previous_items on true
      cross join lateral (
        select walk.offset_items
          + coalesce(previous_items.item_count, 0)::bigint as offset_items
      ) child_offset
     where walk.kind = 'internal'
       and child_offset.offset_items + child.item_count > p_start_index
       and (
         p_limit is null
         or child_offset.offset_items < p_start_index + p_limit
       )
  )
  select
    walk.offset_items + entry.item_index,
    namespace.namespace_name,
    entry.ordinal,
    encode(payload.payload_hash, 'hex'),
    payload.payload_json
    from walk
    join state_leaf_entries entry
      on entry.scope_id = p_scope_id
     and entry.node_id = walk.node_id
    join item_namespaces namespace
      on namespace.namespace_id = entry.namespace_id
    join payloads payload
      on payload.scope_id = entry.scope_id
     and payload.payload_id = entry.payload_id
   where walk.kind = 'leaf'
     and walk.offset_items + entry.item_index >= p_start_index
     and (
       p_limit is null
       or walk.offset_items + entry.item_index < p_start_index + p_limit
     )
   order by walk.offset_items + entry.item_index;
$$;

create function responses.concat_state_trees(
  p_scope_id uuid,
  p_left_state_root_id bigint,
  p_right_state_root_id bigint
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_left_kind text;
  v_left_height smallint;
  v_left_child_count smallint;
  v_right_kind text;
  v_right_height smallint;
  v_right_child_count smallint;
  v_left_children bigint[];
  v_right_children bigint[];
  v_child_count integer;
  v_joined_id bigint;
  v_joined_height smallint;
  v_joined_children bigint[];
begin
  if p_left_state_root_id is null then
    return p_right_state_root_id;
  end if;
  if p_right_state_root_id is null then
    return p_left_state_root_id;
  end if;

  select kind, height, child_count
    into v_left_kind, v_left_height, v_left_child_count
    from state_nodes
   where scope_id = p_scope_id
      and node_id = p_left_state_root_id;
  if v_left_height is null then
    raise exception 'left state root %/% does not exist',
      p_scope_id,
      p_left_state_root_id;
  end if;

  select kind, height, child_count
    into v_right_kind, v_right_height, v_right_child_count
    from state_nodes
   where scope_id = p_scope_id
      and node_id = p_right_state_root_id;
  if v_right_height is null then
    raise exception 'right state root %/% does not exist',
      p_scope_id,
      p_right_state_root_id;
  end if;

  if v_left_height = v_right_height then
    if v_left_kind = 'internal'
      and v_right_kind = 'internal'
      and v_left_child_count + v_right_child_count <= 64 then
      select array_agg(child_node_id order by child_index)
        into v_left_children
        from state_node_children
       where scope_id = p_scope_id
         and parent_node_id = p_left_state_root_id;

      select array_agg(child_node_id order by child_index)
        into v_right_children
        from state_node_children
       where scope_id = p_scope_id
         and parent_node_id = p_right_state_root_id;

      return responses.create_state_internal_node(
        p_scope_id,
        v_left_children || v_right_children
      );
    end if;

    -- Relaxed B+ join: when equal-height internal roots cannot be merged under
    -- one 64-child parent, keep both roots as children of a new parent. A more
    -- tightly packed join could redistribute their child arrays, but that would
    -- rewrite more edges and reduce structural sharing without changing the
    -- logical sequence. Add redistribution later only if workload data shows
    -- underfilled internal nodes are a real read/write bottleneck.
    return responses.build_state_tree_from_roots(
      p_scope_id,
      array[p_left_state_root_id, p_right_state_root_id]::bigint[]
    );
  end if;

  if v_left_height > v_right_height then
    select array_agg(child_node_id order by child_index)
      into v_left_children
      from state_node_children
     where scope_id = p_scope_id
       and parent_node_id = p_left_state_root_id;
    v_child_count = array_length(v_left_children, 1);
    v_joined_id = responses.concat_state_trees(
      p_scope_id,
      v_left_children[v_child_count],
      p_right_state_root_id
    );
    select height
      into v_joined_height
      from state_nodes
     where scope_id = p_scope_id
       and node_id = v_joined_id;

    if v_joined_height = v_left_height - 1 then
      return responses.build_state_tree_from_roots(
        p_scope_id,
        coalesce(v_left_children[1:v_child_count - 1], '{}'::bigint[])
          || v_joined_id
      );
    end if;

    select array_agg(child_node_id order by child_index)
      into v_joined_children
      from state_node_children
     where scope_id = p_scope_id
       and parent_node_id = v_joined_id;
    return responses.build_state_tree_from_roots(
      p_scope_id,
      coalesce(v_left_children[1:v_child_count - 1], '{}'::bigint[])
        || v_joined_children
    );
  end if;

  select array_agg(child_node_id order by child_index)
    into v_right_children
    from state_node_children
   where scope_id = p_scope_id
     and parent_node_id = p_right_state_root_id;
  v_child_count = array_length(v_right_children, 1);
  v_joined_id = responses.concat_state_trees(
    p_scope_id,
    p_left_state_root_id,
    v_right_children[1]
  );
  select height
    into v_joined_height
    from state_nodes
   where scope_id = p_scope_id
     and node_id = v_joined_id;

  if v_joined_height = v_right_height - 1 then
    return responses.build_state_tree_from_roots(
      p_scope_id,
      array[v_joined_id]::bigint[]
        || coalesce(v_right_children[2:v_child_count], '{}'::bigint[])
    );
  end if;

  select array_agg(child_node_id order by child_index)
    into v_joined_children
    from state_node_children
   where scope_id = p_scope_id
     and parent_node_id = v_joined_id;
  return responses.build_state_tree_from_roots(
    p_scope_id,
    v_joined_children
      || coalesce(v_right_children[2:v_child_count], '{}'::bigint[])
  );
end;
$$;

create function responses.split_state_tree(
  p_scope_id uuid,
  p_root_id bigint,
  p_index bigint
)
returns table (
  left_state_root_id bigint,
  right_state_root_id bigint
)
language plpgsql
set search_path = responses, public
as $$
declare
  v_node_kind text;
  v_item_count bigint;
  v_split_slot smallint;
  v_split_child_id bigint;
  v_before_count bigint;
  v_child_left_state_root_id bigint;
  v_child_right_state_root_id bigint;
  v_prefix_children bigint[];
  v_suffix_children bigint[];
  v_prefix_state_root_id bigint;
  v_suffix_state_root_id bigint;
  v_left_entries jsonb;
  v_right_entries jsonb;
begin
  if p_root_id is null then
    if p_index <> 0 then
      raise exception 'split index % is out of range for empty state', p_index;
    end if;
    left_state_root_id = null;
    right_state_root_id = null;
    return next;
    return;
  end if;

  select kind, item_count
    into v_node_kind, v_item_count
    from state_nodes
   where scope_id = p_scope_id
     and node_id = p_root_id;

  if v_item_count is null then
    raise exception 'state root %/% does not exist', p_scope_id, p_root_id;
  end if;
  if p_index < 0 or p_index > v_item_count then
    raise exception 'split index % is out of range 0..%', p_index, v_item_count;
  end if;

  if p_index = 0 then
    left_state_root_id = null;
    right_state_root_id = p_root_id;
    return next;
    return;
  end if;
  if p_index = v_item_count then
    left_state_root_id = p_root_id;
    right_state_root_id = null;
    return next;
    return;
  end if;

  if v_node_kind = 'leaf' then
    select jsonb_agg(
      jsonb_build_object(
        'namespace', item.namespace,
        'ordinal', item.ordinal,
        'payload_hash', item.payload_hash,
        'payload', item.payload
      ) order by item.item_position
    ) into v_left_entries
      from responses.list_state_items(p_scope_id, p_root_id) item
     where item.item_position < p_index;

    select jsonb_agg(
      jsonb_build_object(
        'namespace', item.namespace,
        'ordinal', item.ordinal,
        'payload_hash', item.payload_hash,
        'payload', item.payload
      ) order by item.item_position
    ) into v_right_entries
      from responses.list_state_items(p_scope_id, p_root_id) item
     where item.item_position >= p_index;

    left_state_root_id = responses.build_state_tree(p_scope_id, v_left_entries);
    right_state_root_id = responses.build_state_tree(p_scope_id, v_right_entries);
    return next;
    return;
  end if;

  select child_index, child_node_id, before_count
    into v_split_slot, v_split_child_id, v_before_count
    from (
      select
        edge.child_index,
        edge.child_node_id,
        coalesce(
          sum(edge.child_item_count) over (
            order by edge.child_index rows between unbounded preceding and 1 preceding
          ),
          0
        )::bigint as before_count,
        edge.child_item_count
        from state_node_children edge
       where edge.scope_id = p_scope_id
         and edge.parent_node_id = p_root_id
    ) edge
   where p_index >= before_count
     and p_index < before_count + child_item_count
   order by child_index
   limit 1;

  select split_result.left_state_root_id, split_result.right_state_root_id
    into v_child_left_state_root_id, v_child_right_state_root_id
    from responses.split_state_tree(
      p_scope_id,
      v_split_child_id,
      p_index - v_before_count
    ) split_result;

  select array_agg(child_node_id order by child_index)
    into v_prefix_children
    from state_node_children
   where scope_id = p_scope_id
     and parent_node_id = p_root_id
     and child_index < v_split_slot;

  select array_agg(child_node_id order by child_index)
    into v_suffix_children
    from state_node_children
   where scope_id = p_scope_id
     and parent_node_id = p_root_id
     and child_index > v_split_slot;

  v_prefix_state_root_id = responses.build_state_tree_from_roots(
    p_scope_id,
    v_prefix_children
  );
  v_suffix_state_root_id = responses.build_state_tree_from_roots(
    p_scope_id,
    v_suffix_children
  );
  left_state_root_id = responses.concat_state_trees(
    p_scope_id,
    v_prefix_state_root_id,
    v_child_left_state_root_id
  );
  right_state_root_id = responses.concat_state_trees(
    p_scope_id,
    v_child_right_state_root_id,
    v_suffix_state_root_id
  );
  return next;
end;
$$;

create function responses.splice_state_tree(
  p_scope_id uuid,
  p_root_id bigint,
  p_start_index bigint,
  p_delete_count bigint,
  p_insert_root_id bigint default null
)
returns bigint
language plpgsql
set search_path = responses, public
as $$
declare
  v_left_state_root_id bigint;
  v_tail_state_root_id bigint;
  v_deleted_state_root_id bigint;
  v_right_state_root_id bigint;
  v_combined_state_root_id bigint;
begin
  if p_delete_count < 0 then
    raise exception 'delete_count must not be negative';
  end if;

  select split_result.left_state_root_id, split_result.right_state_root_id
    into v_left_state_root_id, v_tail_state_root_id
    from responses.split_state_tree(p_scope_id, p_root_id, p_start_index) split_result;

  select split_result.left_state_root_id, split_result.right_state_root_id
    into v_deleted_state_root_id, v_right_state_root_id
    from responses.split_state_tree(
      p_scope_id,
      v_tail_state_root_id,
      p_delete_count
    ) split_result;

  v_combined_state_root_id = responses.concat_state_trees(
    p_scope_id,
    v_left_state_root_id,
    p_insert_root_id
  );

  return responses.concat_state_trees(
    p_scope_id,
    v_combined_state_root_id,
    v_right_state_root_id
  );
end;
$$;

create function responses.create_response_record(
  p_scope_id uuid,
  p_response_id text,
  p_prev_response_id text,
  p_state_root_id bigint,
  p_namespace_cursors jsonb,
  p_checkpoints jsonb default '[]'::jsonb,
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
  v_counter record;
  v_checkpoint record;
  v_checkpoint_counter record;
  v_namespace_name text;
  v_namespace_id smallint;
  v_checkpoint_id bigint;
begin
  perform responses.validate_namespace_cursors(
    p_namespace_cursors,
    'response'
  );

  if jsonb_typeof(p_checkpoints) is distinct from 'array' then
    raise exception 'checkpoints must be a JSON array';
  end if;

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

  insert into response_records (
    scope_id,
    response_id,
    prev_response_id,
    state_root_id,
    status,
    completed_at,
    fields
  ) values (
    p_scope_id,
    p_response_id,
    p_prev_response_id,
    p_state_root_id,
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

  for v_counter in
    select value
      from jsonb_array_elements(p_namespace_cursors)
  loop
    v_namespace_name = v_counter.value ->> 'namespace';
    select namespace_id
      into v_namespace_id
      from item_namespaces
     where namespace_name = v_namespace_name;

    insert into response_namespace_cursors (
      scope_id,
      response_id,
      namespace_id,
      next_ordinal
    ) values (
      p_scope_id,
      p_response_id,
      v_namespace_id,
      (v_counter.value ->> 'next_ordinal')::bigint
    );
  end loop;

  for v_checkpoint in
    select value
      from jsonb_array_elements(p_checkpoints)
  loop
    insert into response_checkpoints (scope_id, response_id, state_root_id)
    values (
      p_scope_id,
      p_response_id,
      (v_checkpoint.value ->> 'state_root_id')::bigint
    )
    returning checkpoint_id into v_checkpoint_id;

    if not (v_checkpoint.value ? 'namespace_cursors') then
      raise exception 'checkpoint namespace_cursors are required';
    end if;

    perform responses.validate_namespace_cursors(
      v_checkpoint.value -> 'namespace_cursors',
      'checkpoint'
    );

    for v_checkpoint_counter in
      select value
        from jsonb_array_elements(v_checkpoint.value -> 'namespace_cursors')
    loop
      v_namespace_name = v_checkpoint_counter.value ->> 'namespace';
      select namespace_id
        into v_namespace_id
        from item_namespaces
       where namespace_name = v_namespace_name;

      insert into checkpoint_namespace_cursors (
        scope_id,
        response_id,
        checkpoint_id,
        namespace_id,
        next_ordinal
      ) values (
        p_scope_id,
        p_response_id,
        v_checkpoint_id,
        v_namespace_id,
        (v_checkpoint_counter.value ->> 'next_ordinal')::bigint
      );
    end loop;
  end loop;

  return p_response_id;
end;
$$;

create function responses.append_response(
  p_scope_id uuid,
  p_response_id text,
  p_prev_response_id text,
  p_items jsonb,
  p_namespace_cursors jsonb,
  p_checkpoints jsonb default '[]'::jsonb,
  p_retention interval default interval '30 days',
  p_status text default 'completed',
  p_completed_at timestamptz default null,
  p_fields jsonb default '{}'::jsonb
)
returns table (
  response_id text,
  state_root_id bigint
)
language plpgsql
set search_path = responses, public
as $$
declare
  v_items_root_id bigint;
  v_prev_root_id bigint;
  v_root_id bigint;
begin
  v_items_root_id = responses.build_state_tree(p_scope_id, p_items);

  if p_prev_response_id is null then
    v_root_id = v_items_root_id;
  else
    select record.state_root_id
      into v_prev_root_id
      from response_records record
     where record.scope_id = p_scope_id
       and record.response_id = p_prev_response_id;

    if v_prev_root_id is null then
      raise exception 'previous response %/% does not exist',
        p_scope_id,
        p_prev_response_id;
    end if;

    v_root_id = responses.concat_state_trees(
      p_scope_id,
      v_prev_root_id,
      v_items_root_id
    );
  end if;

  perform responses.create_response_record(
    p_scope_id,
    p_response_id,
    p_prev_response_id,
    v_root_id,
    p_namespace_cursors,
    p_checkpoints,
    p_retention,
    p_status,
    p_completed_at,
    p_fields
  );

  response_id = p_response_id;
  state_root_id = v_root_id;
  return next;
end;
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
  p_response_id text
)
returns void
language plpgsql
set search_path = responses, public
as $$
begin
  insert into conversations (scope_id, conversation_id, current_response_id)
  values (p_scope_id, p_conversation_id, p_response_id)
  on conflict (scope_id, conversation_id) do update
     set current_response_id = excluded.current_response_id,
         last_used_at = now();
end;
$$;
"""


DOWNGRADE_SQL = r"""
drop function if exists responses.move_conversation_head(uuid, text, text);
drop function if exists responses.release_response_lease(uuid, text, text);
drop function if exists responses.create_or_refresh_response_lease(
  uuid,
  text,
  text,
  text,
  timestamptz
);
drop function if exists responses.append_response(
  uuid,
  text,
  text,
  jsonb,
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
  bigint,
  jsonb,
  jsonb,
  interval,
  text,
  timestamptz,
  jsonb
);
drop function if exists responses.splice_state_tree(
  uuid,
  bigint,
  bigint,
  bigint,
  bigint
);
drop function if exists responses.split_state_tree(uuid, bigint, bigint);
drop function if exists responses.concat_state_trees(uuid, bigint, bigint);
drop function if exists responses.list_state_items(uuid, bigint, bigint, bigint);
drop function if exists responses.build_state_tree(uuid, jsonb);
drop function if exists responses.build_state_tree_from_roots(uuid, bigint[]);
drop function if exists responses.create_state_internal_node(uuid, bigint[]);
drop function if exists responses.create_state_leaf(uuid, jsonb);
drop function if exists responses.get_or_create_payload(uuid, bytea, jsonb);
drop function if exists responses.validate_namespace_cursors(jsonb, text);
"""
