"""Add response state GC procedures.

Revision ID: 20260424_0003
Revises: 20260424_0002
Create Date: 2026-04-24 00:03:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260424_0003"
down_revision: str | None = "20260424_0002"
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
create schema if not exists response_state;

create extension if not exists pg_cron;

set search_path = response_state, public;

create procedure response_state.gc_try_delete_state_node(
  p_scope_id uuid,
  p_node_id bigint
)
language plpgsql
set search_path = response_state, public
as $$
declare
  v_kind text;
  v_left_id bigint;
  v_right_id bigint;
  v_refcount bigint;
  v_payload_ids uuid[];
begin
  select kind, left_id, right_id, refcount
    into v_kind, v_left_id, v_right_id, v_refcount
    from state_nodes
   where scope_id = p_scope_id
     and node_id = p_node_id
   for update;

  if not found then
    return;
  end if;

  if v_refcount > 0 then
    return;
  end if;

  if v_kind = 'leaf' then
    select coalesce(array_agg(payload_id), '{}'::uuid[])
      into v_payload_ids
      from state_leaf_entries
     where scope_id = p_scope_id
       and node_id = p_node_id;

    delete from state_leaf_entries
     where scope_id = p_scope_id
       and node_id = p_node_id;

    delete from state_leaves
     where scope_id = p_scope_id
       and node_id = p_node_id;
  end if;

  delete from state_nodes
   where scope_id = p_scope_id
     and node_id = p_node_id;

  if v_kind = 'concat' then
    call response_state.gc_try_delete_state_node(p_scope_id, v_left_id);
    call response_state.gc_try_delete_state_node(p_scope_id, v_right_id);
  elsif v_kind = 'leaf' then
    delete from payload_objects
     where scope_id = p_scope_id
       and payload_id = any(v_payload_ids)
       and refcount = 0;
  end if;
end;
$$;

create procedure response_state.gc_try_delete_response(
  p_scope_id uuid,
  p_response_id text
)
language plpgsql
set search_path = response_state, public
as $$
declare
  v_prev_response_id text;
  v_full_state_root_id bigint;
  v_child_refcount bigint;
  v_lease_refcount bigint;
  v_checkpoint_root_ids bigint[];
  v_root_id bigint;
begin
  select
    prev_response_id,
    full_state_root_id,
    child_refcount,
    lease_refcount
  into
    v_prev_response_id,
    v_full_state_root_id,
    v_child_refcount,
    v_lease_refcount
    from responses
   where scope_id = p_scope_id
     and response_id = p_response_id
   for update;

  if not found then
    return;
  end if;

  if (v_child_refcount + v_lease_refcount) > 0 then
    return;
  end if;

  select coalesce(array_agg(root_id), '{}'::bigint[])
    into v_checkpoint_root_ids
    from response_checkpoints
   where scope_id = p_scope_id
     and response_id = p_response_id;

  delete from response_leases
   where scope_id = p_scope_id
     and response_id = p_response_id
     and status <> 'live';

  delete from responses
   where scope_id = p_scope_id
     and response_id = p_response_id;

  call response_state.gc_try_delete_state_node(p_scope_id, v_full_state_root_id);

  foreach v_root_id in array v_checkpoint_root_ids
  loop
    call response_state.gc_try_delete_state_node(p_scope_id, v_root_id);
  end loop;

  if v_prev_response_id is not null then
    call response_state.gc_try_delete_response(p_scope_id, v_prev_response_id);
  end if;
end;
$$;

create procedure response_state.gc_expire_leases(
  p_batch_size integer default 200
)
language plpgsql
set search_path = response_state, public
as $$
declare
  v_lease record;
begin
  for v_lease in
    with claimed as (
      select scope_id, lease_id
        from response_leases
       where status = 'live'
         and expires_at is not null
         and expires_at <= now()
       order by expires_at
       for update skip locked
       limit p_batch_size
    )
    update response_leases l
       set status = 'expiring'
      from claimed c
     where l.scope_id = c.scope_id
       and l.lease_id = c.lease_id
     returning l.scope_id, l.lease_id, l.response_id
  loop
    delete from response_leases
     where scope_id = v_lease.scope_id
       and lease_id = v_lease.lease_id;

    call response_state.gc_try_delete_response(
      v_lease.scope_id,
      v_lease.response_id
    );
  end loop;
end;
$$;

create procedure response_state.gc_prune_conversations(
  p_batch_size integer default 200,
  p_max_idle interval default interval '30 days'
)
language plpgsql
set search_path = response_state, public
as $$
declare
  v_conversation record;
begin
  for v_conversation in
    with doomed as (
      select scope_id, conversation_id
        from conversations
       where last_used_at <= now() - p_max_idle
       order by last_used_at
       for update skip locked
       limit p_batch_size
    )
    delete from conversations c
      using doomed d
     where c.scope_id = d.scope_id
       and c.conversation_id = d.conversation_id
     returning c.scope_id, c.current_response_id
  loop
    call response_state.gc_try_delete_response(
      v_conversation.scope_id,
      v_conversation.current_response_id
    );
  end loop;
end;
$$;

create procedure response_state.gc_prune_unreferenced_responses(
  p_batch_size integer default 200
)
language plpgsql
set search_path = response_state, public
as $$
declare
  v_response record;
begin
  for v_response in
    select scope_id, response_id
      from responses
     where child_refcount = 0
       and lease_refcount = 0
     order by created_at
      for update skip locked
      limit p_batch_size
  loop
    call response_state.gc_try_delete_response(
      v_response.scope_id,
      v_response.response_id
    );
  end loop;
end;
$$;

create procedure response_state.gc_prune_payloads(
  p_batch_size integer default 500
)
language plpgsql
set search_path = response_state, public
as $$
begin
  with claimed as (
    select scope_id, payload_id
      from payload_objects
     where refcount = 0
     order by created_at
     for update skip locked
     limit p_batch_size
  )
  delete from payload_objects p
    using claimed c
   where p.scope_id = c.scope_id
     and p.payload_id = c.payload_id;
end;
$$;

do $$
begin
  if not exists (
    select 1 from cron.job where jobname = 'expire-response-leases'
  ) then
    perform cron.schedule(
      'expire-response-leases',
      '* * * * *',
      'call response_state.gc_expire_leases(200);'
    );
  end if;

  if not exists (
    select 1 from cron.job where jobname = 'prune-stale-conversations'
  ) then
    perform cron.schedule(
      'prune-stale-conversations',
      '5 * * * *',
      'call response_state.gc_prune_conversations(200, interval ''30 days'');'
    );
  end if;

  if not exists (
    select 1 from cron.job where jobname = 'prune-unreferenced-responses'
  ) then
    perform cron.schedule(
      'prune-unreferenced-responses',
      '10 * * * *',
      'call response_state.gc_prune_unreferenced_responses(200);'
    );
  end if;

  if not exists (
    select 1 from cron.job where jobname = 'prune-zero-ref-payloads'
  ) then
    perform cron.schedule(
      'prune-zero-ref-payloads',
      '15 * * * *',
      'call response_state.gc_prune_payloads(500);'
    );
  end if;
end;
$$;
"""


DOWNGRADE_SQL = r"""
do $$
declare
  v_job record;
begin
  if to_regclass('cron.job') is not null then
    for v_job in
      select jobid
        from cron.job
       where jobname in (
         'expire-response-leases',
         'prune-stale-conversations',
         'prune-unreferenced-responses',
         'prune-zero-ref-payloads'
       )
    loop
      perform cron.unschedule(v_job.jobid);
    end loop;
  end if;
end;
$$;

drop procedure if exists response_state.gc_prune_payloads(integer);
drop procedure if exists response_state.gc_prune_unreferenced_responses(integer);
drop procedure if exists response_state.gc_prune_conversations(integer, interval);
drop procedure if exists response_state.gc_expire_leases(integer);
drop procedure if exists response_state.gc_try_delete_response(uuid, text);
drop procedure if exists response_state.gc_try_delete_state_node(uuid, bigint);
"""
