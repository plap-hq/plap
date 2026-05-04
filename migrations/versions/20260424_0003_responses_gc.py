"""Add responses delta-log GC procedures.

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
create schema if not exists responses;

create extension if not exists pg_cron;

set search_path = responses, public;

create function responses.gc_delete_response_if_unreferenced(
  p_scope_id uuid,
  p_response_id text,
  p_max_steps integer default 500
)
returns integer
language plpgsql
set search_path = responses, public
as $$
declare
  v_current_response_id text := p_response_id;
  v_prev_response_id text;
  v_child_refcount bigint;
  v_lease_refcount bigint;
  v_touched_payload_ids uuid[];
  v_deleted integer := 0;
begin
  if p_max_steps is null or p_max_steps <= 0 then
    return 0;
  end if;

  while v_current_response_id is not null and v_deleted < p_max_steps loop
    select prev_response_id, child_refcount, lease_refcount
      into v_prev_response_id, v_child_refcount, v_lease_refcount
      from response_records
     where scope_id = p_scope_id
       and response_id = v_current_response_id
     for update;

    if not found then
      exit;
    end if;

    if (v_child_refcount + v_lease_refcount) > 0 then
      exit;
    end if;

    select coalesce(array_agg(distinct payload_id), '{}'::uuid[])
      into v_touched_payload_ids
      from (
        select payload_id
          from response_input_items
         where scope_id = p_scope_id
           and response_id = v_current_response_id
        union
        select payload_id
          from response_output_items
         where scope_id = p_scope_id
           and response_id = v_current_response_id
      ) payload_ids;

    delete from response_leases
     where scope_id = p_scope_id
       and response_id = v_current_response_id
       and status <> 'live';

    delete from response_records
     where scope_id = p_scope_id
       and response_id = v_current_response_id;

    if not found then
      exit;
    end if;

    if array_length(v_touched_payload_ids, 1) is not null then
      delete from payloads
       where scope_id = p_scope_id
         and payload_id = any(v_touched_payload_ids)
         and refcount = 0;
    end if;

    v_deleted := v_deleted + 1;
    v_current_response_id := v_prev_response_id;
  end loop;

  return v_deleted;
end;
$$;

create procedure responses.gc_expire_response_leases(
  p_batch_size integer default 200
)
language plpgsql
set search_path = responses, public
as $$
declare
  v_lease record;
  v_deleted integer;
  v_remaining integer := greatest(coalesce(p_batch_size, 0), 0);
begin
  while v_remaining > 0 loop
    with claimed as (
      select scope_id, lease_id
        from response_leases
       where status = 'live'
         and expires_at is not null
         and expires_at <= now()
       order by expires_at
       for update skip locked
       limit 1
    )
    delete from response_leases l
      using claimed c
     where l.scope_id = c.scope_id
       and l.lease_id = c.lease_id
     returning l.scope_id, l.lease_id, l.response_id
      into v_lease;

    exit when not found;

    v_deleted := responses.gc_delete_response_if_unreferenced(v_lease.scope_id, v_lease.response_id, v_remaining);
    v_remaining := greatest(v_remaining - greatest(v_deleted, 1), 0);
  end loop;
end;
$$;

create procedure responses.gc_prune_conversations(
  p_batch_size integer default 200
)
language plpgsql
set search_path = responses, public
as $$
declare
  v_conversation record;
  v_deleted integer;
  v_remaining integer := greatest(coalesce(p_batch_size, 0), 0);
begin
  while v_remaining > 0 loop
    with doomed as (
      select scope_id, conversation_id
        from conversations
       where retention_expires_at is not null
         and retention_expires_at <= now()
       order by retention_expires_at
       for update skip locked
       limit 1
    )
    delete from conversations c
      using doomed d
     where c.scope_id = d.scope_id
       and c.conversation_id = d.conversation_id
     returning c.scope_id, c.current_response_id
      into v_conversation;

    exit when not found;

    v_deleted := responses.gc_delete_response_if_unreferenced(v_conversation.scope_id, v_conversation.current_response_id, v_remaining);
    v_remaining := greatest(v_remaining - greatest(v_deleted, 1), 0);
  end loop;
end;
$$;

create procedure responses.gc_prune_unreferenced_responses(
  p_batch_size integer default 200
)
language plpgsql
set search_path = responses, public
as $$
declare
  v_response record;
  v_deleted integer;
  v_remaining integer := greatest(coalesce(p_batch_size, 0), 0);
begin
  while v_remaining > 0 loop
    select scope_id, response_id
      into v_response
      from response_records
     where child_refcount = 0
       and lease_refcount = 0
     order by created_at
     for update skip locked
     limit 1;

    exit when not found;

    v_deleted := responses.gc_delete_response_if_unreferenced(v_response.scope_id, v_response.response_id, v_remaining);
    v_remaining := greatest(v_remaining - greatest(v_deleted, 1), 0);
  end loop;
end;
$$;

create procedure responses.gc_prune_payloads(
  p_batch_size integer default 500
)
language plpgsql
set search_path = responses, public
as $$
begin
  with claimed as (
    select scope_id, payload_id
      from payloads
     where refcount = 0
     order by created_at
     for update skip locked
     limit p_batch_size
  )
  delete from payloads p
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
      'call responses.gc_expire_response_leases(200);'
    );
  end if;

  if not exists (
    select 1 from cron.job where jobname = 'prune-stale-conversations'
  ) then
    perform cron.schedule(
      'prune-stale-conversations',
      '5 * * * *',
      'call responses.gc_prune_conversations(200);'
    );
  end if;

  if not exists (
    select 1 from cron.job where jobname = 'prune-unreferenced-responses'
  ) then
    perform cron.schedule(
      'prune-unreferenced-responses',
      '10 * * * *',
      'call responses.gc_prune_unreferenced_responses(200);'
    );
  end if;

  if not exists (
    select 1 from cron.job where jobname = 'prune-zero-ref-payloads'
  ) then
    perform cron.schedule(
      'prune-zero-ref-payloads',
      '15 * * * *',
      'call responses.gc_prune_payloads(500);'
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

drop procedure if exists responses.gc_prune_payloads(integer);
drop procedure if exists responses.gc_prune_unreferenced_responses(integer);
drop procedure if exists responses.gc_prune_conversations(integer);
drop procedure if exists responses.gc_expire_response_leases(integer);
drop function if exists responses.gc_delete_response_if_unreferenced(uuid, text, integer);
"""
