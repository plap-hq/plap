You are reviewing a completed benchmark trial and writing a concise critique of the source agent's performance.

The source trial name is `{source_trial_name}`.
The task name is `{task_name}`.

Inputs available to you:
- Task files: `{task_dir}`
- Source trial files: `{trial_dir}`

Read the source trial evidence before judging it.

Important files to inspect:
- `{trial_dir}/result.json`
- `{trial_dir}/trial.log`
- `{trial_dir}/agent/trajectory.json` if present
- `{trial_dir}/agent/mini-swe-agent.trajectory.json` if present
- `{trial_dir}/verifier/test-stdout.txt` if present
- `{trial_dir}/exception.txt` if present

Also read task context when helpful:
- `{task_dir}/instruction.md`
- `{task_dir}/task.toml`

Your job:
1. Determine what the source agent actually attempted.
2. Determine whether it made meaningful progress toward the task.
3. Identify the main reason it succeeded, partially succeeded, or failed.
4. Distinguish agent mistakes from environment, harness, or task-spec issues when possible.
5. Keep the critique evidence-based and specific.

Write a valid JSON object to `{critique_result_path}` with at least these required fields:

```json
{
  "feedback": "3-8 sentence critique with specific evidence references",
  "rating": "good",
  "tags": ["progress", "correctness"]
}
```

Rules for required fields:
- `feedback` must be a non-empty string.
- `rating` must be exactly `good` or `bad`.
- `tags` must contain at least one short string tag.

Suggested tag vocabulary:
- `correctness`
- `progress`
- `agent-loop`
- `tool-use`
- `environment`
- `verifier`
- `task-spec`
- `partial-fix`
- `wrong-fix`
- `no-progress`

Rating guidance:
- Use `good` when the source trial made strong or promising progress, even if it ultimately failed, or when the failure appears mostly due to verifier/task/environment issues.
- Use `bad` when the source trial made little useful progress, followed the wrong path, introduced obviously incorrect changes, or failed in a way that is mainly attributable to the source agent.

You may include extra JSON fields if useful, for example:
- `confidence`
- `root_cause`
- `evidence`
- `next_step`

If you include extra fields, keep them compact and JSON-serializable.

Optionally, write a human-readable Markdown version of the critique to `{critique_markdown_path}`.

Do not print the final critique only to stdout. The critique is only considered complete when the JSON file has been written to `{critique_result_path}`.
