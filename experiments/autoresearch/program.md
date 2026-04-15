# ADN AutoResearch Program (v0)

This file defines the first research loop policy for autonomous improvement of ADN/MATDO-E.

## Mission

Improve long-context inference quality-efficiency trade-offs for the ADN stack under fixed compute/runtime budgets.

Primary optimization targets:

1. Increase long-context quality (especially Needle 128K behavior).
2. Maintain or improve serving efficiency (throughput / latency / memory).
3. Avoid regressions in stability and reproducibility.

## Scope

Allowed edit regions (first phase):

- `src/attnres/`
- `src/qttt/`
- `src/gating/`
- `experiments/matdo/`
- lightweight config/runner glue required to test a change

Disallowed edits (first phase):

- broad repo refactors
- dependency upgrades unrelated to trial goal
- destructive git history operations

## Trial Policy

Each trial must follow:

1. Propose a **single focused hypothesis**.
2. Implement the smallest change required.
3. Run a fixed, short evaluation command.
4. Score against the current best checkpoint/metric policy.
5. Keep only if it improves objective and passes constraints.

## Objective and Constraints

Default objective for v0:

- Primary metric: `needle_128k_accuracy` (maximize)

Default soft secondary signal:

- `throughput_tokens_per_sec` (maximize)

Default hard constraints:

- `p99_latency_ms<=400`
- `peak_memory_gb<=78`

## Experiment hygiene

- One hypothesis per trial.
- Log every trial in JSONL.
- Keep a patch artifact for every trial.
- Promote only tested improvements.

## Suggested first search space

1. qTTT step schedule:
   - adaptive cap
   - warm-start / early-stop thresholds
2. AttnRes block/window policy:
   - modest block count adjustments
   - read-window strategy
3. Gating threshold calibration:
   - target-rate smoothing
   - entropy/max-probability blending

## Example run command

From repo root:

```bash
# One-time login for Cursor CLI agent:
cursor-agent login

# Optional: configure your external agent command template once.
# Placeholders: {workspace} {program} {trial_id} {trial_dir} {prompt}
export AUTORESEARCH_AGENT_CMD_TEMPLATE='echo "Plug in your real agent CLI here: {workspace}"'

python3 experiments/autoresearch/run_agent.py \
  --iterations 5 \
  --trial-cmd "python3 experiments/run_experiments_unified.py --category core --quick" \
  --primary-metric needle_128k_accuracy \
  --primary-direction max \
  --constraint "p99_latency_ms<=400" \
  --constraint "peak_memory_gb<=78"
```

Dry-run check (plans commands without executing agent/training/scoring):

```bash
python3 experiments/autoresearch/run_agent.py \
  --iterations 1 \
  --trial-cmd "python3 experiments/run_experiments_unified.py --category core --quick" \
  --dry-run
```

If `AUTORESEARCH_AGENT_CMD_TEMPLATE` is not set, `run_agent.py` invokes
`agent_driver.py`, which defaults to:
`cursor-agent -p --trust --workspace <trial_worktree> "<generated_prompt>"`.
