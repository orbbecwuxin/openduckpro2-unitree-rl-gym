# Auto Training Framework Plan

## Implementation Slices

- [x] Add reward override training entry point.
  Acceptance: a JSON file can override reward scales without editing `openduckpro2_config.py`.
  Verify: `python -m py_compile legged_gym/scripts/train_reward_variant.py`.

- [x] Add headless evaluator.
  Acceptance: an existing checkpoint can be rolled out and scored using multiple metric nodes.
  Verify: `python -m py_compile legged_gym/scripts/evaluate_policy.py`.

- [x] Add orchestrator.
  Acceptance: candidates can be scheduled across GPU 0 and GPU 1, and generated run files can be committed without staging unrelated files.
  Verify: `./auto_train_openduckpro2.sh --dry-run --no-commit`.

- [x] Add default config and operator docs.
  Acceptance: the default command is documented and configurable.
  Verify: read `docs/auto_training_framework.md`.

## Follow-Up Tasks

- [ ] Run a one-candidate smoke train with low `max_iterations`.
- [ ] Run evaluator on a known good checkpoint from `logs/openduckpro2`.
- [ ] Tune score node weights after inspecting trajectory samples.
- [ ] Expand mutation policy after collecting at least one full cycle.
