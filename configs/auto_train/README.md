# OpenDuckPro2 Training Configuration

`openduckpro2_default.json` is a safe 10K template. It intentionally contains no candidate and cannot start until a candidate has a matching open draft pull request in the training-control repository.

Each candidate must run as one process, one `run_name`, one log directory, and one TensorBoard series through iteration 10000. Iterations 1000 through 9000 are checkpoint milestones only; evaluation starts after training reaches 10000.

Only existing nonzero reward scales may be overridden. Do not place host paths, checkpoints, logs, tokens, or generated PR metadata from completed experiments in this repository.

For a controlled command-distribution experiment, a candidate may also provide
`command_ranges`. This changes command sampling only; it does not change the G1
reward contract. A fixed forward command uses identical lower and upper bounds:
`"lin_vel_x": [0.3, 0.3]`.

Candidate shape:

```json
{
  "name": "descriptive-candidate-name",
  "seed": 1,
  "reward_scales": {
    "tracking_lin_vel": 1.0
  },
  "command_ranges": {
    "lin_vel_x": [0.3, 0.3],
    "lin_vel_y": [0.0, 0.0],
    "ang_vel_yaw": [0.0, 0.0]
  },
  "training_pr": {
    "repo": "orbbecwuxin/openduck-training-control",
    "number": 123,
    "url": "https://github.com/orbbecwuxin/openduck-training-control/pull/123",
    "head_ref": "train/example-candidate",
    "base_ref": "main",
    "created_with": "gh",
    "state": "OPEN",
    "is_draft": true
  }
}
```
