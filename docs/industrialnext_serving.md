# Industrial Next PI0.5 serving and rollout comparison

Use a selected EMA export from [training](industrialnext_training.md). It carries
normalization, tokenizer, profile and receipts; serving does not access training
Parquet/video data. Loading rejects mismatched variant/contracts or changed assets.
The server performs finite real-model warmup before opening its socket.

```bash
CUDA_VISIBLE_DEVICES=0 ./inx_serve.sh 100 \
  --checkpoint-dir checkpoints/pi05_taro_exp_100/RUN/exports/14999
# Stop the first process before switching the shared endpoint:
CUDA_VISIBLE_DEVICES=0 ./inx_serve.sh full \
  --checkpoint-dir checkpoints/pi05_taro_exp_full/RUN/exports/14999
```

Both default to `127.0.0.1:10012`; host, port and explicit timing/postprocessing overrides are available (`--help`). `--num-steps` records an explicit flow-sampling experiment; qualify it separately. Test instances
use explicit isolated ports, such as 10014. No server in this repository publishes
ROS commands or starts robot motion.

## Contract and timing

The implementation ports GR00T `3177bf0`'s async protocol and deadline behavior,
with RTC restricted to off. It uses the pinned Industrial Next DirectRPC transport,
one active session, one inference worker and one replaceable pending observation.
Registration replaces generations; stale in-flight results cannot enter a newer
session. Temporary `action=null` holds position at the client. A terminal
`error=session_unusable` latches stop; a new session is an explicit user action.

| Contract | Value |
|---|---|
| State | left/right EEF pose and right native hand, 38 coordinates |
| Action | right EEF pose plus continuous native right hand, 29 coordinates |
| Frame / rotation | per-arm Flexiv base; wire rot6d columns, internal rows |
| Cameras | head RGB and left/right top fisheye RGB, 256×256 |
| Model resize | 224×224, same serving/training transform |
| Horizon / labels | 40 rows at 50 Hz, target offset D=0 |
| Execution | K=2: model row i maps to source tick + D + i − K, i≥K |
| Ensemble | temporal exponential, coefficient 0.1, newest 3 chunks |
| Transition | 4 frames, SO(3) smoothstep; deadlines are never extended |
| Clock | elapsed time, skipped missed 50 Hz slots; separate request sequence |
| Deadlines | lateness 0.04 s, clock drift 0.1 s, command gap 0.2 s |
| Images | at most 5 slots and 0.1 s old |
| Sampling | 10 flow steps; RTC off |

There are no left-arm predictions, scalar gripper substitutions or hand clipping.
Relative model actions are made absolute against the exact captured inference
state before ensemble/transition processing. Rotation means and transitions use
SO(3); six coordinates are not averaged as Euclidean rotations.

## Local qualification

Run smoke only against an isolated instance: it registers/replaces sessions and
deliberately triggers a terminal command gap. This imports the real ROS transport
module without importing a ROS node or publishing commands.

```bash
JAX_PLATFORMS=cpu uv run --no-sync scripts/smoke_industrialnext.py \
  --config-name pi05_taro_exp_100 --host 127.0.0.1 --port 10014 \
  --output-dir checkpoints/NEW_RPC_SMOKE \
  --ros-client-source-dir /home/azureuser/industrialnext_ros2/src/industrialnext_operator_ros2/industrialnext_operator_policy_client
```

Local qualification on 2026-09-18 used five-update batch-64 smoke checkpoints for
both variants, not trained task policies. Both passed DirectRPC and ROS transport
checks including null, finite native-hand actions, terminal latch and close.
Synchronized warm inference was approximately 97–98 ms median and 108–109 ms p95
on one A100 per model while other validation jobs ran. Six matched replay recipes
emitted 83 actions with six initial nulls and no terminals over their recorded
100-frame traces (missed requests retained). These results validate integration;
repeat evaluation and transport checks for selected trained exports.

## Comparing real robot rollouts

Before any motion, deploy the same camera ordering, 20-coordinate hand declaration,
Flexiv frame calibration and command limits used by GR00T. Run a publication-disabled
shadow check (`inference_mode=true`) on the deployment machine. Verify null/terminal
handling, stop/new-run generation protection, real camera ages, 50 Hz pacing, network
jitter and command-gap coverage. Local RPC tests do not establish these properties
on the robot machine. Motion trials require separate authorization.

Use a 2×2 comparison: GR00T/PI0.5 × Taro 100/full. Prefer checkpoints at the same completed-update budget; otherwise label the
budget difference explicitly. The restarted PI0.5 runs target 15,000 updates,
whereas the existing GR00T final checkpoints completed 40,000. Retain exact
checkpoint, EMA/processor/stats, code and profile identities. Use the same K=2 production timing recipe, camera preprocessing
at the wire, playback 1×, observation/action frame declarations and client limits.
Keep each model's required internal image transforms and normalization attached to
its checkpoint. If comparing a different postprocessing or flow-step setting, give
it a separate named condition and rerun timing/quality qualification.

Pair trials by object/color, conveyor speed, initial placement, lighting and robot
start state. Randomize model order within each block and reset consistently. Set
success, failure, timeout, intervention and safety-stop rules before collecting
results. Record every trial, including null/terminal failures; do not retry failures
silently or select checkpoints using the final rollout set. Report success counts
and confidence intervals, completion time, interventions, command gaps, deadline
expiry and latency by condition. Keep operator changes and any model-specific
limits visible. Smooth trajectories or low offline loss do not establish task success.

The current GR00T standalone replay benchmark omits its newer clock/gap profile
fields; do not treat its default replay as matched until its effective configuration
is verified. This fork's six-recipe replay calls the same resolver as its server.
