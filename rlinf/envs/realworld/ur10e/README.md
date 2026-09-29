# UR10e online TD3 device protocol

The UR10e device server lives in the separate VLA-Tele-ur checkout at
`scripts/inference/ur10e_rlinf_device_server.py`. It owns `URRobot`, two Orbbec
cameras, the Robotiq gripper, and the 500 Hz UR servo thread. Start it on the
robot computer with an interactive terminal:

```bash
python scripts/inference/ur10e_rlinf_device_server.py \
  --robot-ip <robot-ip> --endpoint tcp://0.0.0.0:5555
```

Set `UR10E_DEVICE_ENDPOINT=tcp://<robot-computer-ip>:5555` on the RLinf
computer. Set `PROJECT_ROOT` to the RLinf checkout so the Stage 1 checkpoint,
OpenPI quantile statistics and offline replay manifest resolve. From RLinf,
run the preflight before starting training:

```bash
export PROJECT_ROOT=/app
export UR10E_DEVICE_ENDPOINT=tcp://<robot-computer-ip>:5555
# local docker:
export UR10E_DEVICE_ENDPOINT=tcp://127.0.0.1:5555
PYTHONPATH=. python toolkits/standalone_eval_scripts/openpi/check_ur10e_online_td3.py
bash examples/embodiment/run_embodiment.sh ur10e_rlt_stage2_td3_mlp
```

The preflight checks the resolved UR configuration, Stage 1 checkpoint file,
OpenPI statistics and transforms, replay manifest, Stage 1 metadata, exact
offline twin-Q weight mapping, device `HEALTH`, and an initial state and dual-camera
observation. It sends no motion command. The observation request starts a new
device-side episode and clears pending robot-terminal labels, so run preflight
before training while the robot is stationary. Use `--skip-device` for local
files only or `--skip-observation` for local files plus `HEALTH`. Its default
device timeout is 5 seconds. If you edit the YAML, rerun preflight before
training. The Stage 1 weight file is checked for presence
and nonzero size; full model loading occurs when training starts.

The runner terminal asks for `start` before every episode. Physically reset the
robot, then enter `start` there. `GET_INITIAL_OBSERVATION` only captures state and
images; it never moves the robot. The robot computer terminal reads `c` for
success, `b` for task failure or confirmed collision, and `x` to discard an
interrupted episode. A controller or device exception is not automatically
labeled as collision. Motion interruptions wait for `b` or `x` on the robot
computer. `INVALID_ABORT` discards the **entire current episode**, while earlier
completed episodes remain in replay.
The generic `run_embodiment.sh` launcher sets `ROBOT_PLATFORM=UR10E` for this
configuration. The actual environment and action shape come from the UR10e
YAML; `ROBOT_PLATFORM` is only used by other model families such as StarVLA.

The actor controls every chunk from the first chunk. The frozen Stage 1 model
only supplies `z_rl`, proprioception, and a 30-step reference for BC. Actor,
critic, BC and replay actions use the same OpenPI q01/q99 normalized relative
space. The UR environment converts only the outgoing actor chunk to physical
relative `[15,10]` float32. The device server converts it to absolute TCP targets
using the actual state at chunk start. The existing UR step limits and control
thread remain responsible for motion limits. No pure VLA action is sent.
The configuration imports only the offline twin-Q weights after checking its
manifest hash, action quantiles, model dimensions, and exact weight keys. The
offline replay itself is not mixed with online trajectory replay.

## Protocol v2

ZeroMQ REQ/REP requests are `HEALTH`, `GET_INITIAL_OBSERVATION`, and
`EXECUTE_CHUNK`. All include `protocol_version=2`, `request_id`, `type`, and
`chunk_id`. `EXECUTE_CHUNK` includes one raw float32 frame and an `actions`
descriptor for `[15,10]`. Replies echo IDs and use `<TYPE>_RESULT`.

Observation replies contain JSON descriptors and three raw frames: `state`
float32 `[10]`, and `base_image` and `wrist_image` uint8 RGB `[224,224,3]`.
The robot side applies the existing crop. The reply includes capture and robot
timestamps and the fixed task text `Plug the Ethernet cable into the Ethernet
port.` A terminal chunk may omit the observation. Every chunk reply includes
`executed_steps`, `rewards[15]`, `terminations[15]`, `truncations[15]`, `outcome`,
`record_transition`, `terminal_reason`, and `message`.

`executed_steps` counts targets whose execution began. A partly interrupted
seventh target gives 7. Normal and successful executed steps reward `-1`;
failure changes the final executed reward to `-750`. Unexecuted positions have
zero reward and false flags. A full episode ends at 450 started 30 Hz steps
(30 chunks): RLinf marks the last step as `FAILURE` with
`terminal_reason=time_limit`. This is a policy-step limit, not wall-clock time.

`RUNNING` needs a real final observation. On `SUCCESS` or `FAILURE`, the learner
uses a terminal observation placeholder and does not bootstrap. An invalid
observation, malformed reply, timeout, or transport error aborts the episode.
A timed-out `EXECUTE_CHUNK` is **never retried**: the client cannot infer whether
the robot executed it. The operator must resolve the robot state and confirm a
new physical reset before the next episode.
