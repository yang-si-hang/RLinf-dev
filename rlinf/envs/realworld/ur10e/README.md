# UR10e external device protocol

RLinf loads the OpenPI policy and sends its complete prediction chunk to an
external device service. The service, rather than RLinf, owns `URRobot`, the
gripper, cameras, interpolation, control frequency, safety checks, and
`send_action()` calls. It also selects the execution horizon and may consume
only part of the prediction chunk before returning the next observation.

The transport is ZeroMQ REQ/REP. Every request starts with a JSON frame that
contains `protocol_version`, `request_id`, `type`, and `chunk_id`.
`EXECUTE_CHUNK` adds an `actions` descriptor and a second frame containing a
contiguous float32 array with shape `[model_action_horizon, 10]`. The action order
is `[xyz, rot6d, gripper]`.

Supported request types are:

- `HEALTH`: checks reachability without changing device state.
- `GET_INITIAL_OBSERVATION`: reads state and images without commanding or
  resetting the robot.
- `EXECUTE_CHUNK`: executes one chunk and replies only after execution ends.

An observation response consists of JSON metadata followed by raw `state`,
`base_image`, and `wrist_image` frames. Metadata must include matching protocol,
request, and chunk IDs; `ok`; capture and robot timestamps; and dtype/shape
descriptors for all arrays. An execution response also includes
`executed_steps`, `terminated`, `truncated`, and an optional `message`.
Every observation supplies the task description used for the next model call.
Both images must already be uint8 RGB arrays with shape `[224, 224, 3]`; RLinf
does not crop or resize them. Camera serials remain private to the device service.

Only one request can be in flight. Timeout, malformed data, stale images, ID
mismatch, or unexplained partial execution aborts RLinf inference. Closing RLinf
only closes its local RPC resources; no stop or disconnect request is sent.

Future RTC support can add progress observations and chunk updates on another
socket or protocol mode. The blocking `EXECUTE_CHUNK` contract remains valid and
backward compatible.
