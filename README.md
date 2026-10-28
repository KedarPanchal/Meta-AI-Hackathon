# Fly a drone with a fly's brain

A connectome-constrained simulation of the *Drosophila* male central nervous system
([FLYCNS](https://github.com/IONOFIELD/FLYCNS)) flying an X-UAV multirotor in ROS 2.

The drone's camera and microphone drive real sensory cell types. Spikes from descending and
motor neurons become velocity. Two behaviours:

* **Escape** — a looming object drives LC4 and LPLC2, the giant fibre (DNp01) fires, and the
  drone dives away from the threat. The giant-fibre spike is a *trigger event, not a rate*,
  which is what makes this work; see [REFLEX.md](REFLEX.md).
* **Pursuit** — LC10a target neurons, split left and right, steer toward a target while the
  P1/pC1 courtship cells act as a "pursue now" gate. The gate holds a *state*, not a
  direction: in FLYCNS those cells drive the song command neurons, not locomotion, so the
  heading comes from vision.

An MCP server exposes 15 tools so an agent can inspect, configure, experiment with and stop
the whole thing.

## Status

| part | state |
|---|---|
| Brain, encoders, motor layer, reflex | working, calibrated on the FLYCNS toy graph |
| ROS 2 node, detector, Twist bridge | working, verified live in a node with a synthetic camera feed |
| MCP server (15 tools, 2 resources, 2 prompts) | working |
| Escape reflex | calibrated: fires 0.9–1.1 s before contact at every bearing |
| Pursuit decoder | **not wired.** `outputs.steer`/`outputs.forward` are empty, so pursuit hovers |
| Real MaleCNS connectome | **not run.** Needs a neuPrint token |
| PX4 / MAVROS bridges | written, **not exercised** — neither message package is in this environment |

Nothing moves until you arm it: the output gate starts closed.

## Layout

```
src/fly_brain/fly_brain/     the brain, with NO ROS dependency, so it is testable offline
  config.py                  FLYDRONE_* environment settings
  schemas.py                 pydantic models; these are also the MCP tool schemas
  circuit.json / circuit.py  channel -> cell type map, resolved against the loaded graph
  brain.py                   FlycnsBrain (real spiking net), MockBrain, preimport_flycns()
  encoders.py                percept -> Poisson rates per channel, split left/right
  motor.py                   ESCAPE > COOLDOWN > AVOID > PURSUE > HOLD, plus the safety clamp
  perception.py              LoomTracker, AudioFrontEnd, synthetic approach generator
  controller.py              the shared singleton: encode -> step -> decode -> bridge
  bridge.py                  Bridge protocol, NullBridge, RecordingBridge
src/fly_drone/fly_drone/     the ROS 2 side
  node.py                    rclpy node: Image + Odometry -> tick -> Twist
  detect.py                  colour / motion detector, and a ground-truth projector
  bridge.py                  TwistBridge (default), Px4Bridge, MavrosBridge
mcp-server/
  mcp_server.py              FastMCP.from_fastapi(server), resources and prompts
  app/main.py                the FastAPI app; every operation_id is an MCP tool name
  app/routers/               brain (7 tools), reflex (4), drone (4)
  tests/test_smoke.py        18 tests, run against either backend
launch/launch-fly.yaml       ros2 launch fly_drone launch-fly.yaml
REFLEX.md                    the reflex design note, with every measured number
```

The brain is deliberately free of `rclpy`. That is what lets the reflex be calibrated and
tested without a simulator, which is most of why it got calibrated at all.

## Setup

FLYCNS is not vendored. Clone it next to this repository and build the toy graph:

```bash
git clone https://github.com/IONOFIELD/FLYCNS.git ../FLYCNS
cd ../FLYCNS && python scripts/make_testdata.py && cd -   # 912-neuron toy graph, no account
pixi install
```

For the real connectome, run `python scripts/setup.py` in the clone; it asks for a neuPrint
token and downloads `data/`. Then use `FLYDRONE_DATA=data`.

```bash
export FLYDRONE_REPO=$PWD/../FLYCNS
export FLYDRONE_DATA=testdata        # or: data
export BRAIN_BACKEND=flycns          # or: mock
```

## Run it

```bash
# the tests, on the mock brain and then on the real spiking network
pixi run python mcp-server/tests/test_smoke.py
BRAIN_BACKEND=flycns pixi run python mcp-server/tests/test_smoke.py

# the MCP server
pixi run python mcp-server/mcp_server.py              # stdio
pixi run python mcp-server/mcp_server.py --http 8799  # HTTP

# the ROS 2 node, against a simulator
pixi run colcon build --symlink-install
source install/setup.bash
ros2 launch fly_drone launch-fly.yaml backend:=flycns flycns_repo:=$PWD/../FLYCNS
```

Run it with output disabled first (the default) and watch `drone_state`, then arm it with the
`enable_output` tool or `ros2 param set /fly_brain_pilot enable_output true`.

## The ROS 2 interface

The vehicle interface was not specified, so the default is what this workspace actually has
message packages for: a body-frame `geometry_msgs/Twist` on `/xuav/cmd_vel`, which is what
plain Gazebo gives you through `ros_gz_bridge`. Body frame is FLU and `Twist` is already FLU,
so the mapping needs no sign changes.

Switch with `ROS_BRIDGE` / `bridge:=`:

| value | interface | status |
|---|---|---|
| `twist` (default) | `geometry_msgs/Twist` | works |
| `twist_stamped` | `geometry_msgs/TwistStamped` | works |
| `px4` | `px4_msgs/TrajectorySetpoint` + `OffboardControlMode`, FLU→NED | needs `px4_msgs` in `pixi.toml` |
| `mavros` | `mavros_msgs` setpoint_velocity | needs MAVROS running |

Topics are all environment-configurable (`ROS_TOPIC_CMD_VEL`, `ROS_TOPIC_ODOM`,
`ROS_TOPIC_IMAGE`, `ROS_TOPIC_AUDIO`). **If the real X-UAV sim uses something else, point
these at it or add an adapter to `fly_drone/bridge.py` — that is the only file that needs to
change.**

Audio is optional: `audio_common_msgs` is not in this environment, so the node synthesises a
level from the target's apparent size unless you set `audio_mode:=topic`.

## Conventions

Getting these wrong is the most common source of a drone that flies *into* things:

* Body frame is **FLU**: `x` forward, `y` left, `z` up.
* `yaw_rate` positive is **left** (counter-clockwise from above).
* `bearing_deg` positive is **RIGHT** of the nose. A target at `+30` needs a *negative*
  `yaw_rate` to centre it.

## MCP tools

| | |
|---|---|
| `brain_status` | what is loaded, how big, and `realtime_factor` |
| `load_brain` | build or rebuild, optionally ablated or pruned |
| `find_neurons` | regex search over cell types — **use this before naming a type anywhere** |
| `get_circuit` / `update_circuit` | the channel → cell type map, and what resolved |
| `step_brain` | run on a percept you supply; see the spikes and the command |
| `stimulate_neurons` | drive arbitrary types on a throwaway copy of the network |
| `get_motor_config` / `update_motor_config` | reflex timing, pursuit gains, input gains |
| `reflex_events` | the log of escapes and side-steps |
| `test_reflex` | replay a synthetic approach and report exactly when it fired |
| `drone_state` | vehicle state, behaviour, arming |
| `enable_output` | the arming switch |
| `set_pursuit` | open the courtship gate, set decoder gains |
| `stop_drone` | zero velocity, disarm, clear the escape clock |

Resources: `flycns://docs/reflex`, `flycns://circuit`. Prompts: `calibrate_reflex`,
`find_pursuit_neurons`.

## Traps, all of them load-bearing

* **Import Brian2 on the main thread.** It installs a SIGINT handler when imported, which
  Python only permits from the main thread, and FastAPI, MCP and ROS handlers all run on
  worker threads. `preimport_flycns()` is called from every entry point — including the test
  file, which fails loudly if it cannot. The error message now says so explicitly.
* **Never name a setting `FLYCNS_*`.** FLYCNS's `announce_overrides()` treats any such
  variable as a declared perturbation of the model and stamps it into every report. Ours are
  `FLYDRONE_*`.
* **No startup events.** `FastMCP.from_fastapi` drives the app in-process over an httpx ASGI
  transport, which never runs FastAPI's startup or lifespan hooks. The brain loads lazily.
* **`model.stim` is in bodyId order, not the order you asked for.** Rates are one entry per
  stimulated *neuron* and must be mapped per row, never zipped.
* **Stimulated types leave the network.** CNSModel replaces them with Poisson sources, so a
  type cannot be both driven and read back. `stimulate_neurons` uses a separate copy for
  exactly this reason.
* **Exact-name matching.** MaleCNS splits populations into lettered subtypes — `DNg02` does
  not exist, `DNg02_a`..`DNg02_g` do. Search with `find_neurons` first. `circuit.json`
  supports a trailing `*` to expand by prefix.
* **Keep input rates in single digits.** FLYCNS anchors its loom protocol at 5 Hz per cell.
  Driving these populations at hundreds of Hz makes DNp01 fire continuously and destroys the
  trigger-event property the reflex depends on.
* **`run()` has ~74 ms of fixed overhead per call.** See the table in REFLEX.md. This, not
  graph size, is what sets the control rate.

## What to do next

1. **Point it at the real sim.** Set the four topic variables, run with output disabled, and
   confirm `drone_state` tracks. Then arm.
2. **Wire the pursuit decoder.** Run the `find_pursuit_neurons` prompt: find bilateral
   descending types that respond to LC10a, set `outputs.steer`/`outputs.forward`, verify the
   left/right imbalance changes sign with bearing *before* fitting any gain. Fit forward
   speed with no intercept.
3. **Run the real connectome.** `scripts/setup.py` with a neuPrint token, then re-measure
   everything in REFLEX.md — especially the loom gain and the sign of the audio effect.
4. **Check the latency budget** against the real graph's `realtime_factor` and lower Gazebo's
   real-time factor to match if needed.
