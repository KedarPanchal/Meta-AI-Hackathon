# AGENTS.md

Working guide for AI coding agents and humans in this repository. It is grounded in the
source, configuration, and tests as of **2026-11-03**. `README.md` is the project overview but
is not authoritative when it disagrees with the code, `package.xml`, `pixi.toml`, launch files,
or tests. This file records the post-refactor state; the remaining historical and unverified
claims are called out explicitly.

## 1. What this project is

A connectome-constrained simulation of the *Drosophila* nervous system flying an X3 multirotor
in Gazebo through ROS 2, with a mission layer that searches a forest for a person. The reflex
substrate is a **vendored, real hemibrain v1.2.1 subset**: 5,573 neurons, 712,973 signed
chemical edges, and about 4.6 million EM synapses. It is loaded from the repository; there is
no FLYCNS checkout, no sibling clone, and no Brian2 import in the runtime.

Two layers, joined by one message type:

- **Brain / pilot** (`src/fly_brain`, `src/fly_drone`): reads sensors, runs the spiking
  connectome and the motor policy, and produces body-frame velocity. It owns flying, obstacle
  avoidance, and the escape reflex. `fly_brain` has **no ROS dependency** and runs offline.
  `fly_drone` is the ROS 2 adapter.
- **Mission layer** (`mcp-server/`): an MCP server with five tools. It plans coverage, sends
  the brain one world-position `Goal` at a time, and runs the person detector and (optionally)
  the Muse navigator over the downward camera. **It never commands velocity.**

```
MCP client
   | MCP
   v
mission layer -- Goal JSON (x, y, z, arousal) --> brain --> Twist --> drone_controller
   ^                                                                    --> Gazebo / vehicle
   '--- odometry, downward camera, GoalStatus ----------------------------'

brain <-- forward camera, 3-D lidar, odometry
```

Muse has three separate jobs, all optional and none of them a velocity source:

1. **Detector:** decide whether a downward frame contains a person (`PERSON_DETECTOR`).
2. **Navigator:** choose the next mission goal from real candidate goals, or choose a pixel
   to investigate (`MISSION_NAVIGATOR`).
3. **Dodge advisor:** predict where a tracked moving object will go and pick a dodge
   direction ahead of an escape (`FLYDRONE_DODGE_ADVISOR`, on by default when a key is set).
   The request runs on a worker thread and is cached per track; `fly_brain.dodge` supplies
   the geometry when the answer is late, and `MotorLayer._clamp()` still governs the result.

A goal is an explicit instruction to fly. The brain's output is armed when one arrives
(`arm_on_goal`). With no goal and no arming, nothing moves.

## 2. Read these first

| Path | Why it matters |
|---|---|
| `src/fly_brain/fly_brain/motor.py` | Behaviour priority, escape/avoid/navigation, pC1 commitment multiplier, safety clamp. The core policy. |
| `src/fly_brain/fly_brain/controller.py` | Shared singleton: encode → brain tick → motor → bridge; applies `Goal.arousal`. |
| `src/fly_brain/fly_brain/perception.py` | `Threat`, `LoomTracker` (per-object expansion, urgency, scenery rules). |
| `src/fly_brain/fly_brain/dodge.py` | Closest-approach prediction, dodge choice, escape vector. |
| `src/fly_brain/fly_brain/encoders.py` | Percept → Poisson rates; loom gate and gains. |
| `src/fly_brain/fly_brain/brain.py` | Backend factory: `ConnectomeBrain` or `MockBrain`; no ROS. |
| `src/fly_brain/fly_brain/connectome.py` | NumPy LIF runtime over the vendored hemibrain subset. |
| `src/fly_brain/fly_brain/data/hemibrain_subset.json.gz` | The vendored graph; do not replace with a downloaded checkout. |
| `src/fly_brain/fly_brain/circuit.json` | Channel → exact cell-type map, resolved against the loaded graph. |
| `tools/build_connectome_subset.py` | Reproducible extractor for the vendored subset. |
| `src/fly_drone/fly_drone/node.py` | `fly_brain_pilot`: sensors in, one tick, telemetry/status out. |
| `src/fly_drone/fly_drone/lidar.py` | 3-D point cloud → clusters → `Threat`s. Ground mask, frozen width, motion gate. |
| `src/fly_drone/fly_drone/dodge_advisor.py` | Muse dodge advisor: framed request, defensive parse, per-track cache. |
| `src/fly_drone/fly_drone/drone_controller.py` | Arms the X3 and forwards `/drone/drone_motion` → `/drone/cmd_vel`. |
| `src/fly_drone/fly_drone/bridge.py` | `TwistBridge`, `Px4Bridge`, `MavrosBridge`, enable semantics. |
| `src/fly_bringup/launch/*.py` | What actually starts, in what order. |
| `src/fly_bringup/config/bridge.yaml` | Every ROS ↔ Gazebo topic. |
| `mcp-server/mission/runner.py` | Mission thread, goal protocol, sampling, confirmation, abort, service lifecycle. |
| `mcp-server/mission/coverage.py` | Boustrophedon decomposition, lanes, cell tour, routing. |
| `mcp-server/mission/detector.py` | Muse Spark detector, colour stand-in, defensive parsing. |
| `mcp-server/mission/muse_navigator.py` | General Muse navigation contract (`goto`/`investigate`/`hold`) and HTTP client. |
| `mcp-server/mission/muse_runner.py` | Mission runner that uses Muse-chosen goals; coverage supplies candidates. |
| `mcp-server/mission/link.py` | `RosLink` and `SimLink`. The mission-side protocol. |
| `mcp-server/mission/config.py` | `MISSION_*`, `MUSE_*`, `CAMERA_DOWN_*`, `SIM_*` defaults. |
| `REFLEX.md` | Design note with historical measured numbers and caveats. Read the caveats, not just the tables. |

## 3. Setup and commands

### Environment

The project uses **pixi** (verified with pixi 0.84.0). It provides ROS 2 Lyrical, Gazebo,
OpenCV, FastAPI/FastMCP, NumPy, and the build tools. Use `pixi run` for project commands; do
not assume the system Python or a system ROS installation. `brian2` is still listed in
`pixi.toml` from the old FLYCNS backend, but no module imports it; removing it is optional
cleanup that requires a normal pixi lock update.

```bash
pixi install
pixi run colcon build --symlink-install
source install/setup.bash        # zsh: source install/setup.zsh
```

`pixi.toml` has an empty `[tasks]` section: there is no `pixi run test` or `pixi run build`.
Use the explicit commands below or the Makefile.

The connectome is **vendored** at
`src/fly_brain/fly_brain/data/hemibrain_subset.json.gz` (about 3.1 MB compressed). There is no
clone step. The source is the public Janelia hemibrain v1.2.1 release, CC-BY 4.0. To reproduce
the subset:

```bash
pixi run python tools/build_connectome_subset.py
# or, if the source feathers are already local:
pixi run python tools/build_connectome_subset.py \
    --meta /path/to/hemibrain_121_meta.feather \
    --edges /path/to/hemibrain_121_simple_edgelist.feather
```

The extractor keeps an **induced one-hop chemical subgraph** around LC4, LPLC2, LC10a, Giant
Fiber, DNp02, DNp06, DNp11, DNg30, DNg79, DNg92_a, pC1a–pC1e, and DNp13. The output file
contains provenance metadata (source, license, extraction method, counts). Do not hand-edit
the graph; regenerate it.

### Makefile

The Makefile is now portable: `SHELL := /bin/sh`, the launch/test targets run through `bash`
inside `pixi run` and source `install/setup.bash`, and `make build` calls pixi directly. The
main targets are:

| Target | What it runs |
|---|---|
| `make` / `make all` | Default = `demo-search`: Gazebo + connectome brain + autonomous search |
| `make run` | Gazebo + connectome brain + MCP HTTP server on `127.0.0.1:8799` |
| `make run-sim` | MCP server + in-process brain; no Gazebo, no ROS |
| `make mission` | MCP server alone against an already-running ROS stack |
| `make demo-search` | Search without an MCP client using `tools/search_test.py` |
| `make test` | `test-brain` + `test-mission` |
| `make test-brain` | Offline connectome/perception/lidar/threat tests |
| `make test-mission` | In-process planner/runner/MCP tests |
| `make forest` / `cave` / `building` | `sim_launch.py` for that world |
| `make brain` | Gazebo + connectome brain |
| `make brain-mock` | Gazebo + mock brain + threat launcher |
| `make build` | `pixi run colcon build --symlink-install` |

`make test` was run against this tree: **33 offline tests + 41 mission tests passed**. Those
counts cover the five files in `test-brain` plus `mcp-server/tests`; `test_dodge.py` and
`test_dodge_advisor.py` are separate and are run explicitly (see below). The Gazebo/MCP
launch targets dry-run correctly with `make -n`, but a live Gazebo flight was not re-run after
the refactor; do not claim otherwise without doing it.

### Tests

Offline brain, lidar, threat, connectome, and dodge tests. `PYTHONPATH` is required because
the ROS packages are not importable from the repository root without it:

```bash
PYTHONPATH=src/fly_brain:src/fly_drone pixi run python -m pytest \
    src/fly_brain/test/test_perception.py \
    src/fly_brain/test/test_encoders.py \
    src/fly_brain/test/test_connectome.py \
    src/fly_drone/test/test_lidar.py \
    src/fly_drone/test/test_threats.py -q
# 33 passed
```

`test_connectome.py` covers the vendored graph itself:

- 5,000+ neurons and 500,000+ edges, with real `Giant Fiber`, `LC4`, `LPLC2`, `pC1a`, and
  `DNp13` types;
- empty sky produces no GF spikes;
- a 40° / 200 °/s loom fires the GF, while a 12° / 3 °/s loom does not;
- arousal drives the `courtship` readout and `MotorLayer.courtship_drive`.

Escape direction and the Muse dodge advisor are separate from `make test-brain`:

```bash
PYTHONPATH=src/fly_brain:src/fly_drone pixi run python -m pytest \
    src/fly_brain/test/test_dodge.py \
    src/fly_drone/test/test_dodge_advisor.py -q
# 13 passed
```

They cover head-on/sideways/vertical dodge choice, closest approach, Muse advice precedence
and freshness, the advisor's request shape, per-track caching, and quiet failure.

Mission tests use the in-process `SimLink`, the real planner, real `MotorLayer` navigation, a
synthetic downward camera, and the colour detector. They need **no Gazebo, no ROS, and no
Muse key**:

```bash
pixi run python -m pytest mcp-server/tests -q
# 41 passed
```

The mission suite includes:

- `test_coverage.py` — planner and coverage;
- `test_navigate.py` — goal navigation and safety clamp;
- `test_mission.py` — search, scout, abort, keep-outs, one-mission-at-a-time, look, and a
  fake-Muse navigation integration test;
- `test_smoke.py` — five-tool/one-prompt MCP surface over HTTP and the in-process client;
- `test_detector.py` — Muse detector against a fake API server;
- `test_muse_navigator.py` — Muse navigation JSON, choices, pixel geometry, fail-closed
  parsing, and empty-answer errors.

`conftest.py` sets `MISSION_LINK=sim`, `PERSON_DETECTOR=color`,
`MISSION_NAVIGATOR=coverage`, `BRAIN_BACKEND=mock`, `SIM_SPEEDUP=40`, `SIM_DT=0.05`, and
`MISSION_POLL_S=0.01` before importing anything. Keep that property: the sim tests are the
only automated proof that the planner, runner, navigation, detector pipeline, and MCP surface
work together. The settings loader uses `setdefault`, so avoid putting conflicting
`PERSON_DETECTOR`/`MISSION_NAVIGATOR` values in a developer `.env` if you expect the test
defaults to win.

Do not run a bare `pytest src/` as a quick check. Each ROS package also contains ament lint
tests (`test_flake8.py`, `test_mypy.py`, `test_pep257.py`, `test_copyright.py`,
`test_xmllint.py`) that are not part of the fast loop. Select the named files above unless you
are deliberately changing packaging or style.

### Run the simulation

```bash
# Gazebo + bridges + drone_controller. The vehicle is armed and hovers; no brain output.
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup sim_launch.py'

# Full stack: sim_launch + brain_pilot. Connectome by default; disarmed until a goal.
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py'

# Fast mock stand-in, with spheres thrown at the drone every 20 s.
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py \
    backend:=mock threats:=true'

# Headless / CI. Cameras and lidar still render off-screen; video is off.
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py \
    headless:=true video_recording:=false'
```

`brain_launch.py` takes `backend:=connectome|mock`, `enable_output`, `arm_on_goal`, `threats`,
`threat_period_s` (0 means only on request), `threat_delay_s`, `threat_distance_m`,
`threat_speed_mps`, `use_sim_time`, and the pass-through arguments `gazebo_simulation`,
`headless`, `camera_tracking`, `camera_tracking_target`, `video_recording`, and `video_dir`.
There are no `flycns_repo` or `flycns_data` arguments anymore.

Worlds under `worlds/` are `forest_search`, `cave_rescue`, and `collapsed_building`; defaults,
keep-outs, tests, and historical measured results target `forest_search`.

`drone_controller` is not optional in the simulation. It arms the X3 and republishes the
latest body-frame twist at 20 Hz. The world spawns the vehicle about 2.5 m above the floor;
without this node it falls onto a slope, and the brain's commands reach nothing.

### Run the MCP server

```bash
# stdio, for an MCP client
pixi run python mcp-server/mcp_server.py

# HTTP on 127.0.0.1:8799
pixi run python mcp-server/mcp_server.py --http 8799

# Brain in-process over a flat kinematic world: no Gazebo, no ROS
MISSION_LINK=sim pixi run python mcp-server/mcp_server.py
```

Important: `MISSION_LINK=auto` selects `ros` whenever `rclpy` imports. Inside the pixi
environment `rclpy` **does** import, even if no Gazebo or brain node is running. Set
`MISSION_LINK=sim` explicitly when you want the in-process world.

The top-level `.env` is loaded automatically with `setdefault` semantics by
`fly_brain.config._load_dotenv()`. An explicitly exported variable wins. The API key is not
printed by the code; do not add logging that exposes it. `MUSE_API_KEY` or `MODEL_API_KEY`
enables all three Muse roles. Without a key:

- `PERSON_DETECTOR=auto` falls back to the colour blob;
- `MISSION_NAVIGATOR=auto` falls back to coverage.

Read `mission_status.detector` and `mission_status.navigator` before believing a result.

### Manual topics and live harnesses

```bash
ros2 topic pub --once /fly_brain/goal std_msgs/msg/String \
    "{data: '{\"x\": 30, \"y\": -7, \"z\": 15, \"label\": \"east\"}'}"
ros2 topic echo /fly_brain/telemetry
ros2 topic pub --once /threats/launch std_msgs/msg/String "{data: '{\"bearing_deg\": 60}'}"

pixi run python tools/flight_test.py rates
pixi run python tools/flight_test.py reaction --launch '{"bearing_deg": 70}'
pixi run python tools/search_test.py --look-only
```

The harnesses need the live sim and a sourced install. They are not part of the automated
suite. `flight_test.py` waits for odometry and telemetry, so start the launch first.

## 4. Architecture

### 4.1 Brain, connectome, and pilot

```
forward camera --detect.py--> Boxes --+
3-D lidar -----lidar.py-----> Threats +--> LoomTracker --> Percept
odometry --------------------------------> altitude/attitude
                                              |
                                              v
                                         encoders.encode
                                              |
                                              v
                         ConnectomeBrain.step  (or MockBrain.step)
                                              |
                                              v
                          MotorLayer.decide(percept, spikes, state)
                                              |
                                              v
                                   safety clamp -> Bridge.send
```

- `fly_brain` must stay importable without `rclpy`. It uses pydantic for schemas that cross
  the ROS/MCP boundary and dataclasses for internal settings and parameter bundles.
- `Controller` is a process-wide singleton with a reentrant lock. The ROS node, FastAPI
  worker threads, and the MCP process all go through it. Do not create a second controller
  for convenience; tests use `reset_controller()`.
- `Controller.ensure_brain()` builds the network lazily. There are no FastAPI startup/lifespan
  events anywhere, deliberately.
- `Controller.tick()` applies `Goal.arousal` to the percept as
  `max(percept.arousal, goal.arousal)`, encodes it, runs the brain, then calls the motor.
  This is how Muse triggers pC1; the goal remains a position, not a velocity.

The backend factory in `brain.py` accepts only `connectome` (or `hemibrain`) and `mock`.
Unknown names raise `ValueError`; there is no silent fallback.

### 4.2 The vendored connectome runtime

`connectome.py` is a deliberately small NumPy LIF simulator, not a general connectome
framework.

- `LIFParams` defaults: 1 ms step, 20 ms membrane, 5 ms synaptic, −60 mV rest, −50 mV
  threshold, −62 mV reset, 2 ms refractory, and `syn_gain_mv=16.0`.
- `syn_gain_mv` is one global scale; individual weights are the **real EM synapse counts**
  with a transmitter sign. ACh is excitatory; GABA, glutamate, and histamine are inhibitory;
  amines have zero fast current in this model.
- Input channels are Poisson spike trains into the real sensory cells. Input cells still exist
  in the network; they are not deleted or replaced.
- The network runs one 1 ms LIF step per simulated millisecond, with a sparse compressed-row
  weight table. One 50 ms brain tick is therefore 50 steps.
- Injected sensory spikes are counted in the readout as well as driving the edges. This is
  deliberate: pC1 is both the arousal input and part of the `courtship` readout.
- `ConnectomeBrain.status()` reports graph counts, tick time, realtime factor, missing types,
  and ablations. `ConnectomeBrain.stimulate()` runs a separate model copy and never mutates the
  flying network.

The 16 mV gain was calibrated on this subset: a strong loom (40°, 200 °/s) fires Giant Fiber;
a weak but above-gate loom (12°, 3 °/s) and empty sky do not. If you replace the graph or
change the encoder scale, re-run `test_connectome.py` and re-measure the trigger. The graph is
a control substrate, not an assertion that every membrane property matches the fly.

### 4.3 Circuit map and the two output groups

`circuit.json` is matched against **exact cell-type names**, with `Name*` prefixes where
declared. Resolve it against the loaded graph; a wrong name silently drives nothing otherwise.

| Channel | Types | Meaning |
|---|---|---|
| `loom_speed` | `LC4` | Expansion-sensitive input. |
| `loom_size` | `LPLC2` | Looming-size input. |
| `target` | `LC10a` | Small-object tracking; no calibrated pursuit. |
| `sound` | empty | No Johnston's-organ type in this brain-only subset. |
| `arousal` | `pC1*` | Reproductive/approach gate driven by Muse. |
| `gf` | `Giant Fiber` | Escape trigger; event, not rate. |
| `jump` | `DNp02`, `DNp06`, `DNp11` | Descending escape outputs; not TTMn. |
| `flight_power` | `DNg30`, `DNg79`, `DNg92_a` | Power-recruitment readout. |
| `steer` | empty | Pursuit not calibrated. |
| `forward` | empty | Pursuit not calibrated. |
| `courtship` | `pC1*`, `DNp13` | Reproductive commitment readout; not a heading. |

The subset does **not** contain `DNp01`, `TTMn`, `PSI`, `pMP2`, or `pIP10`. Do not claim those
cells are being simulated. `Giant Fiber` is the GF readout; `jump` is the descending escape
group. The pC1→DNp13 contacts in the hemibrain subset are sparse; the `courtship` channel
therefore includes the injected pC1 spikes and their DNp13 target, and `MotorLayer` uses the
count only as a commitment multiplier.

The behaviour priority is strict:

```
ESCAPE > COOLDOWN > AVOID > NAVIGATE > PURSUE > HOLD
```

- `ESCAPE` is triggered by a `gf` spike, or by a `jump` spike because
  `ReflexParams.accept_ttmn` defaults to true. A spike is an **event**, not a rate: one trigger
  commits the aircraft for `hold_ms` (350 ms), then decays for `decay_ms` (250 ms), then a
  refractory `cooldown_ms` (800 ms).
- The escape direction comes from `fly_brain.dodge.choose_dodge()`, not from a fixed
  "away and up": it predicts the object's closest approach and dodges away from the side it
  will pass. Head-on, overhead, or steeply climbing/diving objects get a **sideways** move;
  only a level crossing from the side gets a vertical one. A vertical escape is pitched
  `climb_deg` above (or below) horizontal, and `away_fraction` still adds a push off the
  object's line. Muse's cached advice wins when it is fresh, confident, about the same track,
  and the reflex's `dodge_prefer` allows its axis; otherwise the geometry is the fallback.
- `ReflexParams.direction_source` defaults to `vision`. `brain` uses the left/right GF
  imbalance instead, which on the toy graph points **toward** the threat; do not switch it
  without measuring.
- Escape speed starts at 3 m/s; extra GF spikes scale it, and a `flight_power` spike adds
  `psi_bonus` (25%).
- `AVOID` is the graded side-step for a loom at least 12° in size and growing at least 3°/s
  that did not fire the GF.
- `NAVIGATE` converts a world-frame `Goal` into body-frame velocity with a saturated P
  controller. It climbs first when well below the goal, faces the direction of travel, and
  latches arrival. A set goal also makes HOLD/AVOID hold the goal altitude, not the 3 m
  pursuit cruise altitude.
- With `Goal.arousal > 0`, `NAVIGATE` multiplies approach speed by
  `1 + 0.35 * courtship_drive`. Direction still comes only from the goal. This is the Muse →
  pC1 → DNp13 → commitment path.
- `_clamp()` is the last stop for every command. It enforces `MAX_SPEED` (5 m/s),
  `MAX_CLIMB` (2 m/s), `MAX_YAW` (2 rad/s), and the `MIN_ALTITUDE_M` (1 m) floor. Never
  bypass it.

`MockBrain` is a rate stand-in, not a connectome. Its thresholds
(`MOCK_GF_THRESHOLD_HZ = 7.0`, `MOCK_GF_DOUBLE_HZ = 16.0`) are tied to the encoder scale. If
you change `FLYDRONE_LOOM_MIN_DPS`, the loom gain, or `FLYDRONE_PEAK_HZ`, update the mock and
its tests, or the two backends stop predicting each other.

### 4.4 Perception and the reflex

- Both the forward camera and the 3-D lidar end in a `fly_brain.perception.Threat`: bearing,
  elevation, angular size, and range when available. The brain does not know which sensor saw
  the object.
- `LoomTracker` differentiates angular size **per `(source, track_id)`**. It needs at least
  three samples before differentiating; the first frames of a track are where merges and new
  ids look like a size leap. It reports the most urgent object: smallest optical
  time-to-contact `size / expansion` among growing objects, else the largest.
- An object growing more slowly than `max_tau_s` (3 s) from contact is scenery: expansion is
  reported as zero. A lidar threat with `moving=False` is also scenery.
- Encoder rates are small on purpose. `FLYDRONE_PEAK_HZ` defaults to 5.0;
  `FLYDRONE_LOOM_GAIN` defaults to 16; `FLYDRONE_LOOM_MIN_DPS` defaults to 3.0. Above the
  gate, both `loom_speed` (LC4, expansion) and `loom_size` (LPLC2, size) are driven; below it
  neither is, even for a large object.
- The forward camera detector (`fly_drone/detect.py`) defaults to colour mode with magenta
  (`FLYDRONE_DETECT_COLOR`), deliberately unlike the forest world. `motion` mode smears when
  the camera moves; only use it roughly station-keeping.
- The brain's lidar is the 3-D `lidar_3d` (32 planes, ±30° elevation, 320 beams per plane,
  10 Hz), not the single-plane lidar. `lidar.py` places returns in the world using odometry
  attitude, drops returns more than 2 m below or 4 m above the airframe, clusters on a polar
  grid, freezes a track's physical width after it settles, and only calls a track moving when
  its world-frame centroid moves consistently. This prevents ground sweeps, cluster
  breathing, and static trees from firing the reflex.

### 4.5 ROS side

`brain_pilot` (`fly_drone/node.py`) subscribes to:

- `/drone/odometry` (`nav_msgs/Odometry`)
- `/drone/camera_image` (`sensor_msgs/Image`) and `/drone/camera_info` (`CameraInfo`)
- `/drone/lidar3d_pointcloud` (`sensor_msgs/PointCloud2`) unless `FLYDRONE_LIDAR=0`
- `/fly_brain/goal` (`std_msgs/String`, JSON `Goal`)

It publishes `/fly_brain/goal_status` and `/fly_brain/telemetry` every tick. It runs one brain
tick on a timer at `LOOP_HZ` (20 Hz), not in the image callback; only the newest frame is
kept, because a tick costs time and queuing frames would back up the executor. Everything
perceptual is stamped with the node clock, so under `use_sim_time` the control period and
escape timing follow Gazebo's clock.

`brain_pilot` publishes velocity on `/drone/drone_motion` through the configured bridge
(default `TwistBridge`). It also publishes `true` on `/drone/enable` while armed; it never
publishes `false`, because Gazebo's `MulticopterVelocityControl` cuts the rotors on `false`.
A stop is a zero twist, which is a hover.

`drone_controller` is a separate node in the `drone` namespace. It:

- subscribes to `drone_motion`, `odometry`, `camera_image`, `camera_info`, and
  `lidar_pointcloud` (the single-plane lidar, for the teammate's sensor-fusion messages);
- publishes `cmd_vel` (bridged to Gazebo), `position`, `raw_sensor_data`, and `sensor_fusion`;
- publishes `true` on `enable` once the Gazebo subscribers connect;
- republishes the latest `drone_motion` twist at 20 Hz.

Do not remove or rename it. The brain never talks to Gazebo directly.

### 4.6 Mission layer and Muse navigation

`mcp-server/mcp_server.py` wraps the FastAPI app in `app/main.py` with
`FastMCP.from_fastapi`. Every route's `operation_id` is the MCP tool name. `/health` is
excluded. There is exactly one prompt, `find_person`.

| Tool | What it does |
|---|---|
| `search_for_person` | Plans coverage, flies it, detects people, confirms a positive from above, ends hovering over the person. |
| `scout_area` | Flies the same coverage but keeps every sighting and finishes the whole area; optional `return_home`. |
| `mission_status` | State, progress, position, brain goal, navigator decisions, sightings, log. `best.estimated_xy` is the position to report. |
| `abort_mission` | Stops the runner and leaves a station-keeping goal at the current position. |
| `look_for_person` | Runs the detector once on the newest downward frame without moving. |

Both flight tools take `area` (box or polygon in world metres), `altitude_m`, `speed_mps`,
`lane_overlap`, `sweep_heading_deg`, `keepouts`, `min_confidence`, `confirm_detections`,
`navigator` (`auto`, `muse`, or `coverage`), and `dry_run`. `scout_area` adds `return_home`.
`dry_run` is the cheap way to inspect a plan; it does not ask Muse and does not fly.

`MissionService` is single-flight: `start()` refuses a second mission while one is
planned/running. It builds the link, detector, and selected navigator eagerly, plans, then
starts one of two runner threads:

- **Coverage runner** (`MissionRunner`): fixed ordered waypoints. At every
  `sample_fraction` of the along-track camera footprint it takes the newest downward frame
  and calls the detector. A positive in `search` mode is confirmed by flying over the
  estimated ground position and looking down again up to three times. If confirmed, the
  mission ends in `found` with a goal labelled `over person`. In `scout` mode positives merge
  within 6 m and coverage continues.
- **Muse runner** (`MuseMissionRunner`): keeps the coverage plan as a **candidate set**. At
  each decision it sends Muse the downward frame and up to four real candidates. Muse returns
  `goto` (one candidate id), `investigate` (a normalised pixel), or `hold`.
  - `goto` flies the chosen candidate.
  - `investigate` projects the pixel to flat ground with the same camera geometry as the
    detector, flies there with `Goal.arousal = max(0.25, confidence)`, and samples/confirm
    through the normal detector path.
  - `hold` station-keeps; after three holds the runner falls back to the first candidate.
  - A Muse error falls back to the first candidate and is recorded in
    `MissionStatus.nav_errors`/`last_nav_decision`; a bad answer never leaves the aircraft
    goal-less.
  - The final completion path uses the same station-keep/return-home rules as coverage.

The Muse navigator is intentionally restrained: coverage still supplies the candidate
geometry, so Muse chooses the next goal rather than inventing arbitrary world coordinates.
It has no `forest`/`cave` scenario branches. The contract should work with another world and
another camera height. Muse never publishes a velocity; the brain remains the only source of
`Twist`.

`MISSION_NAVIGATOR=auto` is the default. It resolves to `muse` when a Muse key is set,
otherwise `coverage`. A per-call `navigator='auto'` means "use `MISSION_NAVIGATOR`"; an
explicit `muse` without a key raises during `start()` rather than silently changing modes.

Planner details:

- Boustrophedon cellular decomposition. A sweep line crosses the area; a cell boundary is
  placed where the number of free intervals changes. Cells are toured nearest-first.
- Lane spacing is the downward camera footprint at the flight level, less `lane_overlap`
  (default 0.2). Lower altitude means a smaller footprint, more lanes, and a longer flight.
- Transits between cells are routed around keep-outs with a visibility graph.
- `coverage_estimate` is computed by rasterising the free area against the lane footprints.
  Do not report it as a guarantee; it is an estimate.
- Keep-outs only matter if
  `top_z_m + keepout_vertical_clearance_m >= altitude_m`. At the default 15 m, none of the
  fixtures in `mission/keepouts.json` block, so the drone flies straight lanes above them.
  Trees are not keep-outs; unexpected obstacles are the brain's reflexes' problem.

Camera and detector details:

- Downward camera defaults: 80° horizontal FOV, 640×480. Image top is the drone's forward,
  image right is the drone's right. `camera.py` assumes flat ground; the forest has about
  6 m of relief, so a bbox-derived position is good to a few metres, and the confirm pass
  refines it.
- `MUSE_BASE_URL` defaults to `https://api.meta.ai/v1`; all three Muse clients POST to
  `/chat/completions` with the JPEG as an `image_url` part and a defined answer schema.
- The detector, navigator, and dodge advisor are separate clients even though they share the
  model, key, and base URL. The detector returns `Detection`, the navigator `NavDecision`,
  and the advisor a `DodgeAdvice`. The detector and navigator retry transient failures; all
  three drop unsupported optional fields on a 400 and treat an empty answer as an error,
  never as "nobody there".
- `PERSON_DETECTOR`: `muse`, `color`, `none`, or `auto`. `auto` chooses Muse only if
  `MUSE_API_KEY` or `MODEL_API_KEY` is set, else colour.
- The mission colour detector defaults to `DETECT_COLOR=blue` (the Gazebo person's jeans).
  This is separate from the forward camera's `FLYDRONE_DETECT_COLOR=magenta`.

## 5. Repository map

```
src/fly_brain/                 offline-testable brain package (ament_python, no rclpy)
  fly_brain/schemas.py         pydantic models: Goal, GoalStatus, Percept, Command, ...
  fly_brain/perception.py      Threat, LoomTracker, AudioFrontEnd, approach_percepts
  fly_brain/encoders.py        Percept -> Poisson rates, loom gate/gain
  fly_brain/motor.py           priority state machine, escape clock, NAVIGATE, clamp
  fly_brain/dodge.py           closest-approach prediction, dodge choice, escape vector
  fly_brain/brain.py           MockBrain, ConnectomeBrain factory
  fly_brain/connectome.py      LIFNetwork, ConnectomeSubset, ConnectomeBrain
  fly_brain/data/              vendored hemibrain_subset.json.gz
  fly_brain/controller.py      shared singleton and lifecycle
  fly_brain/circuit.json       channel -> exact cell types
  fly_brain/config.py          BRAIN_*, FLYDRONE_*, ROS_*, MAX_*, NAV_* settings
  test/                        unit tests + test_connectome/test_dodge + ament lint tests

src/fly_drone/                 ROS 2 adapter package (ament_python)
  fly_drone/node.py            brain_pilot node
  fly_drone/lidar.py           3-D point cloud -> clusters -> Threats (no rclpy)
  fly_drone/dodge_advisor.py   Muse dodge advice (background, cached per track)
  fly_drone/detect.py          colour/motion detector for the forward camera
  fly_drone/threats.py         launcher geometry + SDF (testable, no rclpy)
  fly_drone/threat_launcher.py spawns/moves spheres through Gazebo services
  fly_drone/bridge.py          TwistBridge, Px4Bridge, MavrosBridge
  fly_drone/drone_controller.py forwards motion, arms vehicle, sensor fusion
  test/                        lidar/threat/dodge-advisor tests + ament lint tests

src/fly_bringup/               launch files, chase camera, video recorder
  launch/sim_launch.py         Gazebo + bridges + drone_controller + camera/video
  launch/brain_launch.py       sim_launch + brain_pilot + optional threats
  config/bridge.yaml           every ROS <-> Gazebo topic
  scripts/                     follow_camera, record_video

src/fly_interfaces/            ament_cmake messages used by drone_controller:
                               GeometricObject, RawSensorData, SensorFusion
src/fly_msgs/                  placeholder package with no messages
models/                        SDF models (X3, trees, rocks, person, buildings, cave, ...)
worlds/                        forest_search, cave_rescue, collapsed_building
mcp-server/
  mcp_server.py                FastMCP.from_fastapi, find_person prompt, CLI
  app/main.py                  FastAPI app, /health
  app/routers/mission.py       the five tool routes
  mission/coverage.py          boustrophedon planner
  mission/runner.py            MissionRunner + MissionService
  mission/muse_navigator.py    Muse navigation contract and HTTP client
  mission/muse_runner.py       MuseMissionRunner (Muse-chosen goals)
  mission/link.py              RosLink / SimLink
  mission/detector.py          MuseDetector / ColorDetector / NullDetector
  mission/camera.py            downward camera geometry
  mission/geometry.py          polygons, intervals, routing helpers
  mission/config.py            mission settings singleton
  mission/keepouts.json        tall fixtures in forest_search
  tests/                       planner, navigation, mission, MCP, detector, Muse navigator
tools/build_connectome_subset.py  reproducible hemibrain extractor
tools/flight_test.py           live sim: rates / reaction / survey
tools/search_test.py           live sim mission without an MCP client
REFLEX.md                      historical reflex design note
pixi.toml / pixi.lock          environment (lock is generated; do not hand-edit)
Makefile                       portable targets; /bin/sh, pixi run bash
```

Generated and gitignored: `.pixi/`, `build/`, `install/`, `log/`, `dist/`, `__pycache__/`,
`.pytest_cache/`. Do not edit build or install trees; rebuild instead. The vendored data file
is source-controlled and is not generated at install time.

## 6. Conventions and invariants

### Frames and units

- **World:** ENU. `x` east, `y` north, `z` up. The map is 120 × 120 m centred on the origin.
  The forest floor spans roughly −2.5..+3 m, so a flight level is a **world z**, not a height
  above ground. `Goal.z` is world z.
- **Body:** FLU. `x` forward, `y` left, `z` up.
- **Yaw rate:** rad/s, positive is **left** (counter-clockwise from above).
- **Bearing:** `bearing_deg` is positive to the **right** of the nose. A target at +30° needs
  a negative yaw rate to centre it.
- **Pitch:** positive is nose **down** (right-hand rule about the left-pointing body y axis).
  This is the opposite of the aerospace convention; the docstrings say so.
- **Downward camera:** image top is forward, image right is the drone's right.
- Angles in schemas are named with units (`_deg`, `_rad`, `_dps`); distances are metres.
  Keep that naming.

### Safety and control boundaries

- Every command goes through `MotorLayer._clamp()`. Do not add a bypass.
- The mission layer only sends `Goal`s. It must never publish a `Twist` or a velocity.
- Muse chooses goals, pixels, detections, and dodge directions only. It must never command
  velocity.
- Muse's dodge advice can change the escape *direction*; it never sets a speed and never
  bypasses the reflex clock or `_clamp()`.
- A goal arms the brain output (`arm_on_goal`), because a goal is an order to fly. A goal
  also makes HOLD/AVOID hold the goal altitude.
- `/drone/enable` is `true` only; never publish `false` from the brain or controller. A
  `false` cuts the rotors. Stop with a zero twist.
- `drone_controller` arms the vehicle as soon as Gazebo subscribers connect. Keep it running
  in the simulation.
- The safety envelope is `MAX_SPEED=5`, `MAX_CLIMB=2`, `MAX_YAW=2`, `MIN_ALTITUDE_M=1`.
- `Goal.arousal` is clamped to 0..1 and only scales approach speed; it is never a heading
  command. `GoalStatus.courtship_drive` is a readout, not an input.

### Code boundaries

- `fly_brain` has no ROS imports. Keep it that way so the MCP server and unit tests can run
  without ROS.
- The mission layer must remain importable without ROS. `RosLink` imports `rclpy` lazily;
  `app/bootstrap.py` adds only `src/fly_brain` to `sys.path`, never `src/fly_drone`.
- `fly_drone` is the ROS adapter; it can import `fly_brain` and `rclpy`.
- Do not add FastAPI startup/lifespan hooks. `FastMCP.from_fastapi` drives the app through an
  in-process ASGI transport and never runs them; links, detectors, and the Muse navigator are
  built on first use.
- Keep `MissionService` single-flight. `start()` raises if a mission is already planned or
  running; callers must `abort_mission` first.

### Style

- Python uses `from __future__ import annotations`, type hints on public functions, dataclasses
  for settings/parameter bundles, and pydantic models for schemas that cross the MCP/ROS
  boundary.
- The code intentionally carries explanatory comments with measured numbers and reasons.
  Preserve that style; when you change a threshold or a behaviour, update the explanation,
  not just the constant.
- Keep telemetry/logging bounded. `Mission._log` is an 80-entry deque and status returns the
  last 15 lines; do not turn it into an unbounded list. The brain publishes telemetry every
  tick, so do not add expensive serialization there.
- Do not add dependencies casually. ROS dependencies belong in the relevant `package.xml`;
  Python/environment dependencies belong in `pixi.toml`, and pixi updates `pixi.lock`. Do not
  hand-edit the lock file.
- Prefer editing existing modules over creating parallel abstractions. `fly_brain`,
  `fly_drone`, and `mission` already separate the concerns.
- Keep the vendored graph and its provenance together. A changed graph should come with a
  regenerated data file, a metadata note, and updated `test_connectome.py` counts/behaviour.

## 7. Configuration

Settings are read from the environment **once per process**:

- `fly_brain.config.Settings` → `settings()` / `reload_settings()`
- `mission.config.MissionSettings` → `mission_settings()` / `reload_mission_settings()`

Tests call the `reload_*` functions after setting variables. Do not expect a running process
to pick up an env change; restart it. `fly_brain.config` loads the top-level `.env` at import
with `setdefault`; an explicitly exported variable wins.

### Brain and vehicle

| Variable | Default | Meaning |
|---|---|---|
| `BRAIN_BACKEND` | `connectome` | `connectome`/`hemibrain` or `mock`. |
| `BRAIN_TICK_MS` | `50` | Simulated ms per brain tick; the LIF step is 1 ms. |
| `FLYDRONE_CONNECTOME` | vendored `data/hemibrain_subset.json.gz` | Subset path; no clone. |
| `FLYDRONE_CIRCUIT` | empty (`circuit.json`) | Alternative channel map. |
| `LOOP_HZ` | `20` | ROS control timer. |
| `CAMERA_HFOV_DEG` | `60` | Fallback only; the node reads the real HFOV from `camera_info`. |
| `FLYDRONE_LIDAR` | `1` | `0` flies on the camera alone. |
| `FLYDRONE_LIDAR_RANGE_M` | `20` | Lidar cluster range limit. |
| `FLYDRONE_LIDAR_MAX_BELOW_M` | `2` | Drop returns this far below the airframe. |
| `FLYDRONE_LIDAR_MAX_ABOVE_M` | `4` | Drop returns this far above it. |
| `FLYDRONE_PEAK_HZ` | `5.0` | Encoder anchor for the connectome rate. |
| `FLYDRONE_LOOM_GAIN` | `16.0` | Loom input gain, calibrated on the vendored subset. |
| `FLYDRONE_LOOM_MIN_DPS` | `3.0` | Expansion gate; below it nothing looms. |
| `FLYDRONE_DETECT` | `color` | Forward camera detector: `color` or `motion`. |
| `FLYDRONE_DETECT_COLOR` | `magenta` | Forward camera marker colour. |
| `FLYDRONE_DODGE_ADVISOR` | `1` | Build the Muse dodge advisor when a key is set; `0` disables it. |
| `MAX_SPEED` / `MAX_CLIMB` / `MAX_YAW` | `5` / `2` / `2` | Safety envelope. |
| `MIN_ALTITUDE_M` | `1.0` | Hard floor. |
| `NAV_SPEED_MPS` / `NAV_ARRIVE_RADIUS_M` | `2.5` / `1.5` | Default navigation. |
| `ROS_BRIDGE` | `twist` | `twist`, `twist_stamped`, `px4`, `mavros`. |
| `ROS_TOPIC_*` | see config.py | Every topic name; keep mission and brain in agreement. |
| `ROS_FRAME_ID` | `base_link` | Twist frame for stamped bridges. |

### Mission layer

| Variable | Default | Meaning |
|---|---|---|
| `MISSION_LINK` | `auto` | `ros`, `sim`, or `auto` (ROS if rclpy imports). |
| `MISSION_NAVIGATOR` | `auto` | `coverage`, `muse`, or `auto` (Muse if a key is set). |
| `MISSION_ALTITUDE_M` | `15.0` | Default flight level, world z. |
| `MISSION_GROUND_Z_M` | `0.0` | Flat ground for geolocation. |
| `MISSION_FOUND_ALTITUDE_M` | `7.0` | Descent altitude over a confirmed person. |
| `MISSION_SPEED_MPS` | `2.5` | Lane speed. |
| `MISSION_TRANSIT_SPEED_MPS` | `4.0` | Transit speed. |
| `MISSION_ARRIVE_RADIUS_M` / `MISSION_TRANSIT_ARRIVE_RADIUS_M` | `1.5` / `2.5` | Arrival radii. |
| `MISSION_AREA_HALF_SIZE_M` | `58.0` | Default area is `[-h, h]²`. |
| `MISSION_LANE_OVERLAP` | `0.2` | Fraction of camera footprint shared by lanes. |
| `MISSION_KEEPOUT_CLEARANCE_M` | `4.0` | Horizontal growth of a keep-out. |
| `MISSION_KEEPOUT_VERTICAL_CLEARANCE_M` | `3.0` | Height margin for "blocks the flight level". |
| `MISSION_KEEPOUTS` | `mission/keepouts.json` | Default keep-out file. |
| `MISSION_SAMPLE_FRACTION` | `0.5` | Detector sample spacing as a fraction of along-track footprint. |
| `MISSION_MIN_CONFIDENCE` | `0.6` | Detection threshold. |
| `MISSION_CONFIRM` | `true` | Confirm positives from above. |
| `MISSION_LEG_TIMEOUT_FACTOR` | `3.0` | Leg timeout multiplier. |
| `MISSION_LEG_TIMEOUT_MIN_S` | `20.0` | Minimum leg timeout. |
| `MISSION_POLL_S` | `0.05` | Runner poll interval. |
| `MISSION_MAX_FRAME_AGE_S` | `2.0` | A frame older than this is not sampled. |
| `CAMERA_DOWN_HFOV_DEG` / `CAMERA_DOWN_WIDTH` / `CAMERA_DOWN_HEIGHT` | `80` / `640` / `480` | Downward camera model. |
| `PERSON_DETECTOR` | `auto` | `muse`, `color`, `none`, `auto`. |
| `MUSE_API_KEY` / `MODEL_API_KEY` | empty | Meta Model API key; shared by all three Muse clients. |
| `MUSE_BASE_URL` | `https://api.meta.ai/v1` | OpenAI-compatible endpoint. |
| `MUSE_MODEL` | `muse-spark-1.3` | Model name. |
| `MUSE_TIMEOUT_S` | `60` | Per-request timeout for the detector and navigator; the dodge advisor defaults to 45 s. |
| `MUSE_MAX_IMAGE_W` | `768` | Detector and navigator JPEG downscale width; the dodge advisor uses 640. |
| `MUSE_REASONING_EFFORT` | `low` | Optional request field; dropped on a 400. |
| `DETECT_COLOR` | `blue` | Mission colour stand-in (the person's jeans). |
| `SIM_PERSON_XY` | `2,-38` | In-process sim person; only the synthetic camera knows. |
| `SIM_START` | `-9,-7,2.5` | In-process sim start pose. |
| `SIM_SPEEDUP` / `SIM_DT` | `1.0` / `0.05` | Sim speed and step. |

`MUSE_NAV_MAX_TOKENS = 1500` and the detector's `MUSE_MAX_TOKENS = 1500` are constants, not
environment variables. Both are sized because reasoning tokens count against the completion
budget.

`brain_launch.py`'s `enable_output`, `arm_on_goal`, `threats`, and the threat geometry are
**ROS parameters**, not environment variables. It passes `BRAIN_BACKEND` to `brain_pilot` as
`additional_env`. `FLYDRONE_CONNECTOME` is only needed if you move the vendored file.

Do not introduce a `FLYCNS_*` setting; the old backend treated that prefix as a model
perturbation. Use `FLYDRONE_*`, `MISSION_*`, `MUSE_*`, `CAMERA_DOWN_*`, or `SIM_*`.

## 8. Testing guidance

There are three levels of verification:

1. **Offline unit/component tests** (fast, no ROS):
   `test_perception.py`, `test_encoders.py`, `test_connectome.py`, `test_lidar.py`,
   `test_threats.py`, plus `test_dodge.py` and `test_dodge_advisor.py`.
   They cover tracking, the loom gate, the real-subset GF trigger, arousal readout, escape
   direction and Muse advice precedence, lidar clustering/ground masking/width freezing, and
   launcher geometry. Run these for any change to perception, encoding, the graph, lidar, the
   connectome runtime, or dodge behaviour. The first five are what `make test-brain` runs;
   the two dodge files are run explicitly because they are not in that Make target.
2. **In-process mission tests** (fast, no Gazebo/Muse key):
   `mcp-server/tests`. They run the real planner, real `MotorLayer` navigation, real runner,
   real camera model, detector pipeline, and FastMCP surface against `SimLink`. The Muse
   navigator tests use a fake HTTP server; the mission integration test injects a fake
   navigator. Run the whole folder for changes to planning, runner, mission schemas, MCP
   routes, navigation, Muse integration, or mission config. Run individual files while
   iterating.
3. **Live Gazebo tests** (slow, manual):
   `tools/flight_test.py` and `tools/search_test.py` against a running launch. These prove
   sensor rates, the reaction, and the full ROS/Gazebo path. They are not part of the
   automated suite and should not be claimed as run unless they were.

What the automated tests do **not** cover: ROS topic wiring in a live launch, Gazebo physics,
the live Muse API call, PX4/MAVROS, or the full MaleCNS graph. Live Gazebo was not re-run after
the connectome/navigator refactor. The current automated evidence is the 33 offline + 41
mission tests. Treat README/REFLEX Gazebo measurements as historical, not as reproduced by
this test suite.

The ament lint tests are in `src/fly_brain/test` and `src/fly_drone/test`. They are useful when
changing package metadata or style, but they are not part of the quick loop. Do not "fix" a
failing functional test by weakening it or skipping it; fix the behaviour.

When changing the MCP surface, update `mcp-server/tests/test_smoke.py`'s `EXPECTED_TOOLS` and
assertions in the same change. When changing a does-everything tool, keep its body schema a
plain `BaseModel` with a default instance so FastMCP flattens it into named arguments; the
smoke test explicitly asserts that there is no opaque `req` property. New fields such as
`navigator` must remain flat and optional.

For a change to the vendored graph:

1. Regenerate `hemibrain_subset.json.gz` with the extractor.
2. Update the counts/metadata and `test_connectome.py` if the graph size changes.
3. Re-check the GF trigger at the strong and weak loom cases; the 16 mV gain is calibrated,
   not arbitrary.
4. Re-check that `circuit.json` still resolves every declared type; missing names must be
   visible in `BrainStatus.missing_types`.
5. Do not claim the full connectome or live Gazebo from unit tests.

## 9. Traps, all of them load-bearing

1. **The runtime is NumPy-only.** No module imports Brian2, there is no SIGINT handler, and
   there is no main-thread import ritual. `brian2` is still listed in `pixi.toml` as a
   leftover from the old backend; removing it is optional cleanup that needs a normal pixi
   lock update. Do not add an import or `preimport_flycns`; neither is needed to load the
   graph.
2. **The graph is vendored.** Do not add a clone/download step to the runtime or Makefile.
   `FLYDRONE_CONNECTOME` points at `src/fly_brain/fly_brain/data/hemibrain_subset.json.gz` by
   default.
3. **Do not present the subset as the whole MaleCNS.** It is a real hemibrain one-hop
   subset. It lacks DNp01, TTMn, PSI, pMP2, pIP10, and Johnston's-organ cells. Use the exact
   names in `circuit.json`; `Giant Fiber` is the GF readout.
4. **pC1/DNp13 is a commitment gate, not a heading decoder.** `courtship` includes pC1 and
   DNp13 spikes; `courtship_drive` only scales `NAVIGATE` approach speed. Muse chooses the
   direction through the goal. Do not use it to steer.
5. **Muse is not a velocity controller.** The detector, navigator, and dodge advisor return a
   detection, a decision, or a direction. None of them may publish a `Twist`; the brain owns
   velocity, the escape clock, and `_clamp()`. The dodge advisor is asked only for a tracked
   moving object and its answer is cached off the control path.
6. **`MISSION_NAVIGATOR=auto` uses Muse if a key exists.** Tests force `coverage`; a
   developer `.env` with a key and no navigator override will use the live navigator outside
   tests. Use `MISSION_NAVIGATOR=coverage` for deterministic demos.
7. **A per-call `navigator='auto'` means the process default, not "always coverage".**
   `MissionService.resolve_navigator()` resolves it against `MISSION_NAVIGATOR`. An explicit
   `muse` without a key fails at `start()`.
8. **No FastAPI startup events.** `FastMCP.from_fastapi` never runs the lifespan. The brain,
   link, detector, and navigator are built lazily. Do not move that work into startup.
9. **`MISSION_LINK=auto` usually means ROS inside pixi.** `rclpy` imports, so `auto` selects
   `ros` even with no brain running. Use `MISSION_LINK=sim` for the standalone demo/tests.
10. **An unpowered X3 falls.** The world spawns it about 2.5 m above the floor.
    `drone_controller` arms it immediately so it hovers. Without that node it drops onto a
    slope, and a tilted lidar sees the hillside move. Keep `sim_launch.py`'s nodes running.
11. **`/drone/enable=false` cuts the rotors.** The brain and controller publish only `true`.
    A stop is a zero twist. Do not add a "disable" path that publishes `false` while airborne.
12. **Simulation time vs wall clock.** `brain_pilot` runs with `use_sim_time`; its control
    period, expansion rates, and escape hold/decay/cooldown follow Gazebo's clock. The mission
    runner's timeouts and polling are wall-clock. Do not mix them.
13. **Flat ground in geolocation.** A bbox or Muse pixel is projected onto
    `MISSION_GROUND_Z_M`; the terrain has ~6 m of relief, so expect a few metres of error and
    let the confirm pass refine it. Report `best.estimated_xy`, not a hand-computed guess.
14. **The lidar must not believe the ground or a breathing cluster.** `lidar.py` uses odometry
    attitude, a vertical band, frozen widths, and a motion gate. A single-plane lidar is blind
    in cruise once the nose drops; do not replace the 3-D path with the single-plane topic.
15. **MockBrain thresholds are tied to the encoder.** If you change `LOOM_MIN_DPS`, the loom
    gain, or `PEAK_HZ`, update `MOCK_GF_THRESHOLD_HZ` / `MOCK_GF_DOUBLE_HZ` and the related
    tests.
16. **FastMCP body flattening.** The tool bodies must stay plain models with default
    instances. `SearchRequest` / `ScoutRequest` are written that way on purpose. A union such
    as `Model | None` collapses the schema and breaks `call_tool(name, {})`. New fields such as
    `navigator` must be plain optional fields.
17. **Do not import `fly_drone` from the mission layer.** It needs `rclpy` and would make the
    MCP server ROS-only. `app/bootstrap.py` adds only `src/fly_brain` for a reason.
18. **`fly_brain` stays ROS-free.** Do not add `rclpy`, `cv_bridge`, or ROS message imports to
    that package. The offline tests and the in-process link depend on it.
19. **`brake`/`stop` semantics.** `Controller.stop()` disarms, clears the goal, resets the
    motor layer, and sends a zero command. It does not publish `/drone/enable=false`.
20. **`steer` and `forward` are empty in `circuit.json`.** Pursuit is not calibrated. The
    status/README may mention it as future work; do not present it as working.
21. **`REFLEX.md`'s calibration section is aspirational.** It names `brain_status`,
    `get_motor_config`, `test_reflex`, `update_motor_config`, and a `calibrate_reflex` prompt.
    Those are not in the current five-tool MCP surface. Do not build instructions or tests
    that assume they exist.
22. **README's historical lidar description is stale.** Use
    `/drone/lidar3d_pointcloud` for the brain; `/drone/lidar_pointcloud` is the single-plane
    lidar used by `drone_controller`'s sensor fusion.
23. **The Makefile is portable now.** It is `/bin/sh` + `pixi run bash`, not zsh-specific.
    Do not reintroduce `/bin/zsh` or `install/setup.zsh` dependencies.
24. **Do not run bare `pytest src/` for a quick check.** It collects the ament lint suites as
    well as the unit tests. Use the named files or the `mcp-server/tests` folder.
25. **One mission at a time.** `MissionService.start()` refuses a second mission while one is
    active. Abort first; abort also leaves a station-keeping goal.
26. **Muse thinks against `max_tokens`.** `muse-spark-1.3` is a reasoning model: its thinking
    tokens count toward the completion budget, and a one-line prompt spends about 190 of them
    (measured 2026-11-03). With a small budget the answer comes back as `content: null` with
    `finish_reason: length`. The detector and navigator ask for 1500 tokens; the dodge advisor
    asks for 2500 because it gets a frame plus motion text. All three report an empty answer as
    an error, never as "nobody there" or "no dodge needed".
27. **A live Muse call can outlast the abort join.** `MissionService.abort()` joins the runner
    for 5 s, while `MUSE_TIMEOUT_S` defaults to 60. If the navigator is mid-request, abort may
    leave the runner finishing that request. Do not claim an immediate abort under a live Muse
    outage; the mission eventually parks, but the HTTP call is not interruptible here.
28. **The dodge advisor is advisory, not synchronous.** It runs one request at a time on a
    worker thread and caches per track; the answer can arrive after the escape has already
    fired. Do not make the control tick wait for Muse, and do not treat a late answer as a
    command.

## 10. Where to change what

| Goal | Main files | Check with |
|---|---|---|
| Change the reflex trigger/timing | `fly_brain/encoders.py`, `fly_brain/motor.py`, `fly_brain/config.py` | `test_encoders.py`, `test_connectome.py`, `test_navigate.py` |
| Change the connectome substrate or data | `fly_brain/connectome.py`, `data/hemibrain_subset.json.gz`, `tools/build_connectome_subset.py` | `test_connectome.py`; re-measure GF trigger |
| Change the channel → cell map | `fly_brain/circuit.json` | `test_connectome.py`; check `BrainStatus.missing_types` |
| Change navigation / goal behaviour | `fly_brain/motor.py` (`_navigate`, `set_goal`), `fly_brain/schemas.py` | `mcp-server/tests/test_navigate.py` |
| Change escape direction / dodge geometry | `fly_brain/dodge.py`, `fly_brain/motor.py` (`_begin_escape`) | `test_dodge.py`; live `flight_test.py reaction` |
| Change the Muse dodge advisor | `fly_drone/dodge_advisor.py`, `fly_drone/node.py`, `fly_brain/dodge.py` | `test_dodge_advisor.py`, `test_dodge.py` |
| Change the pC1 commitment path | `fly_brain/connectome.py` (`arousal`, `courtship`), `fly_brain/motor.py`, `fly_brain/controller.py` | `test_connectome.py`, `test_navigate.py` |
| Add or change a sensor front end | `fly_drone/detect.py`, `fly_drone/lidar.py`, `fly_drone/node.py` | `test_lidar.py`, live `flight_test.py reaction` |
| Change a ROS topic | `fly_brain/config.py`, `fly_bringup/config/bridge.yaml`, launch files | mission config imports the brain settings; update both sides and `test_smoke` health assertions |
| Change the planner | `mission/coverage.py`, `mission/camera.py`, `mission/geometry.py` | `mcp-server/tests/test_coverage.py` |
| Change mission flow/confirmation | `mission/runner.py`, `mission/schemas.py` | `mcp-server/tests/test_mission.py` |
| Change Muse navigation | `mission/muse_navigator.py`, `mission/muse_runner.py`, `mission/config.py`, `mission/schemas.py` | `test_muse_navigator.py`, `test_mission.py` |
| Change the detector | `mission/detector.py`, `mission/config.py` | `test_detector.py`, `test_mission.py`, `look_for_person` |
| Add an MCP tool | `app/routers/mission.py` with `operation_id=...`; keep the body flat; update `mcp-server/tests/test_smoke.py` | `test_smoke.py` |
| Change the search area default/keep-outs | `mission/config.py`, `mission/keepouts.json`, `worlds/*.world` | `test_coverage.py`, `test_mission.py` |
| Add a world or model | `worlds/*.world`, `models/*/model.sdf`; pass `gazebo_simulation:=...`; add tall fixtures to keep-outs | headless launch; planner tests use synthetic areas, not world geometry |
| Change the vehicle interface | `fly_drone/bridge.py`, `fly_bringup/config/bridge.yaml`, `fly_brain/config.py` | live sim; PX4/MAVROS packages are not installed here |

## 11. Known drift, gaps, and stale docs

### README vs code

| README says | Code says |
|---|---|
| README calls the reflex lidar a 360° **single-plane** sensor and its topic table maps `/drone/lidar_pointcloud` to the brain. | The brain uses the 3-D `lidar_3d` (32 planes, ±30°, 320 beams, 10 Hz) on `/drone/lidar3d_pointcloud`; the single-plane sensor remains for `drone_controller`'s sensor fusion. A later README section does mention the 32 × 320 3-D lidar, so the document is internally inconsistent. |
| `record_video:=false` turns recording off. | The launch argument is `video_recording:=false` (`sim_launch.py` and `brain_launch.py`). |
| The forward camera covers 60 × 47°. | The node reads the real HFOV from `camera_info`; `CAMERA_HFOV_DEG=60` is only a fallback. |
| `src/fly_interfaces/` is "the teammate's sensor-fusion messages". | Correct for `fly_interfaces` (three used messages), but `src/fly_msgs/` also exists as an empty placeholder package. Do not confuse the two. |
| Five MCP tools, `find_person` prompt. | Correct, and asserted by `test_smoke.py`. |
| Gazebo/search results and a "working" status table. | Historical measurements. The current automated suite covers offline and in-process sim, not live Gazebo. Re-run before citing. |
| `pixi run colcon build --symlink-install`. | Correct; `make build` runs `colcon build` inside pixi. Either is fine. |

### REFLEX.md vs current surface

`REFLEX.md` is a valuable design note, but its "Calibrating it yourself" section names MCP
tools and a prompt (`brain_status`, `get_motor_config`, `test_reflex`,
`update_motor_config`, `calibrate_reflex`) that are not implemented in this MCP server. Do not
assume them.

Its measured numbers come from the earlier FLYCNS **toy** graph (912 neurons), not the
vendored hemibrain subset. The new subset's GF trigger is calibrated and unit-tested, but its
Gazebo behaviour has not been re-flown. The historical tables are not current evidence.

### Current gaps and unverified claims

- **Vendored subset, not full MaleCNS.** The launch defaults to the real hemibrain one-hop
  subset. The larger MaleCNS graph still needs a neuPrint token and an export;
  `tools/build_connectome_subset.py` is the reproducible scaling point.
- **Live Gazebo / ROS integration:** not covered by automated tests and not re-run after the
  refactor. Use the harnesses and report exactly what you ran.
- **Live Muse navigator:** the HTTP client, parser, fallback, and mission integration are
  covered against fake servers/fakes. The real navigator prompt has not been exercised against
  the live API in this test run. The detector was previously verified live; that is separate.
- **Muse dodge advisor:** the client, parser, caching, and failure path are covered against a
  fake HTTP server; it has not been exercised against the live API in this test run, and it is
  only consulted for a moving track.
- **Muse detector:** the colour stand-in is what the automated mission tests use.
  `test_detector.py` covers the Muse client against a fake server. Do not call a colour blob a
  person detector in operator-facing output.
- **Pursuit decoder:** not wired. `circuit.json`'s `steer`/`forward` lists are empty;
  `PursuitParams` gains default to 0, so pursuit would not move the drone meaningfully.
- **PX4 / MAVROS:** adapters exist in `bridge.py` but are not exercised. `px4` needs
  `px4_msgs`, which is not in `pixi.toml`, and raises an explanatory `RuntimeError` if
  selected. `mavros` only needs `geometry_msgs`, but there is no running MAVROS and the
  adapter is untested.
- **Other worlds:** `cave_rescue.world` and `collapsed_building.world` exist, but the mission
  defaults, keep-outs, tests, and historical search results target `forest_search.world`.
  `person_target` appears only in the forest world.
- **`fly_msgs`:** placeholder package, no messages.
- **Build and install artifacts:** `build/` and `install/` are generated; do not inspect them
  for source truth.
- **Live abort during a Muse request:** can wait for the current HTTP timeout; see trap 27.
- **Historical README claims:** the status table and Gazebo numbers predate the connectome
  and navigator refactor. Do not copy them into a new claim without re-running.

## 12. Definition of done

Before finishing a change:

- [ ] The relevant offline tests and/or `mcp-server/tests` pass. Prefer `make test`; if you
      run a subset, name the exact files. Do not claim live Gazebo verification unless you
      actually ran it.
- [ ] New behaviour has a test at the same level as the behaviour (unit for perception/
      connectome/encoding/dodge, in-process mission for planning/flow/navigation, smoke for
      MCP surface, fake HTTP for every Muse client).
- [ ] No command bypasses `_clamp()`, and no mission-layer or Muse code publishes velocity.
- [ ] `fly_brain` still imports without ROS; the MCP server still imports without ROS.
- [ ] Any changed topic name is updated in `fly_brain/config.py`, `bridge.yaml`, and the
      mission side's shared settings.
- [ ] Any changed MCP tool set updates `EXPECTED_TOOLS` in `test_smoke.py`; new tool fields
      stay flat and optional.
- [ ] Any changed default or measured claim is reflected in the code/docstring, and
      `REFLEX.md`/README are only updated with numbers you actually measured.
- [ ] A changed graph is regenerated by `tools/build_connectome_subset.py`, has provenance
      metadata, and has updated counts/trigger tests.
- [ ] `MISSION_NAVIGATOR`/`PERSON_DETECTOR` fallback behaviour is documented when changed;
      tests force `coverage`/`color` so they never call a paid API.
- [ ] No generated files (`build/`, `install/`, `log/`, `dist/`, `.pixi/`) are edited or
      committed. The vendored data file is intentional source, not a build artifact.
- [ ] The five-tool/one-prompt MCP surface remains intact unless the change explicitly adds
      or removes a tool.
