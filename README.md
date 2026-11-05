# A search-and-rescue drone with a fly's brain

A person is missing in a forest. This project sends an X3 multirotor to find them. The mission
layer plans a coverage pattern over the search area, flies it at 15 m, looks down through a
camera, and stops above the person once a sighting is confirmed.

The pilot is a spiking model built from a real slice of the *Drosophila* hemibrain: 5,573
neurons, 712,973 signed chemical edges, and about 4.6 million EM synapses, loaded from a file
in the repository. The sensors drive the fly's looming-sensitive cells, and a giant-fibre spike
triggers an escape manoeuvre when something comes at the aircraft. The search resumes after
the escape.

The simulation runs in Gazebo through ROS 2. An MCP server provides the mission interface, so
an agent can start a search, watch its progress, or stop the flight.

## What runs where

```
MCP client or agent
   | MCP
   v
mission layer (mcp-server/) -- Goal JSON (/fly_brain/goal) --> brain --> Twist
                                                                       |
                                                                       v
                                                              drone_controller --> Gazebo
   ^--------------- odometry, downward camera, GoalStatus -----------------'
brain <-- forward camera, 3-D lidar, odometry
```

The brain (`src/fly_brain`, `src/fly_drone`) owns velocity. It reads the forward camera, the
3-D lidar, and the odometry, and it publishes one body-frame `Twist` per tick. A goal is an
explicit instruction to fly, so the first `Goal` arms the output. With no goal and no arming,
nothing moves.

The mission layer (`mcp-server/`) owns the search. It plans the coverage path, sends one
world-position `Goal` at a time, runs the person detector on the downward camera, and (when a
Muse key is set) lets Muse choose the next goal. It never publishes a `Twist` or a velocity.

Every command from the brain passes through `MotorLayer._clamp()`, which enforces 5 m/s,
2 m/s climb, 2 rad/s yaw, and a 1 m floor.

### The connectome

The graph is a one-hop induced subgraph around LC4, LPLC2, LC10a, Giant Fiber, DNp02, DNp06,
DNp11, DNg30, DNg79, DNg92_a, pC1a through pC1e, and DNp13. It comes from the public Janelia
hemibrain v1.2.1 release (CC-BY 4.0), and `tools/build_connectome_subset.py` regenerates it.
The runtime is a small NumPy LIF simulator: 1 ms steps, 50 ms per brain tick, real EM synapse
counts with transmitter sign. `fly_brain` has no ROS imports and runs offline.

This is a real subset, and it is not the whole MaleCNS. DNp01, TTMn, PSI, pMP2, pIP10, and
Johnston's-organ cells are absent. `Giant Fiber` is the escape readout, and the descending
escape group is DNp02/DNp06/DNp11. The pursuit channels in
`src/fly_brain/fly_brain/circuit.json` are empty, so the drone navigates to goals and runs
reflexes; it does not chase a moving target.

Muse has three optional jobs: detect a person in a downward frame, choose the next candidate
goal or a pixel to investigate, and advise on a dodge direction for a tracked moving object.
None of them can command velocity.

## Requirements

You need [pixi](https://pixi.sh) 0.84 or newer. The environment includes ROS 2 Lyrical,
Gazebo, colcon, NumPy, SciPy, pandas, FastAPI, FastMCP, and OpenCV. You do not need a system
ROS installation or a separate Gazebo install.

A GPU and a display are useful for the Gazebo GUI. `headless:=true` renders the cameras and
lidar off-screen, which is enough for CI or a remote machine. `pixi.toml` solves for Linux
x86_64 and macOS arm64.

A Meta Model API key is optional. Set `MUSE_API_KEY` (or `MODEL_API_KEY`) to give the mission
Muse Spark. Without a key, the mission uses a colour-blob detector and fixed coverage order,
and it still finds the person in the forest simulator. The blob follows the simulated person's
blue jeans. It is a stand-in for the vision model, not a person detector for real imagery.

`make` is convenient but not required; every target is a wrapper around `pixi run`.

The connectome is vendored at `src/fly_brain/fly_brain/data/hemibrain_subset.json.gz`. There
is no FLYCNS checkout, sibling clone, or download step. `pixi.toml` still lists `brian2` from
the old backend, but no runtime module imports it.

## Setup

Install pixi if it is not already on your PATH, then build from the repository root:

```bash
curl -fsSL https://pixi.sh/install.sh | bash

pixi install
pixi run colcon build --symlink-install
source install/setup.bash        # zsh: source install/setup.zsh
```

`make build` runs the colcon step. The launch targets run it for you, so after `pixi install`
you can go straight to `make demo-search` or `make forest`.

To enable Muse, either export the key in your shell or put it in a top-level `.env`:

```bash
export MUSE_API_KEY=...
# or, in .env at the repository root:
MUSE_API_KEY=...
```

Both the brain config and the mission config load `.env` with `setdefault`, so an exported
value wins. One `KEY=value` per line; an `export ` prefix is accepted. The file is gitignored.

Check the installation with the fast test suite:

```bash
make test
# 33 passed
# 41 passed
```

The offline tests cover perception, encoders, the vendored connectome, lidar clustering, and
threat geometry. The mission tests cover the planner, navigation, full missions on the
in-process sim, the MCP surface, the detector, and the Muse navigator. No Gazebo and no Muse
key are needed.

## Run a search

To watch the whole search without an MCP client:

```bash
make demo-search
```

This builds the workspace, starts Gazebo with the brain, and flies the default area with
`tools/search_test.py`. The Gazebo window follows the X3. The terminal prints the plan, each
sighting, the confirm pass, and `FOUND` with the estimated position. The flight ends 7 m above
the person. A video lands in `dist/` as `<world>_<timestamp>.mp4`, plus an HD recording when
`HD_VIDEO=true` (the default).

Useful overrides:

```bash
make demo-search AREA='-14 -50 18 -26'   # the clearing where the person lies
make demo-search HEADLESS=true           # no GUI; the sensors still render
make demo-search BACKEND=mock            # rate stand-in instead of the connectome
```

With a key, Muse looks at every sampled frame. Without one, the colour stand-in does. A
full-map search at 15 m is about a kilometre of flight and roughly 100 detector calls.

To drive the search from an agent, start the full stack:

```bash
make run
# MCP HTTP on http://127.0.0.1:8799
```

This starts Gazebo, `brain_pilot`, and the MCP server. The brain stays disarmed until a
mission sends its first goal, so the drone hovers at the spawn point until a tool call.
`make run` waits for a client; `make all` (the default target) is the demo above.

Point an MCP client at the HTTP server and use the `find_person` prompt, or call the tools
directly. Start with `mission_status` to see which detector is active, then call
`search_for_person` with `dry_run: true` and read the plan before flying it.

Without Gazebo or ROS, run the mission layer against the in-process kinematic world:

```bash
make run-sim
```

The planner, runner, brain navigation, camera model, and detector pipeline are real. The world
and its physics are not Gazebo. This is how the automated mission tests fly.

Other entry points:

```bash
make mission                                    # MCP server alone, against an existing stack
pixi run python mcp-server/mcp_server.py        # stdio, for an MCP client
pixi run python mcp-server/mcp_server.py --http 8799
```

A manual goal also arms the output. With a sourced install and the simulation running:

```bash
ros2 topic pub --once /fly_brain/goal std_msgs/msg/String \
    "{data: '{\"x\": 30, \"y\": -7, \"z\": 15, \"label\": \"east\"}'}"
ros2 topic echo /fly_brain/telemetry
ros2 topic echo /fly_brain/goal_status
ros2 topic echo /mission/muse_raw       # raw Muse answers, when a key is set
```

`mission_status.muse_raw` reports whether that topic has a live publisher, how many records
were sent or dropped, and the last publishing error. When a Muse key is configured, the
mission layer creates the topic at server startup: on the mission link node in ROS mode, or on
its own node in the in-process sim. Start the server before echoing.

The launch files can be run directly:

```bash
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup sim_launch.py'
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py'
```

`sim_launch.py` starts Gazebo, the ROS/Gazebo bridges, `drone_controller`, the chase camera,
and the video recorder. `brain_launch.py` adds `brain_pilot` and optionally the threat
launcher.

Key `brain_launch.py` arguments: `backend:=connectome|mock`, `enable_output`,
`arm_on_goal`, `threats`, `threat_period_s`, `threat_delay_s`, `threat_distance_m`,
`threat_speed_mps`, `use_sim_time`, `gazebo_simulation`, `headless`, `camera_tracking`,
`video_recording`, `video_dir`, and `hd_recording`.

## Simulation environments

The repository has three Gazebo worlds. All three spawn the same `x3_search_uav` model, so the
sensors, topics, and brain are identical. The mission defaults, keep-out file, and most tests
target `forest_search`.

| World | Setting | The person | Good for |
|---|---|---|---|
| `forest_search` | 120 x 120 m of rolling forest, 523 pines, about 700 rocks, clearings, a pond, a campsite, towers, and masts | lying under trees at (2, -38) | the autonomous search and all mission defaults |
| `cave_rescue` | a karst cave with a daylit mouth, a 7-9 m gallery, a 1.5 m squeeze, and a chamber under a daylight shaft | standing in the chamber at (6.9, 6.0) | manual flight, tight-space sensor and reflex tests |
| `collapsed_building` | one office floor, 24 x 18 m, 10 m ceiling, collapsed north-east corner | pinned in the rubble at (8.4, 5.2) | manual flight, low light, indoor obstacle tests |

### forest_search

This is the world the mission layer is built around. The forest floor rolls about 6 m, the
pines are about 5 m tall, and the default flight level is a world z of 15 m, so the drone
clears the canopy. The person lies beneath trees south of the spawn point at (2, -38), rolled
onto one side. The X3 starts at (-9, -7), about 2.5 m above the ground.

The default search area is the whole map. Lane spacing follows the downward camera footprint
at the flight level, about 25 m wide at 15 m. `mcp-server/mission/keepouts.json` lists the
lookout towers, the signal mast, and the radio pole. A keep-out only affects the route when
its top plus 3 m of clearance reaches the flight level, and at 15 m none of the forest
fixtures do. Trees are not keep-outs; the drone flies above the canopy, and the reflexes
handle anything unexpected.

```bash
make forest    # Gazebo and the vehicle, no brain
make brain     # forest plus brain_pilot, disarmed until a goal
```

### cave_rescue

The X3 starts inside the entrance at (-24, -5.5, 2.2), nose down the gallery. The mouth is
about 9-11 m across, the gallery 7-9 m, and the squeeze into the chamber has 1.5 m of
clearance. The X3 is 0.47 m across, so it fits but cannot drift through. The chamber is the
only lit place inside, through a shaft in the roof. Stalactites and stalagmites stay clear of
the centre line.

The mission layer's 15 m default is above the cave, and its planner reasons about a flat
polygon, not a rock shell. Fly this world with goals and the reflexes, or use it for sensor and
geometry tests. A coverage mission would need a small area, a low altitude, and a ground plane
that matches the cave floor.

```bash
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py \
    gazebo_simulation:=cave_rescue'
```

### collapsed_building

The world is a single office floor with a 10 m ceiling, raised well above a real office so
neither the drone nor the cameras riding above it can reach the roof. The north-east corner
has collapsed, so daylight enters there. The rest is lit by five working fluorescents and
three dead fixtures, and the far corners are dark. The X3 starts in the south-west aisle at
(-9.5, -6.5, 2.5). The person is reclined into the rubble at (8.4, 5.2), with the head and
shoulders exposed and the legs pinned.

Indoors the drone hovers over the desks at 2.5 m: above the 0.75 m desk tops, their
monitors (about 1.2 m), the 1.55 m partitions and the 1.85 m shelves, so the downward
camera looks over the desks, but below the 2.9 m interior walls, which are flown around
through the doorways. `brain_launch.py` enforces that level for this world with the
brain's `MAX_ALTITUDE_M` ceiling cap (2.5 m; override with `max_altitude_m:=...`): a goal
at the forest's 15 m level is clamped down to 2.5 m instead of driving the drone -- and
the follow camera 3 m above it, now at 5.5 m under the 10 m ceiling -- into the roof.
Give a mission here a tight area inside the 24 x 18 m floor as well; the cap only fixes
the height, not the plan. The interior walls and columns are in
`mcp-server/mission/keepouts_building.json` so the coverage lanes route around them, and
the brain also analyses any stationary lidar object in its path -- over it, under it, or
sideways around it (`fly_brain.dodge.choose_static_avoid`) -- so an unmapped obstacle
gets a pass instead of a pinned drone. The mission's flat-ground geolocation is least
reliable here. A one-command indoor search (brain stack plus the coverage mission):

```bash
MISSION_NAVIGATOR=coverage MISSION_KEEPOUT_CLEARANCE_M=0.4 MISSION_KEEPOUT_VERTICAL_CLEARANCE_M=0.4 pixi run python tools/search_test.py --area -11 -8 11 8 --altitude 2.5 --keepouts-file mcp-server/mission/keepouts_building.json
```

```bash
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py \
    gazebo_simulation:=collapsed_building'
```

`make cave` and `make building` start those worlds with the vehicle but no brain, which is a
quick way to look around.

To change worlds in a mission, pass `gazebo_simulation:=...` to the launch and give the tool
an `area` and `altitude_m` that fit the space. The default keep-outs are the forest fixtures;
pass `[]` or a world-specific list. Detections are projected onto `MISSION_GROUND_Z_M` (0 by
default), which assumes flat ground. That works to a few metres in the forest and is wrong in
a cave or on a multi-level floor.

## How a search runs

1. The planner decomposes the area with a boustrophedon sweep. A cell boundary goes wherever
   the number of free intervals on the sweep line changes. Cells are flown with back-and-forth
   lanes and toured nearest-first, with transits routed around keep-outs.
2. The runner sends the brain one waypoint `Goal` at a time and waits for `goal_status`.
3. Every 0.5 of the along-track camera footprint, it takes the newest downward frame and calls
   the detector.
4. A positive sighting is confirmed by flying over the estimated ground position and looking
   down again, up to three times.
5. A confirmed sighting ends the mission in `found`. The drone descends to
   `MISSION_FOUND_ALTITUDE_M` (7 m) and holds. `mission_status` reports the position in
   `best.estimated_xy`.
6. `abort_mission`, completion, and failure all leave a station-keeping goal at the current
   position. The brain is never left without a goal.

The Muse navigator, when enabled, keeps the coverage plan as a candidate set. At each decision
it sends Muse the downward frame and up to four candidates. Muse can return `goto` (one
candidate), `investigate` (a normalised pixel), or `hold`. An investigated pixel is projected
onto flat ground, flown to with `Goal.arousal` set from the confidence, and sampled through
the normal detector path. A Muse error falls back to the first candidate and is recorded in
`mission_status.last_nav_decision` and `nav_errors`.

The detector is Muse Spark through the Meta Model API (`https://api.meta.ai/v1`, OpenAI
compatible). Each call sends a JPEG and asks for one JSON object with `person_visible`,
`confidence`, `bbox`, and `description`. Parsing is defensive, and an empty answer is an
error, not "nobody there". `PERSON_DETECTOR=auto` uses Muse when a key is set and the colour
stand-in otherwise.

## MCP tools

| Tool | What it does |
|---|---|
| `search_for_person` | Plans coverage, flies it, detects people, confirms a positive, and ends hovering over the person. |
| `scout_area` | Flies the full coverage path and keeps every sighting. Optional `return_home`. |
| `mission_status` | State, progress, position, brain goal, navigator decisions, sightings, and the log. |
| `abort_mission` | Stops the runner and holds position at the current altitude. |
| `look_for_person` | Runs the detector once on the newest downward frame without moving. |

Both flight tools take `area` (a box or polygon in world metres), `altitude_m`, `speed_mps`,
`lane_overlap`, `sweep_heading_deg`, `keepouts`, `min_confidence`, `confirm_detections`,
`navigator` (`auto`, `muse`, or `coverage`), and `dry_run`. `scout_area` adds `return_home`.
`dry_run` is the cheap way to inspect a plan: it does not ask Muse and does not fly.

`MissionService` allows one mission at a time. `start()` refuses a second mission while one is
planned or running, so call `abort_mission` first.

## Configuration

Settings are read once per process. Restart after changing the environment, or call the
relevant `reload_*` function in tests.

| Variable | Default | Meaning |
|---|---|---|
| `MUSE_API_KEY` / `MODEL_API_KEY` | empty | Enables Muse for the detector, navigator, and dodge advisor. |
| `BRAIN_BACKEND` | `connectome` | `connectome` (or `hemibrain`) or `mock`. |
| `MISSION_LINK` | `auto` | `ros`, `sim`, or `auto`. Inside pixi, `rclpy` imports, so `auto` selects ROS. |
| `PERSON_DETECTOR` | `auto` | `muse`, `color`, `none`, or `auto`. |
| `MISSION_NAVIGATOR` | `auto` | `coverage`, `muse`, or `auto`. `auto` uses Muse when a key is set. |
| `MISSION_ALTITUDE_M` | `15.0` | Default flight level as world z. |
| `MISSION_FOUND_ALTITUDE_M` | `7.0` | Descent altitude over a confirmed person. |
| `MISSION_GROUND_Z_M` | `0.0` | Flat-ground plane used to geolocate detections. |
| `MISSION_SPEED_MPS` | `2.5` | Lane speed. |
| `MAX_SPEED` / `MAX_CLIMB` / `MAX_YAW` | `5` / `2` / `2` | Safety envelope. |
| `MIN_ALTITUDE_M` | `1.0` | Hard floor. |
| `MAX_ALTITUDE_M` | `0` (no cap) | Indoor ceiling cap as world z. `brain_launch.py` sets 2.5 in `collapsed_building`. |

The full list is in `src/fly_brain/fly_brain/config.py` and
`mcp-server/mission/config.py`, and `AGENTS.md` documents the configuration in more detail.

Coordinates are world ENU (x east, y north, z up) and body FLU (x forward, y left, z up).
`bearing_deg` is positive to the right of the nose. Yaw rate is positive to the left. Pitch is
positive nose down. On the downward camera, image top is forward and image right is the
drone's right.

## Tests

`make test` runs 33 offline tests and 41 mission tests. Both suites passed in this checkout.
The offline set covers tracking, the loom gate, the giant-fibre trigger on the real subset,
arousal, lidar clustering and ground masking, and launcher geometry. The mission set covers
the planner, goal navigation, full missions on `SimLink`, the MCP surface, the detector
against a fake API server, and the Muse navigator.

Escape direction and the Muse dodge advisor have their own tests, outside `make test`:

```bash
PYTHONPATH=src/fly_brain:src/fly_drone pixi run python -m pytest \
    src/fly_brain/test/test_dodge.py src/fly_drone/test/test_dodge_advisor.py -q
```

13 tests passed in this checkout. They cover head-on, sideways, and vertical dodge choice,
closest approach, advice precedence and freshness, request shape, per-track caching, and quiet
failure.

The automated suite does not cover a live Gazebo flight, Gazebo physics, the live Muse API,
PX4/MAVROS, or the full MaleCNS. For live checks, start a launch and use the harnesses:

```bash
pixi run python tools/flight_test.py rates
pixi run python tools/flight_test.py reaction --launch '{"bearing_deg": 70}'
pixi run python tools/search_test.py --look-only
```

These need a sourced install and a running simulation. The Gazebo measurements in `REFLEX.md`
come from an earlier 912-neuron toy graph and are historical; the current automated evidence
is the test suite above.

Do not run a bare `pytest src/`. Each ROS package also has ament lint tests, and those are not
part of the fast loop.

## Layout

```
src/fly_brain/fly_brain/       brain package, no ROS dependency
  schemas.py                   Goal, GoalStatus, Percept, Command
  perception.py                Threat and LoomTracker
  encoders.py                  percept to Poisson rates, loom gate
  motor.py                     behaviour priority and safety clamp
  dodge.py                     closest-approach prediction and dodge choice
  brain.py, connectome.py      MockBrain and the NumPy LIF runtime
  controller.py                process-wide singleton
  data/hemibrain_subset.json.gz  vendored hemibrain v1.2.1 subset
src/fly_drone/fly_drone/       ROS 2 adapter
  node.py                      brain_pilot
  lidar.py                     3-D point cloud to Threats
  detect.py                    colour and motion detector for the forward camera
  dodge_advisor.py             cached Muse dodge advice
  drone_controller.py          arms the X3 and forwards motion
  bridge.py                    Twist, PX4, and MAVROS bridges
src/fly_bringup/launch/        sim_launch.py, brain_launch.py
src/fly_bringup/config/        ROS and Gazebo topic bridge
src/fly_interfaces/            sensor-fusion messages used by drone_controller
models/                        X3, trees, rocks, person, cave, office, props
worlds/                        forest_search, cave_rescue, collapsed_building
mcp-server/mission/            planner, runner, detector, navigator, link, camera
mcp-server/tests/              planner, navigation, mission, MCP, detector, Muse tests
tools/build_connectome_subset.py  regenerates the vendored graph
tools/flight_test.py           live sensor, reaction, and survey checks
tools/search_test.py           live search without an MCP client
REFLEX.md                      reflex design note, with historical measurements
AGENTS.md                      detailed working guide for the repository
```

## Troubleshooting

| Symptom | What to check |
|---|---|
| The drone falls at startup. | An unpowered X3 falls. `drone_controller` arms it as soon as the Gazebo subscribers connect and republishes the latest motion command at 20 Hz. Keep `sim_launch.py` and its nodes running. |
| Nothing moves after launch. | The brain starts disarmed. Send a goal, or start with `enable_output:=true`. A goal arms the output because it is an instruction to fly. |
| `make run` looks dead. | It starts the stack and waits for an MCP client. Use `make demo-search` or `make all` for a flight without a client. |
| `/mission/muse_raw` has no output or `ros2 topic echo` says "Could not determine the type". | Start the MCP server first. When a Muse key is set, the topic is created at server startup: on the mission link node in ROS mode, or on its own node with `MISSION_LINK=sim`. Check `mission_status.muse_raw` for `publisher`, `sent`, `dropped`, `closed` and `last_error`. |
| `mission_status` says the detector is `color`. | No Muse key was found. The mission will fly and find the blue-jeans stand-in, but call it a colour blob, not a person detection. Set `MUSE_API_KEY` and restart. |
| A mission used Muse when you wanted coverage. | `MISSION_NAVIGATOR=auto` chooses Muse when a key is set. Use `MISSION_NAVIGATOR=coverage` or pass `navigator: "coverage"` for a deterministic demo. |
| The MCP server cannot find the sim. | `MISSION_LINK=auto` selects ROS whenever `rclpy` imports, even if no brain or Gazebo is running. Use `MISSION_LINK=sim` for the in-process world. |
| Gazebo is slow. | It renders three cameras and a 3-D lidar every simulated tenth of a second. Run with `headless:=true` and watch the video, or lower the sensor rates in `models/x3_uav/model.sdf`. Check the real-time factor with `gz topic -e -t /stats -n 1`. |
| A live abort seems to hang. | If a Muse request is in flight, `abort_mission` joins the runner for 5 s while `MUSE_TIMEOUT_S` defaults to 60 s. The mission parks eventually; the HTTP call itself is not interruptible. |

The brain and `drone_controller` publish only `true` on `/drone/enable`. A `false` cuts the
rotors in Gazebo. A stop is a zero twist, which is a hover.

## Further reading

`AGENTS.md` is the working guide: architecture, invariants, the main settings, and the traps
that have bitten this repository. `REFLEX.md` covers the reflex design and its historical
measurements. `src/fly_brain/fly_brain/config.py` and `mcp-server/mission/config.py` are the
source of truth for defaults.
