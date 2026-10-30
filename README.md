# Fly a drone with a fly's brain

A connectome-constrained simulation of the *Drosophila* male central nervous system
([FLYCNS](https://github.com/IONOFIELD/FLYCNS)) flying an X-UAV multirotor in ROS 2, with a
mission layer on top that uses it to **find a person in the forest**.

Two layers, one message between them:

* **The brain** (`fly_brain`, `fly_drone`) flies. The forward camera and microphone drive real
  sensory cell types; spikes from descending and motor neurons become velocity. It has an
  **escape reflex** (a looming object fires the giant fibre DNp01 and the drone dives away), a
  graded **side-step**, and a **pursuit** gate, and now a **goal**: give it a world position and
  it closes on it *under* those reflexes, so an obstacle on the way produces a manoeuvre and
  then the drone resumes. See [REFLEX.md](REFLEX.md) for the reflex design and every measured
  number.
* **The mission layer** (`mcp-server/`) decides *where* to go and *what the camera shows*. It is
  an MCP server with five tools. It plans a **boustrophedon coverage path** over the search
  area, hands the brain one waypoint at a time, and runs Meta's **Muse Spark** vision model over
  the downward camera to spot the person. It never commands velocity.

```
agent --MCP--> mission layer --Goal (JSON, /fly_brain/goal)--> brain --Twist--> drone
                    ^------- odometry, downward camera, GoalStatus -------'
```

## Status

| part | state |
|---|---|
| Brain, encoders, motor layer, reflex | working, calibrated on the FLYCNS toy graph |
| Goal navigation (NAVIGATE, under the reflexes) | working, tested against a kinematic model |
| ROS 2 node: goal topic, status topic, Gazebo enable flag | written, import-checked; **not yet flown in Gazebo** |
| Boustrophedon planner | working, coverage checked by rasterisation in tests |
| Muse Spark detector | written against the Meta Model API; **needs a key to exercise** |
| MCP server (5 tools, 1 prompt) | working; tested end to end on the in-process sim link |
| Gazebo cameras, odometry, bridge | on branch `sim-search-camera`; **not yet launched** |
| Pursuit decoder, real MaleCNS connectome, PX4/MAVROS | as before: not wired, not run, not exercised |

Nothing moves until a goal arrives or something arms the output.

## Layout

```
src/fly_brain/fly_brain/     the brain, NO ROS dependency, testable offline
  schemas.py                 pydantic models incl. Goal, NavParams, GoalStatus
  motor.py                   ESCAPE > COOLDOWN > AVOID > NAVIGATE > PURSUE > HOLD, safety clamp
  controller.py              the shared singleton: encode -> step -> decode -> bridge; set_goal
  brain.py, encoders.py, perception.py, circuit.py, config.py, bridge.py
src/fly_drone/fly_drone/     the ROS 2 side
  node.py                    Image + Odometry + Goal(JSON) -> tick -> Twist, GoalStatus(JSON)
  bridge.py                  TwistBridge (+ Gazebo enable flag), Px4Bridge, MavrosBridge
  detect.py                  colour / motion detector for the loom reflex
mcp-server/
  mcp_server.py              FastMCP.from_fastapi(server), the find_person prompt
  app/main.py                the FastAPI app; every operation_id is an MCP tool name
  app/routers/mission.py     the five tools
  mission/coverage.py        boustrophedon decomposition, lanes, cell tour, transit routing
  mission/detector.py        MuseDetector (Meta Model API), ColorDetector, NullDetector
  mission/link.py            RosLink (topics) and SimLink (brain in-process, kinematic world)
  mission/runner.py          the mission thread and the MissionService the tools talk to
  mission/camera.py          downward camera footprint and pixel <-> ground
  mission/keepouts.json      the world's tall fixtures, with heights
  tests/                     planner, navigation, end-to-end sim flights, MCP surface
launch/launch-fly.yaml       ros2 launch fly_drone launch-fly.yaml
REFLEX.md                    the reflex design note
```

## Setup

FLYCNS is not vendored. Clone it next to this repository and build the toy graph:

```bash
git clone https://github.com/IONOFIELD/FLYCNS.git ../FLYCNS
cd ../FLYCNS && python scripts/make_testdata.py && cd -   # 912-neuron toy graph, no account
pixi install
```

```bash
export FLYDRONE_REPO=$PWD/../FLYCNS
export FLYDRONE_DATA=testdata        # or: data, after scripts/setup.py with a neuPrint token
export BRAIN_BACKEND=flycns          # or: mock
export MUSE_API_KEY=...              # Meta Model API key; MODEL_API_KEY also works
```

## Run it

```bash
# tests: planner, navigation, and whole missions flown on the in-process sim
pixi run python -m pytest mcp-server/tests -q

# the MCP server, talking to the brain over ROS (default when rclpy imports)
pixi run python mcp-server/mcp_server.py              # stdio
pixi run python mcp-server/mcp_server.py --http 8799  # HTTP

# the MCP server with the brain in-process over a flat kinematic world: no Gazebo, no ROS
MISSION_LINK=sim pixi run python mcp-server/mcp_server.py

# the brain node, against the simulator (branch sim-search-camera provides cameras + bridge)
pixi run colcon build --symlink-install
source install/setup.bash
ros2 launch fly_bringup sim_launch.py                 # Gazebo + ros_gz_bridge
ros2 launch fly_drone launch-fly.yaml backend:=flycns flycns_repo:=$PWD/../FLYCNS
```

Then, from an MCP client: `search_for_person` with `dry_run: true`, read the plan, and run it
for real. The `find_person` prompt walks an agent through it.

## The mission tools

| tool | what it does |
|---|---|
| `search_for_person` | plan coverage of an area, fly it, run Muse on the camera every few metres, confirm a sighting from directly above, end hovering over the person |
| `scout_area` | the same coverage, but every sighting is kept and the whole area is flown; optional `return_home` |
| `mission_status` | state, progress, position, the brain's goal, sightings (`best.estimated_xy` is where the person is), the log |
| `abort_mission` | stop and hold position at the current altitude |
| `look_for_person` | run the detector once on the newest frame, without moving |

Both flight tools take an `area` (box or polygon, world metres), `altitude_m` (a world z;
default 15), `lane_overlap`, `sweep_heading_deg`, `keepouts`, `min_confidence`,
`confirm_detections` and `dry_run`.

**The planner.** Boustrophedon cellular decomposition: a sweep line crosses the area, and
wherever the number of free intervals on it changes (an obstacle's extremity, a reflex vertex of
the polygon) a cell boundary is placed. Each cell is covered by back-and-forth lanes spaced by
the downward camera's footprint at the flight height less the overlap, cells are toured
nearest-first, and transits between them are routed around keep-outs through a visibility
graph. The plan reports a `coverage_estimate` computed by rasterising the free area against
the lane footprints. Keep-outs are only obstacles if they reach the flight level; at the default
15 m none of the world's fixtures do, so the drone flies straight lanes above everything and the
brain's reflexes handle anything unexpected.

**The detector.** Muse Spark through the Meta Model API (`https://api.meta.ai/v1`, OpenAI
compatible): one `chat/completions` call per frame with the JPEG as an `image_url` content part,
answer constrained to one JSON object (`person_visible`, `confidence`, `bbox`, `description`),
parsed defensively. Model `muse-spark-1.3` by default (`MUSE_MODEL`). A detection's bounding
box is turned into a ground position from the drone's pose and the camera geometry, assuming
flat ground: good to a few metres from 15 m up, which is what "fly over there and look again"
needs. Without a key, `PERSON_DETECTOR=auto` falls back to a **colour blob** detector, which the
sim is tested with and which the status names as a stand-in, not a vision model.

**Sampling.** A frame goes to the detector every `MISSION_SAMPLE_FRACTION` (0.5) of the
along-track footprint, so consecutive frames overlap and nothing between them is missed. A
120 x 120 m search at 15 m is about 1 km of path and roughly 100 Muse calls.

## The brain's goal interface

The mission layer publishes a `fly_brain.schemas.Goal` as JSON on `std_msgs/String`
`/fly_brain/goal` (`x`, `y`, `z` in world ENU metres, optional `speed_mps`,
`arrive_radius_m`, `id`, `label`). The node answers with a `GoalStatus` every tick on
`/fly_brain/goal_status` (`reached`, distances, the brain's behaviour state, arming). A goal
**arms the output** (`arm_on_goal`), because a goal is an explicit instruction to fly.

Navigation is a saturated P controller with three rules: **climb first** when well below the
flight level (no crossing the forest at tree height), **face the direction of travel** so the
forward camera and the loom reflex look where the drone is going, and **arrival latches**, so a
gust does not un-reach a waypoint. With a goal set, the altitude held during HOLD and AVOID is
the goal's, not the pursuit cruise altitude. The mission never ends with the brain goal-less:
completion, abort and failure all end with a station-keeping goal at the current position.

For Gazebo, the Twist bridge also publishes `std_msgs/Bool` `true` on `/xuav/enable`
(bridged to `gz.msgs.Boolean` `/X3/enable`) when armed, and never `false`: with the
`MulticopterVelocityControl` plugin, `false` cuts the rotors.

## The ROS 2 interface

| topic | type | direction |
|---|---|---|
| `/xuav/cmd_vel` | `geometry_msgs/Twist`, body FLU | brain -> vehicle |
| `/xuav/enable` | `std_msgs/Bool` | brain -> vehicle |
| `/xuav/odom` | `nav_msgs/Odometry` | vehicle -> brain, mission |
| `/xuav/camera/image_raw` | `sensor_msgs/Image`, forward | vehicle -> brain |
| `/xuav/camera_down/image_raw` | `sensor_msgs/Image`, downward | vehicle -> mission |
| `/fly_brain/goal` | `std_msgs/String` (Goal JSON) | mission -> brain |
| `/fly_brain/goal_status` | `std_msgs/String` (GoalStatus JSON) | brain -> mission |

All names are environment-configurable (`ROS_TOPIC_*`); `MISSION_*`, `MUSE_*`,
`CAMERA_DOWN_*` and `SIM_*` configure the mission layer (see `mcp-server/mission/config.py`).
`ROS_BRIDGE` still selects `twist`, `twist_stamped`, `px4` or `mavros`.

## Conventions

* World frame is **ENU**: x east, y north, z up. The map is 120 x 120 m centred on the origin;
  the forest floor spans about -2.5..3 m in z, so a flight level is a world z, not a height.
* Body frame is **FLU**: `x` forward, `y` left, `z` up. `yaw_rate` positive is **left**.
* `bearing_deg` positive is **RIGHT** of the nose.
* Downward camera: image top is forward, image right is the drone's right.

## Traps, all of them load-bearing

* **Import Brian2 on the main thread.** It installs a SIGINT handler at import, which Python
  only permits there; FastAPI, MCP and ROS handlers run on worker threads. Every entry point
  calls `preimport_flycns()` first. The MCP process only needs this with `MISSION_LINK=sim`.
* **Never name a setting `FLYCNS_*`.** FLYCNS treats any such variable as a declared
  perturbation of the model. Ours are `FLYDRONE_*`, `MISSION_*`, `MUSE_*`.
* **No startup events.** `FastMCP.from_fastapi` never runs FastAPI's lifespan. The brain link
  and the detector are built on first use.
* **The brain's escape clock is wall-clock time.** With `SIM_SPEEDUP` above 1 an escape lasts
  its real 1.4 s, which is many simulated seconds. The sim world has nothing to escape from, so
  this only matters if you add looming objects to it.
* **Flat ground in the geolocation.** A bbox is projected onto `MISSION_GROUND_Z_M`. The
  terrain has 6 m of relief, so expect a few metres of error and let the confirm pass refine it.
* **Keep input rates in single digits**, **exact cell-type names**, **`run()` overhead**: see
  REFLEX.md; unchanged.

## What to do next

1. **Fly it in Gazebo.** Merge `sim-search-camera` (cameras, odometry, bridge) with this
   branch, launch both, check `/xuav/odom` and `/xuav/camera_down/image_raw` are live, run
   `search_for_person` with `dry_run: true`, then for real with output disarmed first.
2. **Point Muse at the real frames.** Set `MUSE_API_KEY`, call `look_for_person` over the
   person model, read the description, and tune `min_confidence` and `altitude_m`. The target
   lies under trees: expect to need a lower altitude or a slightly oblique camera.
3. **Wire the pursuit decoder** and **run the real connectome**, as before.
