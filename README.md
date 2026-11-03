# Fly a drone with a fly's brain

A connectome-constrained simulation of the *Drosophila* male central nervous system
([FLYCNS](https://github.com/IONOFIELD/FLYCNS)) flying an X3 multirotor in Gazebo through
ROS 2, with a mission layer on top that uses it to **find a person in the forest**.

Two layers, one message between them:

* **The brain** (`fly_brain`, `fly_drone`) flies. It has two jobs right now:
  * **Reaction.** Anything that comes at the drone -- seen by the forward camera or by the
    360 degree lidar -- drives the fly's looming-sensitive cells (LC4, LPLC2); when the giant
    fibre DNp01 spikes the drone commits to an escape manoeuvre away from the threat, then
    resumes whatever it was doing. A graded **side-step** handles looms too weak to fire it.
  * **Navigation.** Give it a world position and it closes on it *under* those reflexes.
  See [REFLEX.md](REFLEX.md) for the reflex design and every measured number.
* **The mission layer** (`mcp-server/`) decides *where* to go and *what the camera shows*. It is
  an MCP server with five tools. It plans a **boustrophedon coverage path** over the search
  area, hands the brain one waypoint at a time, and runs Meta's **Muse Spark** vision model over
  the downward camera to spot the person. It never commands velocity.

```
agent --MCP--> mission layer --Goal (JSON, /fly_brain/goal)--> brain --Twist--> drone
                    ^------- odometry, downward camera, GoalStatus -------'
                                                                 ^-- forward camera, lidar
```

## Status

| part | state |
|---|---|
| Brain, encoders, motor layer, reflex | working, calibrated on the FLYCNS toy graph |
| Goal navigation (NAVIGATE, under the reflexes) | working, tested against a kinematic model **and flown in Gazebo** |
| Reaction in Gazebo: camera + lidar threats, escape, resume | **flown headless with the FLYCNS toy connectome**: four spheres from four directions, all dodged before contact, goal resumed; table in REFLEX.md |
| Threat launcher (spawns spheres on a collision course) | working; `threats:=true` on the launch |
| ROS 2 node: goal topic, status topic, telemetry, camera_info, lidar | working against the sim |
| Gazebo: cameras, lidar, odometry, bridge, headless rendering | working; `headless:=true` needs no display |
| Search in Gazebo: coverage flight, downward camera, confirm pass | **flown headless**: the lying person found from 15 m at (1.4, -37.7) against a true (2, -38), confirmed from above, 53 s, 7 detector calls, with the colour stand-in on the jeans |
| Boustrophedon planner | working, coverage checked by rasterisation in tests |
| Muse Spark detector | written against the Meta Model API; **needs a key to exercise** |
| MCP server (5 tools, 1 prompt) | working; tested end to end on the in-process sim link |
| Pursuit decoder, real MaleCNS connectome, PX4/MAVROS | not wired, not run, not exercised |

Nothing moves until a goal arrives or something arms the output.

## Layout

```
src/fly_brain/fly_brain/     the brain, NO ROS dependency, testable offline
  schemas.py                 pydantic models incl. Goal, NavParams, GoalStatus, DroneState
  perception.py              Threat (one object in angles, whatever sensed it), LoomTracker
  encoders.py                percept -> Poisson rates; the loom gate (LOOM_MIN_DPS)
  motor.py                   ESCAPE > COOLDOWN > AVOID > NAVIGATE > PURSUE > HOLD, safety clamp
  controller.py              the shared singleton: encode -> step -> decode -> bridge; set_goal
  brain.py, circuit.py, config.py, bridge.py
src/fly_drone/fly_drone/     the ROS 2 side
  node.py                    brain_pilot: Image + CameraInfo + PointCloud2 + Odometry + Goal
                             -> tick -> Twist; GoalStatus and telemetry out
  lidar.py                   scan -> clusters -> Threats with range (no rclpy, testable)
  detect.py                  colour / motion detector for the forward camera
  threats.py                 aim and the sphere's SDF for the threat launcher (testable)
  threat_launcher.py         the node that throws spheres at the drone (Gazebo services)
  bridge.py                  TwistBridge (+ Gazebo enable flag), Px4Bridge, MavrosBridge
  drone_controller.py        forwards /drone/drone_motion to the vehicle; lidar/camera fusion
src/fly_bringup/
  launch/sim_launch.py       Gazebo + bridges + drone_controller (+ chase camera, video)
  launch/brain_launch.py     sim_launch + brain_pilot (+ threat_launcher)
  config/bridge.yaml         every ROS <-> Gazebo topic
src/fly_interfaces/          the teammate's sensor-fusion messages
mcp-server/
  mcp_server.py              FastMCP.from_fastapi(server), the find_person prompt
  app/routers/mission.py     the five tools
  mission/coverage.py        boustrophedon decomposition, lanes, cell tour, transit routing
  mission/detector.py        MuseDetector (Meta Model API), ColorDetector, NullDetector
  mission/link.py            RosLink (topics) and SimLink (brain in-process, kinematic world)
  mission/runner.py          the mission thread and the MissionService the tools talk to
  mission/camera.py          downward camera footprint and pixel <-> ground
  tests/                     planner, navigation, end-to-end sim flights, MCP surface
models/x3_uav/model.sdf      the X3: forward camera, downward camera, chase camera, lidar,
                             velocity controller, odometry
tools/flight_test.py         flight tests over the live sim: sensor rates, the reaction test
                             (spheres thrown while cruising), a survey of a world point
tools/search_test.py         a search mission over the live sim without an MCP client
REFLEX.md                    the reflex design note, with the numbers
```

## Setup

FLYCNS is not vendored. Clone it next to this repository and build the toy graph:

```bash
git clone https://github.com/IONOFIELD/FLYCNS.git ../FLYCNS
cd ../FLYCNS && python scripts/make_testdata.py && cd -   # 912-neuron toy graph, no account
pixi install
pixi run colcon build --symlink-install
```

```bash
export MUSE_API_KEY=...              # Meta Model API key; MODEL_API_KEY also works
```

## Run it

```bash
# unit tests: tracker, encoders, lidar clustering, threat geometry
PYTHONPATH=src/fly_brain:src/fly_drone pixi run python -m pytest \
    src/fly_brain/test/test_perception.py src/fly_brain/test/test_encoders.py \
    src/fly_drone/test/test_lidar.py src/fly_drone/test/test_threats.py -q
# mission tests: planner, navigation, whole missions flown on the in-process sim
pixi run python -m pytest mcp-server/tests -q

# Gazebo + the brain flying the X3 (mock brain; output disarmed until a goal arrives)
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py'
# the real connectome brain (the toy graph), and things thrown at it every 20 s
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py \
    backend:=flycns threats:=true'
# no display (tests, CI): the cameras and the lidar still render off-screen
pixi run bash -c 'source install/setup.bash && ros2 launch fly_bringup brain_launch.py \
    headless:=true video_recording:=false'

# the MCP server, talking to the brain over ROS (default when rclpy imports)
pixi run python mcp-server/mcp_server.py              # stdio
pixi run python mcp-server/mcp_server.py --http 8799  # HTTP

# the MCP server with the brain in-process over a flat kinematic world: no Gazebo, no ROS
MISSION_LINK=sim pixi run python mcp-server/mcp_server.py
```

`brain_launch.py` takes `backend`, `flycns_repo` (default `../FLYCNS`), `flycns_data`,
`enable_output`, `arm_on_goal`, `threats`, `threat_period_s` (0: only on request),
`threat_delay_s`, `threat_distance_m`, `threat_speed_mps`, `use_sim_time`, and passes
`gazebo_simulation`, `headless`, `camera_tracking`, `video_recording` and `video_dir` through
to `sim_launch.py`.

The sim launch records the X3's chase-camera view, framed like the GUI follow camera. Frames
are paced against the wall clock, so a slow-rendering world still produces a video spanning
the whole flight; gaps repeat the previous frame rather than compressing the timeline. Close
Gazebo (or Ctrl-C) and the finished video is moved into `dist/` as
`<world>_<timestamp>.mp4`. `record_video:=false` turns it off, and
`video_dir:=/somewhere/else` writes it elsewhere.

Then, from an MCP client: `search_for_person` with `dry_run: true`, read the plan, and run it
for real. The `find_person` prompt walks an agent through it. Without an MCP client, a goal is
one line:

```bash
ros2 topic pub --once /fly_brain/goal std_msgs/msg/String \
    "{data: '{\"x\": 30, \"y\": -7, \"z\": 15, \"label\": \"east\"}'}"
ros2 topic echo /fly_brain/telemetry          # state, spikes, the percept, the command
ros2 topic pub --once /threats/launch std_msgs/msg/String "{data: '{\"bearing_deg\": 60}'}"
```

Or let the harness do it: with `brain_launch.py threats:=true threat_period_s:=0` up,

```bash
python tools/flight_test.py rates                                   # sensors and brain alive?
python tools/flight_test.py reaction --launch '{"bearing_deg": 70}'  # fly, throw, measure
python tools/search_test.py --look-only                              # the mission layer's eye
```

## Reaction: how a thing coming at the drone becomes a dodge

```
forward camera --colour/motion blobs--+
lidar ---------range clusters---------+--> Threats --LoomTracker--> Percept (bearing, size,
                                                                    expansion, range)
   --encode--> LC4 (expansion), LPLC2 (size) --FLYCNS--> DNp01 spikes --motor--> ESCAPE
```

Both sensors end in the same :class:`Threat` -- bearing, elevation, angular size, and, from
the lidar, range -- and the brain never knows which saw the thing. The tracker differentiates
angular size per object to get the expansion rate the giant fibre responds to, and describes
the **most urgent** object: the smallest optical time-to-contact `size / expansion` among the
objects that are growing, else the biggest. One giant-fibre spike commits the drone to a fixed
manoeuvre away from the threat's bearing and 35 degrees up; the command then decays and a
refractory period follows, after which navigation resumes. The forward camera covers 60 x 47
degrees ahead; the lidar covers the full circle in the airframe's plane. Each catches what
the other cannot: a bird from the side or behind is the lidar's, one from above is the
camera's.

Two things a horizontal lidar on a multirotor does that a loom detector must not believe, both
seen in the first flights and both handled in `fly_drone/lidar.py`: it **sweeps the ground**
when the airframe tilts (so returns more than 2 m below the airframe are dropped, using the
odometry attitude), and its **clusters breathe** by a beam or two per scan (so a track's
physical width is frozen once measured and only a change in range can make it grow). On top
of that nothing counts as a loom below 3 deg/s of expansion (`FLYDRONE_LOOM_MIN_DPS`).

**Testing it.** The threat launcher (`threats:=true`) reads the odometry and spawns a 0.5 m
sphere `distance_m` away at a chosen bearing and elevation, thrown at where the drone will be
at `speed_mps`: a proper intercept, moved 20 times per simulated second through Gazebo's
`set_pose` service (the `VelocityControl` plugin on a spawned model turned out to be inert).
The sphere has a visual and no collision, so the sensors see it and a failed dodge costs
nothing. The launch starts a second `ros_gz_bridge` for the `create`, `remove` and `set_pose`
services; running the launcher by hand needs the same. Every spawn, removal and position is
reported on `/threats/events`, and the brain reports every tick on `/fly_brain/telemetry`, so
a test can line the two up. REFLEX.md has the measured runs.

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

The brain publishes its velocity on `/drone/drone_motion`; the `drone_controller` node forwards
it to the vehicle on `/drone/cmd_vel` at 20 Hz and raises the vehicle's enable flag. The brain
also publishes `std_msgs/Bool` `true` on `/drone/enable` when armed, and never `false`: with
the `MulticopterVelocityControl` plugin, `false` cuts the rotors.

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
sim is tested with and which the status names as a stand-in, not a vision model. Its default
colour is blue, the Gazebo person's jeans: from 15 m they are a 50-70 px blob and the person
lies in a gap in the canopy, so the full mission runs without a key (see Status).

**Sampling.** A frame goes to the detector every `MISSION_SAMPLE_FRACTION` (0.5) of the
along-track footprint, so consecutive frames overlap and nothing between them is missed. A
120 x 120 m search at 15 m is about 1 km of path and roughly 100 Muse calls.

## The ROS 2 interface

| topic | type | direction |
|---|---|---|
| `/drone/drone_motion` | `geometry_msgs/Twist`, body FLU | brain -> drone_controller -> vehicle |
| `/drone/enable` | `std_msgs/Bool` | brain, drone_controller -> vehicle |
| `/drone/odometry` | `nav_msgs/Odometry` | vehicle -> brain, mission, threat launcher |
| `/drone/camera_image`, `/drone/camera_info` | `sensor_msgs/Image`, `CameraInfo`, forward | vehicle -> brain |
| `/drone/lidar_pointcloud` | `sensor_msgs/PointCloud2`, 360 deg single plane | vehicle -> brain |
| `/drone/camera_down_image`, `/drone/camera_down_info` | `sensor_msgs/Image`, `CameraInfo`, downward | vehicle -> mission |
| `/clock` | `rosgraph_msgs/Clock` | Gazebo -> every node with `use_sim_time` |
| `/fly_brain/goal` | `std_msgs/String` (Goal JSON) | mission -> brain |
| `/fly_brain/goal_status` | `std_msgs/String` (GoalStatus JSON) | brain -> mission |
| `/fly_brain/telemetry` | `std_msgs/String` (JSON, every tick) | brain -> anyone watching |
| `/threats/launch`, `/threats/events` | `std_msgs/String` (JSON) | tests -> launcher -> tests |

All names are environment-configurable (`ROS_TOPIC_*`); `MISSION_*`, `MUSE_*`,
`CAMERA_DOWN_*` and `SIM_*` configure the mission layer (see `mcp-server/mission/config.py`).
`ROS_BRIDGE` still selects `twist`, `twist_stamped`, `px4` or `mavros`. `FLYDRONE_LIDAR=0`
flies on the camera alone.

## Conventions

* World frame is **ENU**: x east, y north, z up. The map is 120 x 120 m centred on the origin;
  the forest floor spans about -2.5..3 m in z, so a flight level is a world z, not a height.
  The pines are about 5 m tall.
* Body frame is **FLU**: `x` forward, `y` left, `z` up. `yaw_rate` positive is **left**.
  Positive pitch is nose **down** (right-hand rule about the left-pointing y axis).
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
* **The brain pilot runs on simulation time.** `use_sim_time` is on in `brain_launch.py`, so
  the 20 Hz control period, the expansion rates and the escape's hold and cooldown all follow
  Gazebo's clock; if the world renders slowly, the brain slows with it instead of mis-timing.
  The MCP server's own timeouts are wall-clock.
* **An unpowered X3 falls.** The world spawns it 2.7 m above the floor; `drone_controller`
  arms it at once so it hovers there. Without that node it drops onto a slope, and a tilted
  lidar on the ground sweeps the hillside. Keep `sim_launch.py`'s nodes running.
* **Flat ground in the geolocation.** A bbox is projected onto `MISSION_GROUND_Z_M`. The
  terrain has 6 m of relief, so expect a few metres of error and let the confirm pass refine it.
* **Keep input rates in single digits**, **exact cell-type names**, **`run()` overhead**: see
  REFLEX.md; unchanged.

## What to do next

1. **Point Muse at the real frames.** Set `MUSE_API_KEY`, call `look_for_person` over the
   person model (`python tools/search_test.py --look-only` with the drone at (2, -38, 15)),
   read the description, and tune `min_confidence` and `altitude_m`. The colour stand-in has
   flown the whole mission; Muse only has to say yes to the same frames.
2. **Fly the real connectome.** `backend:=flycns` with `FLYDRONE_DATA=data` after
   `scripts/setup.py` in the FLYCNS clone (a neuPrint token), and re-measure every number in
   REFLEX.md; the toy graph's giant-fibre input is hand-wired.
3. **Wire the pursuit decoder**, as before.
