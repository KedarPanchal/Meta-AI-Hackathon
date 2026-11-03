# The escape reflex

How a looming object becomes an evasive manoeuvre, and the numbers behind each choice.

Everything marked **measured** was measured in this workspace on the FLYCNS **toy** graph
(912 neurons, `scripts/make_testdata.py`), Python 3.14, brian2 2.11, cython engine. Nothing
here has been measured on the real MaleCNS connectome yet — that needs a neuPrint token — and
the toy graph's giant-fibre input is hand-wired, so every number below should be re-measured
before it is trusted.

## The one idea

**A giant-fibre spike is a trigger event, not a rate.**

DNp01 fires one or two spikes to a looming object and then adapts. If you treat its spike
count as a speed command, the drone escapes hardest at the moment the loom begins and stops
escaping exactly as the threat arrives. So the reflex is a *triggered primitive*: one spike
commits the aircraft to a fixed manoeuvre for a fixed time, which then decays, followed by a
refractory period.

**Measured** — a 0.5 m object closing at 5 m/s, head-on, loom gain x16:

| time | distance | angular size | expansion | GF spikes |
|---|---|---|---|---|
| 0.00 s | 6.00 m | 4.8° | 0°/s | 0 |
| 0.30 s | 4.50 m | 6.4° | 6.7°/s | **1** |
| 0.60 s | 3.00 m | 9.5° | 14°/s | 1 |
| 0.85 s | 1.75 m | 16.3° | 38°/s | 2 |

One or two spikes per tick, never a train. That is the property the whole design rests on.

## The primitive

Away from the threat bearing and 35° up, at 3 m/s. A threat at bearing `b` (positive to the
RIGHT) lies along `(cos b, -sin b)` in body `(forward, left)`, so the escape runs along
`(-cos b, +sin b)`, scaled by `cos(climb)`, with `vz = sin(climb)`.

**Measured** — escape velocity in body FLU (x forward, y left, z up):

| threat bearing | vx | vy | vz |
|---|---|---|---|
| 0° (dead ahead) | −2.46 | +0.00 | +1.72 |
| +60° (right) | −1.23 | +2.13 | +1.72 |
| −60° (left) | −1.23 | −2.13 | +1.72 |
| +90° (hard right) | −0.00 | +2.46 | +1.72 |

Scaling: `x(1 + 0.5 per extra GF spike)`, and `+25%` if PSI (flight power) also spiked. A
TTMn (jump motor neuron) spike is accepted as a trigger too, because the GF does not always
drive it and vice versa. **Measured**: 1 spike → 3.00 m/s, 2 → 4.19, 3 → 5.31, 1+PSI → 3.67.

### Threat from above

Above 40° of elevation the threat is overhead, so the escape **dives** instead of climbing:
`climb` flips sign and the drone slides sideways and down. The altitude floor
(`MIN_ALTITUDE_M`) still applies, so this cannot fly the aircraft into the ground.

## Which way

The primitive above ran *away* from the threat's bearing and 35 degrees up, whatever the
threat did. Flown in Gazebo it dodged, but by little: spheres thrown head-on passed within
half a metre, because a retreat along the object's line gains nothing on something faster
than the aircraft, and the climb put the drone under the sphere's path. A fly's escape is a
sideways jump. So the direction is now a decision of its own (`fly_brain/dodge.py`), made
when the spike comes:

* From the object's tracked relative position and velocity (the lidar measures both; a
  camera-only threat is assumed to fly straight at the drone), the point of closest approach
  says which side it will pass on. The dodge goes **away from that side**.
* Anything coming straight at the drone, or steeply from above or below, gets a **sideways**
  dodge, left or right: it leaves the object's line fastest and never puts the airframe under
  or over it (and its downwash).
* Only a level crossing from the side, beyond 45 degrees of bearing, gets a **vertical** dodge:
  up, or down if it will pass above.
* The escape vector is that dodge plus a push straight away from the object, weighted
  `away_fraction` (0.45); a vertical dodge is pitched `climb_deg` as before.

**Muse decides when it can.** With a key set, the ROS node hands Muse Spark the forward
camera frame and the object's tracked motion as soon as a moving object is tracked, off the
control path (`fly_drone/dodge_advisor.py`); Muse predicts where the object goes and names a
dodge. If its answer is about this track and younger than `advice_max_age_s` (6 s) when the
giant fibre fires, it wins; otherwise the geometry decides, as it always can in time. Honest
about latency: a vision model answers in seconds and a sphere from 10 m at 5 m/s arrives in
two, so Muse gets the slow, far or circling things, and the geometry the sudden ones.

**Did it hit?** The GUI cannot show whether the drone passed under a sphere or through it, so
the launcher tracks the distance between the two centres over the flight, logs `HIT` or `MISS`
with the closest distance (touch is 0.6 m for a 0.25 m sphere and the X3's 0.35 m), counts
them, and leaves a self-lit red (hit) or green (miss) ball where the sphere came closest.

## Timing

`hold 350 ms → decay 250 ms → cooldown 800 ms`, and pursuit is damped to 30% during the
cooldown. Priority is strict: `ESCAPE > COOLDOWN > AVOID > PURSUE > HOLD`.

**Measured** — one trigger at t=0, no further input:

| t | state | speed |
|---|---|---|
| +0.00 s | ESCAPE | 3.00 |
| +0.35 s | ESCAPE (hold) | 3.00 |
| +0.45 s | ESCAPE (decay 0.60) | 1.80 |
| +0.60 s | COOLDOWN | 0.00 |
| +1.00 s | COOLDOWN | 0.00 |
| +1.45 s | AVOID (re-armed) | 0.32 |

COOLDOWN outranks AVOID deliberately. Without that, every escape is followed by a side-step
chasing the same object as it recedes.

## Calibrating the trigger

The constraint is that the drone must be able to get out of the way:

```
tau_trigger  >=  latency + sqrt(2 * clearance / deceleration)
             =   0.15 s  + sqrt(2 * 0.9 m / 4 m/s^2)
             =   0.82 s
```

(The X3's `maxLinearAcceleration` was raised from the gz-sim example's 2 m/s^2 to 4, because
a reflex is only as good as the airframe's ability to move a metre in under a second.)

Two knobs set when the giant fibre can fire at all:

* **The loom gate, `LOOM_MIN_DPS` (3 deg/s).** Below this expansion rate neither LC4 nor
  LPLC2 is driven: the object is tracked (LC10a) but is not looming. It exists because both
  real sensors differentiate to a degree or two per second of noise on their own (a camera
  blob's edge jitter, a lidar cluster's beam count), and because LPLC2 is a motion detector,
  not a size meter: a big thing that is not growing must not drive the giant fibre.
* **The loom input gain** (x16, as before), which multiplies FLYCNS's anchor of
  `PEAK_HZ = 5.0` Hz per cell, chosen so the GF fires the one or two spikes von Reyn et al.
  2014 measured.

**Measured** on the toy graph (2026-11-03): with the gate in place, DNp01 fires on the **first
tick either loom input is driven at all**, so the trigger range is set by the gate, not by
the gain. A 0.5 m object closing from 10 m:

| closing speed | object | first GF spike at | angular size | expansion | time to contact |
|---|---|---|---|---|---|
| 5.0 m/s | 0.5 m | 6.5 m | 4.4 deg | 3.3 deg/s | 1.30 s |
| 7.5 m/s | 0.5 m | 7.0 m | 4.1 deg | 4.2 deg/s | 0.93 s |
| 10.0 m/s | 0.5 m | 9.0 m | 3.2 deg | 3.3 deg/s | 0.90 s |
| 5.0 m/s | 0.8 m | 7.75 m | 5.9 deg | 3.7 deg/s | 1.55 s |

The expansion of a sphere of width `w` closing at `v` from range `r` is `w v / r^2`, so a
gate at a fixed expansion fires at `r = sqrt(w v / gate)`: the faster the object, the
farther out. That is the speed-awareness the fly has (the GF's threshold angle scales with
the object's size-to-speed ratio), obtained here without any fitting. The Poisson inputs
make the exact tick stochastic: expect +-1 m run to run. LC4's saturation was lowered from
400 to 60 deg/s at the same time so that the expansion channel carries weight during the
approach rather than only in the last half metre; it does not move the trigger on the toy
graph, which fires at the gate either way.

**The mock brain** (`BRAIN_BACKEND=mock`) was changed to match: it sums the two loom
channels and trips at 7 Hz, which is 4 Hz of LC4 at the gate plus a 3-degree LPLC2 drive.
Offline it fires at 6.75 m for 0.5 m at 5 m/s (1.35 s), 8.25 m at 7.5 m/s (1.10 s), 9.5 m at
10 m/s (0.95 s), 4.3 m at 2 m/s (2.15 s), and never for a static or slowly growing object.

The old sweep of the loom gain against bearing (x12 marginal at +30 degrees, x16 chosen,
x20 firing at harmless things) stands; the gate did not change which gain is the smallest
that clears the budget at every bearing.

Two things that surprised us and are worth remembering:

* **The trigger distance depends on how long the object has been in view.** The membrane
  integrates, so the same object starting at 6 m triggers earlier than one starting at 5 m
  (0.90 s vs 0.70 s at gain x12). There is no single "trigger distance" independent of
  history.
* **The bearings are not symmetric.** The eye split drives one half of the population harder,
  and the toy graph's two sides are not identical populations. Always sweep both signs.

Driving above the physiological anchor is a deliberate trade: a real fly triggers much later
because it accelerates far harder than a multirotor can.

## Latency: the number that constrains everything

brian2's `Network.run()` has a **large fixed cost per call**, independent of how much
simulated time it covers. **Measured** on the toy graph (cython engine):

| chunk | wall time | realtime factor | wall per simulated ms |
|---|---|---|---|
| 5 ms | 76.2 ms | 0.07 | 15.2 |
| 10 ms | 75.2 ms | 0.13 | 7.5 |
| 20 ms | 77.5 ms | 0.26 | 3.9 |
| 50 ms | 82.3 ms | 0.61 | 1.6 |
| 100 ms | 93.8 ms | 1.07 | 0.94 |
| 200 ms | 100.5 ms | 1.99 | 0.50 |

About **74 ms of fixed overhead per call** plus ~0.13 ms per simulated ms. The consequences:

* A 20 ms control tick is 97% overhead and can never reach real time, even on a 912-neuron
  graph. `BRAIN_TICK_MS` therefore defaults to **50 ms** (realtime factor 0.61), which trades
  latency against throughput; **measured** 0.54 in the live ROS node.
* Real time needs chunks of 100 ms or more, which costs 100 ms of reflex latency — a
  significant fraction of the 0.75 s budget.
* Pruning the graph will **not** fix this on its own, because the cost is per call, not per
  neuron. On the full 170k-neuron connectome the per-ms term grows and the fixed term stays,
  so larger chunks help proportionally more.

Practical options, in order of preference: raise `BRAIN_TICK_MS`; lower the simulator's
real-time factor to match (Gazebo makes this easy, and it is honest — the brain really is
that slow); run the reflex on a pruned subgraph in a separate faster loop. Check
`brain_status.realtime_factor` before believing any timing result.

## Two eyes: the camera and the lidar

Every sensor ends in a `Threat` -- bearing, elevation, angular size, and range if the sensor
has it -- and the brain never knows which saw the thing (`fly_brain/perception.py`). The
tracker differentiates angular size per `(source, track id)` and hands the brain the **most
urgent** object: the smallest optical time-to-contact `tau = size / expansion` among the
objects that are growing, else the biggest. Three rules shape what it is allowed to call a
loom, each of which was a false escape in Gazebo before it existed:

* **Three samples before anything is differentiated.** A camera blob's first frames are
  where it merges with a neighbour or gets a fresh track id, and both are a leap in size. The
  lookout tower's red roof fired the reflex that way from 40 m.
* **An object further than `max_tau_s` (3 s) from contact is scenery.** Growing, but too
  slowly to be coming. The climb out of the pine canopy at take-off is the case: the branches
  at the drone's level come about a metre closer per second of climb, a big thing growing
  slowly, which fired the giant fibre four times on the way up to 15 m.
* **A lidar track is unsettled until its width is measured** (three scans), and the width
  is then frozen; see below.

The forward camera covers 60 x 47 degrees ahead and is colour-blind to everything but its
marker colour (`FLYDRONE_DETECT_COLOR`, magenta by default: nothing in the forest world is
magenta, and the tower roof is red). The lidar covers the full circle. Each catches what the
other cannot.

## What the lidar must not believe

The brain has its own 3-D lidar (`lidar_3d` in the X3 model: 32 planes over +-30 degrees of
elevation, 320 beams per plane, 10 Hz). Three lessons from the first flights, all in
`fly_drone/lidar.py`:

* **A single plane is blind in cruise.** The multirotor pitches about 10 degrees nose-down
  to hold 2.5 m/s, so a horizontal scan looks at the ground 80 m ahead and a sphere coming at
  the airframe at its own level stays above the plane until the last metre. Spheres thrown
  from 70 degrees right and from dead ahead passed within 1.3 m without a single lidar return.
  Hence the 3-D sensor; the single-plane lidar the teammate mounted stays for his node.
* **It sweeps the ground.** The airframe tilts, or sits on a slope, and the lower planes cut
  the hillside somewhere in range; as the tilt changes that cut moves at metres per second.
  Sitting unpowered on a 8-degree slope, the hillside at bearing -96 degrees "approached"
  from 16.5 m to 10 m in six seconds and fired the reflex every two seconds. The odometry
  attitude places every return in the world, and returns more than 2 m below or 4 m above
  the airframe are dropped before clustering.
* **Its clusters breathe.** The return count changes by a beam or two per scan and branches
  split a tree into fragments that merge again; differentiated at 10 Hz that is several
  degrees per second of fake expansion, and the median of a width window moving as the window
  fills looked exactly like a loom at start-up. So a track's physical width is measured over
  its first three scans and then **frozen**, the angular size handed to the tracker is
  computed from that width and the median of the last three ranges (which the lidar measures
  to a centimetre), and only a change in range can make an object grow.

Clustering is connected components on a polar grid (one beam by 0.5 m of range, dilated one
cell so a single dropped beam does not split an object), which keeps a far wall whole however
sparse its returns, joins the planes that hit the same trunk, and separates two objects at
the same bearing and different ranges. About 6 ms for a 6,000-return scan.

## Flown in Gazebo

**Measured** 2026-11-03 in the forest world, headless, with `tools/flight_test.py reaction`:
the drone takes off from among the trees, climbs to 15 m, and flies 40 m legs at 2.5 m/s; once
it is cruising a 0.5 m magenta sphere is thrown from 10 m at 5 m/s on an intercept course,
from four directions in turn. "To contact" is the sphere's time of flight to the intercept
less the time the giant fibre took to fire, in simulation time; "closest" is how near the
sphere actually came, from the launcher's reported positions against odometry. Every escape
was followed by the drone resuming and reaching its goal.

| sphere from | FLYCNS toy connectome | | | mock brain | | |
|---|---|---|---|---|---|---|
| | fired by | to contact | closest | fired by | to contact | closest |
| ahead, level | camera, GF x2 | 0.53 s | 1.34 m | camera | 1.26 s | 1.25 m |
| 70 deg right, level | lidar, GF x2 | 1.30 s | 0.73 m | lidar | 0.82 s | 1.24 m |
| behind, level (a chase) | lidar, twice | 2.25 s | 1.59 m | no escape needed | - | 3.85 m |
| ahead, 15 deg up | lidar, GF x2 | 0.84 s | 0.54 m | camera | 1.24 s | 1.14 m |

Take-off through the canopy fired **no** false escape on either run (it fired four per
take-off before the moving-object rule and the time-to-contact gate). The sphere thrown
from behind chases a drone flying away at 2.5 m/s, so it closes at 2.5 m/s and the mock's
gate never opened before the drone outran it; the connectome, with its higher LC4 drive,
fired twice. The X3 accelerates at 4 m/s^2, so an escape with half a second to spare clears
the sphere by about a metre: tight, and the lidar's path is where the time goes (two scans to
settle a width, two to confirm motion, then the expansion must cross the gate). On the real
connectome every number here must be re-measured; the toy graph's giant-fibre input is
hand-wired.

**With the dodge direction** (branch `reflex-dodge-direction`, same test, the launcher now
judging the hit itself: touch is 0.60 m between centres):

| sphere from | FLYCNS: dodge, fired before contact, closest | mock: dodge, fired, closest |
|---|---|---|
| ahead, level | right (camera), 1.04 s, 1.54 m | left, 1.27 s, 2.48 m |
| 70 deg right, level | left (lidar, "passing below us 2.5 m"), 1.32 s, 1.58 m | down (lidar), 0.87 s, 1.92 m |
| behind, level (a chase) | down (lidar), 1.19 s, 3.35 m | down, 1.30 s, 2.79 m |
| ahead, 15 deg up | left (camera), 1.02 s, 2.15 m | left, 1.18 s, 1.97 m |

Eight throws, **no hit**, and the clearances roughly doubled against the climb-and-retreat
primitive above (0.5-1.3 m). Head-on spheres were dodged sideways; the crossing and chasing
ones took the vertical dodge the rules assign to a level approach from beyond 45 degrees.
Muse was not consulted (no key on this machine): every direction came from the geometry, or
from the default when only the camera saw the sphere.

The brain pilot costs about 10% of a core with OpenCV pinned to one thread; the simulator's
real-time factor on this machine's software renderer sits between 0.5 and 1.0 with the three
cameras and two lidars, and both nodes run on the simulation clock, so a slow render slows
the reflex's timing rather than mis-timing it.

## Open problems

* **Sound flips sign between datasets.** FLYCNS reports that on the real connectome sound
  *lowers* the GF loom response (30% → 17%); on the toy graph it *raises* it. The audio
  channel is wired and tunable, but which way it moves the trigger has to be re-measured on
  whichever graph is loaded. Do not assume.
* **The ground looks like a loom.** It expands exactly like an approaching object during
  descent. `reflex.suppress_below_horizon` ignores looms more than 5° below the horizon;
  it is **off** by default because it also blinds the drone to real low threats. For the
  lidar the problem is solved differently (the vertical band, above); for the camera it is
  still open.
* **The lidar sees nothing straight up or down.** Its planes cover +-30 degrees. A threat
  from above is the camera's, and only while the camera's 47-degree vertical field holds it;
  with the nose down in cruise that field looks 10 degrees under the horizon.
* **Escape direction from the brain does not work.** With `direction_source='brain'` the
  left/right GF imbalance points *toward* the threat on the toy graph, because loom input
  reaches both giant fibres. The default is `'vision'`.
* **The escape ignores current velocity.** It is computed in the body frame as though the
  aircraft were stationary. A drone already moving toward the threat needs more.
* **No neuromodulation, no background activity.** The model runs at 0 Hz baseline, so
  inhibition-only and disinhibition effects cannot appear at all, and P1 "gating" can only act
  as extra excitation summing downstream. Several real mechanisms are simply outside what this
  model can express.

## Calibrating it yourself

The `calibrate_reflex` MCP prompt walks through this. The short version:

```
brain_status                  # check realtime_factor first
get_motor_config              # note encoder.gains.loom_speed
test_reflex                   # read ttc_at_trigger_s and verdict
update_motor_config           # raise the loom gain, repeat
```

Sweep bearings 0, ±30, ±60 and re-run with `sound_db: -20`. Keep the smallest gain that
passes everywhere.
