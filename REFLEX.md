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
             =   0.15 s  + sqrt(2 * 0.9 m / 5 m/s^2)
             =   0.75 s
```

The knob is the loom input gain. FLYCNS anchors its own loom protocol at `PEAK_HZ = 5.0` Hz
per cell, chosen so the GF fires the one or two spikes von Reyn et al. 2014 measured, and the
encoders keep that anchor; the gain multiplies it.

**Measured** — time to contact when DNp01 first spiked, 0.5 m object at 5 m/s from 6 m:

| gain | b=0° | b=−30° | b=+30° | b=−60° | b=+60° |
|---|---|---|---|---|---|
| x1 | 0.10 s | — | — | — | — |
| x4 | 0.20 s | — | — | — | — |
| x12 | 0.90 s | 1.10 s | **0.75 s** | 1.10 s | 0.90 s |
| **x16** | 1.10 s | 1.10 s | 0.90 s | 1.10 s | 0.90 s |
| x20 | 1.10 s | 1.10 s | 1.15 s | 1.10 s | 1.10 s |

`x16` is the default: the smallest gain that clears 0.75 s at every bearing with margin. A
higher gain fires earlier but also fires at harmless things.

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

## Open problems

* **Sound flips sign between datasets.** FLYCNS reports that on the real connectome sound
  *lowers* the GF loom response (30% → 17%); on the toy graph it *raises* it. The audio
  channel is wired and tunable, but which way it moves the trigger has to be re-measured on
  whichever graph is loaded. Do not assume.
* **The ground looks like a loom.** It expands exactly like an approaching object during
  descent. `reflex.suppress_below_horizon` ignores looms more than 5° below the horizon;
  it is **off** by default because it also blinds the drone to real low threats.
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
