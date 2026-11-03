"""The threat launcher's geometry and its Gazebo requests."""
from __future__ import annotations

import math

from fly_drone.threats import Launch, aim, threat_sdf


def test_spawn_point_is_at_the_requested_range_and_bearing() -> None:
    drone = (10.0, -5.0, 15.0)
    yaw = math.radians(90.0)                          # nose north
    pos, vel, tof = aim(drone, yaw, (0.0, 0.0, 0.0), 8.0, bearing_deg=0.0, elevation_deg=0.0,
                        speed_mps=4.0, lead=False)
    assert math.isclose(pos[0], 10.0, abs_tol=1e-6) and math.isclose(pos[1], 3.0, abs_tol=1e-6)
    assert math.isclose(pos[2], 15.0)
    assert math.isclose(math.hypot(*vel[:2]), 4.0) and abs(vel[2]) < 1e-9
    assert vel[1] < 0                                 # flying south, back at the drone
    assert math.isclose(tof, 2.0)
    # bearing + is the drone's right: nose north, right is east
    pos_r, _, _ = aim(drone, yaw, (0.0, 0.0, 0.0), 8.0, 90.0, 0.0, 4.0, lead=False)
    assert math.isclose(pos_r[0], 18.0, abs_tol=1e-6) and math.isclose(pos_r[1], -5.0, abs_tol=1e-6)
    pos_up, _, _ = aim(drone, yaw, (0.0, 0.0, 0.0), 8.0, 0.0, 90.0, 4.0, lead=False)
    assert math.isclose(pos_up[2], 23.0, abs_tol=1e-6)


def test_lead_and_miss() -> None:
    drone = (0.0, 0.0, 15.0)
    pos, vel, tof = aim(drone, 0.0, (2.0, 0.0, 0.0), 8.0, 0.0, 0.0, 4.0, lead=True)
    # the drone moves +x at 2 m/s and the object comes back at 4 m/s from x=8: they meet
    # where 2 T = 8 - 4 T, i.e. after 4/3 s at x = 8/3
    assert math.isclose(pos[0], 8.0) and vel[0] < 0
    assert math.isclose(tof, 4.0 / 3.0, abs_tol=1e-6)
    hit = (pos[0] + vel[0] * tof, pos[1] + vel[1] * tof)
    drone_then = (drone[0] + 2.0 * tof, drone[1])
    assert math.isclose(hit[0], drone_then[0], abs_tol=1e-6)
    assert math.isclose(hit[1], drone_then[1], abs_tol=1e-6)
    # a drone flying away faster than the object cannot be intercepted: no lead, straight throw
    _, vel2, tof2 = aim(drone, 0.0, (6.0, 0.0, 0.0), 8.0, 180.0, 0.0, 4.0, lead=True)
    assert math.isclose(tof2, 2.0) and vel2[0] > 0
    pos, vel, tof = aim(drone, 0.0, (0.0, 0.0, 0.0), 8.0, 0.0, 0.0, 4.0, miss_m=1.5, lead=False)
    miss = (pos[0] + vel[0] * tof, pos[1] + vel[1] * tof)
    assert math.isclose(math.hypot(miss[0], miss[1]), 1.5, rel_tol=0.1)
    assert miss[1] > 0                                 # to the drone's left, nose +x


def test_sdf_and_kinematics() -> None:
    sdf = threat_sdf('threat_3', 0.25, 'magenta')
    assert '\n' not in sdf
    assert '<static>true</static>' in sdf and '<radius>0.250</radius>' in sdf
    assert '<collision' not in sdf                     # it must not knock the drone down
    assert '<plugin' not in sdf                        # moved by set_pose, not by physics
    launch = Launch(name='threat_3', position=(10.0, 0.0, 15.0), velocity=(-4.0, 0.0, 0.0),
                    radius_m=0.25, color='magenta', ttl_s=4.5, distance_m=10.0, bearing_deg=0.0,
                    elevation_deg=0.0, speed_mps=4.0, tof_s=2.5, t0=100.0)
    assert launch.position_at(99.0) == (10.0, 0.0, 15.0)
    assert launch.position_at(101.5) == (4.0, 0.0, 15.0)
