SHELL := /bin/zsh

.PHONY: forest cave building brain brain-flycns demo-reflex demo-search demo-quick build

forest: build
	source install/setup.zsh && \
	ros2 launch fly_bringup sim_launch.py gazebo_simulation:=forest_search

cave: build
	source install/setup.zsh && \
	ros2 launch fly_bringup sim_launch.py gazebo_simulation:=cave_rescue

building: build
	source install/setup.zsh && \
	ros2 launch fly_bringup sim_launch.py gazebo_simulation:=collapsed_building

# Gazebo plus the fly brain flying the X3 (mock brain, output disarmed until a goal arrives)
brain: build
	source install/setup.zsh && \
	ros2 launch fly_bringup brain_launch.py gazebo_simulation:=forest_search

# ...with the FLYCNS toy connectome (clone FLYCNS next to this repo first) and spheres thrown
# at the drone every 20 s
brain-flycns: build
	source install/setup.zsh && \
	ros2 launch fly_bringup brain_launch.py gazebo_simulation:=forest_search backend:=flycns threats:=true

# Demo 1: the reflex. The connectome brain flies 40 m legs at 15 m while spheres are thrown
# at it from four directions; the launcher logs HIT or MISS and leaves a red or green ball
# where each came closest. Watch the Gazebo window and the terminal.
demo-reflex: build
	source install/setup.zsh && \
	( ros2 launch fly_bringup brain_launch.py backend:=flycns threats:=true threat_period_s:=0 & \
	  sleep 30; python tools/flight_test.py reaction --leg 40 \
	    --launch '{"bearing_deg": 0}' --launch '{"bearing_deg": 70}' \
	    --launch '{"bearing_deg": 180}' --launch '{"bearing_deg": 0, "elevation_deg": 15}'; wait )

# Demo 2: the search. Lawnmower the area from 15 m with the downward camera, Muse Spark on
# every frame (MUSE_API_KEY; else the colour stand-in on the jeans), confirm a sighting from
# above, then descend to 7 m over the person. Default is the whole 120 x 120 m map: about
# 1 km of lanes, 100 Muse calls, 8 minutes of simulated flight. A quick one, the south
# clearing where the person lies:   make demo-search AREA="-14 -50 18 -26"
AREA ?= -58 -58 58 58
demo-search: build
	source install/setup.zsh && \
	( ros2 launch fly_bringup brain_launch.py backend:=flycns & \
	  sleep 30; python tools/search_test.py --area $(AREA) --altitude 15 --timeout 3600; wait )

# Demo 2, quick: the same search over the strip east of the start, where a second hiker
# lies 16 m away: one lane at 8 m and 4 m/s, FOUND on Muse's first positive (no confirm
# pass), then down to 7 m over them. Measured at 24 s from take-off at real-time factor 0.6.
demo-quick: build
	source install/setup.zsh && \
	( ros2 launch fly_bringup brain_launch.py backend:=flycns & \
	  sleep 30; python tools/search_test.py --area -12 -14 16 0 --altitude 8 --speed 4 --no-confirm --timeout 600; wait )

build:
	colcon build
