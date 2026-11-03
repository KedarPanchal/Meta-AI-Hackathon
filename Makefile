SHELL := /bin/zsh

.PHONY: forest cave building brain brain-flycns demo-reflex demo-search build

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

# Demo 2: the search. Lawnmower the south clearing from 15 m with the downward camera, confirm
# the sighting from above, end hovering over the person. Muse Spark if MUSE_API_KEY is set,
# else the colour stand-in on the jeans.
demo-search: build
	source install/setup.zsh && \
	( ros2 launch fly_bringup brain_launch.py backend:=flycns & \
	  sleep 30; python tools/search_test.py --area -14 -50 18 -26 --altitude 15; wait )

build:
	colcon build
