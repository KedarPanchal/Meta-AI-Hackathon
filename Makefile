SHELL := /bin/zsh

.PHONY: forest cave building brain brain-flycns demo-search build

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

# The demo: search the entire world for the person. The FLYCNS brain lawnmowers the whole
# 120 x 120 m map from 15 m with the downward camera, Muse Spark looks at every frame
# (MUSE_API_KEY in the environment; else the colour stand-in on the jeans), a sighting is
# confirmed from directly above, and the drone descends to 7 m over the person. About 1 km
# of lanes and 8 minutes of simulated flight if the person is on the last lane; they lie
# south of the start and were found on the second lane in testing. Runs through pixi's own
# bash, so it works from any shell. HEADLESS=true for no window (the video still records).
AREA ?= -58 -58 58 58
HEADLESS ?= false
demo-search: build
	pixi run bash -c 'source install/setup.bash && ( ros2 launch fly_bringup brain_launch.py backend:=flycns headless:=$(HEADLESS) & \
	  sleep 30; python tools/search_test.py --area $(AREA) --altitude 15 --timeout 3600; wait )'

build:
	colcon build
