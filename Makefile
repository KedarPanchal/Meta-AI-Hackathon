SHELL := /bin/zsh

.PHONY: forest cave building brain brain-flycns build

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

build:
	colcon build
