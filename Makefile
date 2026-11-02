SHELL := /bin/zsh

.PHONY: forest cave building build

forest: build
	source install/setup.zsh && \
	ros2 launch fly_bringup sim_launch.py gazebo_simulation:=forest_search

cave: build
	source install/setup.zsh && \
	ros2 launch fly_bringup sim_launch.py gazebo_simulation:=cave_rescue

building: build
	source install/setup.zsh && \
	ros2 launch fly_bringup sim_launch.py gazebo_simulation:=collapsed_building

build:
	colcon build
