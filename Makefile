# Every target runs inside the pixi environment through bash, so `make <target>` works from
# any shell (bash, zsh, fish) on Linux or macOS without `pixi shell` first.
#
# Common targets:
#   make all                  fly the autonomous search demo (mock brain)
#   make run                  full stack: Gazebo + brain + MCP HTTP server; no mission until a client sends one
#   make run BACKEND=mock     ...with the fast stand-in instead of the connectome
#   make run-sim              MCP server + in-process brain, no Gazebo and no ROS
#   make demo-search          autonomous search without an MCP client
#   make test                 offline unit tests + in-process mission tests
#   make forest|brain|...     the individual pieces
SHELL := /bin/sh
RUN := pixi run bash -c
SETUP := source install/setup.bash

# Load a local .env if present. The file is gitignored and uses `export KEY=...`.
ENV_FILE ?= .env
LOAD_ENV = if [ -f "$(ENV_FILE)" ]; then . "./$(ENV_FILE)"; fi;

# --- run settings -----------------------------------------------------------
WORLD ?= forest_search
BACKEND ?= connectome
HEADLESS ?= false
THREATS ?= false
VIDEO ?= true

# --- demo-search settings ---------------------------------------------------
AREA ?= -58 -58 58 58
ALTITUDE ?= 15
TIMEOUT ?= 3600

# --- MCP server settings ----------------------------------------------------
MCP_HOST ?= 127.0.0.1
MCP_PORT ?= 8799
MCP_LINK ?= ros

.PHONY: all run run-sim mission demo-search test test-brain test-mission \
        forest cave building brain brain-mock build

.DEFAULT_GOAL := all

# Bare `make` flies the mission demo, not just the stack: `run` starts the MCP server and
# waits for a client, which is correct for an agent but looks like a dead drone at a terminal.
all: demo-search

# Full stack: Gazebo + brain_pilot + the MCP server over HTTP. The brain output is disarmed
# until an MCP goal arrives (arm_on_goal), so this target alone leaves the drone hovering.
# Use `make all`/`make demo-search` for a flight without an MCP client. Ctrl-C, or closing
# Gazebo, stops both halves. The connectome data is vendored in the repository; there is no
# clone step and no sibling checkout.
run: build
	@echo "MCP HTTP on http://$(MCP_HOST):$(MCP_PORT)  (Ctrl-C stops the stack)"
	@$(RUN) 'set -e; $(LOAD_ENV) $(SETUP); set +e; \
	  ros2 launch fly_bringup brain_launch.py gazebo_simulation:=$(WORLD) backend:=$(BACKEND) headless:=$(HEADLESS) threats:=$(THREATS) video_recording:=$(VIDEO) & launch_pid=$$!; \
	  MISSION_LINK=ros python mcp-server/mcp_server.py --http $(MCP_PORT) --host $(MCP_HOST) & mcp_pid=$$!; \
	  cleanup() { kill -INT $$launch_pid $$mcp_pid 2>/dev/null || true; }; \
	  trap cleanup INT TERM; \
	  while kill -0 $$launch_pid 2>/dev/null && kill -0 $$mcp_pid 2>/dev/null; do sleep 1; done; \
	  trap - INT TERM; \
	  if kill -0 $$launch_pid 2>/dev/null; then \
	    kill -INT $$launch_pid 2>/dev/null || true; \
	    wait $$mcp_pid; status=$$?; \
	    wait $$launch_pid 2>/dev/null || true; \
	  else \
	    kill -INT $$mcp_pid 2>/dev/null || true; \
	    wait $$launch_pid; status=$$?; \
	    wait $$mcp_pid 2>/dev/null || true; \
	  fi; \
	  exit $$status'

# The same mission layer, with the brain in-process over the flat kinematic world: no Gazebo,
# no ROS, no drone_controller. Useful for exercising the MCP tools and mission flow.
run-sim:
	@echo "MCP HTTP on http://$(MCP_HOST):$(MCP_PORT)  (in-process sim; Ctrl-C stops)"
	@$(RUN) '$(LOAD_ENV) MISSION_LINK=sim python mcp-server/mcp_server.py --http $(MCP_PORT) --host $(MCP_HOST)'

# The MCP server alone, for a client that expects to connect to an already-running stack.
# Set MCP_LINK=sim for the in-process world.
mission:
	@echo "MCP HTTP on http://$(MCP_HOST):$(MCP_PORT)"
	@$(RUN) '$(LOAD_ENV) MISSION_LINK=$(MCP_LINK) python mcp-server/mcp_server.py --http $(MCP_PORT) --host $(MCP_HOST)'

# The autonomous demo without an MCP client. Stops the launch when the mission ends (or
# when it is interrupted), so the target returns the mission's exit status. Uses BACKEND
# (connectome by default; the data is vendored).
demo-search: build
	@$(RUN) 'set -e; $(LOAD_ENV) $(SETUP); set +e; \
	  ros2 launch fly_bringup brain_launch.py gazebo_simulation:=$(WORLD) backend:=$(BACKEND) headless:=$(HEADLESS) video_recording:=$(VIDEO) & launch_pid=$$!; \
	  kill_tree() { \
	    for child in $$(pgrep -P "$$2" 2>/dev/null); do kill_tree "$$1" "$$child"; done; \
	    kill "-$$1" "$$2" 2>/dev/null || true; \
	  }; \
	  stop_launch() { \
	    kill -INT "$$launch_pid" 2>/dev/null || true; \
	    for _ in 1 2 3 4 5 6 7 8 9 10; do \
	      kill -0 "$$launch_pid" 2>/dev/null || return 0; \
	      sleep 1; \
	    done; \
	    kill_tree TERM "$$launch_pid"; \
	    sleep 2; \
	    kill_tree KILL "$$launch_pid"; \
	    wait "$$launch_pid" 2>/dev/null || true; \
	  }; \
	  trap "stop_launch; exit 130" INT TERM; \
	  MISSION_LINK=ros python tools/search_test.py --area $(AREA) --altitude $(ALTITUDE) --timeout $(TIMEOUT); status=$$?; \
	  trap - INT TERM; \
	  stop_launch; \
	  exit $$status'

# Offline unit/component tests, then the end-to-end in-process mission suite.
test: test-brain test-mission

test-brain:
	@$(RUN) 'PYTHONPATH=src/fly_brain:src/fly_drone python -m pytest src/fly_brain/test/test_perception.py src/fly_brain/test/test_encoders.py src/fly_brain/test/test_connectome.py src/fly_drone/test/test_lidar.py src/fly_drone/test/test_threats.py -q'

test-mission:
	@$(RUN) 'python -m pytest mcp-server/tests -q'

forest: build
	$(RUN) '$(SETUP) && ros2 launch fly_bringup sim_launch.py gazebo_simulation:=forest_search'

cave: build
	$(RUN) '$(SETUP) && ros2 launch fly_bringup sim_launch.py gazebo_simulation:=cave_rescue'

building: build
	$(RUN) '$(SETUP) && ros2 launch fly_bringup sim_launch.py gazebo_simulation:=collapsed_building'

# Gazebo plus the fly brain flying the X3 (mock brain, output disarmed until a goal arrives)
brain: build
	$(RUN) '$(LOAD_ENV) $(SETUP) && ros2 launch fly_bringup brain_launch.py gazebo_simulation:=forest_search'

# ...with the vendored connectome and spheres thrown at the drone every 20 s.
brain-mock: build
	$(RUN) '$(LOAD_ENV) $(SETUP) && ros2 launch fly_bringup brain_launch.py gazebo_simulation:=forest_search backend:=mock threats:=true'

build:
	pixi run colcon build --symlink-install
