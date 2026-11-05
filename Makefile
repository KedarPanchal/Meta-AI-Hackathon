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
# HD_VIDEO records the dedicated HD chase camera (1920x1080 at 60 Hz) to
# dist/<world>_<timestamp>_hd.mp4. It is on by default so a bare `make`
# produces the good video, not just the 320x240 chase-sensor one (which
# still runs as a fallback). Set HD_VIDEO=false if the extra render load
# hurts the real-time factor on a slow GPU. Works headless.
HD_VIDEO ?= true

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
	  (trap - INT; exec ros2 launch fly_bringup brain_launch.py gazebo_simulation:=$(WORLD) backend:=$(BACKEND) headless:=$(HEADLESS) threats:=$(THREATS) video_recording:=$(VIDEO) hd_recording:=$(HD_VIDEO)) & launch_pid=$$!; \
	  (trap - INT; exec env MISSION_LINK=ros python mcp-server/mcp_server.py --http $(MCP_PORT) --host $(MCP_HOST)) & mcp_pid=$$!; \
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
#
# The launch is started through `(trap - INT; exec ...)` for a reason: a plain
# `ros2 launch ... &` in a non-interactive shell inherits SIGINT as *ignored*
# (POSIX async-command rule), so the `kill -INT` in stop_launch below was a
# no-op. The launch never Ctrl-C'd, the video recorder never finalised its
# .partial.mp4, and the tree only died 10 s later on TERM/KILL. Resetting the
# disposition before exec makes the kill a real Ctrl-C: the launch forwards
# SIGINT to every node, record_video drains its queue, closes ffmpeg, and
# renames the partial to dist/<world>_<timestamp>.mp4. stop_launch therefore
# gives that graceful path 30 s (a long recording can need seconds to pad and
# encode its tail) before escalating to TERM/KILL as a last resort.
#
# With HD_VIDEO=true, stop_launch first stops the HD recording over its gz
# service while Gazebo is still rendering, and waits for the _hd.mp4 to
# land. That ordering is load-bearing: Gazebo's CameraVideoRecorder only
# moves its temporary file on a render callback, which no longer happens
# once shutdown begins, so a stop sent during Ctrl-C arrives too late.
demo-search: build
	@$(RUN) 'set -e; $(LOAD_ENV) $(SETUP); set +e; \
	  (trap - INT; exec ros2 launch fly_bringup brain_launch.py gazebo_simulation:=$(WORLD) backend:=$(BACKEND) headless:=$(HEADLESS) video_recording:=$(VIDEO) hd_recording:=$(HD_VIDEO)) & launch_pid=$$!; \
	  hd_marker=$$(mktemp); \
	  kill_tree() { \
	    for child in $$(pgrep -P "$$2" 2>/dev/null); do kill_tree "$$1" "$$child"; done; \
	    kill "-$$1" "$$2" 2>/dev/null || true; \
	  }; \
	  stop_launch() { \
	    if [ "$(HD_VIDEO)" = "true" ]; then \
	      echo "stopping HD recording before shutdown"; \
	      gz service -s /camera_chase_hd/record_video --reqtype gz.msgs.VideoRecord --reptype gz.msgs.Boolean --timeout 20000 --req "stop: true" 2>/dev/null || true; \
	      for _ in $$(seq 1 30); do \
	        find dist -name "$(WORLD)_*_hd.mp4" -newer "$$hd_marker" 2>/dev/null | grep -q . && break; \
	        sleep 1; \
	      done; \
	    fi; \
	    kill -INT "$$launch_pid" 2>/dev/null || true; \
	    for _ in $$(seq 1 30); do \
	      kill -0 "$$launch_pid" 2>/dev/null || return 0; \
	      sleep 1; \
	    done; \
	    echo "launch did not stop on Ctrl-C after 30 s; escalating to TERM/KILL (video may be left partial)"; \
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
	@$(RUN) 'PYTHONPATH=src/fly_brain:src/fly_drone python -m pytest src/fly_brain/test/test_perception.py src/fly_brain/test/test_encoders.py src/fly_brain/test/test_connectome.py src/fly_drone/test/test_lidar.py src/fly_drone/test/test_threats.py src/fly_drone/test/test_node_qos.py -q'

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
