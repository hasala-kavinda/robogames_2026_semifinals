# final_task - RoboGames 2026 Final Round Drone Stack

This folder is a clean, self-contained implementation for the final round.
All runtime paths are relative so this directory can be moved into a new repository without code changes.

## What is included

- `src/`: modular onboard autonomy stack
- `config/`: tunable JSON parameters for field testing
- `scripts/`: startup and diagnostics utilities
- `requirements.txt`: Python dependencies
- `.gitignore`: Python and Raspberry Pi development ignore rules

## Final Round Differences from Semi-final

- AprilTags are now used directly along the path and for final landing alignment.
- Landing logic now fuses AprilTag centering with yellow-line assist, instead of treating them as separate phases.
- Yellow path detection includes adaptive filtering to better handle real lighting variation, camera noise, and motion blur.

## Target Platform

- Raspberry Pi 5
- Alpine Linux
- MAVLink endpoint on UDP port `14550`
- Camera stream on TCP port `8080`

## 1. Connect over SSH

From your host machine:

```bash
ssh root@localhost -p 2222
```

After login, move to the project directory on the Pi:

```bash
cd /path/to/final_task
```

## 2. Install system packages (Alpine `apk`)

Install Python, pip, and common build tools needed for native wheels:

```sh
apk add --no-cache \
  python3 \
  py3-pip \
  python3-dev \
  gcc \
  g++ \
  make \
  musl-dev \
  linux-headers
```

## 3. Create and activate virtual environment

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 4. Configure mission parameters for real hardware

Default config lives at `config/defaults.json`.

For real hardware tuning:

1. Copy `config/hardware.example.json` to `config/local.json`
2. Adjust camera, HSV thresholds, PID gains, and landing tolerances
3. Run mission with the override file

Example:

```sh
cp config/hardware.example.json config/local.json
python3 -m src.mission --config config/local.json --debug
```

## 5. Start onboard mission

### Option A: Startup script (recommended)

```sh
sh scripts/start.sh config/defaults.json
```

With target tag override:

```sh
TARGET_TAG_ID=125 sh scripts/start.sh config/local.json
```

### Option B: Run mission directly

```sh
python3 -m src.mission --config config/defaults.json
```

## 6. Hardware test flow (SSH session)

Run these checks in order before live flight:

1. Validate config and camera socket reachability:

```sh
python3 scripts/diagnostics.py
```

2. Confirm MAVLink heartbeat and telemetry access:

```sh
python3 scripts/test_mavlink.py
```

3. Preview camera and detection overlays (`q` to quit):

```sh
python3 scripts/camera_preview.py
```

4. Run mission with conservative speeds first:

```sh
python3 -m src.mission --config config/local.json --debug
```

## Module overview

- `src/mavlink_client.py`: connection to MAVLink, arm/takeoff/velocity/land commands
- `src/camera_stream.py`: low-latency TCP frame ingestion from port 8080
- `src/apriltag_detector.py`: AprilTag detection tuned for ground path
- `src/line_follower.py`: adaptive yellow-path detection and tracking error output
- `src/landing_controller.py`: landing command fusion (tag alignment + line assist)
- `src/mission.py`: mission state machine integrating all components

## Performance notes for Raspberry Pi 5 + Alpine

- Frame buffering is intentionally small to reduce control lag.
- Vision operations use simple morphology and contour logic to limit CPU usage.
- Avoid running extra GUI tools during flight unless needed for diagnostics.
- Keep camera resolution moderate (for example 640x480) if CPU load is high.

## Portability notes

- No hardcoded absolute paths are used.
- Config and script paths are relative to project root.
- This folder can be copied into a new repository and run as-is after dependency installation.
