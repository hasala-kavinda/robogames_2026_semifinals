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

- Core mission behavior is kept equivalent to the semi-final: AprilTag number decoding, country-based selection, and multi-airport progression.
- The final-round change is that AprilTags are now on the same ground path as line following, so perception and landing timing are tuned for this placement.
- Yellow path detection includes adaptive filtering to better handle real lighting variation, camera noise, and motion blur.

AprilTag mission semantics:

- Tag ID is decoded as `country-status-reachable` (for example `125` means country `1`, safe status `1`, reachable count `5`).
- Landing decisions are based on decoded metadata plus the requested country sequence in `mission.requested_countries`.

## Target Platform

- Raspberry Pi 5
- Alpine Linux
- MAVLink endpoint on UDP port `14550`
- Camera stream on TCP port `9000` (raw RGB frames)

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

For SSH/headless operation on Raspberry Pi, this project uses
`opencv-contrib-python-headless` by default, so GUI/X11 libraries are not
required for mission runtime.

If you previously installed GUI OpenCV wheels, clean them first to avoid mixed
wheel conflicts such as `ImportError: libxcb.so.1`:

```sh
python3 -m pip uninstall -y \
  opencv-python \
  opencv-contrib-python \
  opencv-python-headless \
  opencv-contrib-python-headless
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

If `config/local.json` is missing, mission startup now fails with a hint that
includes the exact `cp` command above.

If evaluators require a different order, set `mission.requested_countries` in the config override.
Example: `[2, 1]` means complete country `2` first, then country `1`.

## 5. Start onboard mission

### Option A: Startup script (recommended)

```sh
sh scripts/start.sh config/defaults.json
```

With country order override:

```sh
COUNTRIES=1,2 sh scripts/start.sh config/local.json
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

1. Confirm MAVLink heartbeat and telemetry access:

```sh
python3 scripts/test_mavlink.py
```

1. Preview camera and detection overlays (`q` to quit):

```sh
python3 scripts/camera_preview.py
```

Note: preview is optional and requires a GUI session. For pure SSH mission
execution, skip this step.

1. Run mission with conservative speeds first:

```sh
python3 -m src.mission --config config/local.json --debug
```

## Module overview

- `src/mavlink_client.py`: connection to MAVLink, arm/takeoff/velocity/land commands
- `src/camera_stream.py`: low-latency TCP frame ingestion from port 9000
- `src/apriltag_detector.py`: AprilTag detection tuned for ground path
- `src/line_follower.py`: adaptive yellow-path detection and tracking error output
- `src/landing_controller.py`: landing command fusion (tag alignment + line assist)
- `src/mission.py`: mission state machine integrating all components

## Performance notes for Raspberry Pi 5 + Alpine

- Frame buffering is intentionally small to reduce control lag.
- Vision operations use simple morphology and contour logic to limit CPU usage.
- Avoid running extra GUI tools during flight unless needed for diagnostics.
- Keep camera resolution moderate (for example 640x480) if CPU load is high.

## Camera stream expectations and troubleshooting

- The mission TCP camera client expects raw RGB frames at 640x480 by default.
- If dimensions differ from the configured expected size, that frame is dropped
  to prevent OpenCV crashes from malformed payloads.
- If you intentionally use a different camera size, override
  `camera.expected_width` and `camera.expected_height` in your local config.

## Portability notes

- No hardcoded absolute paths are used.
- Config and script paths are relative to project root.
- This folder can be copied into a new repository and run as-is after dependency installation.
