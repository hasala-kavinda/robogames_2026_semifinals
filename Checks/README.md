# Checks Camera Preview (Red Detection)

This folder is for quick camera checking with red-color detection.

## Requirements

Create and use a virtual environment (from repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-contrib-python
```

After this, run all commands with the venv active.

## Run

From repository root:

```bash
python Checks/camera_preview.py
```

- Opens two windows:
  - `camera_preview`: live image with detection box and center
  - `color_mask`: binary mask for detected red color
- Press `q` to quit.

## Use TCP camera stream (if needed)

```bash
python Checks/camera_preview.py --source tcp --host 127.0.0.1 --port 8080
```

## Tune red detection

Default is red hue range `0..10`.
If your lighting causes weak detection, try high-red hue range:

```bash
python Checks/camera_preview.py --h-min 170 --h-max 179 --s-min 120 --v-min 70
```

You can tune any HSV values directly with command-line flags.
