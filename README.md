# Focal

Focal is a desktop application that mirrors a selected window in an
always-on-top, resizable overlay. It is the foundation for the window mirror
and tracking project. It runs on Windows (Graphics Capture) and on Linux
(Wayland via the XDG ScreenCast portal).

## Features

### Phase 1 — Mirror + Pin

- Enumerates visible, titled top-level Windows windows.
- Captures the selected window with Windows Graphics Capture.
- Displays live frames in a resizable Qt window.
- Supports moving and resizing using the normal Windows frame.
- Toggles always-on-top behavior with **Pin**.
- Adjusts overlay opacity.
- Refreshes the available source-window list while the app is running.

### Phase 2 — Manual Crop

- Click **Select crop** to enter crop-selection mode.
- Drag over the full source preview to select a focused region.
- Displays the selected region scaled to fill the overlay preview.
- Stores the region as normalized coordinates, so it remains valid when the
  source window changes resolution.
- Click **Reset crop** to return to the full source window.

### Phase 3 — Content Follow

- After selecting a crop, enable **Follow content** to keep its visual content
  centered as it moves in the source window.
- Uses OpenCV pyramidal optical flow; no model download or additional package
  is required.
- Applies a 15% dead zone and smoothing to avoid jitter.
- Tracking pauses safely when the selected region has too few visual features.

### Phase 4 — Person Follow

- Select a crop around the intended subject, then enable **Follow person**.
- Uses the lightweight YOLO11n person detector, including people facing away
  from the camera.
- Runs inference in a separate worker and keeps only the latest source frame,
  so detection cannot backlog or block the mirror UI.
- When multiple people are visible, it chooses the detection closest to the
  selected crop initially, then the detection closest to the last tracked
  person. Enable **Lock person** to associate future detections with that
  person; if they are obscured or leave the frame, the crop holds still rather
  than switching to someone else.
- The `ultralytics` dependency is installed with `uv sync`. YOLO model weights
  are downloaded from Ultralytics on the first use of **Follow person**.

### Split Mirror Workspace

- Focal uses one workspace window with one shared control bar and up to eight
  independently configured mirror tiles.
- Every tile has a title bar showing its mirrored source (or **Empty tile**).
  The active tile uses an accent border and title bar; the source selector,
  crop controls, and follow-mode controls in the top bar apply only to it.
- Click **Add tile** to create an empty placeholder, then select its source.
  **Remove tile** stops and removes the selected tile (at least one tile is
  always retained).
- Use the **Layout** chooser for a row, column, top-pair-and-bottom, or
  two-column grid. Drag splitter handles to set each region's size.
- For a custom nested arrangement, use **Add left**, **Add right**, **Add
  above**, or **Add below** for the active tile. Each button creates an empty
  sibling on that side. The right-click **Split horizontally** and **Split
  vertically** actions remain available.
- Drag a tile preview onto another tile. The highlighted center zone swaps
  their positions; a highlighted left, right, top, or bottom zone splits the
  target and moves the dragged tile to that side. These moves retain each
  tile's capture, crop, and tracking state.
- Pin and opacity apply to the workspace window; each tile has independent
  capture, crop, content/person tracking, and person-lock settings.
- Use **Flip H** and **Flip V** to mirror the active tile horizontally or
  vertically. These display settings are saved per source window.
- Press **F11** to toggle the workspace window between normal and fullscreen
  mode; fullscreen hides the control and status bars. Double-click a tile
  preview to enlarge it to the entire workspace, then double-click it again to
  restore the previous tile layout.

### Phase 5 — Polish

- Saves a preset per source application: crop, follow mode, and overlay
  position/size.
- Uses the source executable as the preset key when available, falling back to
  the window title if Windows denies process metadata access.
- Restores a saved overlay to an available monitor. If a monitor was removed,
  the overlay is placed safely on the primary display instead of opening
  off-screen.
- Retains the existing Pin and opacity controls. A full-window click-through
  mode is intentionally deferred until it has a reliable global shortcut or
  tray control to restore input safely.

Presets are stored at `%APPDATA%\Focal\presets.json` on Windows and
`$XDG_CONFIG_HOME/Focal/presets.json` (defaulting to `~/.config/Focal/presets.json`)
on Linux.

The capture backend is isolated in `src/focal/capture.py`, normalized crop
state is isolated in `src/focal/crop.py`, optical-flow tracking is isolated in
`src/focal/tracking.py`, person detection is isolated in
`src/focal/person.py`, and persisted presets are isolated in
`src/focal/presets.py`.

## Requirements

- Windows 10 version 1903 or later, **or** a modern Linux desktop.
- Python 3.10+, managed through [uv](https://docs.astral.sh/uv/).

### Linux requirements

Focal captures windows through the [XDG ScreenCast portal](https://flatpak.github.io/xdg-desktop-portal/), so it works on any
desktop that provides one — KDE Plasma, GNOME, Gamescope, and others. You need:

- A **Wayland** session with [PipeWire](https://pipewire.org/) running.
- An active `xdg-desktop-portal` with a desktop backend (for example
  `xdg-desktop-portal-kde`). Most distributions install these by default.

On a Wayland session there is no global window list; the source selector opens
the desktop's native picker and the window you choose is streamed to Focal.

### Linux performance

Each mirror tile processes its own PipeWire frames. The **Preview quality**
control is per tile and defaults to **Balanced** (30 FPS, capped at 960×540) to
limit CPU use. Choose **Low power** (20 FPS, 640×360) for several tiles or a
battery-conscious setup, or **High** (45 FPS, 1280×720) when image quality is
more important than CPU use. The setting takes effect for the active mirror
without asking the portal to share the window again.

## Setup and run with uv

Install uv if it is not already available, then from the project directory:

### Windows

```powershell
uv python install 3.12
uv sync
uv run focal
```

### Linux

```sh
uv sync
uv run focal
```

`uv sync` installs CPU-only PyTorch by default, so Focal installs on any
distro and CPU architecture. If the machine has an NVIDIA GPU and you want CUDA
acceleration for **Follow person**, reselect the backend with uv's built-in
support:

```sh
uv sync --torch-backend=auto
# or, non-destructively, export UV_TORCH_BACKEND=auto before future syncs
```

uv probes the installed GPU driver and picks the matching PyTorch index
(see the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)).
The lockfile keeps the CPU wheels as the portable default.

The equivalent module invocation is:

```powershell
uv run python -m focal
```

Run the test suite with:

```powershell
uv run pytest
```

`uv sync` creates or updates `.venv` and `uv.lock`. Commit `uv.lock` so all
developers use the same resolved dependency versions.

The source window must remain open and visible in the Windows window list.
Windows Graphics Capture can capture a window independently of whether it is
currently in front of other windows, subject to Windows and application-level
capture restrictions. On Linux the screen cast is granted per selection by the
desktop portal; closing the picked source window stops that tile's capture.

## Project layout

```text
src/focal/
  capture.py   capture workers: Windows Graphics Capture and Linux ScreenCast portal
  crop.py      normalized crop-region model
  windows.py   Win32 window enumeration + Linux portal source selection
  ui.py        PySide6 mirror overlay and crop interaction
tests/
  test_crop.py
  test_windows.py
```
