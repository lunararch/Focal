# Window Mirror & Tracking Overlay — Project Plan

A desktop application that mirrors a chosen window in an always-on-top,
resizable frame (like OnTopReplica), with the ability to focus on a
sub-region of that window and have the frame automatically follow a
person or the tracked content as it moves.

---

## 1. Goals

- Select any open window and mirror its live content in a separate overlay.
- Overlay window is always-on-top, freely resizable and movable.
- Ability to define a crop region within the source window (focus on a
  sub-area rather than the whole window).
- Two auto-follow modes for that crop region:
  - **Content-follow**: keep whatever was originally inside the selected
    region centered, even as it shifts within the source window.
  - **Person-follow**: detect a person (full body, not just face) and
    keep them centered in the frame.
- Smooth, non-jittery tracking behavior (no frame-to-frame snapping).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────┐
│  Target Window (any app on the OS)           │
└──────────────────┬────────────────────────────┘
                    │ capture API
┌───────────────────▼────────────────────────┐
│  Capture Module                              │
│  - grabs frames at N fps from chosen window  │
└──────────────────┬────────────────────────────┘
                    │ raw frame (numpy / bitmap)
┌───────────────────▼────────────────────────┐
│  Tracking Module (optional, toggleable)      │
│  - detects person / subject bounding box     │
│  - smooths + computes ideal crop rect        │
└──────────────────┬────────────────────────────┘
                    │ cropped region
┌───────────────────▼────────────────────────┐
│  Overlay Renderer                            │
│  - always-on-top window                      │
│  - handles resize / drag / pin / opacity     │
└───────────────────────────────────────────────┘
```

Three independent subsystems, each solvable with a different piece of tech:

1. **Window capture** — grab pixels from another app's window.
2. **Overlay/display** — always-on-top, resizable frame showing the capture.
3. **Auto-tracking crop** — intelligently re-center the frame on a subject.

---

## 3. Subsystem Details

### 3.1 Window Capture

Two realistic platform-level approaches:

**Windows Graphics Capture API (WGC)**
- Modern, GPU-accelerated, per-window (not just per-monitor) capture.
- This is the API modern screen-mirroring tools use instead of the older
  Desktop Duplication API.
- Accessible via:
  - C#/.NET: `Windows.Graphics.Capture` namespace (most mature path).
  - Python: `windows-capture` package, or `dxcam` (note: `dxcam` is
    monitor-based, so per-window capture needs manual cropping to window
    bounds).

**Cross-platform Python fallback**
- `mss` for fast region/screen capture.
- `pygetwindow` or `pywin32` to get the target window's on-screen bounds,
  then crop the region out of a full-screen grab.
- Simpler to build, but not true per-window capture — if another window
  overlaps the target, you'll capture that overlap too.

**Recommendation**: since tracking requires Python + OpenCV anyway, use
`windows-capture` (or `dxcam` if window-specific capture isn't critical)
so the entire pipeline — capture, tracking, rendering — lives in one
Python codebase instead of splitting capture (C#) from tracking (Python).

Alternative path: fork or reference **OnTopReplica's open-source code**
for a working capture + overlay layer on day one, then bolt on tracking
via ONNX Runtime or a local Python process it talks to. More initial
setup, but a proven capture layer.

### 3.2 Always-On-Top Resizable Overlay

Straightforward UI engineering:

- **PyQt6 / PySide6**
  - `Qt.WindowStaysOnTopHint` for always-on-top behavior.
  - Frameless window; Qt doesn't give free resize handles on frameless
    windows, so implement mouse-drag-on-edge/corner resize logic manually
    (a few dozen lines — track mouse position relative to window edges,
    resize accordingly).
  - Render captured frames into a `QLabel` or `QOpenGLWidget`, scaling
    the current crop to fill the overlay window.

- **Controls to include**
  - Pin / unpin (toggle always-on-top).
  - Opacity slider.
  - "Select window" mode (choose which app to mirror).
  - "Select region" mode (drag a rectangle on the preview to define the
    focused sub-area).
  - Toggle for tracking mode: off / content-follow / person-follow.

- **Region selection UX**
  - User drags directly on the mirrored preview to define the crop
    rectangle.
  - Store the rectangle in normalized coordinates (0.0–1.0 relative to
    source window size) so it stays valid if the source window is
    resized.

### 3.3 Auto-Tracking Crop

Two distinct problems: **detection** and **smoothing/follow logic**.

#### Detection — two modes

**Person-follow mode**
- Run a person detector on each captured frame.
- **YOLOv8n / YOLO11n** via the `ultralytics` package: real-time,
  full-body bounding boxes, runs acceptably on CPU at modest resolutions,
  faster with GPU.
- **MediaPipe Pose** as an alternative if skeleton landmarks are useful
  later (e.g., following torso center specifically) — more accurate per
  person but slower and less robust with multiple people in frame.
- If multiple people can appear, add a lightweight tracker
  (**ByteTrack**, or OpenCV's built-in `CSRT` / `KCF`) so the frame
  doesn't jump between people — detect once, track continuously,
  re-detect periodically to correct drift.

**Content-follow mode**
- Not "find a person" but "keep whatever was inside this rectangle
  centered, even as it moves." This is template/feature tracking, not
  object detection.
- OpenCV trackers: `CSRT` (more accurate, slower) or `KCF` / `MOSSE`
  (faster, less accurate) initialized on the user-selected region.
- Alternative: optical flow (`cv2.calcOpticalFlowPyrLK`) on feature
  points inside the region, tracking the centroid's drift over time.

#### Follow / smoothing logic

This is what makes tracking feel natural instead of janky:

- Compute the target crop center from the bounding box (or pose)
  centroid.
- Apply exponential smoothing rather than snapping instantly:
  ```
  new_pos = old_pos + alpha * (target_pos - old_pos)
  ```
  so the frame eases toward the subject.
- Add a **dead zone**: only move the crop once the subject drifts past
  an inner threshold (e.g. 15% from center) — mirrors how camera
  operators avoid constant micro-adjustments.
- Clamp the crop rectangle to the source window's bounds so tracking
  never tries to follow past the edge.
- Handle subject loss gracefully: if detection fails for N consecutive
  frames, either hold the last known position, zoom out to reacquire,
  or fall back to the static manually-selected crop.

---

## 4. Suggested Tech Stack

| Layer | Technology |
|---|---|
| Window capture | `windows-capture` or `dxcam` (Python) |
| Computer vision / tracking | OpenCV + `ultralytics` (YOLO) |
| UI / overlay | PySide6 (Qt for Python) |
| Packaging | PyInstaller for a distributable `.exe` |

Rationale: a single Python codebase covers capture, tracking, and UI.
No server component is needed — a FastAPI layer would be overkill unless
a plugin/extensibility architecture is planned for later.

---

## 5. Phased Build Plan

**Phase 1 — Mirror + Pin**
- Enumerate open windows.
- Capture the chosen window.
- Display it in an always-on-top, frameless, resizable Qt window.
- No cropping yet — this alone replicates core OnTopReplica functionality.

**Phase 2 — Manual Crop**
- Let the user drag a rectangle on the preview to define a sub-region.
- Render only that region, scaled to fill the overlay window.

**Phase 3 — Static-Region "Smart" Follow**
- Add optical-flow / CSRT tracking of the manually selected region's
  content.
- Apply smoothing + dead zone so the crop follows content drift.

**Phase 4 — Person Mode**
- Integrate YOLO person detection as an alternate follow target.
- Add a UI toggle between "follow content" and "follow person."

**Phase 5 — Polish**
- Multi-monitor handling.
- Performance tuning: decouple capture fps from detection fps (e.g. run
  YOLO every 5th frame and interpolate/track between detections rather
  than running it on every frame).
- Click-through / opacity options.
- Save per-source-app presets (crop region, tracking mode, size/position).

---

## 6. Open Design Questions

- **Continuous vs. on-demand tracking**: should detection run in the
  background at all times, or only when explicitly triggered? This
  determines whether a throttled background detection thread is needed
  versus a simple call inside the main render loop.
- **Multi-person handling**: if more than one person appears in
  person-follow mode, what's the selection rule — largest bounding box,
  most centered, first detected, or manual click-to-select?
- **Behavior on window resize/move**: how should the overlay react if
  the source window is resized, minimized, or moved — pause tracking,
  rescale the stored crop, or reset to full-window view?
- **Performance ceiling**: target resolution and fps for both capture
  and detection, especially on CPU-only setups, to keep the app
  responsive without requiring a dedicated GPU.

---

## 7. Next Step

Scaffold Phase 1: window enumeration, window capture, and an
always-on-top resizable Qt overlay — the minimum working "mirror" before
adding any cropping or tracking logic.
