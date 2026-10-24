from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True, slots=True)
class PreviewSettings:
    """Per-tile frame-rate and size cap applied before Qt renders a frame."""

    label: str
    fps: int
    max_width: int
    max_height: int


HIGH_QUALITY_PREVIEW = PreviewSettings("High: 45 FPS · 1280×720", 45, 1280, 720)
BALANCED_PREVIEW = PreviewSettings("Balanced: 30 FPS · 960×540", 30, 960, 540)
LOW_POWER_PREVIEW = PreviewSettings("Low power: 20 FPS · 640×360", 20, 640, 360)
PREVIEW_QUALITIES = (HIGH_QUALITY_PREVIEW, BALANCED_PREVIEW, LOW_POWER_PREVIEW)
DEFAULT_PREVIEW = BALANCED_PREVIEW

# Backwards-compatible defaults for callers of _copy_frame_pixels().
MAX_PREVIEW_WIDTH = DEFAULT_PREVIEW.max_width
MAX_PREVIEW_HEIGHT = DEFAULT_PREVIEW.max_height
MAX_PREVIEW_FPS = DEFAULT_PREVIEW.fps


def _frame_buffer(frame: Any) -> Any:
    """Return pixels from either a capture wrapper or a raw PipeWire array."""
    if isinstance(frame, np.ndarray):
        return frame

    frame_buffer = getattr(frame, "frame_buffer", None)
    if frame_buffer is not None:
        return frame_buffer

    # Keep compatibility with older Windows capture releases that exposed
    # to_numpy(). PipeWire frames are handled as raw NumPy arrays above.
    to_numpy = getattr(frame, "to_numpy", None)
    if to_numpy is None:
        raise AttributeError("Capture frame has no pixel buffer")
    return to_numpy(copy=False)


def _copy_frame_pixels(
    frame: Any,
    max_width: int = MAX_PREVIEW_WIDTH,
    max_height: int = MAX_PREVIEW_HEIGHT,
) -> np.ndarray:
    """Copy and bound a frame to a size suitable for the preview UI."""
    frame_buffer = _frame_buffer(frame)
    height, width = frame_buffer.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return np.array(frame_buffer, copy=True)

    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    try:
        import cv2

        return np.ascontiguousarray(
            cv2.resize(
                frame_buffer,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    except ImportError:
        # This is only a fallback; the capture backends normally install OpenCV.
        y_indices = np.linspace(0, height - 1, target_height, dtype=int)
        x_indices = np.linspace(0, width - 1, target_width, dtype=int)
        return np.array(frame_buffer[y_indices[:, None], x_indices], copy=True)


class CaptureThread(QThread):
    """Base capture worker that forwards at most one owned frame to Qt.

    Subclasses implement the platform backend in ``run()`` and call
    ``_deliver`` for each raw frame. Dropping frames while the previous one is
    still queued for the GUI keeps the overlay responsive on fast sources.
    """

    frame_ready = Signal(object)
    capture_error = Signal(str)
    source_closed = Signal()

    def __init__(
        self,
        parent: Any = None,
        preview_settings: PreviewSettings = DEFAULT_PREVIEW,
    ) -> None:
        super().__init__(parent)
        self._stop_requested = threading.Event()
        self._frame_pending = threading.Event()
        self._last_frame_time = 0.0
        self._preview_settings = preview_settings

    def _deliver(self, frame: Any) -> bool:
        """Rate-limit, drop while pending, and forward an owned pixel copy."""
        if self._stop_requested.is_set():
            return False
        settings = self._preview_settings
        now = time.monotonic()
        if now - self._last_frame_time < 1 / settings.fps:
            return True
        if self._frame_pending.is_set():
            return True
        self._last_frame_time = now
        self._frame_pending.set()
        try:
            pixels = _copy_frame_pixels(frame, settings.max_width, settings.max_height)
            self.frame_ready.emit(pixels)
        except Exception:
            self._frame_pending.clear()
            raise
        return True

    def frame_consumed(self) -> None:
        """Allow the capture callback to deliver the next latest frame."""
        self._frame_pending.clear()

    def set_preview_settings(self, settings: PreviewSettings) -> None:
        """Apply a new cap without interrupting an active portal session."""
        self._preview_settings = settings
        self._last_frame_time = 0.0

    def stop(self) -> None:
        """Request shutdown without blocking the GUI thread."""
        self._stop_requested.set()

    def run(self) -> None:  # pragma: no cover -- implemented by backends
        raise NotImplementedError


class WindowsCaptureThread(CaptureThread):
    """Capture one Windows window with Windows Graphics Capture."""

    def __init__(
        self,
        window_hwnd: int,
        parent: Any = None,
        preview_settings: PreviewSettings = DEFAULT_PREVIEW,
    ) -> None:
        super().__init__(parent, preview_settings)
        self.window_hwnd = window_hwnd
        self._capture_control: Any = None

    def run(self) -> None:
        try:
            from windows_capture import WindowsCapture  # pyright: ignore[reportMissingImports]

            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=False,
                # A handle identifies exactly the entry selected in the UI;
                # window_name is only a substring match and may select another app.
                window_hwnd=self.window_hwnd,
                # Limit callbacks in the native capture layer too, so a fast
                # source cannot keep acquiring the Python GIL between frames.
                minimum_update_interval=round(1000 / self._preview_settings.fps),
            )

            @capture.event
            def on_frame_arrived(frame: Any, capture_control: Any) -> None:
                if self._stop_requested.is_set():
                    capture_control.stop()
                    return
                self._deliver(frame)

            @capture.event
            def on_closed() -> None:
                self._stop_requested.set()
                self.source_closed.emit()

            # Do not call CaptureControl.wait() here. Its native binding holds
            # a borrow for the duration of the wait, making a concurrent stop()
            # fail with "Already borrowed". The event keeps this QThread and
            # the WindowsCapture instance alive without borrowing the control.
            self._capture_control = capture.start_free_threaded()
            if self._stop_requested.is_set():
                self._capture_control.stop()
            self._stop_requested.wait()
        except Exception as exc:  # surface backend/runtime errors in the UI
            if not self._stop_requested.is_set():
                self.capture_error.emit(str(exc))
        finally:
            self._capture_control = None

    def stop(self) -> None:
        """Request shutdown without blocking the GUI thread."""
        super().stop()
        if self._capture_control is not None:
            self._capture_control.stop()


class PortalCaptureThread(CaptureThread):
    """Select a window with the XDG ScreenCast portal and stream its frames.

    Linux window capture is gated behind the desktop portal: a Wayland
    compositor only grants pixel access to windows the user explicitly shares.
    This worker shows the system picker (blocking, GIL released), then consumes
    the PipeWire stream in the same thread.
    """

    selection_cancelled = Signal()

    def __init__(
        self,
        parent: Any = None,
        preview_settings: PreviewSettings = DEFAULT_PREVIEW,
    ) -> None:
        super().__init__(parent, preview_settings)
        self._stream: Any = None
        self._session: Any = None

    def run(self) -> None:
        try:
            from pipewire_capture import CaptureStream, PortalCapture, is_available

            if not is_available():
                self.capture_error.emit(
                    "Window capture needs the XDG ScreenCast portal. Run a "
                    "Wayland session with a portal provider such as "
                    "xdg-desktop-portal-kde or xdg-desktop-portal-gnome."
                )
                return

            portal = PortalCapture()
            session = portal.select_window()
            if self._stop_requested.is_set():
                self._session = session
                return
            if session is None:
                self.selection_cancelled.emit()
                return
            self._session = session
            stream = CaptureStream(
                session.fd,
                session.node_id,
                session.width,
                session.height,
                capture_interval=1.0 / self._preview_settings.fps,
            )
            self._stream = stream
            stream.start()

            # ``CaptureStream.get_frame()`` returns a newly allocated NumPy
            # copy of its latest frame every time it is called. It does not
            # consume that frame, so polling it in a tight loop duplicates a
            # full-resolution buffer hundreds of times per second even though
            # _deliver later drops most of those copies. Pace retrieval before
            # calling into the binding, and do not allocate another frame while
            # one is already queued for Qt.
            next_frame_at = 0.0
            while not self._stop_requested.is_set():
                if stream.window_invalid:
                    self.source_closed.emit()
                    return
                settings = self._preview_settings
                if self._frame_pending.is_set():
                    time.sleep(0.005)
                    continue
                now = time.monotonic()
                if now < next_frame_at:
                    time.sleep(min(0.005, next_frame_at - now))
                    continue
                frame = stream.get_frame()
                if frame is None:
                    # The portal stream has not produced its first frame yet.
                    time.sleep(0.005)
                    continue
                if not self._deliver(frame):
                    return
                next_frame_at = time.monotonic() + 1 / settings.fps
        except Exception as exc:  # surface backend/runtime errors in the UI
            if not self._stop_requested.is_set():
                self.capture_error.emit(str(exc))
        finally:
            self._close_stream()

    def _close_stream(self) -> None:
        stream, session = self._stream, self._session
        self._stream = self._session = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        if session is not None:
            try:
                session.close()
            except Exception:
                pass