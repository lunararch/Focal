from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QDrag,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .capture import (
    DEFAULT_PREVIEW,
    PREVIEW_QUALITIES,
    CaptureThread,
    PortalCaptureThread,
    PreviewSettings,
    WindowsCaptureThread,
)
from .crop import CropRegion
from .person import PersonDetectionWorker
from .presets import PresetStore, SourcePreset, default_preset_path
from .tracking import ContentFollower
from .windows import (
    PICK_WINDOW,
    STOP_MIRRORING,
    WindowInfo,
    is_portal_session,
    list_windows,
    preset_key,
)
from .workspace import LayoutMode, pane_rows

MAX_TILES = 8
TILE_MIME_TYPE = "application/x-focal-mirror-tile"
SPLIT_DIRECTIONS = ("left", "right", "above", "below")


def _split_details(direction: str) -> tuple[Qt.Orientation, bool]:
    """Return a split orientation and whether the new tile belongs first."""
    if direction == "left":
        return Qt.Orientation.Horizontal, True
    if direction == "right":
        return Qt.Orientation.Horizontal, False
    if direction == "above":
        return Qt.Orientation.Vertical, True
    if direction == "below":
        return Qt.Orientation.Vertical, False
    raise ValueError(f"Unknown split direction: {direction}")


def _order_windows(windows: list[WindowInfo], recent_hwnds: list[int]) -> list[WindowInfo]:
    """Put recently mirrored windows first, newest first."""
    recent_positions = {hwnd: index for index, hwnd in enumerate(recent_hwnds)}
    return sorted(
        windows,
        key=lambda window: (
            recent_positions.get(window.hwnd, len(recent_positions)),
            window.title.casefold(),
        ),
    )


class VideoLabel(QLabel):
    """Preview canvas that displays a frame and interactively selects a crop."""

    crop_selected = Signal(object)
    activated = Signal()
    drag_requested = Signal()
    double_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #111318; color: #aab1c2;")
        self.setText("Select a window to begin mirroring")
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._crop = CropRegion.full()
        self._crop_mode = False
        self._flip_horizontal = False
        self._flip_vertical = False
        self._display_rect = QRectF()
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None
        self._tile_drag_start: QPointF | None = None

    @property
    def crop_region(self) -> CropRegion:
        return self._crop

    @property
    def crop_mode(self) -> bool:
        return self._crop_mode

    def set_frame(self, pixels: np.ndarray) -> None:
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            return
        height, width, channels = pixels.shape
        pixels = np.ascontiguousarray(pixels)
        image_format = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
        pixels = np.ascontiguousarray(pixels[:, :, [2, 1, 0, 3]] if channels == 4 else pixels[:, :, ::-1])
        self._image = QImage(pixels.data, width, height, pixels.strides[0], image_format).copy()
        self._draw_image()

    def set_crop_region(self, crop: CropRegion) -> None:
        self._crop = crop
        self._draw_image()

    def reset_crop(self) -> None:
        self.set_crop_region(CropRegion.full())

    @property
    def flip_horizontal(self) -> bool:
        return self._flip_horizontal

    @property
    def flip_vertical(self) -> bool:
        return self._flip_vertical

    def set_flipped(self, horizontal: bool, vertical: bool) -> None:
        self._flip_horizontal = horizontal
        self._flip_vertical = vertical
        self._draw_image()
        self.update()

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        self._drag_start = self._drag_current = None
        self._draw_image()
        self.update()

    def clear(self) -> None:
        self._image = None
        self._display_rect = QRectF()
        self._drag_start = self._drag_current = None
        super().clear()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._draw_image()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            if self._crop_mode and self._display_rect.contains(event.position()):
                self._drag_start = self._drag_current = event.position()
                self.update()
                return
            if not self._crop_mode:
                self._tile_drag_start = event.position()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._crop_mode:
            self.activated.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            self._drag_current = event.position()
            self.update()
            return
        if self._tile_drag_start is not None:
            if (event.position() - self._tile_drag_start).manhattanLength() >= QApplication.startDragDistance():
                self._tile_drag_start = None
                self.drag_requested.emit()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._tile_drag_start = None
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_current = event.position()
            selection = self._selection_rect()
            self._drag_start = self._drag_current = None
            if selection.width() >= 6 and selection.height() >= 6:
                crop = self._display_to_crop(selection)
                if crop.right - crop.left >= 0.01 and crop.bottom - crop.top >= 0.01:
                    self._crop = crop
                    self.crop_selected.emit(crop)
            self.update()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if not self._crop_mode or self._display_rect.isEmpty():
            return
        selection = self._selection_rect()
        if selection.isEmpty():
            return
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        shade, display = QColor(0, 0, 0, 105), self._display_rect
        painter.fillRect(QRectF(display.left(), display.top(), display.width(), max(0.0, selection.top() - display.top())), shade)
        painter.fillRect(QRectF(display.left(), selection.bottom(), display.width(), max(0.0, display.bottom() - selection.bottom())), shade)
        painter.fillRect(QRectF(display.left(), selection.top(), max(0.0, selection.left() - display.left()), selection.height()), shade)
        painter.fillRect(QRectF(selection.right(), selection.top(), max(0.0, display.right() - selection.right()), selection.height()), shade)
        painter.setPen(QPen(QColor(91, 199, 255), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(selection)
        painter.end()

    def _draw_image(self) -> None:
        if self._image is None:
            self._display_rect = QRectF()
            return
        source = self._image
        if not self._crop_mode and not self._crop.is_full:
            x1, y1, x2, y2 = self._crop.pixel_bounds(source.width(), source.height())
            source = source.copy(x1, y1, x2 - x1, y2 - y1)
        if self._flip_horizontal or self._flip_vertical:
            source = source.mirrored(self._flip_horizontal, self._flip_vertical)
        pixmap = QPixmap.fromImage(source).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._display_rect = QRectF((self.width() - pixmap.width()) / 2, (self.height() - pixmap.height()) / 2, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)
        self.setText("")
        self.update()

    def _selection_rect(self) -> QRectF:
        if self._drag_start is not None and self._drag_current is not None:
            return QRectF(self._drag_start, self._drag_current).normalized().intersected(self._display_rect)
        return self._crop_to_display(self._crop)

    def _crop_to_display(self, crop: CropRegion) -> QRectF:
        left, top, right, bottom = crop.left, crop.top, crop.right, crop.bottom
        if self._flip_horizontal:
            left, right = 1 - right, 1 - left
        if self._flip_vertical:
            top, bottom = 1 - bottom, 1 - top
        display = self._display_rect
        return QRectF(
            display.left() + left * display.width(),
            display.top() + top * display.height(),
            (right - left) * display.width(),
            (bottom - top) * display.height(),
        )

    def _display_to_crop(self, selection: QRectF) -> CropRegion:
        display = self._display_rect
        left = (selection.left() - display.left()) / display.width()
        top = (selection.top() - display.top()) / display.height()
        right = (selection.right() - display.left()) / display.width()
        bottom = (selection.bottom() - display.top()) / display.height()
        if self._flip_horizontal:
            left, right = 1 - right, 1 - left
        if self._flip_vertical:
            top, bottom = 1 - bottom, 1 - top
        return CropRegion.from_bounds(left, top, right, bottom)


class DropZoneOverlay(QWidget):
    """Non-interactive overlay that makes tile drop destinations visible."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._zone: str | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def set_zone(self, zone: str | None) -> None:
        self._zone = zone
        self.setVisible(zone is not None)
        self.update()

    def paintEvent(self, event: Any) -> None:
        if self._zone is None:
            return
        rect = QRectF(self.rect())
        edge_width = rect.width() * 0.25
        edge_height = rect.height() * 0.25
        zones = {
            "left": QRectF(rect.left(), rect.top(), edge_width, rect.height()),
            "right": QRectF(rect.right() - edge_width, rect.top(), edge_width, rect.height()),
            "above": QRectF(rect.left() + edge_width, rect.top(), rect.width() - 2 * edge_width, edge_height),
            "below": QRectF(rect.left() + edge_width, rect.bottom() - edge_height, rect.width() - 2 * edge_width, edge_height),
            "center": QRectF(rect.left() + edge_width, rect.top() + edge_height, rect.width() - 2 * edge_width, rect.height() - 2 * edge_height),
        }
        painter = QPainter(self)
        for zone, zone_rect in zones.items():
            active = zone == self._zone
            painter.setPen(QPen(QColor(91, 199, 255, 230 if active else 120), 2))
            painter.setBrush(QColor(91, 199, 255, 105 if active else 35))
            painter.drawRect(zone_rect)
        painter.end()


class MirrorTile(QWidget):
    """One independent capture, crop, and tracking session in the workspace."""

    selected = Signal(object)
    status_changed = Signal(str)
    swap_requested = Signal(str, str)
    split_requested = Signal(object, object)
    edge_drop_requested = Signal(str, str, str)
    fullscreen_requested = Signal(object)

    def __init__(self, presets: PresetStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tile_id = str(uuid4())
        self._presets = presets
        self._capture: CaptureThread | None = None
        self._person_worker: PersonDetectionWorker | None = None
        self._content_follower = ContentFollower()
        self._preset_key: str | None = None
        self._source_hwnd: int | None = None
        self._source_title = ""
        self._tracking_mode = "none"
        self._person_locked = False
        self._preview_settings = DEFAULT_PREVIEW
        self._drop_zone: str | None = None
        self._title = QLabel("Empty tile", self)
        self._title.setObjectName("tileTitle")
        self._title.setToolTip("Empty tile")
        self._video = VideoLabel(self)
        self._video.crop_selected.connect(self._crop_selected)
        self._video.activated.connect(lambda: self.selected.emit(self))
        self._video.drag_requested.connect(self._start_drag)
        self._video.double_clicked.connect(lambda: self.fullscreen_requested.emit(self))
        self.setAcceptDrops(True)
        self.setMinimumSize(190, 120)
        self.setStyleSheet("MirrorTile { border: 2px solid #2b3140; background: #111318; } #tileTitle { background: #1c2130; color: #cbd3e5; font-weight: 600; padding: 3px 6px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._video, 1)
        self._drop_overlay = DropZoneOverlay(self)
        self._drop_overlay.setGeometry(self.rect())
        self._drop_overlay.raise_()

    @property
    def source_hwnd(self) -> int | None:
        return self._source_hwnd

    @property
    def source_title(self) -> str:
        return self._source_title

    @property
    def is_mirroring(self) -> bool:
        """True while a source is assigned or its capture thread is alive."""
        return self._source_hwnd is not None or self._capture is not None

    @property
    def crop_region(self) -> CropRegion:
        return self._video.crop_region

    @property
    def tracking_mode(self) -> str:
        return self._tracking_mode

    @property
    def crop_mode(self) -> bool:
        return self._video.crop_mode

    @property
    def person_locked(self) -> bool:
        return self._person_locked

    @property
    def preview_settings(self) -> PreviewSettings:
        return self._preview_settings

    @property
    def flip_horizontal(self) -> bool:
        return self._video.flip_horizontal

    @property
    def flip_vertical(self) -> bool:
        return self._video.flip_vertical

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                "MirrorTile { border: 4px solid #5bc7ff; background: #182a3a; } #tileTitle { background: #24516b; color: #f2fbff; font-weight: 700; padding: 3px 6px; }"
            )
        else:
            self.setStyleSheet(
                "MirrorTile { border: 2px solid #2b3140; background: #111318; } #tileTitle { background: #1c2130; color: #cbd3e5; font-weight: 600; padding: 3px 6px; }"
            )

    def configure_source(self, hwnd: int | None, title: str = "") -> None:
        if hwnd == self._source_hwnd:
            return
        self._save_preset()
        self._stop_capture()
        self._source_hwnd, self._source_title = hwnd, title
        self._title.setText(title or "Empty tile")
        self._title.setToolTip(title or "Empty tile")
        self._preset_key = preset_key(hwnd, title) if hwnd is not None else None
        self._content_follower.reset()
        self._tracking_mode = "none"
        self._person_locked = False
        self._video.set_crop_mode(False)
        if hwnd is None:
            self._video.set_flipped(False, False)
            self._video.clear()
            self._video.setText("Select a window to begin mirroring")
            self.status_changed.emit("Choose a window to mirror")
            return
        # A preset key is assigned whenever ``hwnd`` is present above.
        preset_key_value = self._preset_key
        if preset_key_value is None:
            return
        preset = self._presets.get(preset_key_value)
        self._video.set_crop_region(preset.crop)
        self._video.set_flipped(preset.flip_horizontal, preset.flip_vertical)
        self._person_locked = preset.person_lock
        self._video.clear()
        self._video.setText(f"Starting capture of {title}…")
        capture = WindowsCaptureThread(hwnd, self, self._preview_settings)
        capture.frame_ready.connect(self._frame_received, Qt.ConnectionType.QueuedConnection)
        capture.capture_error.connect(self._capture_failed, Qt.ConnectionType.QueuedConnection)
        capture.source_closed.connect(self._source_closed, Qt.ConnectionType.QueuedConnection)
        capture.finished.connect(self._capture_finished)
        self._capture = capture
        capture.start()
        self.status_changed.emit(f"Mirroring: {title}")
        if preset.tracking_mode != "none" and not preset.crop.is_full:
            self.set_tracking_mode(preset.tracking_mode)

    def pick_portal_window(self) -> None:
        """Start a Linux portal picker; a chosen window becomes this tile's source."""
        self._save_preset()
        self._stop_capture()
        self._source_hwnd, self._source_title = None, "Portal window"
        self._preset_key = preset_key(0, self._source_title)
        self._content_follower.reset()
        self._tracking_mode = "none"
        self._person_locked = False
        self._video.set_crop_mode(False)
        self._video.set_flipped(False, False)
        self._video.clear()
        self._video.setText("Waiting for window selection…")
        self.status_changed.emit("Pick a window in the system dialog")
        capture = PortalCaptureThread(self, self._preview_settings)
        capture.frame_ready.connect(self._frame_received, Qt.ConnectionType.QueuedConnection)
        capture.capture_error.connect(self._capture_failed, Qt.ConnectionType.QueuedConnection)
        capture.source_closed.connect(self._source_closed, Qt.ConnectionType.QueuedConnection)
        capture.selection_cancelled.connect(self._portal_cancelled, Qt.ConnectionType.QueuedConnection)
        capture.finished.connect(self._capture_finished)
        self._capture = capture
        capture.start()

    def set_preview_settings(self, settings: PreviewSettings) -> None:
        """Set this tile's render cap without resetting its portal session."""
        self._preview_settings = settings
        if self._capture is not None:
            self._capture.set_preview_settings(settings)
        self.status_changed.emit(f"Preview quality: {settings.label}")

    def set_crop_mode(self, enabled: bool) -> None:
        self._video.set_crop_mode(enabled)
        if enabled:
            self.set_tracking_mode("none")
            self.status_changed.emit("Drag over the selected tile to select a source region")
        elif self._capture is not None:
            self.status_changed.emit("Mirroring selected crop")

    def set_flipped(self, horizontal: bool, vertical: bool) -> None:
        self._video.set_flipped(horizontal, vertical)
        self._save_preset()
        if horizontal and vertical:
            direction = "horizontally and vertically"
        elif horizontal:
            direction = "horizontally"
        elif vertical:
            direction = "vertically"
        else:
            direction = "normally"
        self.status_changed.emit(f"Displaying the source flipped {direction}")

    def set_person_lock(self, locked: bool) -> None:
        self._person_locked = locked
        if self._person_worker is not None:
            self._person_worker.set_person_lock(locked)
        self._save_preset()
        self.status_changed.emit(
            "Person lock enabled" if locked else "Person lock disabled"
        )

    def reset_crop(self) -> None:
        self._content_follower.reset()
        self.set_tracking_mode("none")
        self._video.reset_crop()
        self._video.set_crop_mode(False)
        self._save_preset()
        self.status_changed.emit("Showing the full source window")

    def set_tracking_mode(self, mode: str) -> None:
        if mode not in {"none", "content", "person"}:
            mode = "none"
        if mode != "none" and self._video.crop_region.is_full:
            self.status_changed.emit("Select a crop before enabling tracking")
            mode = "none"
        if mode == "person" and self._capture is None:
            self.status_changed.emit("Select a window before enabling person follow")
            mode = "none"
        if mode != "person":
            self._stop_person_worker()
        if mode == "content":
            self._content_follower.reset()
        if mode == "person" and self._tracking_mode != "person":
            self._start_person_worker()
            self.status_changed.emit("Loading person detector…")
        self._tracking_mode = mode
        if mode != "none":
            self._video.set_crop_mode(False)
        self._save_preset()

    def shutdown(self) -> None:
        self._save_preset()
        self._stop_person_worker(wait=True)
        self._stop_capture()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._drop_overlay.setGeometry(self.rect())
        self._drop_overlay.raise_()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        self.selected.emit(self)
        menu = QMenu(self)
        split_horizontal = menu.addAction("Split horizontally")
        split_vertical = menu.addAction("Split vertically")
        action = menu.exec(event.globalPos())
        if action is split_horizontal:
            self.split_requested.emit(self, Qt.Orientation.Horizontal)
        elif action is split_vertical:
            self.split_requested.emit(self, Qt.Orientation.Vertical)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(TILE_MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: Any) -> None:
        if not event.mimeData().hasFormat(TILE_MIME_TYPE):
            event.ignore()
            return
        source_id = bytes(event.mimeData().data(TILE_MIME_TYPE)).decode("utf-8")
        zone = None if source_id == self.tile_id else self._drop_zone_at(event.position())
        if zone != self._drop_zone:
            self._drop_zone = zone
            self._drop_overlay.set_zone(zone)
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: Any) -> None:
        self._drop_zone = None
        self._drop_overlay.set_zone(None)
        event.accept()

    def dropEvent(self, event: Any) -> None:
        source_id = bytes(event.mimeData().data(TILE_MIME_TYPE)).decode("utf-8")
        zone = self._drop_zone_at(event.position())
        self._drop_zone = None
        self._drop_overlay.set_zone(None)
        if source_id != self.tile_id:
            if zone == "center":
                self.swap_requested.emit(source_id, self.tile_id)
            else:
                self.edge_drop_requested.emit(source_id, self.tile_id, zone)
        event.acceptProposedAction()



    def _drop_zone_at(self, position: QPointF) -> str:
        rect = self.rect()
        edge_width = rect.width() * 0.25
        edge_height = rect.height() * 0.25
        if position.x() < edge_width:
            return "left"
        if position.x() >= rect.width() - edge_width:
            return "right"
        if position.y() < edge_height:
            return "above"
        if position.y() >= rect.height() - edge_height:
            return "below"
        return "center"


    @Slot(object)
    def _frame_received(self, pixels: np.ndarray) -> None:
        capture = self.sender()
        if not isinstance(capture, CaptureThread):
            return
        try:
            if capture is self._capture:
                if self._tracking_mode == "content" and not self._video.crop_mode and not self._video.crop_region.is_full:
                    tracked = self._content_follower.update(pixels, self._video.crop_region)
                    if tracked is not None:
                        self._video.set_crop_region(tracked)
                elif self._tracking_mode == "person" and not self._video.crop_mode and self._person_worker is not None:
                    self._person_worker.submit(pixels, self._video.crop_region)
                self._video.set_frame(pixels)
        finally:
            capture.frame_consumed()

    @Slot(object)
    def _crop_selected(self, crop: CropRegion) -> None:
        self._content_follower.reset()
        if self._person_worker is not None:
            self._person_worker.reset()
        self._video.set_crop_mode(False)
        self._save_preset()
        self.status_changed.emit(f"Crop selected: {crop.left:.0%}, {crop.top:.0%} to {crop.right:.0%}, {crop.bottom:.0%} of source")

    def _save_preset(self) -> None:
        if self._preset_key is None:
            return
        existing = self._presets.get(self._preset_key)
        self._presets.set(
            self._preset_key,
            SourcePreset(
                self._video.crop_region,
                self._tracking_mode,
                existing.geometry,
                self._video.flip_horizontal,
                self._video.flip_vertical,
                self._person_locked,
            ),
        )

    def _start_person_worker(self) -> None:
        self._stop_person_worker()
        worker = PersonDetectionWorker(self)
        worker.set_person_lock(self._person_locked)
        worker.crop_ready.connect(self._person_crop_ready, Qt.ConnectionType.QueuedConnection)
        worker.detection_error.connect(self._person_detection_failed, Qt.ConnectionType.QueuedConnection)
        self._person_worker = worker
        worker.start()

    def _stop_person_worker(self, wait: bool = False) -> None:
        if self._person_worker is not None:
            self._person_worker.stop()
            if wait:
                self._person_worker.wait(2000)
            self._person_worker = None

    def _stop_capture(self) -> None:
        self._stop_person_worker()
        if self._capture is not None:
            self._capture.stop()
            # A portal picker dialog can stay open while the user decides, so
            # grant that backend more time before abandoning its thread.
            timeout = 2000 if isinstance(self._capture, PortalCaptureThread) else 250
            self._capture.wait(timeout)
            self._capture = None

    @Slot(object)
    def _person_crop_ready(self, crop: CropRegion) -> None:
        if self.sender() is self._person_worker and self._tracking_mode == "person":
            self._video.set_crop_region(crop)
            self._save_preset()
            self.status_changed.emit("Following person")

    @Slot(str)
    def _person_detection_failed(self, message: str) -> None:
        if self.sender() is self._person_worker:
            self._tracking_mode = "none"
            self._stop_person_worker()
            self.status_changed.emit(f"Person detection unavailable: {message}")

    @Slot(str)
    def _capture_failed(self, message: str) -> None:
        self._tracking_mode = "none"
        self._video.clear()
        self._video.setText("Capture unavailable")
        self.status_changed.emit(f"Capture error: {message}")

    @Slot()
    def _source_closed(self) -> None:
        self._tracking_mode = "none"
        self._video.clear()
        self._video.setText("The source window was closed")
        self.status_changed.emit("Source window closed — choose another window")

    @Slot()
    def _portal_cancelled(self) -> None:
        self._tracking_mode = "none"
        self._source_hwnd = None
        self._source_title = ""
        self._preset_key = None
        self._video.clear()
        self._video.setText("Select a window to begin mirroring")
        self.status_changed.emit("Window selection cancelled")

    @Slot()
    def _capture_finished(self) -> None:
        if self._capture is not None and self._capture.isFinished():
            self._capture = None

    def _start_drag(self) -> None:
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(TILE_MIME_TYPE, self.tile_id.encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)


class FocalWindow(QMainWindow):
    """Single-window, splitter-based workspace for up to eight mirror tiles."""

    mirror_closed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Focal")
        self.resize(1120, 720)
        self.setMinimumSize(620, 420)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._presets = PresetStore(default_preset_path())
        self._tiles: list[MirrorTile] = []
        self._selected: MirrorTile | None = None
        self._recent_hwnds: list[int] = []
        self._refreshing = False
        self._splitter: QSplitter | None = None
        self._fullscreen_tile: MirrorTile | None = None
        self._fullscreen_tile_parent: QSplitter | None = None
        self._fullscreen_tile_index: int | None = None
        self._window_fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self._window_fullscreen_shortcut.activated.connect(self._toggle_window_fullscreen)

        self._source_combo = QComboBox()
        self._source_combo.setMinimumWidth(180)
        self._source_combo.currentIndexChanged.connect(self._source_changed)
        # On Linux the picker is an action, not a persistent source value.
        # ``currentIndexChanged`` is not emitted when a user re-selects the
        # already-current "Pick a window…" entry; ``activated`` always is.
        self._source_combo.activated.connect(self._source_activated)
        refresh = QPushButton("↻")
        refresh.setToolTip("Refresh available source windows")
        refresh.clicked.connect(self.refresh_windows)

        self._pane_button = QPushButton("Pane ▾")
        pane_menu = QMenu(self._pane_button)
        self._add_action = pane_menu.addAction("Add tile")
        self._add_action.triggered.connect(self._add_tile)
        pane_menu.addSeparator()
        self._split_actions = []
        for label, direction in (("Add left", "left"), ("Add right", "right"), ("Add above", "above"), ("Add below", "below")):
            action = pane_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, value=direction: self._split_selected_tile(value)
            )
            self._split_actions.append(action)
        pane_menu.addSeparator()
        remove = pane_menu.addAction("Remove selected tile")
        remove.triggered.connect(self._remove_selected_tile)
        self._pane_button.setMenu(pane_menu)
        self._layout_combo = QComboBox()
        self._layout_combo.addItem("Row", LayoutMode.ROW)
        self._layout_combo.addItem("Column", LayoutMode.COLUMN)
        self._layout_combo.addItem("Top pair + bottom", LayoutMode.TOP_PAIR)
        self._layout_combo.addItem("2-column grid", LayoutMode.GRID)
        self._layout_combo.currentIndexChanged.connect(self._rebuild_splitters)
        self._crop_button = QPushButton("Crop")
        self._crop_button.setToolTip("Select a crop region in the active tile")
        self._crop_button.setCheckable(True)
        self._crop_button.toggled.connect(self._set_crop_mode)
        reset = QPushButton("Reset")
        reset.setToolTip("Reset the active tile crop")
        reset.clicked.connect(self._reset_crop)
        self._flip_horizontal_button = QPushButton("Flip H")
        self._flip_horizontal_button.setCheckable(True)
        self._flip_horizontal_button.setToolTip("Flip the active tile horizontally")
        self._flip_horizontal_button.toggled.connect(self._set_horizontal_flip)
        self._flip_vertical_button = QPushButton("Flip V")
        self._flip_vertical_button.setCheckable(True)
        self._flip_vertical_button.setToolTip("Flip the active tile vertically")
        self._flip_vertical_button.toggled.connect(self._set_vertical_flip)
        self._tracking_combo = QComboBox()
        self._tracking_combo.addItem("Track: Off", "none")
        self._tracking_combo.addItem("Track: Content", "content")
        self._tracking_combo.addItem("Track: Person", "person")
        self._tracking_combo.setToolTip("Tracking mode for the active tile")
        self._tracking_combo.currentIndexChanged.connect(self._tracking_changed)
        self._person_lock = QCheckBox("Lock person")
        self._person_lock.setToolTip(
            "Keep following the selected person; hold position instead of switching people"
        )
        self._person_lock.toggled.connect(self._set_person_lock)
        self._preview_quality_combo = QComboBox()
        for settings in PREVIEW_QUALITIES:
            self._preview_quality_combo.addItem(settings.label, settings)
        self._preview_quality_combo.setToolTip(
            "Per-tile frame-rate and resolution cap. Lower settings reduce CPU use."
        )
        self._preview_quality_combo.currentIndexChanged.connect(self._preview_quality_changed)
        self._pin = QCheckBox("Pin")
        self._pin.setChecked(True)
        self._pin.setToolTip("Keep the workspace above other windows")
        self._pin.toggled.connect(self._set_pinned)
        opacity = QSlider(Qt.Orientation.Horizontal)
        opacity.setRange(35, 100)
        opacity.setValue(100)
        opacity.setMaximumWidth(80)
        opacity.setToolTip("Workspace opacity")
        opacity.valueChanged.connect(lambda value: self.setWindowOpacity(value / 100.0))
        self._status = QLabel("Finding open windows…")
        self._status.setStyleSheet("color: #7f8799;")

        controls = QHBoxLayout()
        controls.setContentsMargins(10, 8, 10, 8)
        controls.addWidget(self._source_combo, 1)
        controls.addWidget(refresh)
        controls.addWidget(self._pane_button)
        controls.addWidget(self._layout_combo)
        controls.addWidget(self._crop_button)
        controls.addWidget(reset)
        controls.addWidget(self._flip_horizontal_button)
        controls.addWidget(self._flip_vertical_button)
        controls.addWidget(self._tracking_combo)
        controls.addWidget(self._person_lock)
        controls.addWidget(self._preview_quality_combo)
        controls.addWidget(self._pin)
        controls.addWidget(opacity)
        self._workspace = QWidget()
        self._workspace_layout = QVBoxLayout(self._workspace)
        self._workspace_layout.setContentsMargins(8, 2, 8, 2)
        self._controls = QWidget()
        self._controls.setLayout(controls)
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 6)
        root_layout.setSpacing(4)
        root_layout.addWidget(self._controls)
        root_layout.addWidget(self._workspace, 1)
        root_layout.addWidget(self._status)
        self.setCentralWidget(root)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(2500)
        self._refresh_timer.timeout.connect(self.refresh_windows)
        self._refresh_timer.start()
        self._add_tile()
        self.refresh_windows()

    @Slot()
    def refresh_windows(self) -> None:
        if self._source_combo.view().isVisible():
            return
        current = self._selected.source_hwnd if self._selected is not None else None
        self._refreshing = True
        self._source_combo.clear()
        if not is_portal_session():
            try:
                windows = _order_windows(list_windows(exclude_hwnd=int(self.winId())), self._recent_hwnds)
            except Exception as exc:
                self._status.setText(f"Unable to enumerate windows: {exc}")
                self._refreshing = False
                return
            self._source_combo.addItem("Select a window…", None)
            for window in windows:
                self._source_combo.addItem(window.title, window.hwnd)
            selected_index = next((index for index, window in enumerate(windows, 1) if window.hwnd == current), 0)
            self._source_combo.setCurrentIndex(selected_index)
            self._refreshing = False
            if not windows:
                self._status.setText("No visible windows found")
            return

        # Linux: windows are picked through the system XDG ScreenCast dialog.
        self._source_combo.addItem("Pick a window…", PICK_WINDOW)
        self._source_combo.addItem("Stop mirroring", STOP_MIRRORING)
        active = self._selected is not None and self._selected.is_mirroring
        self._source_combo.setCurrentIndex(1 if active else 0)
        self._refreshing = False
        if not active and not any(tile.is_mirroring for tile in self._tiles):
            self._status.setText("Pick a window to start mirroring")

    @Slot(int)
    def _source_changed(self, index: int) -> None:
        if self._refreshing or index < 0 or self._selected is None or is_portal_session():
            return
        value = self._source_combo.currentData()
        hwnd = int(value) if value is not None else None
        title = self._source_combo.currentText() if hwnd is not None else ""
        if hwnd is not None:
            if hwnd in self._recent_hwnds:
                self._recent_hwnds.remove(hwnd)
            self._recent_hwnds.insert(0, hwnd)
            QTimer.singleShot(100, self.refresh_windows)
        self._selected.configure_source(hwnd, title)
        self._sync_controls()

    @Slot(int)
    def _source_activated(self, index: int) -> None:
        """Run portal source actions after a deliberate user selection."""
        if self._refreshing or index < 0 or self._selected is None or not is_portal_session():
            return
        value = self._source_combo.itemData(index)
        if value == PICK_WINDOW:
            self._selected.pick_portal_window()
        elif value == STOP_MIRRORING:
            self._selected.configure_source(None, "")
        self._sync_controls()

    @Slot()
    def _add_tile(self) -> None:
        if len(self._tiles) >= MAX_TILES:
            self._status.setText(f"A workspace can contain at most {MAX_TILES} tiles")
            return
        tile = self._new_tile()
        self._tiles.append(tile)
        self._select_tile(tile)
        if self._splitter is None:
            self._rebuild_splitters()
        else:
            self._splitter.addWidget(tile)

    @Slot()
    def _remove_selected_tile(self) -> None:
        if self._selected is None or len(self._tiles) == 1:
            self._status.setText("The workspace must keep one tile")
            return
        tile = self._selected
        parent = tile.parentWidget()
        self._tiles.remove(tile)
        tile.shutdown()
        tile.setParent(None)
        tile.deleteLater()
        if isinstance(parent, QSplitter):
            self._compact_splitters(parent)
        self._select_tile(self._tiles[-1])

    @Slot(object)
    def _select_tile(self, tile: MirrorTile) -> None:
        if tile not in self._tiles:
            return
        self._selected = tile
        for candidate in self._tiles:
            candidate.set_selected(candidate is tile)
        self.refresh_windows()
        self._sync_controls()

    @Slot(str, str)
    def _swap_tiles(self, first_id: str, second_id: str) -> None:
        first = next((tile for tile in self._tiles if tile.tile_id == first_id), None)
        second = next((tile for tile in self._tiles if tile.tile_id == second_id), None)
        if first is None or second is None:
            return
        first_parent, second_parent = first.parentWidget(), second.parentWidget()
        if not isinstance(first_parent, QSplitter) or not isinstance(second_parent, QSplitter):
            return
        first_index = first_parent.indexOf(first)
        second_index = second_parent.indexOf(second)
        if first_index < 0 or second_index < 0:
            return
        if first_parent is second_parent:
            replacements = {first_index: second, second_index: first}
            first.setParent(None)
            second.setParent(None)
            for index in sorted(replacements):
                first_parent.insertWidget(index, replacements[index])
        else:
            first.setParent(None)
            second.setParent(None)
            first_parent.insertWidget(first_index, second)
            second_parent.insertWidget(second_index, first)
        self._select_tile(first)

    @Slot(object, object)
    def _split_tile(self, tile: MirrorTile, orientation: Qt.Orientation) -> None:
        """Keep the context-menu behavior: add an empty tile after this one."""
        self._split_with_empty_sibling(tile, orientation, sibling_first=False)

    @Slot(str)
    def _split_selected_tile(self, direction: str) -> None:
        if self._selected is None:
            return
        orientation, sibling_first = _split_details(direction)
        self._split_with_empty_sibling(self._selected, orientation, sibling_first)

    def _split_with_empty_sibling(
        self, tile: MirrorTile, orientation: Qt.Orientation, sibling_first: bool
    ) -> None:
        if tile not in self._tiles:
            return
        if len(self._tiles) >= MAX_TILES:
            self._status.setText(f"A workspace can contain at most {MAX_TILES} tiles")
            return
        sibling = self._new_tile()
        if not self._split_around_tile(tile, sibling, orientation, sibling_first):
            sibling.deleteLater()
            return
        self._tiles.append(sibling)
        self._select_tile(sibling)

    @Slot(str, str, str)
    def _drop_tile_on_edge(self, source_id: str, target_id: str, direction: str) -> None:
        source = next((tile for tile in self._tiles if tile.tile_id == source_id), None)
        target = next((tile for tile in self._tiles if tile.tile_id == target_id), None)
        if source is None or target is None or source is target:
            return
        orientation, source_first = _split_details(direction)
        source_parent = source.parentWidget()
        if not isinstance(source_parent, QSplitter):
            return
        source.setParent(None)
        self._compact_splitters(source_parent)
        if self._split_around_tile(target, source, orientation, source_first):
            self._select_tile(source)

    def _split_around_tile(
        self,
        tile: MirrorTile,
        sibling: MirrorTile,
        orientation: Qt.Orientation,
        sibling_first: bool,
    ) -> bool:
        """Replace a tile with a nested splitter containing it and its sibling."""
        parent = tile.parentWidget()
        if not isinstance(parent, QSplitter):
            return False
        index = parent.indexOf(tile)
        if index < 0:
            return False
        splitter = QSplitter(orientation)
        tile.setParent(None)
        parent.insertWidget(index, splitter)
        if sibling_first:
            splitter.addWidget(sibling)
            splitter.addWidget(tile)
        else:
            splitter.addWidget(tile)
            splitter.addWidget(sibling)
        return True

    def _new_tile(self) -> MirrorTile:
        tile = MirrorTile(self._presets)
        tile.selected.connect(self._select_tile)
        tile.status_changed.connect(self._tile_status)
        tile.swap_requested.connect(self._swap_tiles)
        tile.split_requested.connect(self._split_tile)
        tile.edge_drop_requested.connect(self._drop_tile_on_edge)
        tile.fullscreen_requested.connect(self._toggle_tile_fullscreen)
        return tile

    def _compact_splitters(self, splitter: QSplitter) -> None:
        """Collapse one-child nested splitters while preserving the root splitter."""
        for index in range(splitter.count()):
            child = splitter.widget(index)
            if isinstance(child, QSplitter):
                self._compact_splitters(child)
        if splitter is self._splitter or splitter.count() != 1:
            return
        parent = splitter.parentWidget()
        if not isinstance(parent, QSplitter):
            return
        index = parent.indexOf(splitter)
        only_child = splitter.widget(0)
        if index < 0 or only_child is None:
            return
        only_child.setParent(None)
        parent.insertWidget(index, only_child)
        splitter.setParent(None)
        splitter.deleteLater()
        self._compact_splitters(parent)

    @Slot()
    def _rebuild_splitters(self, _index: int = 0) -> None:
        for tile in self._tiles:
            tile.setParent(None)
        if self._splitter is not None:
            self._workspace_layout.removeWidget(self._splitter)
            self._splitter.deleteLater()
        mode = self._layout_combo.currentData()
        if not isinstance(mode, LayoutMode):
            mode = LayoutMode.GRID
        root_orientation = Qt.Orientation.Horizontal if mode is LayoutMode.ROW else Qt.Orientation.Vertical
        root = QSplitter(root_orientation, self._workspace)
        rows = pane_rows(self._tiles, mode)
        if mode in (LayoutMode.ROW, LayoutMode.COLUMN):
            for tile in rows[0]:
                root.addWidget(tile)
        else:
            for row in rows:
                if len(row) == 1:
                    root.addWidget(row[0])
                else:
                    pair = QSplitter(Qt.Orientation.Horizontal, root)
                    for tile in row:
                        pair.addWidget(tile)
                    root.addWidget(pair)
        self._splitter = root
        self._workspace_layout.addWidget(root)

    @Slot(object)
    def _toggle_tile_fullscreen(self, tile: MirrorTile) -> None:
        expanded_tile = self._fullscreen_tile
        if expanded_tile is not None:
            self._restore_tile_fullscreen()
            if tile is expanded_tile:
                return
        if tile not in self._tiles:
            return
        parent = tile.parentWidget()
        if not isinstance(parent, QSplitter):
            return
        index = parent.indexOf(tile)
        if index < 0 or self._splitter is None:
            return
        self._fullscreen_tile = tile
        self._fullscreen_tile_parent = parent
        self._fullscreen_tile_index = index
        tile.setParent(None)
        self._splitter.hide()
        self._controls.hide()
        self._status.hide()
        self._workspace_layout.addWidget(tile)
        tile.show()
        self._status.setText("Double-click the enlarged tile to restore the workspace")

    def _restore_tile_fullscreen(self) -> None:
        tile = self._fullscreen_tile
        parent = self._fullscreen_tile_parent
        index = self._fullscreen_tile_index
        if tile is None or parent is None or index is None:
            return
        self._workspace_layout.removeWidget(tile)
        tile.setParent(None)
        parent.insertWidget(index, tile)
        if self._splitter is not None:
            self._splitter.show()
        if not self.isFullScreen():
            self._controls.show()
            self._status.show()
        self._fullscreen_tile = None
        self._fullscreen_tile_parent = None
        self._fullscreen_tile_index = None
        self._sync_controls()

    @Slot()
    def _toggle_window_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            if self._fullscreen_tile is None:
                self._controls.show()
                self._status.show()
        else:
            self._controls.hide()
            self._status.hide()
            self.showFullScreen()

    @Slot(bool)
    def _set_crop_mode(self, enabled: bool) -> None:
        if self._selected is not None:
            self._selected.set_crop_mode(enabled)
        self._sync_controls()

    @Slot()
    def _reset_crop(self) -> None:
        if self._selected is not None:
            self._selected.reset_crop()
        self._sync_controls()

    @Slot(bool)
    def _set_horizontal_flip(self, enabled: bool) -> None:
        if self._selected is not None:
            self._selected.set_flipped(enabled, self._selected.flip_vertical)
        self._sync_controls()

    @Slot(bool)
    def _set_vertical_flip(self, enabled: bool) -> None:
        if self._selected is not None:
            self._selected.set_flipped(self._selected.flip_horizontal, enabled)
        self._sync_controls()

    @Slot(int)
    def _tracking_changed(self, _index: int) -> None:
        mode = self._tracking_combo.currentData()
        if isinstance(mode, str):
            self._set_tracking_mode(mode)

    def _set_tracking_mode(self, mode: str) -> None:
        if self._selected is not None:
            self._selected.set_tracking_mode(mode)
        self._sync_controls()

    @Slot(bool)
    def _set_person_lock(self, locked: bool) -> None:
        if self._selected is not None:
            self._selected.set_person_lock(locked)
        self._sync_controls()

    @Slot(int)
    def _preview_quality_changed(self, _index: int) -> None:
        settings = self._preview_quality_combo.currentData()
        if isinstance(settings, PreviewSettings) and self._selected is not None:
            self._selected.set_preview_settings(settings)
        self._sync_controls()

    @Slot(bool)
    def _set_pinned(self, pinned: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
        self.show()

    @Slot(str)
    def _tile_status(self, message: str) -> None:
        if self.sender() is self._selected:
            self._status.setText(message)
            self._sync_controls()

    def _sync_controls(self) -> None:
        tile = self._selected
        if tile is None:
            return
        self._crop_button.blockSignals(True)
        self._crop_button.setChecked(tile.crop_mode)
        self._crop_button.blockSignals(False)
        self._crop_button.setText("Cancel crop" if tile.crop_mode else "Crop")
        self._flip_horizontal_button.blockSignals(True)
        self._flip_horizontal_button.setChecked(tile.flip_horizontal)
        self._flip_horizontal_button.blockSignals(False)
        self._flip_vertical_button.blockSignals(True)
        self._flip_vertical_button.setChecked(tile.flip_vertical)
        self._flip_vertical_button.blockSignals(False)
        tracking_index = self._tracking_combo.findData(tile.tracking_mode)
        self._tracking_combo.blockSignals(True)
        self._tracking_combo.setCurrentIndex(max(0, tracking_index))
        self._tracking_combo.blockSignals(False)
        self._person_lock.blockSignals(True)
        self._person_lock.setChecked(tile.person_locked)
        self._person_lock.blockSignals(False)
        self._person_lock.setEnabled(tile.tracking_mode == "person")
        preview_index = self._preview_quality_combo.findData(tile.preview_settings)
        self._preview_quality_combo.blockSignals(True)
        self._preview_quality_combo.setCurrentIndex(max(0, preview_index))
        self._preview_quality_combo.blockSignals(False)
        can_split = len(self._tiles) < MAX_TILES
        self._add_action.setEnabled(can_split)
        for action in self._split_actions:
            action.setEnabled(can_split)

    def closeEvent(self, event: Any) -> None:
        if self._fullscreen_tile is not None:
            self._restore_tile_fullscreen()
        self._refresh_timer.stop()
        for tile in self._tiles:
            tile.shutdown()
        self.mirror_closed.emit(self)
        super().closeEvent(event)
