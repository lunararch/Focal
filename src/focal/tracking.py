from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .crop import CropRegion


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    """Tuning values for optical-flow content following."""

    smoothing: float = 0.35
    dead_zone: float = 0.15
    min_features: int = 8
    max_features: int = 80


class ContentFollower:
    """Follow the visual features originally contained in a crop region.

    The follower estimates per-frame motion with pyramidal Lucas-Kanade optical
    flow, then moves the crop towards that target with a dead zone and
    exponential smoothing. It intentionally leaves the current crop unchanged
    when a source frame does not contain enough reliable features.
    """

    def __init__(self, settings: TrackingSettings | None = None) -> None:
        self._settings = settings or TrackingSettings()
        self.reset()

    def reset(self) -> None:
        self._previous_gray: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self._target_crop: CropRegion | None = None

    def update(self, pixels: np.ndarray, crop: CropRegion) -> CropRegion | None:
        """Return an adjusted crop, or ``None`` while initializing/lost."""
        gray = _to_gray(pixels)
        if gray is None:
            self.reset()
            return None

        if self._previous_gray is None or self._target_crop is None:
            self._initialize(gray, crop)
            return None

        if self._points is None or len(self._points) < self._settings.min_features:
            self._initialize(gray, crop)
            return None

        next_points, status, _errors = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            self._previous_gray,
            gray,
            self._points,
            None,  # type: ignore[arg-type]  # OpenCV accepts None to allocate output.
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
        if next_points is None or status is None:
            self._initialize(gray, crop)
            return None

        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < self._settings.min_features:
            self._initialize(gray, crop)
            return None

        previous_points = self._points[valid]
        current_points = next_points[valid]
        shift_x, shift_y = np.median(current_points - previous_points, axis=0).reshape(2)
        self._points = current_points.reshape(-1, 1, 2)
        self._previous_gray = gray

        height, width = gray.shape
        self._target_crop = _translate_crop(
            self._target_crop, float(shift_x) / width, float(shift_y) / height
        )
        return _smooth_crop(crop, self._target_crop, self._settings)

    def _initialize(self, gray: np.ndarray, crop: CropRegion) -> None:
        height, width = gray.shape
        x1, y1, x2, y2 = crop.pixel_bounds(width, height)
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._settings.max_features,
            qualityLevel=0.01,
            minDistance=7,
            mask=mask,
            blockSize=7,
        )
        self._previous_gray = gray
        self._points = points
        self._target_crop = crop


def _to_gray(pixels: np.ndarray) -> np.ndarray | None:
    if pixels.ndim != 3:
        return None
    if pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2GRAY)
    if pixels.shape[2] == 3:
        return cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    return None


def _translate_crop(crop: CropRegion, dx: float, dy: float) -> CropRegion:
    width = crop.right - crop.left
    height = crop.bottom - crop.top
    left = min(max(0.0, crop.left + dx), 1.0 - width)
    top = min(max(0.0, crop.top + dy), 1.0 - height)
    return CropRegion.from_bounds(left, top, left + width, top + height)


def _smooth_crop(
    current: CropRegion, target: CropRegion, settings: TrackingSettings
) -> CropRegion | None:
    current_center_x = (current.left + current.right) / 2
    current_center_y = (current.top + current.bottom) / 2
    target_center_x = (target.left + target.right) / 2
    target_center_y = (target.top + target.bottom) / 2
    width = current.right - current.left
    height = current.bottom - current.top

    if (
        abs(target_center_x - current_center_x) < width * settings.dead_zone
        and abs(target_center_y - current_center_y) < height * settings.dead_zone
    ):
        return None

    left = current.left + (target.left - current.left) * settings.smoothing
    top = current.top + (target.top - current.top) * settings.smoothing
    return _translate_crop(current, left - current.left, top - current.top)
