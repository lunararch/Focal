from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from .crop import CropRegion
from .tracking import TrackingSettings, _smooth_crop, _translate_crop


PERSON_CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class PersonBox:
    left: float
    top: float
    right: float
    bottom: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2


class PersonFollower:
    """Select one detected person consistently and center the crop on them."""

    def __init__(self, settings: TrackingSettings | None = None) -> None:
        self._settings = settings or TrackingSettings()
        self.reset()

    def reset(self) -> None:
        self._last_center: tuple[float, float] | None = None
        self._last_box: PersonBox | None = None
        self._locked_box: PersonBox | None = None
        self._person_locked = False

    def set_person_lock(self, locked: bool) -> None:
        """Lock to the current person, or let selection choose freely again."""
        self._person_locked = locked
        self._locked_box = self._last_box if locked else None

    def update(
        self,
        boxes: list[PersonBox],
        crop: CropRegion,
        frame_width: int,
        frame_height: int,
    ) -> CropRegion | None:
        candidates = [box for box in boxes if box.confidence >= 0.4]
        if not candidates or frame_width < 1 or frame_height < 1:
            return None

        reference = self._last_center or (
            (crop.left + crop.right) * frame_width / 2,
            (crop.top + crop.bottom) * frame_height / 2,
        )
        if self._person_locked and self._locked_box is not None:
            candidates = [
                box for box in candidates if _is_same_person(box, self._locked_box)
            ]
            # Holding the crop is preferable to jumping to a different person
            # while the locked subject is briefly obscured or leaves the frame.
            if not candidates:
                return None

        person = min(candidates, key=lambda box: _squared_distance(box.center, reference))
        self._last_center = person.center
        self._last_box = person
        if self._person_locked:
            self._locked_box = person
        target = _translate_crop(
            crop,
            person.center[0] / frame_width - (crop.left + crop.right) / 2,
            person.center[1] / frame_height - (crop.top + crop.bottom) / 2,
        )
        return _smooth_crop(crop, target, self._settings)


def _is_same_person(candidate: PersonBox, previous: PersonBox) -> bool:
    """Conservatively associate a detection with a locked person."""
    intersection_left = max(candidate.left, previous.left)
    intersection_top = max(candidate.top, previous.top)
    intersection_right = min(candidate.right, previous.right)
    intersection_bottom = min(candidate.bottom, previous.bottom)
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    candidate_area = (candidate.right - candidate.left) * (candidate.bottom - candidate.top)
    previous_area = (previous.right - previous.left) * (previous.bottom - previous.top)
    union = candidate_area + previous_area - intersection
    if union > 0.0 and intersection / union >= 0.1:
        return True

    # Permit motion between inference frames, but reject a person who is clearly
    # separated from the locked subject.
    max_displacement = 0.8 * max(
        previous.right - previous.left, previous.bottom - previous.top
    )
    return _squared_distance(candidate.center, previous.center) <= max_displacement**2


def _squared_distance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


class PersonDetectionWorker(QThread):
    """Run bounded-rate YOLO inference off the Qt UI thread."""

    crop_ready = Signal(object)
    detection_error = Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._condition = threading.Condition()
        self._latest: tuple[np.ndarray, CropRegion] | None = None
        self._stop_requested = False
        self._reset_requested = False
        self._person_lock_requested = False
        self._follower = PersonFollower()

    def submit(self, pixels: np.ndarray, crop: CropRegion) -> None:
        """Replace any queued frame so inference never builds a backlog."""
        with self._condition:
            self._latest = (pixels, crop)
            self._condition.notify()

    def reset(self) -> None:
        with self._condition:
            self._reset_requested = True
            self._latest = None

    def set_person_lock(self, locked: bool) -> None:
        with self._condition:
            self._person_lock_requested = locked

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._latest = None
            self._condition.notify()

    def run(self) -> None:
        try:
            detector = _YoloPersonDetector()
            while True:
                with self._condition:
                    while not self._stop_requested and self._latest is None:
                        self._condition.wait()
                    if self._stop_requested:
                        return
                    job = self._latest
                    self._latest = None
                    if job is None:
                        continue
                    pixels, crop = job
                    person_locked = self._person_lock_requested
                    if self._reset_requested:
                        self._follower.reset()
                        self._reset_requested = False

                boxes = detector.detect(pixels)
                self._follower.set_person_lock(person_locked)
                next_crop = self._follower.update(
                    boxes, crop, pixels.shape[1], pixels.shape[0]
                )
                if next_crop is not None:
                    self.crop_ready.emit(next_crop)
        except Exception as exc:
            self.detection_error.emit(str(exc))


def _inference_device() -> str:
    """Pick the CUDA device when one is available, otherwise fall back to CPU."""
    import torch  # pyright: ignore[reportMissingImports]

    if torch.cuda.is_available():
        return "0"
    return "cpu"


class _YoloPersonDetector:
    """Lazy Ultralytics adapter; importing/loading happens in the worker."""

    def __init__(self) -> None:
        from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

        # Ultralytics downloads this small model on first use if it is not
        # already cached. It recognizes people facing toward or away from view.
        self._model = YOLO("yolo11n.pt")
        self._device = _inference_device()

    def detect(self, pixels: np.ndarray) -> list[PersonBox]:
        source = pixels[:, :, :3] if pixels.shape[2] == 4 else pixels
        results = self._model.predict(
            source=source,
            classes=[PERSON_CLASS_ID],
            conf=0.4,
            imgsz=320,
            device=self._device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        return [
            PersonBox(
                left=float(left),
                top=float(top),
                right=float(right),
                bottom=float(bottom),
                confidence=float(confidence),
            )
            for (left, top, right, bottom), confidence in zip(coordinates, confidences)
        ]
