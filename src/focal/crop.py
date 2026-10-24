from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CropRegion:
    """A crop rectangle expressed as normalized source-image coordinates."""

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0

    @classmethod
    def full(cls) -> CropRegion:
        return cls()

    @classmethod
    def from_bounds(
        cls, left: float, top: float, right: float, bottom: float
    ) -> CropRegion:
        left, top, right, bottom = (
            max(0.0, min(1.0, value)) for value in (left, top, right, bottom)
        )
        return cls(
            left=min(left, right),
            top=min(top, bottom),
            right=max(left, right),
            bottom=max(top, bottom),
        )

    @property
    def is_full(self) -> bool:
        return self == CropRegion.full()

    def pixel_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return a valid half-open pixel rectangle for an image size."""
        if width < 1 or height < 1:
            return 0, 0, 0, 0

        x1 = max(0, min(width - 1, math.floor(self.left * width)))
        y1 = max(0, min(height - 1, math.floor(self.top * height)))
        x2 = max(x1 + 1, min(width, math.ceil(self.right * width)))
        y2 = max(y1 + 1, min(height, math.ceil(self.bottom * height)))
        return x1, y1, x2, y2
