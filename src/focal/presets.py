from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .crop import CropRegion


@dataclass(frozen=True, slots=True)
class SourcePreset:
    crop: CropRegion = field(default_factory=CropRegion.full)
    tracking_mode: str = "none"
    geometry: tuple[int, int, int, int] | None = None
    flip_horizontal: bool = False
    flip_vertical: bool = False
    person_lock: bool = False

    @classmethod
    def from_json(cls, value: object) -> SourcePreset:
        if not isinstance(value, dict):
            return cls()
        crop_value = value.get("crop", {})
        if not isinstance(crop_value, dict):
            crop_value = {}
        crop = CropRegion.from_bounds(
            float(crop_value.get("left", 0.0)),
            float(crop_value.get("top", 0.0)),
            float(crop_value.get("right", 1.0)),
            float(crop_value.get("bottom", 1.0)),
        )
        mode = value.get("tracking_mode", "none")
        tracking_mode = mode if mode in {"none", "content", "person"} else "none"
        geometry_value = value.get("geometry")
        geometry = None
        if (
            isinstance(geometry_value, list)
            and len(geometry_value) == 4
            and all(isinstance(item, int) for item in geometry_value)
            and geometry_value[2] > 0
            and geometry_value[3] > 0
        ):
            geometry = tuple(geometry_value)  # type: ignore[assignment]
        return cls(
            crop,
            tracking_mode,
            geometry,
            bool(value.get("flip_horizontal", False)),
            bool(value.get("flip_vertical", False)),
            bool(value.get("person_lock", False)),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "crop": {
                "left": self.crop.left,
                "top": self.crop.top,
                "right": self.crop.right,
                "bottom": self.crop.bottom,
            },
            "tracking_mode": self.tracking_mode,
            "geometry": list(self.geometry) if self.geometry is not None else None,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "person_lock": self.person_lock,
        }


class PresetStore:
    """Small, failure-tolerant JSON store for source-specific overlay state."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._presets = self._load()

    def get(self, key: str) -> SourcePreset:
        return self._presets.get(key, SourcePreset())

    def set(self, key: str, preset: SourcePreset) -> None:
        self._presets[key] = preset
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {name: value.to_json() for name, value in self._presets.items()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self._path)

    def _load(self) -> dict[str, SourcePreset]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            return {str(key): SourcePreset.from_json(value) for key, value in raw.items()}
        except (OSError, ValueError, TypeError):
            return {}


def default_preset_path() -> Path:
    if sys.platform == "win32":
        app_data = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return app_data / "Focal" / "presets.json"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "Focal" / "presets.json"
