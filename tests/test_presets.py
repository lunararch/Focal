import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from focal.crop import CropRegion
from focal.presets import PresetStore, SourcePreset, default_preset_path
from focal.windows import preset_key


class PresetStoreTests(unittest.TestCase):
    def test_round_trips_crop_mode_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "presets.json"
            store = PresetStore(path)
            preset = SourcePreset(
                crop=CropRegion.from_bounds(0.1, 0.2, 0.7, 0.8),
                tracking_mode="person",
                geometry=(50, 60, 800, 500),
                flip_horizontal=True,
                flip_vertical=True,
                person_lock=True,
            )

            store.set("exe:example", preset)

            self.assertEqual(PresetStore(path).get("exe:example"), preset)

    def test_invalid_preset_file_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "presets.json"
            path.write_text("not valid json", encoding="utf-8")

            self.assertEqual(PresetStore(path).get("anything"), SourcePreset())

    def test_non_windows_key_uses_normalized_title(self) -> None:
        with patch("focal.windows.sys.platform", "linux"):
            self.assertEqual(preset_key(1, "Example Window"), "title:example window")

    def test_default_path_on_windows_uses_appdata(self) -> None:
        with patch("focal.presets.sys.platform", "win32"), patch.dict(
            "focal.presets.os.environ", {"APPDATA": r"C:\Users\Test\AppData\Roaming"}
        ):
            expected = Path(r"C:\Users\Test\AppData\Roaming") / "Focal" / "presets.json"
            self.assertEqual(default_preset_path(), expected)

    def test_default_path_on_linux_uses_xdg_config_home(self) -> None:
        with patch("focal.presets.sys.platform", "linux"), patch.dict(
            "focal.presets.os.environ", {"XDG_CONFIG_HOME": "/home/test/.config"}
        ):
            self.assertEqual(
                default_preset_path(), Path("/home/test/.config") / "Focal" / "presets.json"
            )

    def test_default_path_on_linux_falls_back_to_home(self) -> None:
        with patch("focal.presets.sys.platform", "linux"), patch.dict(
            "focal.presets.os.environ", {}, clear=True
        ), patch("focal.presets.Path.home", return_value=Path("/home/test")):
            self.assertEqual(
                default_preset_path(), Path("/home/test/.config") / "Focal" / "presets.json"
            )


if __name__ == "__main__":
    unittest.main()
