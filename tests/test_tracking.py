import unittest

import numpy as np

from focal.crop import CropRegion
from focal.tracking import ContentFollower, TrackingSettings, _translate_crop


class ContentFollowerTests(unittest.TestCase):
    def test_translated_features_move_crop_after_dead_zone(self) -> None:
        rng = np.random.default_rng(42)
        first = np.zeros((120, 160, 4), dtype=np.uint8)
        first[30:80, 40:90, :3] = rng.integers(
            0, 256, size=(50, 50, 3), dtype=np.uint8
        )
        first[:, :, 3] = 255

        second = np.zeros_like(first)
        second[30:80, 60:110, :3] = first[30:80, 40:90, :3]
        second[:, :, 3] = 255

        crop = CropRegion.from_bounds(0.25, 0.25, 0.5625, 2 / 3)
        follower = ContentFollower(TrackingSettings(smoothing=0.5, dead_zone=0.1))

        self.assertIsNone(follower.update(first, crop))
        updated = follower.update(second, crop)

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertGreater(updated.left, crop.left)
        self.assertAlmostEqual(updated.top, crop.top, places=2)

    def test_translation_clamps_crop_at_source_edge(self) -> None:
        crop = CropRegion.from_bounds(0.7, 0.2, 1.0, 0.8)

        translated = _translate_crop(crop, 0.5, 0.5)
        self.assertAlmostEqual(translated.left, 0.7)
        self.assertAlmostEqual(translated.top, 0.4)
        self.assertAlmostEqual(translated.right, 1.0)
        self.assertAlmostEqual(translated.bottom, 1.0)


if __name__ == "__main__":
    unittest.main()
