import unittest

from focal.crop import CropRegion


class CropRegionTests(unittest.TestCase):
    def test_bounds_are_normalized_and_clamped(self) -> None:
        crop = CropRegion.from_bounds(0.8, 0.7, -0.2, 1.4)
        self.assertEqual(crop, CropRegion(0.0, 0.7, 0.8, 1.0))

    def test_pixel_bounds_are_half_open(self) -> None:
        crop = CropRegion.from_bounds(0.25, 0.1, 0.75, 0.6)
        self.assertEqual(crop.pixel_bounds(1000, 500), (250, 50, 750, 300))

    def test_tiny_regions_still_produce_one_pixel(self) -> None:
        crop = CropRegion.from_bounds(0.999, 0.999, 1.0, 1.0)
        self.assertEqual(crop.pixel_bounds(100, 100), (99, 99, 100, 100))


if __name__ == "__main__":
    unittest.main()
