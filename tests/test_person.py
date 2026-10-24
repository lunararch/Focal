import unittest

from focal.crop import CropRegion
from focal.person import PersonBox, PersonFollower
from focal.tracking import TrackingSettings


class PersonFollowerTests(unittest.TestCase):
    def test_initial_detection_nearest_crop_moves_crop_towards_person(self) -> None:
        crop = CropRegion.from_bounds(0.2, 0.2, 0.4, 0.6)
        follower = PersonFollower(TrackingSettings(smoothing=0.5, dead_zone=0.1))
        near_crop = PersonBox(25, 30, 45, 80, 0.9)
        other_person = PersonBox(75, 20, 95, 90, 0.95)

        updated = follower.update([other_person, near_crop], crop, 100, 100)

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertGreater(updated.left, crop.left)
        self.assertLess(updated.left, 0.5)

    def test_following_prefers_person_near_previous_detection(self) -> None:
        crop = CropRegion.from_bounds(0.2, 0.2, 0.4, 0.6)
        follower = PersonFollower(TrackingSettings(smoothing=1.0, dead_zone=0.0))
        selected = PersonBox(25, 30, 45, 80, 0.9)
        follower.update([selected], crop, 100, 100)

        nearby = PersonBox(35, 30, 55, 80, 0.7)
        distant = PersonBox(80, 20, 99, 90, 0.99)
        updated = follower.update([distant, nearby], crop, 100, 100)

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertLess(updated.left, 0.5)

    def test_person_lock_holds_crop_when_only_another_person_is_detected(self) -> None:
        crop = CropRegion.from_bounds(0.2, 0.2, 0.4, 0.6)
        follower = PersonFollower(TrackingSettings(smoothing=1.0, dead_zone=0.0))
        locked_person = PersonBox(25, 30, 45, 80, 0.9)
        follower.update([locked_person], crop, 100, 100)
        follower.set_person_lock(True)

        other_person = PersonBox(75, 20, 95, 90, 0.99)

        self.assertIsNone(follower.update([other_person], crop, 100, 100))

    def test_person_lock_follows_a_matching_detection(self) -> None:
        crop = CropRegion.from_bounds(0.2, 0.2, 0.4, 0.6)
        follower = PersonFollower(TrackingSettings(smoothing=1.0, dead_zone=0.0))
        follower.update([PersonBox(25, 30, 45, 80, 0.9)], crop, 100, 100)
        follower.set_person_lock(True)

        updated = follower.update([PersonBox(30, 30, 50, 80, 0.9)], crop, 100, 100)

        self.assertIsNotNone(updated)

    def test_ignores_low_confidence_persons(self) -> None:
        follower = PersonFollower()
        crop = CropRegion.from_bounds(0.2, 0.2, 0.4, 0.6)

        self.assertIsNone(
            follower.update([PersonBox(20, 20, 40, 80, 0.2)], crop, 100, 100)
        )


if __name__ == "__main__":
    unittest.main()
