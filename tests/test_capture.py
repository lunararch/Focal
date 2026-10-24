from types import SimpleNamespace
import unittest

import numpy as np

from focal.capture import (
    BALANCED_PREVIEW,
    LOW_POWER_PREVIEW,
    WindowsCaptureThread,
    _copy_frame_pixels,
)


class CaptureFrameTests(unittest.TestCase):
    def test_copies_windows_capture_frame_buffer(self) -> None:
        source = np.zeros((2, 3, 4), dtype=np.uint8)
        frame = SimpleNamespace(frame_buffer=source)

        copied = _copy_frame_pixels(frame)
        source[0, 0, 0] = 255

        self.assertEqual(copied.shape, (2, 3, 4))
        self.assertEqual(copied[0, 0, 0], 0)

    def test_supports_legacy_numpy_converter(self) -> None:
        source = np.ones((2, 2, 4), dtype=np.uint8)
        frame = SimpleNamespace(to_numpy=lambda copy=False: source)

        copied = _copy_frame_pixels(frame)
        self.assertTrue(np.array_equal(copied, source))
        self.assertIsNot(copied, source)

    def test_copies_raw_pipewire_numpy_frame(self) -> None:
        # pipewire-capture returns BGRA pixels directly, unlike windows-capture
        # which exposes a wrapper with ``frame_buffer``.
        source = np.zeros((2, 3, 4), dtype=np.uint8)

        copied = _copy_frame_pixels(source)
        source[0, 0, 0] = 255

        self.assertEqual(copied.shape, (2, 3, 4))
        self.assertEqual(copied[0, 0, 0], 0)

    def test_capture_thread_keeps_selected_window_handle(self) -> None:
        capture = WindowsCaptureThread(12345)

        self.assertEqual(capture.window_hwnd, 12345)
        self.assertEqual(capture._preview_settings, BALANCED_PREVIEW)

    def test_capture_thread_applies_updated_preview_settings(self) -> None:
        capture = WindowsCaptureThread(12345)

        capture.set_preview_settings(LOW_POWER_PREVIEW)

        self.assertEqual(capture._preview_settings, LOW_POWER_PREVIEW)

    def test_bounds_large_frames_for_preview(self) -> None:
        source = np.zeros((8, 6, 4), dtype=np.uint8)
        frame = SimpleNamespace(frame_buffer=source)

        copied = _copy_frame_pixels(frame, max_width=4, max_height=4)

        self.assertEqual(copied.shape, (4, 3, 4))


if __name__ == "__main__":
    unittest.main()
