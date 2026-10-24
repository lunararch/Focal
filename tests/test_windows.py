import unittest
from unittest.mock import patch

from focal.ui import _order_windows
from focal.windows import WindowInfo, list_windows


class WindowTests(unittest.TestCase):
    def test_window_info_is_value_like(self) -> None:
        self.assertEqual(WindowInfo(42, "Example"), WindowInfo(42, "Example"))

    def test_orders_recently_viewed_windows_first(self) -> None:
        windows = [
            WindowInfo(1, "Alpha"),
            WindowInfo(2, "Bravo"),
            WindowInfo(3, "Charlie"),
        ]

        self.assertEqual(
            _order_windows(windows, [3, 1]),
            [WindowInfo(3, "Charlie"), WindowInfo(1, "Alpha"), WindowInfo(2, "Bravo")],
        )

    def test_non_windows_enumeration_is_safe(self) -> None:
        with patch("focal.windows.sys.platform", "linux"):
            self.assertEqual(list_windows(), [])


if __name__ == "__main__":
    unittest.main()
