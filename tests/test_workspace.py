import unittest

from PySide6.QtCore import Qt

from focal.ui import _split_details
from focal.workspace import LayoutMode, pane_rows


class WorkspaceLayoutTests(unittest.TestCase):
    def test_row_and_column_keep_all_tiles_in_one_splitter(self) -> None:
        tiles = ["a", "b", "c"]

        self.assertEqual(pane_rows(tiles, LayoutMode.ROW), [tiles])
        self.assertEqual(pane_rows(tiles, LayoutMode.COLUMN), [tiles])

    def test_top_pair_places_remaining_tiles_below_pair(self) -> None:
        self.assertEqual(
            pane_rows(["a", "b", "c", "d"], LayoutMode.TOP_PAIR),
            [["a", "b"], ["c"], ["d"]],
        )

    def test_grid_groups_tiles_into_horizontal_pairs(self) -> None:
        self.assertEqual(
            pane_rows([1, 2, 3, 4, 5], LayoutMode.GRID),
            [[1, 2], [3, 4], [5]],
        )

    def test_empty_workspace_has_no_panes(self) -> None:
        self.assertEqual(pane_rows([], LayoutMode.GRID), [])

    def test_directional_splits_place_the_new_tile_on_requested_side(self) -> None:
        self.assertEqual(_split_details("left"), (Qt.Orientation.Horizontal, True))
        self.assertEqual(_split_details("right"), (Qt.Orientation.Horizontal, False))
        self.assertEqual(_split_details("above"), (Qt.Orientation.Vertical, True))
        self.assertEqual(_split_details("below"), (Qt.Orientation.Vertical, False))

    def test_unknown_split_direction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _split_details("diagonal")


if __name__ == "__main__":
    unittest.main()
