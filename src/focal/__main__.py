from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui import FocalWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Focal")
    app.setApplicationDisplayName("Focal")

    window = FocalWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
