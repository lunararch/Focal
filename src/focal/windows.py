from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """A selectable top-level Windows window."""

    hwnd: int
    title: str


# Special source-combo values used on Linux, where windows are selected through
# the XDG ScreenCast portal picker instead of a Win32 enumeration.
PICK_WINDOW = "__focal_pick_window__"
STOP_MIRRORING = "__focal_stop_mirroring__"


def is_portal_session() -> bool:
    """Return True when the app should use the Wayland portal window picker.

    On core Windows this returns False (hwnd-based enumeration is used there).
    The check intentionally is not tied to DISPLAY/WAYLAND_DISPLAY so the UI can
    prepare the picker flow before the portal is truly reachable.
    """
    return sys.platform != "win32"


def preset_key(hwnd: int, title: str) -> str:
    """Return a stable per-application key, falling back to the window title."""
    if sys.platform != "win32":
        return f"title:{title.casefold()}"
    try:
        import win32api
        import win32con
        import win32process

        _thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        process = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
            False,
            process_id,
        )
        try:
            executable = win32process.GetModuleFileNameEx(process, 0)
        finally:
            win32api.CloseHandle(process)
        return f"exe:{os.path.normcase(executable)}"
    except Exception:
        return f"title:{title.casefold()}"


def list_windows(exclude_hwnd: int | None = None) -> list[WindowInfo]:
    """Return visible, titled top-level windows sorted by title.

    The Win32 import is intentionally lazy so the rest of the package can be
    imported for documentation or static checks on non-Windows machines.
    """
    if sys.platform != "win32":
        return []

    import win32gui

    windows: list[WindowInfo] = []

    def collect(hwnd: int, _unused: object) -> None:
        if exclude_hwnd is not None and hwnd == exclude_hwnd:
            return
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        if right <= left or bottom <= top:
            return

        windows.append(WindowInfo(hwnd=hwnd, title=title))

    win32gui.EnumWindows(collect, None)
    return sorted(windows, key=lambda window: window.title.casefold())
