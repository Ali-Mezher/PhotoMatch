"""
Interface package — the local kiosk app students and staff use to
search an event and review matches.
"""

from .app import PhotoMatchApp, launch_app

__all__ = ["PhotoMatchApp", "launch_app"]
