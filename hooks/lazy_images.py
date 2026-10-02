"""MkDocs hook: load content images lazily.

Every figure exists twice, a light and a dark SVG, and the theme hides one of them. Lazy images that
are not displayed are not fetched, so a page downloads only the figures of the active colour scheme,
and only when they come near the screen.
"""

from __future__ import annotations

import re

_IMG = re.compile(r"<img(?![^>]*\sloading=)")


def on_page_content(html, **kwargs):
    return _IMG.sub('<img loading="lazy" decoding="async"', html)
