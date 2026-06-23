"""Version string sanity checks."""

import re

from crashlab import __version__


def test_version_is_pep440_like() -> None:
    assert re.match(r"^\d+\.\d+\.\d+", __version__)
