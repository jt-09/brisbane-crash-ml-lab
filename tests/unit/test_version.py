"""Bootstrap sanity tests — expanded by the autonomous build agent."""

from crashlab import __version__


def test_version_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts)
