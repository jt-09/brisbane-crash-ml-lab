"""Streamlit app import smoke test."""

from __future__ import annotations


def test_streamlit_app_imports() -> None:
    from crashlab.app import streamlit_app

    assert hasattr(streamlit_app, "main")
    assert hasattr(streamlit_app, "PAGES")
    assert len(streamlit_app.PAGES) == 7
