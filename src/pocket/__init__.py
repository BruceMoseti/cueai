"""Pocket Physics — physics-informed billiards simulation + ML trajectory prediction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pocket-physics")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"
