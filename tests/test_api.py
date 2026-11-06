"""The planning structures are imported from planning_structs, not from rosame (clean break in 0.2.0)."""
import importlib

import pytest

import rosame

MOVED = ["Domain", "Instance", "ActionModel", "ActionSchema", "Predicate", "Type", "Trace"]


def test_structures_are_not_exported_by_rosame():
    for name in MOVED:
        assert not hasattr(rosame, name), f"rosame.{name} should not exist since 0.2.0"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rosame.planning_structs")


def test_structures_come_from_planning_structs():
    import planning_structs
    for name in MOVED:
        assert hasattr(planning_structs, name)
