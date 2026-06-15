"""Maintenance Wizard backend package.

The package avoids importing the full agent at module import time so lightweight
submodules such as ``backend.llm`` can be patched before pandas/model code loads.
"""

__all__ = ["MaintenanceWizard"]


def __getattr__(name):
    if name == "MaintenanceWizard":
        from .agent import MaintenanceWizard

        return MaintenanceWizard
    raise AttributeError(name)
