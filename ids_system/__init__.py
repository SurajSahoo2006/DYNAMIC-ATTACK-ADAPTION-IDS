"""
ids_system package
==================
Consolidated intrusion detection and tactical response pipeline.
Unifies all components from Phases 1 through 12 into a production-ready system.
"""

from ids_system.pipeline import IDSPipeline

__all__ = ["IDSPipeline"]
