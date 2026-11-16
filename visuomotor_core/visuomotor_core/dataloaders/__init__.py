# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
dataloaders module alias for visuomotor_core.
Provides backward and forward compatibility with visuomotor_core.dataset.
"""
from ..dataset.zarr_dataset import ZarrDataset

__all__ = ["ZarrDataset"]
