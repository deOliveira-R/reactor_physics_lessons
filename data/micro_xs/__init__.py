"""Microscopic cross section data loading.

Loads isotope data from HDF5 files produced by ``convert_gxs_to_hdf5.py``.
"""

from pathlib import Path

from .isotope import Isotope
from .hdf5_io import load_isotope_h5

_HDF5_DIR = Path(__file__).resolve().parent


def load_isotope(name: str, temp_K: int) -> Isotope:
    """Load an isotope from the HDF5 data store."""
    h5_path = _HDF5_DIR / f"{name}.h5"
    return load_isotope_h5(h5_path, temp_K)


__all__ = ["Isotope", "load_isotope"]
