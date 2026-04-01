#!/usr/bin/env python3
"""Convert all GENDF (.GXS) files to HDF5 format.

Reads from ``01.Micro.XS.421g/*.GXS`` and writes one ``.h5`` file per
element into ``data/micro_xs/``.  Each HDF5 file contains all temperatures.
"""

from pathlib import Path

import h5py

from data.micro_xs.gendf import convert_gxs, _GXS_DIR
from data.micro_xs.hdf5_io import save_isotope

OUTPUT_DIR = Path(__file__).resolve().parent  # same directory as this script


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gxs_files = sorted(_GXS_DIR.glob("*.GXS"))
    print(f"Found {len(gxs_files)} GXS files in {_GXS_DIR}\n")

    for gxs_path in gxs_files:
        name = gxs_path.stem  # e.g. "H_001", "U_235"
        h5_path = OUTPUT_DIR / f"{name}.h5"

        print(f"Converting {name}:")
        isotopes = convert_gxs(name)

        with h5py.File(h5_path, "w") as f:
            f.attrs["element"] = name
            f.attrs["n_temperatures"] = len(isotopes)
            for iso in isotopes:
                save_isotope(iso, f)
                temp_K = int(round(iso.temp))
                print(f"    {temp_K}K written")

        size_mb = h5_path.stat().st_size / 1e6
        print(f"  -> {h5_path.name} ({size_mb:.1f} MB, {len(isotopes)} temperatures)\n")

    print("Done. All HDF5 files in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
