"""Plot the per-point data in ``tise2_2_00027_master.hdf`` directly with h5py.

This is a NeXus master file written by ``MyNXWriter`` (see
``callbacks/nexus_data_file_writer.py``), not a databroker catalog entry, so
it is read directly rather than through ``databroker``.

Caveat: the scanned motor (``psic_eta``) and the hinted Lambda 250K detector
field (``lambda250k_stats1_total``) are NOT present as datasets under
``entry/data`` in this file -- only the scaler and ah2700a scalar channels
are. So there is no true per-point motor readback to plot against; the point
index is used as the x-axis instead, and this is called out explicitly in the
printed report.
"""

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = "tise2_2_00027_master.hdf"
OUT_PATH = "tise2_2_00027_plot.png"


def main():
    with h5py.File(FILE_PATH, "r") as f:
        data_group = f["entry/data"]
        fields = sorted(data_group.keys())

        columns = {field: np.asarray(data_group[field][()], dtype=float) for field in fields}

        meta = f["entry/instrument/bluesky/metadata"]

        def _meta(name, default=None):
            if name not in meta:
                return default
            value = meta[name][()]
            if isinstance(value, bytes):
                value = value.decode()
            return value

        plan_name = _meta("plan_name")
        motors = _meta("motors")
        num_points = _meta("num_points")

    n = len(next(iter(columns.values())))
    x = np.arange(n)

    fig, axes = plt.subplots(len(fields), 1, figsize=(9, 2.2 * len(fields)), sharex=True)
    for ax, field in zip(axes, fields):
        ax.plot(x, columns[field], marker="o", markersize=2, linewidth=1)
        ax.set_ylabel(field, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("point index (no per-point motor readback in this file)")
    fig.suptitle(f"{FILE_PATH}  (plan={plan_name}, motors={motors}, num_points={num_points})")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)

    print(f"plan_name={plan_name}  motors={motors}  num_points={num_points}")
    print(f"entry/data fields ({n} points each): {fields}")
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
