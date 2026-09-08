"""Run every experiment and write results/summary.json."""
from common import RES_DIR, dump  # noqa: F401

import two_volumes_3d
import two_volumes_4d
import dimension_sweep
import n_volumes

if __name__ == "__main__":
    summary = {
        "A_two_volumes_3d": two_volumes_3d.run(),
        "B_two_volumes_4d": two_volumes_4d.run(),
        "C_dimension_sweep": dimension_sweep.run(),
        "D_n_volumes": n_volumes.run(),
    }
    dump(summary, "summary.json")
