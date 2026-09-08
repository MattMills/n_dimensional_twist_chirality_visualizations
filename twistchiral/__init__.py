"""twistchiral: twist-chirality interfaces of interfering n-dimensional wave volumes.

The interference of N localised wave volumes in R^n is treated as a map
``x -> [Psi(x)] in CP^{N-1}`` (the relative amplitudes and phases at each
point).  Its pulled-back quantum geometry -- Berry connection A, quantum
metric g, Berry curvature (twist) F -- and the ladder of Chern-Simons /
Chern forms ``A, F, A^F, F^F, ...`` give an intrinsic, phase-invariant
notion of twist and chirality that is a pseudoscalar only when the ladder
reaches degree n, and is otherwise a multi-component, direction-valued
chirality.
"""
from .exterior import ExteriorAlgebra
from .waves import WaveSystem, WaveVolume
from .qgt import quantum_geometry, chirality_ladder, rotation_planes, twist_rank
from .interface import sample_interface, sample_junction, project_to_interface, project_to_junction, visibility
from .analysis import (geometry_at, analytic_two_volume_chirality, screw_prediction, direction_spectrum,
                       localization_profile, relative_phase_invariance, mirror_comparison, ladder_existence,
                       interface_chirality_summary)
from .systems import twisted_pair, cluster, wave_set
from . import exact, geometry, highdim, slices, style, viz

__all__ = [
    "ExteriorAlgebra", "WaveSystem", "WaveVolume", "quantum_geometry", "chirality_ladder", "rotation_planes",
    "twist_rank", "sample_interface", "sample_junction", "project_to_interface", "project_to_junction",
    "visibility", "geometry_at", "analytic_two_volume_chirality", "screw_prediction", "direction_spectrum",
    "localization_profile", "relative_phase_invariance", "mirror_comparison", "ladder_existence",
    "interface_chirality_summary", "twisted_pair", "cluster", "wave_set", "exact", "geometry", "highdim", "slices", "style", "viz",
]
__version__ = "0.1.0"
