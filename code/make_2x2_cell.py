#!/usr/bin/env python3
"""
make_2x2_cell.py — Stage B setup: in-plane 2×2 replication of the 130-atom Σ5(021)
aimsgb cell -> 520 atoms. Removes the single-CSL-unit cross-section limitation
(the main size caveat in the draft) while keeping the same boundary structure,
plane positions (c-fractions 0.0 and 0.77) and analysis conventions.

Usage:
    python make_2x2_cell.py result_130atom/Ni_S5_021_relaxed.vasp
    python build_NiMoCrFe_GB.py Ni_S5_021_2x2.vasp --seeds 3
    # then MC per seed (see run_M3_remaining.sh)
"""
import sys
from ase.io import read, write

src = sys.argv[1] if len(sys.argv) > 1 else "result_130atom/Ni_S5_021_relaxed.vasp"
at = read(src)
rep = at.repeat((2, 2, 1))          # a, b are in-plane; c is the GB normal
write("Ni_S5_021_2x2.vasp", rep, format="vasp")
L = rep.get_cell().lengths()
print(f"Ni_S5_021_2x2.vasp: {len(rep)} atoms, |a|={L[0]:.2f} |b|={L[1]:.2f} "
      f"|c|={L[2]:.2f} Å (GB planes still at c-fractions 0.0 and 0.77)")
