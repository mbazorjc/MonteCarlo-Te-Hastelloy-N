#!/usr/bin/env python3
"""
build_NiMoCrFe_GB.py — Manuscript 3 starting structures.

Extends Manuscript 1's build_NiCrMo_GB.py to the full Hastelloy-N quaternary:
Ni-16Mo-7Cr-4Fe wt%  ~=  Ni-10Mo-8Cr-4Fe at%.

Turns a relaxed pure-Ni grain boundary into several RANDOM quaternary
realizations (seeds). These are (a) the t=0 inputs to mc_swap_equilibrate.py
and (b) the "random substitution" baseline that the equilibrated boundaries
are compared against (the paper's central contrast).

Usage:
    python build_NiMoCrFe_GB.py ../Manuscript1_Te/TeNi_sim/Ni_S5_021_relaxed.vasp \
        --mo 0.10 --cr 0.08 --fe 0.04 --seeds 5
Outputs NiMoCrFe_S5_021_seed0.vasp ... and prints achieved compositions.
"""
import argparse
import numpy as np
from ase.io import read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure", help="relaxed pure-Ni GB (e.g. Ni_S5_021_relaxed.vasp)")
    ap.add_argument("--mo", type=float, default=0.10, help="Mo atomic fraction")
    ap.add_argument("--cr", type=float, default=0.08, help="Cr atomic fraction")
    ap.add_argument("--fe", type=float, default=0.04, help="Fe atomic fraction")
    ap.add_argument("--seeds", type=int, default=5, help="number of random realizations")
    ap.add_argument("--protect", type=int, nargs="*", default=[],
                    help="atom indices to keep as Ni (e.g. a Te test site)")
    a = ap.parse_args()

    base = read(a.structure)
    N = len(base)
    nMo = int(round(a.mo * N))
    nCr = int(round(a.cr * N))
    nFe = int(round(a.fe * N))
    pool = [i for i in range(N) if i not in set(a.protect)]
    tag = a.structure.rsplit('.', 1)[0].split('/')[-1].split('\\')[-1]
    tag = tag.replace('Ni_', '').replace('_relaxed', '')

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        pick = rng.choice(pool, size=nMo + nCr + nFe, replace=False)
        mo_idx = pick[:nMo]
        cr_idx = pick[nMo:nMo + nCr]
        fe_idx = pick[nMo + nCr:]
        at = base.copy()
        s = at.get_chemical_symbols()
        for i in mo_idx:
            s[i] = 'Mo'
        for i in cr_idx:
            s[i] = 'Cr'
        for i in fe_idx:
            s[i] = 'Fe'
        at.set_chemical_symbols(s)
        syms = at.get_chemical_symbols()
        comp = {e: 100 * syms.count(e) / N for e in ('Ni', 'Mo', 'Cr', 'Fe')}
        out = f"NiMoCrFe_{tag}_seed{seed}.vasp"
        write(out, at, format='vasp')
        print(f"{out}: Ni {comp['Ni']:.1f} / Mo {comp['Mo']:.1f} / "
              f"Cr {comp['Cr']:.1f} / Fe {comp['Fe']:.1f} at%  ({N} atoms)")

    print("\nNext: python mc_swap_equilibrate.py NiMoCrFe_S5_021_seed0.vasp --temp 900 --steps 20000")


if __name__ == "__main__":
    main()
