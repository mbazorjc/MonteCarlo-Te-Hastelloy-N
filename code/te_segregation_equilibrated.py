#!/usr/bin/env python3
"""
te_segregation_equilibrated.py — Stage A of the remaining M3 runs (the punchline).

Te segregation spectrum on MC-EQUILIBRATED Ni-Mo-Cr-Fe Σ5 boundaries, and on the
matching RANDOM-substitution baselines, so the two spectra can be compared directly
(same cells, same sites, same potential as Manuscript 1's method).

Definition (identical to Manuscript 1, substitutional, same supercell):
    E_seg(site) = E(Te @ GB site) - E(Te @ bulk reference site)
Te replaces a *Ni* atom in both cases (keeps the alloy content identical across
sites, so spectra are comparable between configurations). Negative = Te prefers GB.

Geometry-aware: distances to the GB planes are measured along any lattice vector
(--axis, fractional coordinates), so the monoclinic 130-atom aimsgb cell
(GB normal = c, planes at 0.0 and 0.77) is handled correctly.

Per Te site the CSV also records the first-shell chemistry (n_Ni, n_Mo, n_Cr, n_Fe
within 3.0 Å of the site BEFORE substitution) — this is what links the spectrum to
the deep-trap analysis of Manuscript 1 (are deep traps still solute-rich motifs?).

Resumable: reruns skip (structure, site) pairs already present in the output CSV.

Usage (from the folder holding the configurations):
    # equilibrated (10 configs):
    python te_segregation_equilibrated.py \
        NiMoCrFe_S5_021_seed*_T900K_final.vasp NiMoCrFe_S5_021_seed*_T900K_snap_1000.vasp \
        --axis 2 --gb-fracs 0.0 0.77 --out te_seg_equilibrated.csv
    # random baseline (5 configs):
    python te_segregation_equilibrated.py NiMoCrFe_S5_021_seed?.vasp \
        --axis 2 --gb-fracs 0.0 0.77 --out te_seg_random.csv

Cost: one FIRE relax per site; ~15-20 Ni sites/config in the GB window of the
130-atom cell -> a few hours per 10-config set on a 12 GB GPU.
"""
import argparse
import csv
import os

import numpy as np
from ase.io import read

KB_EV = 8.617333262e-5


def get_calc(device=None):
    from mace.calculators import mace_mp
    kwargs = dict(model="medium", dispersion=False, default_dtype="float64")
    if device:
        kwargs["device"] = device
    return mace_mp(**kwargs)


def fire_relax_E(atoms, calc, fmax=0.03, steps=300):
    from ase.optimize import FIRE
    a = atoms.copy()
    a.calc = calc
    FIRE(a, logfile=None).run(fmax=fmax, steps=steps)
    return a.get_potential_energy()


def dist_from_gb(atoms, gb_fracs, axis):
    L = atoms.get_cell().lengths()[axis]
    s = atoms.get_scaled_positions()[:, axis] % 1.0
    d = np.full(len(atoms), np.inf)
    for f in gb_fracs:
        df = np.abs(s - f)
        df = np.minimum(df, 1.0 - df)
        d = np.minimum(d, df * L)
    return d


def first_shell(atoms, idx, cutoff=3.0):
    from ase.neighborlist import neighbor_list
    ii, jj = neighbor_list("ij", atoms, cutoff)
    sym = np.asarray(atoms.get_chemical_symbols())
    nb = sym[jj[ii == idx]]
    return {s: int((nb == s).sum()) for s in ("Ni", "Mo", "Cr", "Fe")}


def substitute(atoms, idx, sym="Te"):
    a = atoms.copy()
    s = a.get_chemical_symbols()
    s[int(idx)] = sym
    a.set_chemical_symbols(s)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structures", nargs="+")
    ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--gb-fracs", type=float, nargs="*", default=[0.0, 0.77])
    ap.add_argument("--gb-halfwidth", type=float, default=3.0,
                    help="GB site window (Å) for Te candidate sites")
    ap.add_argument("--bulk-min", type=float, default=8.0,
                    help="bulk reference: Ni site farthest from both planes, must be > this (Å)")
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="te_seg_results.csv")
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["structure"], int(r["site"])))
        f = open(a.out, "a", newline="")
        w = csv.writer(f)
    else:
        f = open(a.out, "w", newline="")
        w = csv.writer(f)
        w.writerow(["structure", "site", "d_gb_A", "nNi", "nMo", "nCr", "nFe",
                    "E_TeGB_eV", "E_Tebulk_eV", "E_seg_eV"])

    calc = get_calc(a.device)

    for fn in a.structures:
        at = read(fn)
        name = os.path.basename(fn)
        sym = np.asarray(at.get_chemical_symbols())
        d = dist_from_gb(at, a.gb_fracs, a.axis)

        # bulk reference: the Ni atom farthest from both boundaries
        ni = np.where(sym == "Ni")[0]
        ref = ni[np.argmax(d[ni])]
        if d[ref] < a.bulk_min:
            print(f"WARNING {name}: farthest Ni site is only {d[ref]:.1f} Å from a "
                  f"boundary (< {a.bulk_min}); cell has no clean bulk region")
        E_bulk = fire_relax_E(substitute(at, ref), calc, a.fmax, a.steps)
        print(f"{name}: bulk ref site {ref} (d={d[ref]:.1f} Å), "
              f"E(Te@bulk) = {E_bulk:.4f} eV")

        sites = [i for i in ni if d[i] <= a.gb_halfwidth]
        print(f"{name}: {len(sites)} Ni GB sites within ±{a.gb_halfwidth} Å")
        for i in sites:
            if (name, int(i)) in done:
                continue
            shell = first_shell(at, i)
            E_gb = fire_relax_E(substitute(at, i), calc, a.fmax, a.steps)
            E_seg = E_gb - E_bulk
            w.writerow([name, int(i), f"{d[i]:.2f}", shell["Ni"], shell["Mo"],
                        shell["Cr"], shell["Fe"], f"{E_gb:.6f}", f"{E_bulk:.6f}",
                        f"{E_seg:.6f}"])
            f.flush()
            print(f"  site {i:4d}  d={d[i]:4.1f}  shell Ni{shell['Ni']}/Mo{shell['Mo']}"
                  f"/Cr{shell['Cr']}/Fe{shell['Fe']}  E_seg = {E_seg:+.3f} eV")
    f.close()
    print(f"\nDone -> {a.out}")
    print("Return this CSV (equilibrated AND random) for spectrum comparison "
          "against Manuscript 1.")


if __name__ == "__main__":
    main()
