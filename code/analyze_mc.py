#!/usr/bin/env python3
"""
analyze_mc.py — convergence check + the paper's first two results figures.

Three modes (combine freely):

1. Convergence (reviewer gate):
       python analyze_mc.py --logs *_mc_log.csv
   Plots total energy, acceptance ratio and GB-window composition vs MC
   trial for every log (overlay = independent seeds). Both E and GB
   composition must plateau before any structure is used downstream.

2. Enrichment profiles (Result 1):
       python analyze_mc.py --profile NiMoCrFe_S5_021_seed0_T900K_final.vasp \
           [more_finals.vasp ...] --random NiMoCrFe_S5_021_seed0.vasp
   Composition of Mo/Cr/Fe vs distance from the nearest GB plane
   (x = GB normal, planes at --gb-fracs), averaged over the given files,
   against the flat random-substitution baseline.

3. Warren-Cowley short-range order (Result 2):
       python analyze_mc.py --sro <finals...> --random <seed.vasp>
   First-shell alpha_ij (i,j in Ni,Mo,Cr,Fe) for GB-window atoms and for
   bulk-region atoms, equilibrated vs random. alpha<0 = i-j attraction
   (ordering), alpha>0 = repulsion (clustering of like atoms).

Outputs PNG + CSV next to the inputs (prefix mc_analysis_*).
"""
import argparse
import csv
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

SPECIES = ("Ni", "Mo", "Cr", "Fe")
COLORS = {"Ni": "0.4", "Mo": "tab:blue", "Cr": "tab:green", "Fe": "tab:red"}


def dist_from_gb(atoms, gb_fracs, axis=0):
    """Distance (Å) to nearest GB plane, measured along lattice vector `axis`
    in fractional coordinates (works for non-orthogonal / aimsgb cells)."""
    L = atoms.get_cell().lengths()[axis]
    s = atoms.get_scaled_positions()[:, axis] % 1.0
    d = np.full(len(atoms), np.inf)
    for f in gb_fracs:
        df = np.abs(s - f)
        df = np.minimum(df, 1.0 - df)
        d = np.minimum(d, df * L)
    return d


# ---------- 1. convergence ----------
def plot_convergence(log_files):
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for lf in log_files:
        rows = list(csv.DictReader(open(lf)))
        t = np.array([int(r["trial"]) for r in rows])
        E = np.array([float(r["E_eV"]) for r in rows])
        acc = np.array([float(r["acc_ratio"]) for r in rows])
        lab = lf.replace("_mc_log.csv", "")
        axes[0].plot(t, E - E[0], lw=1, label=lab)
        axes[1].plot(t, acc, lw=1)
        for s in ("Mo", "Cr", "Fe"):
            axes[2].plot(t, [100 * float(r[f"gb_{s}"]) for r in rows],
                         lw=1, color=COLORS[s],
                         label=s if lf == log_files[0] else None)
    axes[0].set_ylabel("E - E$_0$ (eV)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("acceptance ratio")
    axes[1].axhspan(0.1, 0.4, alpha=0.1, color="green")
    axes[2].set_ylabel("GB-window comp. (at%)")
    axes[2].set_xlabel("MC trial")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("mc_analysis_convergence.png", dpi=300)
    print("wrote mc_analysis_convergence.png")


# ---------- 2. enrichment profile ----------
def profile(files, random_file, gb_fracs, bin_width, axis=0):
    def binned(fnames):
        d_all, sym_all = [], []
        for fn in fnames:
            at = read(fn)
            d_all.append(dist_from_gb(at, gb_fracs, axis))
            sym_all.append(np.asarray(at.get_chemical_symbols()))
        d = np.concatenate(d_all)
        sym = np.concatenate(sym_all)
        edges = np.arange(0, d.max() + bin_width, bin_width)
        mid = 0.5 * (edges[:-1] + edges[1:])
        prof = {}
        for s in SPECIES:
            frac = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (d >= lo) & (d < hi)
                frac.append(100 * (sym[m] == s).mean() if m.sum() else np.nan)
            prof[s] = np.array(frac)
        return mid, prof

    mid, eq = binned(files)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in ("Mo", "Cr", "Fe"):
        ax.plot(mid, eq[s], "o-", color=COLORS[s], label=f"{s} (MC-equilibrated)")
    if random_file:
        _, rnd = binned([random_file])
        for s in ("Mo", "Cr", "Fe"):
            ax.plot(mid, rnd[s], "--", color=COLORS[s], alpha=0.5,
                    label=f"{s} (random)")
    ax.set_xlabel("distance from GB plane (Å)")
    ax.set_ylabel("composition (at%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("mc_analysis_enrichment_profile.png", dpi=300)
    with open("mc_analysis_enrichment_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_A"] + [f"{s}_at%" for s in SPECIES])
        for k, m in enumerate(mid):
            w.writerow([f"{m:.2f}"] + [f"{eq[s][k]:.3f}" for s in SPECIES])
    print("wrote mc_analysis_enrichment_profile.png/.csv")


# ---------- 3. Warren-Cowley SRO ----------
def warren_cowley(files, gb_fracs, gb_halfwidth, cutoff, axis=0):
    """First-shell alpha_ij = 1 - P(j|i)/c_j, GB window vs bulk region."""
    results = {}
    for region in ("GB", "bulk"):
        pair_counts = {(i, j): 0 for i in SPECIES for j in SPECIES}
        region_syms = []
        for fn in files:
            at = read(fn)
            sym = np.asarray(at.get_chemical_symbols())
            d = dist_from_gb(at, gb_fracs, axis)
            in_gb = d <= gb_halfwidth
            sel = in_gb if region == "GB" else (d >= 2 * gb_halfwidth)
            ii, jj = neighbor_list("ij", at, cutoff)
            for a_, b_ in zip(ii, jj):
                if sel[a_]:
                    pair_counts[(sym[a_], sym[b_])] += 1
            region_syms.append(sym)  # concentrations from whole cell
        sym_all = np.concatenate(region_syms)
        conc = {s: (sym_all == s).mean() for s in SPECIES}
        alpha = {}
        for i in SPECIES:
            tot_i = sum(pair_counts[(i, j)] for j in SPECIES)
            for j in SPECIES:
                p = pair_counts[(i, j)] / tot_i if tot_i else np.nan
                alpha[(i, j)] = 1 - p / conc[j] if conc[j] else np.nan
        results[region] = alpha
    return results


def plot_sro(files, random_file, gb_fracs, gb_halfwidth, cutoff, axis=0):
    eq = warren_cowley(files, gb_fracs, gb_halfwidth, cutoff, axis)
    rnd = warren_cowley([random_file], gb_fracs, gb_halfwidth, cutoff, axis) \
        if random_file else None
    pairs = [(i, j) for i, j in itertools.combinations_with_replacement(SPECIES, 2)]
    labels = [f"{i}-{j}" for i, j in pairs]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    ax.bar(x - w, [eq["GB"][p] for p in pairs], w, label="GB (MC)", color="tab:blue")
    ax.bar(x, [eq["bulk"][p] for p in pairs], w, label="bulk (MC)", color="tab:cyan")
    if rnd:
        ax.bar(x + w, [rnd["GB"][p] for p in pairs], w,
               label="GB (random)", color="0.6")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel(r"Warren-Cowley $\alpha_{ij}$ (1st shell)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("mc_analysis_sro.png", dpi=300)
    with open("mc_analysis_sro.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["pair", "alpha_GB_MC", "alpha_bulk_MC", "alpha_GB_random"])
        for p, lab in zip(pairs, labels):
            wcsv.writerow([lab, f"{eq['GB'][p]:.4f}", f"{eq['bulk'][p]:.4f}",
                           f"{rnd['GB'][p]:.4f}" if rnd else ""])
    print("wrote mc_analysis_sro.png/.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[], help="*_mc_log.csv files")
    ap.add_argument("--profile", nargs="*", default=[],
                    help="equilibrated .vasp files for enrichment profile")
    ap.add_argument("--sro", nargs="*", default=[],
                    help="equilibrated .vasp files for Warren-Cowley SRO")
    ap.add_argument("--random", default=None,
                    help="random-substitution baseline .vasp")
    ap.add_argument("--gb-fracs", type=float, nargs="*", default=[0.0, 0.5])
    ap.add_argument("--axis", type=int, default=0,
                    help="lattice-vector index of the GB normal (0=a, 1=b, 2=c); "
                         "aimsgb cells usually have the GB normal along c -> --axis 2")
    ap.add_argument("--gb-halfwidth", type=float, default=4.0)
    ap.add_argument("--bin-width", type=float, default=2.0)
    ap.add_argument("--cutoff", type=float, default=3.0,
                    help="1st-shell cutoff (Å); fcc Ni d1 = 2.49 Å")
    a = ap.parse_args()

    if a.logs:
        plot_convergence(a.logs)
    if a.profile:
        profile(a.profile, a.random, a.gb_fracs, a.bin_width, a.axis)
    if a.sro:
        plot_sro(a.sro, a.random, a.gb_fracs, a.gb_halfwidth, a.cutoff, a.axis)
    if not (a.logs or a.profile or a.sro):
        ap.print_help()


if __name__ == "__main__":
    main()
