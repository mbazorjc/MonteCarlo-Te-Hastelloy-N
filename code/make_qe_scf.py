#!/usr/bin/env python3
"""
make_qe_scf.py — build FORCE-FREE single-point QE SCF inputs from the MACE-relaxed
geometries in relaxed_tiny/ (written by mace_energies.py).

Why: your GPU run converged the SCF fine and only OOM'd at the FORCE step
(addusforce, the ultrasoft augmentation term, ~5.6 GB extra). A parity of ENERGIES
does not need forces or a QE relaxation. Both codes evaluate the SAME (MACE-relaxed)
geometry, so the energy differences are directly comparable and QE finishes the instant
SCF converges — no addusforce, no OOM.

Run AFTER mace_energies.py:
    python make_qe_scf.py                       # ecutwfc 45 / ecutrho 360 (default)
    python make_qe_scf.py --ecutwfc 30 --ecutrho 240   # tightest GPU memory
Creates qe_crosscheck/qeS_*.in  (S = single-point scf). Then: bash run_qe_docker.sh
Needs: ASE. Pseudos in qe_crosscheck/pseudo (Ni,Te,Cr,Mo .pbe-n-rrkjus_psl.1.0.0.UPF).
"""
import argparse, os
from ase.io import read, write

ap = argparse.ArgumentParser()
ap.add_argument("--ecutwfc", type=float, default=45)
ap.add_argument("--ecutrho", type=float, default=360)
ap.add_argument("--src", default="relaxed_tiny")
a = ap.parse_args()
QE = "qe_crosscheck"

pseudos = {"Ni":"Ni.pbe-n-rrkjus_psl.1.0.0.UPF","Te":"Te.pbe-n-rrkjus_psl.1.0.0.UPF",
           "Cr":"Cr.pbe-n-rrkjus_psl.1.0.0.UPF","Mo":"Mo.pbe-n-rrkjus_psl.1.0.0.UPF",
           "Fe":"Fe.pbe-spn-kjpaw_psl.0.2.1.UPF"}   # M3: PAW spn (in pseudo/); USPP+PAW mix is fine in QE
mag = {"Ni":0.6,"Cr":0.4,"Mo":0.0,"Te":0.0,"Fe":0.7}  # M3: Fe strongly polarized in Ni matrix
kp  = lambda n: (2,2,2) if n.startswith("bulk") else (1,3,3)

# calculation='scf', tprnfor/tstress OFF -> no force call -> no addusforce OOM
base = dict(
    control=dict(calculation="scf", restart_mode="from_scratch",
                 tprnfor=False, tstress=False, pseudo_dir="./pseudo", outdir="./tmp"),
    system=dict(ecutwfc=a.ecutwfc, ecutrho=a.ecutrho, occupations="smearing",
                smearing="mp", degauss=0.01, nspin=2),
    electrons=dict(conv_thr=1e-6, mixing_beta=0.2, mixing_mode="local-TF", electron_maxstep=200),
)

if not os.path.isdir(a.src):
    raise SystemExit(f"{a.src}/ not found — run mace_energies.py first to create it.")
n = 0
for c in sorted(os.listdir(a.src)):
    if not c.endswith(".vasp"): continue
    name = c[:-5]; at = read(f"{a.src}/{c}"); syms = at.get_chemical_symbols()
    at.set_initial_magnetic_moments([mag.get(s,0.0) for s in syms]); sp = sorted(set(syms))
    inp = {k:dict(v) for k,v in base.items()}; inp["control"]["prefix"] = name
    write(f"{QE}/qeS_{name}.in", at, format="espresso-in", input_data=inp,
          pseudopotentials={s:pseudos[s] for s in sp}, kpts=kp(name)); n += 1
print(f"wrote {n} force-free SCF inputs {QE}/qeS_*.in  (ecutwfc={a.ecutwfc}, ecutrho={a.ecutrho})")
print("next: cd qe_crosscheck && bash run_qe_docker.sh")
