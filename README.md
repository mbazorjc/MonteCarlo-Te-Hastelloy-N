# MonteCarlo-Te-Hastelloy-N
Codes for Equilibrium grain-boundary chemistry suppresses deep tellurium traps in Hastelloy-N paper
# Equilibrium grain-boundary chemistry and Te trapping in Hastelloy-N

This repository contains the data and computational workflows supporting the
study:

**Equilibrium grain-boundary chemistry suppresses deep tellurium traps in
Hastelloy-N**

## Overview

Canonical swap Monte Carlo and the MACE-MP-0 foundation interatomic potential
were used to compare random and chemically equilibrated Ni-10Mo-8Cr-4Fe
Sigma5(021) grain boundaries.

Equilibration produces Ni-Mo short-range order and Mo enrichment at the
boundary. These changes remove the solute-poor Ni environments responsible
for the deepest Te traps in the random-alloy model. Work-of-separation
calculations show that protection arises primarily from reduced Te occupancy,
not from a smaller cohesion penalty after Te reaches the boundary.

Selected energy differences were cross-checked using spin-polarized PBE-DFT.

## Repository contents

- `code/`: simulation, analysis, DFT, and plotting scripts
- `structures/`: starting, equilibrated, control, and parity structures
- `data/`: numerical data used in the manuscript



## Key data files

- `data/te_seg_equilibrated.csv`
- `data/te_seg_random.csv`
- `data/te_occupancy_table.csv`
- `data/wsep_v2.csv`
- `data/mc2x2_sro.csv`
- `data/parity_results.csv`

## Software

The calculations use:

- Python 3.10
- Atomic Simulation Environment (ASE)
- MACE-MP-0
- PyTorch
- Quantum ESPRESSO
- OVITO

Package versions are listed in `requirements.txt`.


## Data notes

Only successful calculations used in the manuscript are included. The
superseded first work-of-separation campaign and overwritten 700 K runs are
excluded. The final cohesion dataset is `data/wsep_v2.csv`.

Third-party model files and pseudopotentials are not redistributed. Their
identifiers and sources are documented in the runbook.


## License

Code is released under the MIT License. Repository data are released under
CC BY 4.0.
