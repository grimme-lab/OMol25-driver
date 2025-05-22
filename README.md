# OMol25 Driver

This repository contains a command-line utility for executing energy and gradient calculations, as well as molecular geometry optimization, using pretrained machine-learned interatomic potentials (MLIPs) from the FAIRChem framework.

## Features

- **Structure Input:** Supports any structure format readable by ASE.
- **Charge & Multiplicity Handling:**
  - Priority given to CLI arguments.
  - Falls back to `.CHRG` and `.UHF` files if available.
  - Defaults to charge = 0 and multiplicity = 1 otherwise.
- **Model Selection:** Uses a pretrained FAIRChem MLIP model (default: `"uma-sm"`).
- **Geometry Optimization:** Optional LBFGS optimization with customizable `fmax`.
- **Energy and Gradient Output:** Results written to files named `energy` and `gradient`.
- **Performance Reporting:** Verbose mode prints wall times for key computational steps.

## Installation

**TODO**

## Usage

```bash
python driver.py [options] structure.xyz
```

### Options

- `--charge`: Override molecular charge (int)
- `--multiplicity`: Override spin multiplicity (int)
- `--fmax`: Optimization convergence threshold in eV/Å (default: 5e-4)
- `--verbose`: Print charge/multiplicity source info and wall times
- `--opt`: Enable geometry optimization
- `--model`: Specify a different MLIP model (default: "uma-sm")

### Positional Arguments

- `structure_file`: Path to the input structure file (required)

## Output

- `energy`: Total energy in Hartree
- `gradient`: Nuclear gradient information and gradient norm
- `omol25-opt.xyz`: Optimized geometry (if --opt is set)
- `trajectory.out`: Optimization trajectory file

## Example

```bash
python driver.py --charge 1 --multiplicity 2 --fmax 0.001 --opt --model uma-sm struc.xyz
```

### Notes

- Gradient and energy values are converted to Hartree units.
- Optimization uses the LBFGS algorithm with up to 1000 steps.
- Forces are internally converted for output as gradients.
