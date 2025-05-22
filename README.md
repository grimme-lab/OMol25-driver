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

### Prerequisites

> [!IMPORTANT]
> We should test if a general installation for all of us via a `conda`/`mamba` environment could also work.

- Create a Hugging Face account and request access to the `FAIRChem` models:
  - [Dataset](https://huggingface.co/facebook/OMol25)
  - [Model](https://huggingface.co/facebook/UMA)

- Generate a token for access to the models:
  1. Go to your [Hugging Face account settings](https://huggingface.co/settings/tokens).
  2. Create a new token with the `read` scope.
  3. **Leave the window open for the next step.**

### Installation of the FAIRChem framework

You can install the project in a new virtual environment (provided for example by the package managers `conda` or `mamba` (see also [here](https://github.com/conda-forge/miniforge) and [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html))).
With `mamba`, a matching Python environment can be set up and activated as follows:

```
mamba create -n fairchem-omol25 python=3.12
mamba activate fairchem-omol25
```

Afterwards, the package can be installed by downloading the package from `PyPi`:

```bash
pip install fairchem-core
```

Now, you need to login to the Hugging Face Hub to access the pretrained models. You can do this by running:

```bash
huggingface-cli login
```

In the command-line prompt, paste the token you generated in the previous step. This will authenticate your local environment with the Hugging Face Hub.

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
