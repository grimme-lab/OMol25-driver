import argparse
from pathlib import Path
import time

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import Bohr, Hartree
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

DEFAULT_MODEL = "uma-sm"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OMol25 driver for CLI.")
    parser.add_argument("--charge", type=int, help="Override molecular charge.")
    parser.add_argument("--multiplicity", type=int, help="Override spin multiplicity.")
    parser.add_argument(
        "--fmax", type=float, default=5e-4, help="Optimization threshold / eV/Å."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print charge/multiplicity source info."
    )
    parser.add_argument(
        "--opt", action="store_true", default=False, help="Optimize the structure."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name for the MLIP.",
    )
    parser.add_argument("structure_file", type=str, help="Input structure file.")
    return parser.parse_args()


def read_charge_and_multiplicity(
    curr_dir: Path,
    cli_charge: int | None,
    cli_mult: int | None,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Determines the charge and multiplicity.
    CLI flags take precedence over file-based values.
    """
    chrg_file = curr_dir / ".CHRG"
    uhf_file = curr_dir / ".UHF"

    # Charge
    if cli_charge is not None:
        charge = cli_charge
        if verbose:
            print(f"Using charge from CLI: {charge}")
    elif chrg_file.exists():
        charge = int(chrg_file.read_text().strip())
        if verbose:
            print(f"Using charge from file {chrg_file.name}: {charge}")
    else:
        charge = 0
        if verbose:
            print("Using default charge: 0")

    # Multiplicity
    if cli_mult is not None:
        multiplicity = cli_mult
        if verbose:
            print(f"Using multiplicity from CLI: {multiplicity}")
    elif uhf_file.exists():
        try:
            uhf = float(uhf_file.read_text().strip())
            multiplicity = int(2 * (uhf * 0.5) + 1)
            if verbose:
                print(
                    f"Using multiplicity from file {uhf_file.name} (UHF={uhf}): {multiplicity}"
                )
        except ValueError as e:
            raise ValueError(f"Invalid float in {uhf_file}") from e
    else:
        multiplicity = 1
        if verbose:
            print("Using default multiplicity: 1")

    return charge, multiplicity


def optimize(
    atoms: Atoms, fmax: float = 5e-4, steps: int = 1000, traj: str | None = None
) -> None:
    optimizer = LBFGS(atoms, trajectory=traj)
    optimizer.run(fmax, steps)


def write_gradient_block(
    file: Path,
    energy_hartree: float,
    forces: np.ndarray,
    atoms: Atoms,
) -> None:
    gradient = -forces * Bohr / Hartree
    grad_norm = np.linalg.norm(gradient)

    with file.open("w", encoding="utf-8") as f:
        f.write("$grad\n")
        f.write(
            f" cycle = 1   SCF energy = {energy_hartree:.10f}  |dE/dxyz| = {grad_norm:.10f}\n"
        )

        for (x, y, z), symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            f.write(
                f"{x / Bohr:20.14f}  {y / Bohr:20.14f}  {z / Bohr:20.14f} {symbol.lower():>2}\n"
            )

        np.savetxt(f, gradient, fmt="%20.14f")
        f.write("$end\n")

    print(f"Gradient Norm: {grad_norm:.10f}")


def write_energy_block(file: Path, energy_hartree: float) -> None:
    with file.open("w", encoding="utf-8") as f:
        f.write(
            f"$energy\n     1     {energy_hartree:.10f}     {energy_hartree:.10f}     {energy_hartree:.10f}\n$end\n"
        )


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.structure_file)
    working_dir = input_path.parent
    grad_file = Path("gradient")
    energy_file = Path("energy")

    charge, multiplicity = read_charge_and_multiplicity(
        working_dir,
        cli_charge=args.charge,
        cli_mult=args.multiplicity,
        verbose=args.verbose,
    )
    predictor = pretrained_mlip.get_predict_unit(args.model, device="cuda")
    calc = FAIRChemCalculator(predictor, task_name="omol")

    atoms = read(input_path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = multiplicity
    atoms.calc = calc

    if args.opt:
        original_atoms = atoms.copy()
        optimize(atoms, fmax=args.fmax, steps=1000, traj="trajectory.out")
        # TODO: Calculate RMSD between original and optimized structures
        write("omol25-opt.xyz", atoms)

    # Check run time for "get_potential_energy"
    # and "get_forces" methods
    start_time = time.time()
    etot = atoms.get_potential_energy() / Hartree  # NumPy array with one element
    elapsed_time_sp = time.time() - start_time
    # print time in H:M:S format
    elapsed_time_sp_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sp))
    if args.verbose:
        print(f"Wall time for single-point energy: {elapsed_time_sp_str}")
    energy = float(np.atleast_1d(etot)[0])

    # Check run time for "get_forces" method
    start_time_forces = time.time()
    forces = atoms.get_forces(apply_constraint=True, md=False)
    elapsed_time_forces = time.time() - start_time_forces
    elapsed_time_forces_str = time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time_forces)
    )
    if args.verbose:
        print(f"Wall time for nuclear gradient: {elapsed_time_forces_str}")

    # Check total run time
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    if args.verbose:
        print(f"Total wall time: {total_time_str}")

    print(f"Total energy: {energy:.10f}")
    if args.verbose:
        # Write name of employed checkpoint file into the output
        print(f"Used model: {DEFAULT_MODEL}")

    write_gradient_block(grad_file, energy, forces, atoms)
    write_energy_block(energy_file, energy)


if __name__ == "__main__":
    main()
