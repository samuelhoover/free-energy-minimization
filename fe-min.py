# pyright: basic

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import optimize
from typing import Any

from f import objective, constraints


### CONSTANTS
eps0: float = 625000.0 / (
    22468879468420441.0 * np.pi
)  # vacuum permittivity [A^2 s^4 kg^-1 m^-3]
e: float = 1.602e-19  # elementary charge [A s]
kB: float = 1.380694e-23  # boltzmann constant [kg m^2 s^-2 K^-1]

### PARAMETERS
N: float = 100.0  # degree of polymerization of polyelectrolyte
ell: float = 0.55e-9  # kuhn length [m]
theta: float = 257.0  # theta temperature [K]
p: float = ell  # dipole length on the monomer [m]
eps: float = 80.0  # dielectric constant of water


def get_sys_params(T: float, phi_p: float, phi_s: float) -> tuple[float, ...]:
    """
    Get system parameters.

    Args:
      - T [float]: absolute temperature [K]
      - phi_p [float]: total polymer species volume fraction
      - phi_s [float]: total salt ions volume fraction
    """
    lB: float = (e**2.0) / (4.0 * np.pi * eps0 * eps * kB * T)  # bjerrum length [m]
    chi: float = theta / (2.0 * T)  # solvent-polyelectrolyte interaction parameter
    phi_pi: float = phi_s / 2.0
    phi_ni: float = phi_s / 2.0

    sys_params: tuple[float, ...] = (
        N,
        chi,
        lB,
        ell,
        p,
        phi_p,
        phi_pi,
        phi_ni,
    )

    return sys_params


def solve_f(
    *sys_params: tuple[float, ...], **solver_options: dict[str, Any]
) -> optimize.OptimizeResult:
    """
    Solve the free energy expression.

    Args:
      - sys_params [tuple[float, ...]]: system parameters
      - solver_options [dict[str, Any]]: parameters for the minimization algorithm
    """
    phi_p, phi_pi = sys_params[-3:-1]
    results: optimize.OptimizeResult = optimize.minimize(
        objective,
        x0=(phi_p, phi_pi, 0.3),
        sys_params=sys_params,
        method="Nelder-Mead",
        options=solver_options,
    )

    return results


def fe_min(**solver_options: dict[str, Any]) -> npt.NDArray[np.float64]:
    """
    Define dependent variable search space and call `get_sys_params()` and
    `solve_f()` to determine phase diagram.

    Args:
      - solver_options [dict[str, Any]]: parameters for the minimization algorithm
    """
    TEMPS: npt.NDArray[np.float64] = np.array([25, 30, 34]) + 273.0
    PHI_P: npt.NDArray[np.float64] = np.arange(0, 0.4, 1e-2)
    PHI_S: npt.NDArray[np.float64] = np.arange(0, 0.08, 1e-3)
    phase: npt.NDArray[np.float64] = np.zeros(
        (len(TEMPS) * len(PHI_P) * len(PHI_S), 11)
    )

    for i, (T, phi_p, phi_s) in enumerate(product(TEMPS, PHI_P, PHI_S)):
        sys_params: tuple[float, ...] = get_sys_params(T, phi_p, phi_s)
        results: optimize.OptimizeResult = solve_f(*sys_params, **solver_options)

        if results.success:  # if minimization solution is found
            res: npt.NDArray[np.float64] = constraints(results.x, *sys_params)

            if np.abs(res[0] - res[4]) > 1e-3:
                phase[i, :] = np.array(
                    [
                        T,
                        res[0],
                        res[4],
                        res[1],
                        res[5],
                        res[2],
                        res[6],
                        res[3],
                        res[7],
                        res[8],
                        results.fun,
                    ]
                )

    return phase


def phase_diagram(phase: npt.NDArray[np.float64], save_fig: bool = True) -> None:
    """
    Plot resulting phase diagram from free energy expression solutions.
    """
    # discard empty rows (i.e., rows with no solutions)
    phase_keep: npt.NDArray[np.float64] = phase[phase.any(axis=1)]

    fig, ax = plt.subplots()

    ax.scatter(
        phase_keep[:, 1], phase_keep[:, 3] + phase_keep[:, 5], c=phase_keep[:, 0], s=1
    )
    ax.scatter(
        phase_keep[:, 2], phase_keep[:, 4] + phase_keep[:, 6], c=phase_keep[:, 0], s=1
    )

    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$\phi_{p}$")
    ax.set_ylabel(r"$\phi_{i}$")

    fig.tight_layout()
    if save_fig:
        fig.savefig("phase_diagram.pdf")

    plt.show()


def main(
    save: bool = True, save_fig: bool = True, **solver_options: dict[str, Any]
) -> None:
    phase: npt.NDArray[np.float64] = fe_min(**solver_options)
    if save:
        np.savetxt("phase.txt", phase, fmt="%.6e", delimiter=" ")

    phase_diagram(phase, save_fig)


if __name__ == "__main__":
    solver_options: dict[str, Any] = {
        "disp": False,
        "maxiter": 1600,
        "maxfev": 1200,
        "fatol": 1e-12,
        "xatol": 1e-12,
        "adaptive": True,
    }
    save = True
    main(save, **solver_options)
