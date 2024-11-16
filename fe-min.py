# pyright: basic

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from typing import Any

from f import objective


# constants
eps0 = 625000.0 / (
    22468879468420441.0 * np.pi
)  # vacuum permittivity [A^2 s^4 kg^-1 m^-3]
e = 1.602e-19  # elementary charge [A s]
kB = 1.380694e-23  # boltzmann constant [kg m^2 s^-2 K^-1]

# parameters
N = 100.0  # degree of polymerization of polyelectrolyte
ell = 0.55e-9  # kuhn length [m]
theta = 257.0  # theta temperature [K]
p = ell  # dipole length on the monomer [m]
eps = 80.0  # dielectric constant of water


def fe_min(**solver_options: dict[str, Any]) -> npt.NDArray[np.float64]:
    TEMPS = np.array([25, 30, 34]) + 273.0
    PHI_P = np.arange(0, 0.4, 1e-2)
    PHI_S = np.arange(0, 0.08, 1e-3)
    phase: npt.NDArray[np.float64] = np.zeros(
        (len(TEMPS) * len(PHI_P) * len(PHI_S), 11)
    )

    for i, (T, phi_p, phi_s) in enumerate(product(TEMPS, PHI_P, PHI_S)):
        lB = (e**2) / (4 * np.pi * eps0 * eps * kB * T)  # bjerrum length [m]
        chi = theta / (2 * T)  # solvent-polyelectrolyte interaction parameter
        phi_pi = phi_s / 2
        phi_ni = phi_s / 2

        args = (
            N,
            chi,
            lB,
            ell,
            p,
            phi_p,
            phi_pi,
            phi_ni,
        )
        results = minimize(
            objective,
            x0=(phi_p, phi_pi, 0.3),
            args=args,
            method="Nelder-Mead",
            options=solver_options,
        )
        if results.success:  # if minimization solution is found
            pa, pia, xa = results.x

            # phase A
            nia = pia
            p0a = 1 - pa - pia - nia

            # phase B
            pb = (phi_p - (xa * pa)) / (1 - xa)
            pib = (phi_pi - (xa * pia)) / (1 - xa)
            nib = (phi_ni - (xa * nia)) / (1 - xa)
            p0b = 1 - pb - pib - nib

            ftotal = results.fun

            if np.abs(pa - pb) > 1e-3:
                phase[i, :] = np.array(
                    [T, pa, pb, pia, pib, nia, nib, p0a, p0b, xa, ftotal]
                )

    return phase


def phase_diagram(phase: npt.NDArray[np.float64], save_fig: bool = True) -> None:
    # discard empty rows (i.e., rows with no solutions)
    phase_keep = phase[phase.any(axis=1)]

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
