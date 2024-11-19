# pyright: basic

import numpy as np
import numpy.typing as npt


def xlogx(x: float) -> float:
    return x * np.log(x) if x > 0 else 0 if x == 0.0 else np.inf


def entropic(
    phi_p: float, phi_pi: float, phi_ni: float, phi_0: float, *args
) -> tuple[float, ...]:
    """
    Entopric contributions from mobile species.
    """
    N: float = args[0]

    fSp: float = xlogx(phi_p) / (2.0 * N)
    fSi: float = xlogx(phi_pi) + xlogx(phi_ni)
    fS0: float = xlogx(phi_0)

    return fSp, fSi, fS0


def excluded_volume(phi_p: float, phi_0: float, *args) -> float:
    """
    Excluded volume effects between polymer and solvent.
    """
    chi: float = args[1]

    fex: float = chi * phi_p * phi_0

    return fex


def electrostatic(phi_p: float, kl, *args) -> float:
    """
    Electrostatic interactions: charge-charge, charge-dipole, and
    dipole-dipole.
    """
    lB, ell, p = args[2:5]

    vdd: float = (
        -(np.pi / 9.0)
        * ((lB**2.0) * (p**4.0) / (ell**6.0))
        * np.exp(-2.0 * kl)
        * (4.0 + (8.0 * kl) + (4.0 * (kl**2.0)) + (kl**3.0))
    )

    fel: float = 0.5 * vdd * (phi_p / 2.0) ** 2.0

    return fel


def elec_correlations(kl: float) -> float:
    """
    Correlations among ions near the polymer backbone.
    """
    ffli: float = -(1.0 / (4.0 * np.pi)) * (np.log(1.0 + kl) - kl + 0.5 * (kl**2.0))

    return ffli


def f(
    phi_p: float,
    phi_pi: float,
    phi_ni: float,
    *args: tuple[float, ...],
) -> float | tuple[float, ...]:
    lB: float = args[2]
    ell: float = args[3]

    phi_0: float = 1.0 - phi_p - phi_pi - phi_ni
    kl: float = np.sqrt(4.0 * np.pi * (lB / ell) * (phi_pi + phi_ni))

    # entropic
    fSp, fSi, fS0 = entropic(phi_p, phi_pi, phi_ni, phi_0, *args)

    # excluded volume
    fex = excluded_volume(phi_p, phi_0, *args)

    # electrostatics
    fel = electrostatic(phi_p, kl, *args)

    # electrostatic correlations
    ffli = elec_correlations(kl)

    return fSp + fSi + fS0 + fex + fel + ffli


def constraints(s: npt.NDArray[np.float64], *args) -> npt.NDArray[np.float64]:
    phi_pa, phi_pia, x = s
    phi_p, phi_pi, phi_ni = args[-3:]

    # phase A
    phi_nia = phi_pia
    phi_0a = 1.0 - phi_pa - phi_pia - phi_nia

    # phase B
    phi_pb = (phi_p - (x * phi_pa)) / (1.0 - x)
    phi_pib = (phi_pi - (x * phi_pia)) / (1.0 - x)
    phi_nib = (phi_ni - (x * phi_nia)) / (1.0 - x)
    phi_0b = 1.0 - phi_pb - phi_pib - phi_nib

    res: npt.NDArray[np.float64] = np.array(
        [
            phi_pa,
            phi_pia,
            phi_nia,
            phi_0a,
            phi_pb,
            phi_pib,
            phi_nib,
            phi_0b,
            x,
        ]
    )

    return res


def objective(s: npt.NDArray[np.float64], *args: tuple[float, ...]) -> float:
    res: npt.NDArray[np.float64] = constraints(s, *args)
    phi_pa, phi_pia, phi_nia, _, phi_pb, phi_pib, phi_nib, _, x = res

    if (np.abs(res - 0.5) < 0.5).all():  # if within 0 and 1, calculate free energy
        ftotal = x * f(phi_pa, phi_pia, phi_nia, *args) + (1.0 - x) * f(
            phi_pb, phi_pib, phi_nib, *args
        )
        return ftotal if ((ftotal < 1000.0) and (np.abs(x - 0.5) < 0.499)) else np.inf

    else:  # if not, return infinity
        # print(
        #     f"error #1: aphysical phi value --> {np.asarray(np.abs(res - 0.5) >= 0.5).nonzero()}, {res, s}"
        # )
        return np.inf
