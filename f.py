# pyright: basic

import numpy as np
import numpy.typing as npt


def xlogx(x: float) -> float:
    return x * np.log(x) if x > 0 else 0 if x == 0.0 else np.inf


def f(
    phi_p: float,
    phi_pi: float,
    phi_ni: float,
    *args: tuple[float, ...],
) -> float | tuple[float, ...]:
    N, chi, lB, ell, p = args[:5]

    # entropic
    fSp: float = xlogx(phi_p) / (2 * N)
    fSi: float = xlogx(phi_pi) + xlogx(phi_ni)

    phi_0: float = 1 - phi_p - phi_pi - phi_ni
    fS0: float = xlogx(phi_0)

    # excluded volume
    fex: float = chi * phi_p * phi_0

    # electrostatics
    kl2: float = 4 * np.pi * (lB / ell) * (phi_pi + phi_ni)
    kl = np.sqrt(kl2)

    vdd: float = (
        -(np.pi / 9)
        * ((lB**2) * (p**4) / (ell**6))
        * np.exp(-2 * kl)
        * (4 + (8 * kl) + (4 * kl2) + (kl**3))
    )

    fel: float = 0.5 * vdd * (phi_p / 2) ** 2

    # electrostatic correlations
    ffli: float = -(1 / (4 * np.pi)) * (np.log(1 + kl) - kl + 0.5 * kl2)

    return fSp + fSi + fS0 + fex + fel + ffli


def objective(s: npt.NDArray[np.float64], *args: tuple[float, ...]) -> float:
    _phi_pa, _phi_pia, _x = s
    phi_p, phi_pi, phi_ni = args[-3:]

    # phase A
    _phi_nia = _phi_pia
    _phi_0a = 1 - _phi_pa - _phi_pia - _phi_nia

    # phase B
    _phi_pb = (phi_p - (_x * _phi_pa)) / (1 - _x)
    _phi_pib = (phi_pi - (_x * _phi_pia)) / (1 - _x)
    _phi_nib = (phi_ni - (_x * _phi_nia)) / (1 - _x)
    _phi_0b = 1 - _phi_pb - _phi_pib - _phi_nib

    _phis: npt.NDArray[np.float64] = np.array(
        [
            _phi_pa,
            _phi_pia,
            _phi_nia,
            _phi_0a,
            _phi_pb,
            _phi_pib,
            _phi_nib,
            _phi_0b,
            _x,
        ]
    )

    if (np.abs(_phis - 0.5) < 0.5).all():  # if within 0 and 1, calculate free energy
        ftotal = _x * f(_phi_pa, _phi_pia, _phi_nia, *args) + (1 - _x) * f(
            _phi_pb, _phi_pib, _phi_nib, *args
        )
        return ftotal if ((ftotal < 1000) and (np.abs(_x - 0.5) < 0.499)) else np.inf
    else:  # if not, return infinity
        # print(
        #     f"error #1: aphysical phi value --> {np.asarray(np.abs(_phis - 0.5) >= 0.5).nonzero()}, {_phis, s}"
        # )
        return np.inf
