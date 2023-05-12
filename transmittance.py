import numpy as np
from scipy.optimize import curve_fit

# se vedete le formule molto lunghe formattate in modo strano...
# Uso un formatter che si chiama Black...
# non ho disabilitato l'accorciamento automatico


def transmittance(lmbd, n, k, n_0, n_1, t):
    """
    Uso la formula indicata dalla Francesca nella sua ultima mail...
    Piano Piano la ottimizzo un po\' per l\'uso, intanto:
    - lmbd : lambda, la lunghezza d'onda della luce incidente
    - n    : parte reale dell'indice di rifrazione complesso del film sottile
    - k    : parte complessa dell'indice di rifrazione complesso del film sottile
    - n_1  : indice di rifrazione della superfice superiore al film sottile (aria)
    - n_2  : indice di rifrazione della superfice inferiore al film sottile (vetro)
    - t    : spessore del film sottile
    """
    # Controllo se in input n e k siano funzioni oppure siano dei valori...
    # se scopro che tutti usano almeno python 3.10 modifico i valori
    if isinstance(n, callable) and isinstance(k, callable):
        c_1 = (n(lmbd) + n_0)(n_1 + n(lmbd))
        c_2 = (n(lmbd) - n_0)(n_1 - n(lmbd))
        alpha = np.exp(-4 * np.pi * k(lmbd) * t / lmbd)
        T_num = 16 * n(lmbd) ** 2 * n_0 * n_1 * alpha
        T_denom = (
            c_1**2
            + c_2**2 * alpha**2
            + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * n(lmbd) * t / lmbd)
        )
    else:
        c_1 = (n + n_0)(n_1 + n)
        c_2 = (n - n_0)(n_1 - n)
        alpha = np.exp(-4 * np.pi * k * t / lmbd)
        T_num = 16 * n**2 * n_0 * n_1 * alpha
        T_denom = (
            c_1**2
            + c_2**2 * alpha**2
            + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * n * t / lmbd)
        )

    T = T_num / T_denom
    return T


def transmittane(lmbd, eta: complex | callable, n_0, n_1, t):
    """
    versione di tranmittance che usa eta complesso,
    invece che l'indice di rifrazione diviso nelle sue componenti
    """

    if isinstance(eta, callable):
        n = eta(lmbd).real()
        k = -eta(lmbd).imag()
    else:
        n = eta.real()
        k = -eta.imag()
    transmittance(lmbd=lmbd, n=n, k=k, n_0=n_0, n_1=n_1, t=t)


def Beer_Lambert(lmbd: float | np.array, k: callable | np.ndarray, t: float):
    """
    - lmbd : lunghezza d'onda incidente
    - t    : spessore ipotizzato
    - k    : coefficente di assorbimento, può essere una funzione o un numero
    """
    if isinstance(k, callable):
        return np.exp(-4 * np.pi * k(lmbd) * t / lmbd)
    else:
        return np.exp(-4 * np.pi * k * t / lmbd)


def fit_thickness(x_data, y_data, function_to_fit, p0 : tuple | list | np.array):
    """
    Funzione che si occupa di fare il fit dello spessore partendo dai dati del
    profilometro.
    """
    curve_fit(function_to_fit, xdata=x_data, ydata=y_data)
