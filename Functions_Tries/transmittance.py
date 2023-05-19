# %%

import numpy as np
from scipy.optimize import curve_fit

from typing import Callable

# se vedete le formule molto lunghe formattate in modo strano...
# Uso un formatter che si chiama Black...
# non ho disabilitato l'accorciamento automatico
# Vorrei aggiungere che le colonne sono lunghe circa 90... Forse dovrei sul serio aumetare la lunghezza,
# visto che arrivo sempre a circa 105 caratteri di lunghezza... O va be


# def transmittance(lmbd, /, n: float | Callable, k: float | Callable, n_0, n_1, t):
#     """
#     Uso la formula indicata dalla Francesca nella sua ultima mail...
#     Piano Piano la ottimizzo un po\' per l\'uso, intanto:
#     - lmbd : lambda, la lunghezza d'onda della luce incidente
#     - n    : parte reale dell'indice di rifrazione complesso del film sottile
#     - k    : parte complessa dell'indice di rifrazione complesso del film sottile
#     - n_1  : indice di rifrazione della superfice superiore al film sottile (aria)
#     - n_2  : indice di rifrazione della superfice inferiore al film sottile (vetro)
#     - t    : spessore del film sottile
#     """
#     # Controllo se in input n e k siano funzioni oppure siano dei valori...
#     # se scopro che tutti usano almeno python 3.10 modifico i valori
#     if isinstance(n, Callable) and isinstance(k, Callable):
#         c_1 = (n(lmbd) + n_0)(n_1 + n(lmbd))
#         c_2 = (n(lmbd) - n_0)(n_1 - n(lmbd))
#         alpha = np.exp(-4 * np.pi * k(lmbd) * t / lmbd)
#         T_num = 16 * n(lmbd) ** 2 * n_0 * n_1 * alpha
#         T_denom = (
#             c_1**2
#             + c_2**2 * alpha**2
#             + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * n(lmbd) * t / lmbd)
#         )
#     else:
#         c_1 = (n + n_0)(n_1 + n)
#         c_2 = (n - n_0)(n_1 - n)
#         alpha = np.exp(-4 * np.pi * k * t / lmbd)
#         T_num = 16 * n**2 * n_0 * n_1 * alpha
#         T_denom = (
#             c_1**2
#             + c_2**2 * alpha**2
#             + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * n * t / lmbd)
#         )

#     T = T_num / T_denom
#     return T


# def transmittane(lmbd, eta: Callable | complex, n_0, n_1, t):
#     """
#     versione di tranmittance che usa eta complesso,
#     invece che l'indice di rifrazione diviso nelle sue componenti
#     """

#     if isinstance(eta, Callable):
#         n = eta(lmbd).real()
#         k = -eta(lmbd).imag()
#     else:
#         n = eta.real()
#         k = -eta.imag()
#     transmittance(lmbd=lmbd, n=n, k=k, n_0=n_0, n_1=n_1, t=t)


# def Beer_Lambert(lmbd: float | np.ndarray,/, k: Callable | np.ndarray, t: float):
#     """
#     - lmbd : lunghezza d'onda incidente
#     - t    : spessore ipotizzato
#     - k    : coefficente di assorbimento, può essere una funzione o un numero
#     """
#     if isinstance(k, Callable):
#         return np.exp(-4 * np.pi * k(lmbd) * t / lmbd)
#     else:
#         return np.exp(-4 * np.pi * k * t / lmbd)


# Nuova sezione, mi sono visto alcuni problemi nell'implementazione del curve fit, la roba più semplice che mi è venuta in mente
# è di riscrivere la sezione precedente usando delle classi. In questo modo posso fissare alcuni parametri a priori.


class Transmittance:
    """
    Classe che racchiude vari metodi per il calcolo della Trasmittanza, comprende:
    - Beer Lambert
    - Trasmittance secondo l'ultima mail mandata dalla Fra

    La riscrittura si è resa necessaria per semplificare alcuni problemi avuti con il fitting dei parametri.
    """

    def __init__(self, n=None, k=None, n_0=None, n_1=None) -> None:
        """
        Parametri iniziali da fornire alla classe perchè questa possa funzionare
        - n    : Indice di rifrazione del materiale da analizzare
        - k    : Coefficene di estinzione del materiale da analizzare
        - n_0  : Indice di rifrazione della superfice superiore al film sottile (aria)
        - n_1  : Indice di rifrazione della superfice inferiore al film sottile (vetro)
        """

        self.n: float | Callable[[np.ndarray | float], np.ndarray | float] = n
        self.k: float | Callable[[np.ndarray | float], np.ndarray | float] = k
        self.n_0: float | Callable[[np.ndarray | float], np.ndarray | float] = n_0
        self.n_1: float | Callable[[np.ndarray | float], np.ndarray | float] = n_1

    def beer_lambert(
        self,  # La classe stessa
        lmbd: float | np.ndarray,  # La lunghezza d'onda incledente
        t: float  # Lo spessore da fittare
    ) -> float | np.ndarray:
        """
        Calcola mediante la formula di Beer Lambert
        .. math::
            \\exp (-4 \\pi k(\\lambda) t / \\lambda)

        - lmbd : lunghezza d'onda incidente
        - t    : spessore ipotizzato
        - k    : coefficente di assorbimento, può essere una funzione o un numero, è un parametro della classe stessa
        """
        if isinstance(self.k, Callable):
            # print("Calling with k function")
            return np.exp(-4.0 * np.pi * self.k(lmbd) * t / lmbd)
        else:
            # print("Calling with k scalar")
            return np.exp(-4.0 * np.pi * self.k * t / lmbd)

    def transmittance(
        self,  # La classe stessa
        lmbd,  # La lunghezza d'onda
        t  # Spessore
    ) -> float | np.ndarray:
        """
        Uso la formula indicata dalla Francesca nella sua ultima mail...
        Piano Piano la ottimizzo un po\' per l\'uso, intanto:
        - lmbd : lambda, la lunghezza d'onda della luce incidente
        - n    : parte reale dell'indice di rifrazione complesso del film sottile
        - k    : parte complessa dell'indice di rifrazione complesso del film sottile
        - t    : spessore del film sottile
        """
        # Controllo se in input n e k siano funzioni oppure siano dei valori...
        # se scopro che tutti usano almeno python 3.10 modifico i valori
        if isinstance(self.n, Callable) and isinstance(self.k, Callable):
            c_1 = (self.n(lmbd) + self.n_0) * (self.n_1 + self.n(lmbd))
            c_2 = (self.n(lmbd) - self.n_0) * (self.n_1 - self.n(lmbd))
            alpha = np.exp(-4 * np.pi * self.k(lmbd) * t / lmbd)
            T_num = 16 * self.n(lmbd) ** 2 * self.n_0 * self.n_1 * alpha
            T_denom = (
                c_1**2
                + c_2**2 * alpha**2
                + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * self.n(lmbd) * t / lmbd)
            )
        else:
            c_1 = (self.n + self.n_0) * (self.n_1 + self.n)
            c_2 = (self.n - self.n_0) * (self.n_1 - self.n)
            # Ma guarda chi si rivede. La vecchia Beer Lambert... alla fine avevi qualche uso
            alpha = np.exp(-4 * np.pi * self.k * t / lmbd)
            # Calcolo numeratore
            T_num = 16 * self.n**2 * self.n_0 * self.n_1 * alpha
            # Calcolo denominatore
            T_denom = (
                c_1**2
                + c_2**2 * alpha**2
                + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * self.n * t / lmbd)
            )
        T = T_num / T_denom
        return T

    def transmittance_n_free(
        self,  # La classe stessa
        lmbd,  # La lunghezza d'onda
        n_1,  # Indice di rifrazione del primo materiale (Vetro)
        t  # Spessore
    ) -> float | np.ndarray:
        """
        Uso la formula indicata dalla Francesca nella sua ultima mail...
        Piano Piano la ottimizzo un po\' per l\'uso, intanto:
        - lmbd : lambda, la lunghezza d'onda della luce incidente
        - n    : parte reale dell'indice di rifrazione complesso del film sottile
        - k    : parte complessa dell'indice di rifrazione complesso del film sottile
        - t    : spessore del film sottile
        """
        # Controllo se in input n e k siano funzioni oppure siano dei valori...
        # se scopro che tutti usano almeno python 3.10 modifico i valori
        if isinstance(self.n, Callable) and isinstance(self.k, Callable):
            c_1 = (self.n(lmbd) + self.n_0) * (n_1 + self.n(lmbd))
            c_2 = (self.n(lmbd) - self.n_0) * (n_1 - self.n(lmbd))
            alpha = np.exp(-4 * np.pi * self.k(lmbd) * t / lmbd)
            T_num = 16 * self.n(lmbd) ** 2 * self.n_0 * n_1 * alpha
            T_denom = (
                c_1**2
                + c_2**2 * alpha**2
                + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * self.n(lmbd) * t / lmbd)
            )
        else:
            c_1 = (self.n + self.n_0) * (n_1 + self.n)
            c_2 = (self.n - self.n_0) * (n_1 - self.n)
            # Ma guarda chi si rivede. La vecchia Beer Lambert... alla fine avevi qualche uso
            alpha = np.exp(-4 * np.pi * self.k * t / lmbd)
            # Calcolo numeratore
            T_num = 16 * self.n**2 * self.n_0 * n_1 * alpha
            # Calcolo denominatore
            T_denom = (
                c_1**2
                + c_2**2 * alpha**2
                + 2 * c_1 * c_2 * alpha * np.cos(4 * np.pi * self.n * t / lmbd)
            )
        T = T_num / T_denom
        return T

    def transmittance_complex(self, lmbd, eta: Callable | complex, n_0, n_1, t):
        """
        versione di tranmittance che usa eta complesso,
        invece che l'indice di rifrazione diviso nelle sue componenti
        """
        if isinstance(eta, Callable):
            n = eta(lmbd).real()
            k = -eta(lmbd).imag()
        else:
            n = eta.real()
            k = -eta.imag()
        return self.transmittance(lmbd=lmbd, n=n, k=k, n_0=n_0, n_1=n_1, t=t)


# %% Testiamo se tutto funziona
def test():
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline

    np.set_printoptions(precision=4)

    df = pd.read_csv(
        r"/Users/margheritapolgati/lab_mat/data/9_05_spettrofotometro/ELAB/gold_glass_trasm_4_cm_1.csv"
    )
    johns = pd.read_csv(r"../data/book_data/Johnson.csv")

    df["lambda"] *= 1e-9

    spl_john = CubicSpline(johns["wl"], johns["k"])
    new_BL = Transmittance(n=None, k=spl_john)
    popt, _ = curve_fit(
        new_BL.beer_lambert,
        df[(df["lambda"] < 800e-9) & (df["lambda"] > 300e-9)]["lambda"],
        df[(df["lambda"] < 800e-9) & (df["lambda"] > 300e-9)]["polished"],
        p0=60e-9,
    )

    # (df["lambda"] < 800) & (df["lambda"] > 300)
    # plt.plot("wl", "k", "o",data=johns, label = "data")
    # plt.plot(johns.wl, spl_john(johns.wl), label = "fit")
    # plt.plot(johns.wl, new_BL.k(johns.wl), label="Function transfered")
    plt.plot(
        "lambda",
        "polished",
        data=df[(df["lambda"] < 800e-9) & (df["lambda"] > 300e-9)],
        label="data",
    )
    plt.plot(
        df["lambda"], new_BL.beer_lambert(df["lambda"], popt), label=f"fit: {popt} m"
    )
    # plt.plot(df["lambda"], new_BL.beer_lambert(df["lambda"], 50e-9), label="hyp: 4 nm")

    plt.legend()


if __name__ == "__main__":
    test()

# %%
