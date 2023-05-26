# %%
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from Functions_Tries.transmittance import Transmittance
from pathlib import Path
from typing import Callable


plt.style.use("seaborn")

# # Il codice seguente serve al fit dello spessore dagli spettri ottenuti dallo spettrofotometro
#
# - Usiamo la libreria ```numpy``` per avere la matematica corretta e dei contenitori efficenti.
# - Usiamo la libreria ```scipy.interpolate``` per avere le curve di Bezier per interpolare i dati ottenuti dal Johnson
# - Usiamo la libreria ```scipy.optimize``` per fittare gli spettri e ottenere il valore di spessore e l'errore
# - Usiamo la libreria ```pandas``` per leggere i file che contengono gli spettri e per l'output
# - Stefano ha scritto il modulo ```transmittance``` per avere le funzioni su cui fare il fit e per fare il primo test di fattibilità sui fit
# - Usiamo il modulo ```pathlib``` per gestire i Path dei vari file in maniera automatizzata
# - Usiamo la libreria ```matplotlib``` per graficare gli spettri

# Imposto la precisione per l'output dei numeri
np.set_printoptions(precision=2)

# Path entro cui fare le ricerche
dir = Path("./data/")

# Nome colonne for pretty printing
colonne = ["NomeFile", "ValoreFit", "ErrFit", "χ_2_Rid", "GdL"]

# Liste vuote, servono per salvare i vari dati del fit
data_Beer_Lambert = dict(NomeFile=[], ValoreFit=[], ErrFit=[], χ_2_Rid=[], GdL=[])

data_Transmittance = dict(NomeFile=[], ValoreFit=[], ErrFit=[], χ_2_Rid=[], GdL=[])

data_Transmittance_n_free = dict(
    NomeFile=[], ValoreFit=[], ErrFit=[], n_obt=[], err_n=[], χ_2_Rid=[], GdL=[]
)


# leggo dati del johnny
john = pd.read_csv("./data/book_data/Johnson.csv")
n_spl_john = CubicSpline(john["wl"], john["n"])
k_spl_john = CubicSpline(john["wl"], john["k"])

# Inizializziamo la classe che contiene le transmittance per i successivi fit
Trans = Transmittance(
    n=n_spl_john,
    # n=1,
    k=k_spl_john,
    n_0=1.0,
    n_1=1.52,
)


# Funzione che va a fare i fit in maniera automatica a partire dai dati ottenuti per beer lambert...
def general_optimizer(
    path: Path,
    dest_name=None,
    fit_func: Callable = Trans.beer_lambert,
    wavelen_bound: tuple | list = (300e-9, 800e-9),
    y_bound: tuple | list = (0.0, 1.0),
    graph_title: str = "",
):
    """
    Funzione che va a fare i fit in maniera automatica, indipendentemente dalla funzione in ingresso
    Input Parameters:
    - path : Input File
    - dest_name: Graph Output Directory
    - fit_func : Function to fit
    - wavelen_bound : Limiti sulla lunghezza d'onda
    - y_bound : Limite sulla trasmittanza
    - graph_title : Titolo del grafico
    """

    # Leggiamo i dati
    df = pd.read_csv(path)
    # Ci sono capitati dati uguali a zero... questi creano problemi, via dal dataframe
    # Revisione successiva introduce delle condizioni più strette... I dati che si possono usare sono quelli che
    # sono maggiori di 0 e minori di 1... Le altre condizioni sono solo spiacevoli infortuni.
    df_clean = df(df["polished"] > y_bound[0] & (df["polished"] < y_bound[1]))

    # Filtro per le lunghezze d'onda
    filter_λ = (df_clean["lambda"] > wavelen_bound[0]) & (
        df_clean["lambda"] < wavelen_bound[1]
    )

    # Calcoliamo lo spessore e il suo errore
    popt, pcov = curve_fit(
        fit_func,
        df_clean[filter_λ]["lambda"],
        df_clean[filter_λ]["polished"],
        p0=60e-9,
        sigma=df_clean[filter_λ]["trasm_error"],
    )

    # Calcolo del Chi quadro... Potrebbe essere inserito nel grafico, ma non ho voglia
    chisq_rid = (
        np.sum(
            (
                df_clean[filter_λ]["polished"]
                - fit_func(df_clean[filter_λ]["lambda"], *popt)
            )
            ** 2
            / df_clean[filter_λ]["trasm_error"]**2
        )
        / df_clean[filter_λ].size
    )

    # Grafichiamo... Innanzitutto controlliamo se c'è la directory dove buttare fuori i dati
    # Se non ci fosse la creiamo, ho recentemente scoperto che si può fare tutto in un solo comando
    if dest_name is None:
        dest = Path("./images/beer_lambert/" + path.parent.parts[-2])
        dest.mkdir(parents=True, exist_ok=True)
    # questo serve solo nel caso in cui sia lanciato iteratore_spettro() per cui dest_name non è nullo
    else:
        dest = Path("./images/beer_lambert/").joinpath(
            dest_name[0] + "/" + dest_name[1]
        )
        # Quello che è contenuto in join path serve ad aggiugnere *_spettrofotometro/vetrino_*
        # Nella mia stanchezza non mi è venuto in mente niente di meglio...
        dest.mkdir(parents=True, exist_ok=True)

    # Ora Grafichiamo
    fig, ax = plt.subplots()
    # Aggiungiamo i dati, ho riscalato le misure per avere sull'asse x dei nm e non dei metri illeggibili
    # Dati originali
    ax.plot(
        df_clean[filter_λ]["lambda"] * 1e9,
        df_clean[filter_λ]["polished"],
        "go",
        label="original data",
    )

    # La funzione fittata
    if len(popt) == 1:
        ax.plot(
            df_clean[filter_λ]["lambda"] * 1e9,
            fit_func(df_clean[filter_λ]["lambda"], *popt),
            label=f"fit : {popt*1e9} nm",
        )
    else:
        ax.plot(
            df_clean[filter_λ]["lambda"] * 1e9,
            fit_func(df_clean[filter_λ]["lambda"], *popt),
            label=f"fit : {popt[0]*1e9} nm\nn_1 : {popt[1]}",
        )

    # Plotto gli scarti (y - f(x)), chiesto dalla Fra
    ax.plot(
        df_clean[filter_λ]["lambda"] * 1e9,
        np.abs(fit_func(df_clean[filter_λ]["lambda"]) - df_clean[filter_λ]["polished"]),
        "r--",
        label="residues",
    )

    # Questioni estetiche relative agli assi
    ax.set_xlabel("$\\lambda$ [nm]")
    ax.set_ylabel("Transmittance")
    ax.set_title(graph_title)
    ax.legend()

    # Salviamo l'immagine, sia in formato svg che in formato pdf...
    for sur in (".svg", ".pdf", ".png"):
        fig.savefig(dest / path.with_suffix(sur).name, format=sur)

    # Per qualche motivo non resetta i canvas... lo forziamo a pulirsi
    plt.clf()
    plt.cla()
    plt.close()

    # Riportiamo finalmente i risulati, in ordine sono il parametro ottimizzato, il suo errore,
    # il chi quadro ridotto, i gradi di libertà
    return popt, np.sqrt(np.diag(pcov)), chisq_rid, df_clean[filter_λ].size


# Funzione per salvare dati, in questo modo non dobbiamo ogni volta riscrivere lo stesso codice 2 volte
def saving_res(file, data: dict, fit_func: Callable, i=None, **kwargs):
    # Ottengo i risultati dell'ottimizzatore e li carico in un Liste a parte
    if i is None:
        res = general_optimizer(file, fit_func=fit_func, **kwargs)
    else:
        # Questo lo uso quando capita iteratore_spettro... che è un filo più rompiballe
        res = general_optimizer(
            file, [i.parts[-3], i.parts[-2]], fit_func=fit_func, **kwargs
        )

    # SALVATAGGIO DATI
    match len(data.keys):
        case 5:
            data["NomeFile"].append(file)
            data["ValoreFit"].append(
                float(res[0])
            )  # Secondo me è un qualcosa di brutto quel cast a float
            data["ErrFit"].append(
                float(res[1])
            )  # Ma non mi è venuto in mente niente di meglio per risolvere la questione
            data["χ_2_Rid"].append(res[2])
            data["GdL"].append(res[3])
        case 7:
            data["NomeFile"].append(file)
            data["ValoreFit"].append(
                float(res[0][1])
            )  # Secondo me è un qualcosa di brutto quel cast a float
            data["ErrFit"].append(
                float(res[1][1])
            )  # Ma non mi è venuto in mente niente di meglio per risolvere la questione
            data["n_obt"].append(float(res[0][0]))
            data["err_n"].append(float(res[1][0]))
            data["χ_2_Rid"].append(res[2])
            data["GdL"].append(res[3])


# Funzione che itera su tutti i file contenuti nella cartella "./data"
def iterazione(path):
    # Ci interessano solo le cartelle che contengono la parola spettrofotometro
    data = [
        i for i in path.iterdir() if i.match("*spettrofotometro")
    ]  # purtroppo per filtrare non abbiamo ancora scoperto un modo migliore...
    # Così non posso sfruttare i vantaggi dei generatori
    for i in data:
        # Itero solo sulle sottocartelle e non su file che sono in giro
        subfolder = [j for j in i.iterdir() if j.is_dir()]
        for folder in subfolder:
            # Dobbiamo trovarci nella cartella ELAB per fare qualsiasi cosa
            if folder.match("ELAB"):
                # Itero sui file di ELAB
                for file in folder.iterdir():
                    # Piccole correzioni per evitare di fare Step inutili
                    if file.match("*.png"):
                        continue
                    if file.match("Aria*") or file.match("aria*") or file.match("air*"):
                        continue
                    saving_res(
                        file,
                        data_Beer_Lambert,
                        fit_func=Trans.beer_lambert,
                        graph_title="Fit Beer Lambert",
                    )
                    saving_res(
                        file,
                        data_Transmittance,
                        fit_func=Trans.transmittance,
                        graph_title="Fit Transmittance",
                    )
                    saving_res(
                        file,
                        data_Transmittance_n_free,
                        fit_func=Trans.transmittance_n_free,
                        graph_title="Fit Transmittance, n fitted",
                    )
            else:
                # Ci sono cartelle che contengono spettrofotometro... Ma hanno sottocartelle sottostanti
                # print("La cartella non è Elab... gestiamo logica dopo")
                # print(folder)
                iterazione_spettro(folder)


# In alcuni casi non si trova subito la cartella "ELAB", in quei casi bisogna scendere ancora di un livello
def iterazione_spettro(path):
    # print("launching iterazione_spettro")
    # print(path)
    elab = [i for i in path.iterdir() if i.match("ELAB")]
    # print(elab)
    for i in elab:
        for file in i.iterdir():
            if file.match("*.png"):
                continue
            if file.match("Aria*") or file.match("aria*") or file.match("air*"):
                continue
            saving_res(
                file,
                data_Beer_Lambert,
                fit_func=Trans.beer_lambert,
                graph_title="Fit Beer Lambert",
            )
            saving_res(
                file,
                data_Transmittance,
                fit_func=Trans.transmittance,
                graph_title="Fit Transmittance",
            )
            saving_res(
                file,
                data_Transmittance_n_free,
                fit_func=Trans.transmittance_n_free,
                graph_title="Fit Transmittance, n fitted",
            )



if __name__ == "__main__":
    # Finalmente lanciamo la cazzo di funzione
    iterazione(dir)
    # SALVATAGGIO DATI BEER LAMBERT SU FILE
    res_df_beer_lambert = pd.DataFrame(data_Beer_Lambert)
    res_df_beer_lambert.to_csv("Risultati_Beer_Lambert_spettrofotometro.csv", index=False)

    # SALVATAGGIO DATI TRANSMITTANCE SU FILE
    res_transmittance = pd.DataFrame(data_Transmittance)
    res_df_beer_lambert.to_csv("Risultati_Transmittance_spettrofotometro.csv", index=False)

    res_transmittance_n_free = pd.DataFrame(data_Transmittance_n_free)
    res_transmittance_n_free.to_csv(
        "Risultati_Transmittance_n_free_spettrofotometro.csv", index=False
    )
