# lab_mat

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Dove Stefano, Margherita  e Francesco tengono i suoi codici separati dal resto delle cose, ma comunque accessibili

L'ambiente python usato è anaconda, con versione di python 3.10, Stefano ha testato con python 3.11 funzionava. Si potrebbe scrivere il codice meglio? Sicuramente. Abbiamo voglia? Certamente NO.

## Cosa c'è in ogni file

Siccome ormai c'è  un po' di disordine, facciamo in modo di metterlo dichiarando almeno in ogni file cosa ci sia

### File .py

Sono file sorgenti di python, in questo caso sono tutte funzioni scritte ad oc per fare alcune cose particolari:

- `cleaner.py`, serve alla pulizia dei dati dello spettrofotometro, presa in input la cartella da pulire e il file da usare come zero ci pensa lui a fare su tutti gli altri file le operazioni opportune per la pulizia
- `plotter.py`, in teoria automatizza il processo di plotting, se siamo molto pigri si può utilizzare e dovrebbe semplificare e ripulire il lavoro
- `transmittance.py`, funzioni per la trasmittanza su cui in futuro si faranno dei fit

### Notebook

- `absorbance_graph.ipynb` dovrebbe contenere tutti i file dell'assorbanza... alla fine della fiera ne contiene uno
- `fit_lab.ipynb` contiene vari fit dei dati del palik, presto conterrà anche quelli del Johnson
- `gradino_3D.ipynb` contiene i grafici 3D dei vari scalini e i relativi istogrammi
- `grafici_spettri.ipynb` contiene i primi spettri che abbiamo plottato, quindi anche il prototipo per la pulizia dei dati, contiene per qualche motivo le informazioni riguardanti la $k_\lambda$ e contiene per qualche altro motivo i fit della legge dei coseni... In pratica contiene tutto quello che dovrebbe contenere più roba che non c'entra un cazzo
- `lab_mat_distribution.ipynb` altro notebook che potrebbe venire buttato, se non fosse che ha un po' di teoria utile all'interno... si potrebbe trasferire tutto da un'altra parte... contiene anche il primo disegno del coseno alla quarta
- `new_graph_21_04.ipynb` contiene gli ultimi grafici fatti, ma neanche tutti
- `probable_dist.ipynb` si potrebbe anche buttare, dovrebbe rappresentare le varie distribuzioni coseno in base alla distanza dalla sorgente, idealmente il fit del coseno dovrebbe essere fatto qua dentro
- `Spettrofotometro.ipynb`, dentro ci sono tutte le stampe fatte

## TODO

- [ ] riordinare i notebook, come notava stefano c'è un po' di merda che si può sistemare
- [ ] Fare tutti i grafici mancanti... evidentemente ne mancano un po'
- [x] Dare ai file python una loro cartella... Idealmente chiamata src...
- [ ] La cartella Data si potrebbe riordinare per dargli un filo di chiarezza in più
- [x] Eliminare il makefile che in questo frangente non serve a un cazzo.
