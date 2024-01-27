from common_imports import np, Counter
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances
from sklearn.base import BaseEstimator, ClassifierMixin
 
class KNN_Classifier_Custom(BaseEstimator, ClassifierMixin):
    """
    Importando i mixin BaseEstimator e ClassifierMixin e utilizzandoli come superclassi, 
    si fornisce alla classe una struttura e un'implementazione che rendono il modello 
    compatibile con l'API di scikit-learn. Questo tornerà particolarmente utile quando 
    andrò a fare il tuning di k, potendo interfacciarmi direttamente con la funzione 
    cross_validate offerta da sklearn.
    """

    def __init__(self, k=8, distance_type="Manhattan"):
        # Inizializzazione dell'istanza del classificatore con i parametri specificati
        self.k = k
        self.distance_type = distance_type

    def fit(self, train_x, train_y):
        # Memorizzazione dei dati di addestramento
        self.train_x = train_x
        self.train_y = train_y

    def chebyshev_distances(self, row_test):
        # Calcolo delle distanze di Chebyshev tra una riga di test e tutte le righe di addestramento
        distanza_chebyshev = []
        for i in range(len(self.train_x)):
            row_train = self.train_x.iloc[i, :]
            distanza_chebyshev.append(np.abs(row_train - row_test).max())
        return distanza_chebyshev

    def calcola_NN(self, distances):
        # Identificazione delle etichette delle k istanze più vicine
        etichette = []
        for i in range(0, self.k):
            indice_dist_minima = np.argmin(distances)
            etichette.append(self.train_y.iloc[indice_dist_minima])
            distances[indice_dist_minima] = float('inf')
        # Determinazione dell'etichetta di classe più comune tra le k istanze più vicine
        label_classe = Counter(etichette).most_common(1)[0][0]
        return label_classe

    def predict(self, test_x):
        # Predizione delle etichette di classe per le istanze di test
        predizioni = []
        for i in range(len(test_x)):
            row = test_x.iloc[i, :]
            # Calcolo delle distanze in base al tipo specificato
            if self.distance_type == "Euclidea":
                distances = euclidean_distances(self.train_x, [row])
            elif self.distance_type == "Manhattan":
                distances = manhattan_distances(self.train_x, [row])
            elif self.distance_type == "Chebyshev":
                distances = self.chebyshev_distances(row)
            else:
                # Tipo di distanza non supportato
                return None
            # Determinazione delle predizioni in base al valore di k
            if self.k > 1:
                predizioni.append(self.calcola_NN(distances))
            elif self.k == 1:
                indice_dist_minima = np.argmin(distances)
                predizioni.append(self.train_y.iloc[indice_dist_minima])
            else:
                # Valore di k non valido
                return None
        return predizioni

    def fit_predict(self, train_x, train_y, test_x):
        # Addestramento del modello e predizione sulle istanze di test
        self.fit(train_x, train_y)
        return self.predict(test_x)
