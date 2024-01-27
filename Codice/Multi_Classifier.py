import random
from sklearn.tree import DecisionTreeClassifier
from common_imports import Counter,np
from sklearn.model_selection import train_test_split

 
class RandomForestCustom:
    def __init__(self, numero_alberi = 28, depth = 6, random_state = 0):
        # Inizializzazione con la profondità massima degli alberi
        self.max_depth = depth #profondità massima dei singoli alberi
        self.alberi_foresta = []  # Lista per contenere gli alberi della foresta
        self.colonne_alberi = []  # Lista per contenere le colonne utilizzate da ciascun albero
        self.max_alberi = numero_alberi #numero massimo di alberi della foresta
        self.seed_random = random_state #seme per la random
        
    def fit(self, train_x, train_y):

        for i in range(0, self.max_alberi):

            # Imposta il seme casuale
            random.seed(i + self.seed_random)
            np.random.seed(i + self.seed_random)

            # Determina il numero casuale di colonne da estrarre
            numero_colonne_da_estrarre = random.randint(1, len(train_x.columns))

            # Estrarre un numero casuale di colonne dal DataFrame
            colonne_casuali = np.random.choice(train_x.columns, size = numero_colonne_da_estrarre, replace=False)

            #salvo le colonne selezionate. Torneranno utili in fase di test
            self.colonne_alberi.append(colonne_casuali)

            # Creare un nuovo DataFrame con le colonne estratte
            df_campione_colonne = train_x[colonne_casuali]

            # Esegui lo split del dataset per selezionare record random dal nuovo dataframe campione
            train_x_campione, _, train_y_campione, _ = train_test_split(df_campione_colonne, train_y, test_size=0.35, random_state=i + self.seed_random)

            #istanza di un albero di classificazione con una profondità di defaulf
            dTree_clf = DecisionTreeClassifier(max_depth = self.max_depth, random_state = i + self.seed_random)

            #salvo l'albero della foresta generato
            self.alberi_foresta.append(dTree_clf)

            # Addestramento
            dTree_clf.fit(train_x_campione, train_y_campione)

    def predict(self, test_x):
        labels_finali = []

        for i in range(len(test_x)):
            #seleziono la riga del test set
            row = test_x.iloc[i, :]

            #calcola etichetta
            labels_finali.append(self.hard_voting(row))

        return labels_finali

    def hard_voting(self, record):
        # Votazione rigida (hard voting) per ottenere l'etichetta di classe più comune tra gli alberi
        etichette = []

        for i in range(len(self.alberi_foresta)):
            colonne_utilizzate = self.colonne_alberi[i]
            row_selected = record[colonne_utilizzate].to_frame().transpose()
            etichette.append(self.alberi_foresta[i].predict(row_selected)[0])

        # Trova l'elemento con l'occorrenza massima (se sono uguali, ne sceglie uno casuale)
        label_classe = Counter(etichette).most_common(1)[0][0]
        return label_classe
    
    def fit_predict(self, train_x, train_y, test_x):
        # Addestramento del modello e predizione sulle istanze di test
        self.fit(train_x, train_y)
        return self.predict(test_x)

