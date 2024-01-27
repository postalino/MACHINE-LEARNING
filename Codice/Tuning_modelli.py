from Knn import KNN_Classifier_Custom
from sklearn.tree import DecisionTreeClassifier
from common_imports import split_dataset, accuracy_score, StandardScaler, SVC, MinMaxScaler, warnings, np
from sklearn.model_selection import cross_validate
from Valutazione_Modello import plot_tuning_knn, dataset_senza_preprocessing, plot_tuning_AlberoDecisionale, plot_tuning_Random_forest
from sklearn.model_selection import GridSearchCV
from SVM import SVM_classifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from ANN import Artificial_Neural_Network
from Multi_Classifier  import RandomForestCustom


def tuning_KNN_custom():
    # Suddividi il dataset in training set e test set
    train_x, test_x, train_y, test_y = split_dataset()

    # Definisci i range di valori per k e tipi di distanza
    k_range = list(range(1, 26))
    dist_range = ['Euclidea', 'Manhattan', 'Chebyshev']

    # Inizializza array per memorizzare le accuratezze durante il tuning
    acc_train = np.empty((len(dist_range), len(k_range)))
    acc_val = np.empty((len(dist_range), len(k_range)))
    acc_test = np.empty((len(dist_range), len(k_range)))

    # Loop attraverso le diverse distanze e valori di k
    for i, dist in enumerate(dist_range):
        for j, k in enumerate(k_range):
            # Crea un classificatore KNN personalizzato con la configurazione corrente
            clf = KNN_Classifier_Custom(k, distance_type=dist)
            print("Distanza: ",dist, "  K: ", k)
            # Esegui una 5-fold cross-validation
            scores = cross_validate(estimator=clf, X=train_x, y=train_y, cv=5, n_jobs=10, return_train_score=True, return_estimator=True) 
            score_train = scores['train_score']
            score_val = scores['test_score']
            
            # Calcola l'accuratezza media sul set di test
            score_test = np.mean([estimator.score(test_x, test_y) for estimator in scores['estimator']])
            
            # Memorizza le accuratezze medie nei rispettivi array
            acc_train[i, j] = score_train.mean()
            acc_val[i, j] = score_val.mean()
            acc_test[i, j] = score_test

    
    # Chiama la funzione per visualizzare i risultati del tuning del KNN
    plot_tuning_knn(acc_train, acc_val, acc_test, dist_range)


def tuning_Albero_decisionale():
    # Suddividi il dataset in training set e test set
    train_x, test_x, train_y, test_y = split_dataset()

    # Definisci la gamma di profondità massima degli alberi da esplorare
    max_depth_range = list(range(1, 26))

    # Liste per memorizzare le accuratezze durante il tuning
    acc_train = []
    acc_val = []
    acc_test = []

    # Loop attraverso le diverse profondità degli alberi
    for depth in max_depth_range:
        # Crea un classificatore ad albero decisionale con la profondità corrente
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)

        # Esegui una 10-fold cross-validation
        scores = cross_validate(estimator=clf, X=train_x, y=train_y, cv=10, n_jobs=10, return_train_score=True, return_estimator=True) 
    
        # Memorizza le accuratezze medie per allenamento, validazione e test
        acc_train.append(np.mean(scores['train_score']))
        acc_val.append(np.mean(scores['test_score']))

        # Calcola l'accuratezza media sul set di test
        acc_test.append(np.mean([estimator.score(test_x, test_y) for estimator in scores['estimator']]))

    # Plotta i risultati del tuning
    plot_tuning_AlberoDecisionale(acc_train, acc_val, acc_test, max_depth_range)


def gridSearchSVM(train_x, train_y, param_grid):
    # Creazione di un classificatore di tipo SVM
    svm_clf = SVC()
    # Numero di fold per la Cross-validation
    n_folds = 5
    # Creazione di un oggetto di tipo GridSearchCV
    grid_search_cv = GridSearchCV(svm_clf, param_grid, cv=n_folds)
    # Esecuzione della ricerca degli iperparametri 
    grid_search_cv.fit(train_x, train_y)
    # Stampa risultati
    migliori_parametri = grid_search_cv.best_params_
    # Ottieni l'accuratezza associata alla combinazione ottimale
    best_accuracy = grid_search_cv.best_score_

    return migliori_parametri, best_accuracy

def tuning_SVM():
    train_x, test_x, train_y, test_y = split_dataset()
    # Creazione della griglia di iperparametri per SVM lineare
    param_grid_linear = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

    # Creazione della griglia di iperparametri per SVM con RBF
    param_grid_rbf = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.025,0.05,]}]

    #dataset moons
    scaler = StandardScaler()
    scld_train_x1 = scaler.fit_transform(train_x)

    #risultato per tuning SVM lineare con ottimizzazione di C
    param_tuning_linear, acc_tuning_linear = gridSearchSVM(scld_train_x1, train_y, param_grid_linear)

    #risultato per tuning SVM con kernel RBF con ottimizzazione di C e gamma
    param_tuning_rbf, acc_tuning_rbf = gridSearchSVM(scld_train_x1, train_y, param_grid_rbf)

    type_model = "SVM"
    #trovo il migliore e lo restituisco 
    if(acc_tuning_linear > acc_tuning_rbf):
        esito_cancro = SVM_classifier(train_x, train_y, test_x , param_tuning_linear)
        #plot
        dataset_senza_preprocessing(esito_cancro, test_y, type_model,best_c = param_tuning_linear.get('C'), best_kernel = param_tuning_linear.get('kernel'))
    else:
        esito_cancro = SVM_classifier(train_x, train_y, test_x , param_tuning_rbf)
        #plot
        dataset_senza_preprocessing(esito_cancro, test_y, type_model,best_c = param_tuning_rbf.get('C'), best_kernel = param_tuning_rbf.get('kernel'), best_gamma= param_tuning_rbf.get('gamma'))

def tuning_ANN():
    # Suddivide il dataset in dati di addestramento e test
    train_x, test_x, train_y, test_y = split_dataset()
    
    # Ignora i FutureWarning per mantenere l'output pulito
    warnings.filterwarnings("ignore")

    # Definisci il set di parametri da esplorare nella ricerca della griglia
    GRID = [
        {'scaler': [StandardScaler(), MinMaxScaler()],
         'estimator': [MLPClassifier(max_iter=100,
                                     solver="sgd",
                                     random_state=0,
                                     learning_rate_init=0.2,
                                     early_stopping=True)],
         'estimator__hidden_layer_sizes': [(20), (30), (40), (40, 20), (50, 30), (50, 30, 10), (50, 40, 30, 20)],
         'estimator__alpha': [0.001, 0.01]
         }
    ]
    
    # Crea un pipeline con uno scaler e un classificatore di rete neurale
    PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])
    
    # Specifica il numero di fold per la cross-validation
    n_folds = 5
    
    # Esegui la ricerca della griglia con cross-validation
    grid_search_cv = GridSearchCV(PIPELINE, param_grid=GRID, cv=n_folds)
    grid_search_cv.fit(train_x, train_y)
    
    # Trova la combinazione migliore di parametri ottenuti dalla ricerca della griglia
    best_combination = grid_search_cv.best_params_
    
    # Addestra la rete neurale con i migliori parametri trovati
    esito_cancro = Artificial_Neural_Network(train_x, train_y, test_x, 
                                             best_combination.get('estimator__hidden_layer_sizes'), 
                                             best_combination.get('scaler'), 
                                             best_combination.get('estimator__alpha'))
    
    # Plot della valutazione del modello migliore trovato con il tuning
    dataset_senza_preprocessing(esito_cancro, test_y, 'Rete Neurale', 
                                scaler=best_combination.get('scaler'), 
                                alpha_value=best_combination.get('estimator__alpha'), 
                                hidden_layer=best_combination.get('estimator__hidden_layer_sizes'))
    

def tuning_Random_Forest(max_num_classifier=125):
    """
    Tuning del modello Random Forest variando il numero di alberi nel forest.

    Parameters:
    - max_num_classifier (int): Numero massimo di alberi da testare nel tuning.
    """
    # Suddivide il dataset in training set e test set
    train_x, test_x, train_y, test_y = split_dataset()

    # Liste per memorizzare le performance del modello per diversi numeri di alberi
    all_score_Random_Forest = []
    all_numbers_trees_forest = []

    # Loop attraverso i diversi numeri di alberi nel forest
    for i in range(2, max_num_classifier + 1):
        print(i)
        rf = RandomForestCustom(numero_alberi = i, random_state = 0)

        # Calcola l'accuratezza del modello Random Forest con il numero corrente di alberi
        accuracy = accuracy_score(test_y, rf.fit_predict(train_x, train_y,test_x))
        # Memorizza l'accuratezza e il numero corrente di alberi
        all_score_Random_Forest.append(accuracy)
        all_numbers_trees_forest.append(i)

    # Chiama la funzione per visualizzare i risultati del tuning del Random Forest
    plot_tuning_Random_forest(all_score_Random_Forest, max_num_classifier, all_numbers_trees_forest)


# *********************** Attivazione Tuning per i vari modelli *********************** #
'''
Per eseguire il tuning dei vari modelli è sufficiente togliere # posizionato davanti alla funzione.
Questa non necessita di alcun parametro da passare.
Alla fine verrà mostrato il plot per i vari tuning.
'''

# KNN --> Plot
# Attivare o disattivare il tuning per KNN
# tuning_KNN_custom()

# ALBERO DECISIONALE --> Plot
# Attivare o disattivare il tuning per l'Albero Decisionale
# tuning_Albero_decisionale()

# RANDOM FOREST --> Plot
# Attivare o disattivare il tuning per Random Forest
# tuning_Random_Forest()

# SVM --> Plot
# Attivare o disattivare il tuning per SVM
# tuning_SVM()

# ANN --> Plot
# Attivare o disattivare il tuning per ANN
# tuning_ANN()

# ************************************************************************************ #
