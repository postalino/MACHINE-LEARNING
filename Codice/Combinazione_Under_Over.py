from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from Pre_Processing import stratificazione
from common_imports import import_dataset, accuracy_score,warnings
from Decision_Tree import decision_tree
from Multi_Classifier import RandomForestCustom
from Knn import KNN_Classifier_Custom
from SVM import SVM_classifier
from ANN import Artificial_Neural_Network

# Funzione per l'undersampling con una percentuale del 50%
def Undersampling_50(tecnica,X, y):
     # ... [Implementazione delle diverse tecniche di undersampling] ...
    if tecnica == "Random":
        rus = RandomUnderSampler(sampling_strategy={'B':285},random_state = 42)
        X_resampled, y_resampled = rus.fit_resample(X, y) 
        
    elif tecnica == "IHT":
        # Mappiamo le etichette da stringhe a numeri interi (M --> 1 e B--> 0)
        y = y.replace({'M': 1, 'B': 0})
            
        iht = InstanceHardnessThreshold(sampling_strategy={0:285},random_state = 42)
        X_resampled, y_resampled = iht.fit_resample(X, y)
            
        # Rimappiamo le etichette numeriche a stringhe
        y_resampled = y_resampled.replace({1: 'M', 0: 'B'})
        
    elif tecnica == "NearMiss_v1":
        nm = NearMiss(sampling_strategy={'B':285},version=1)
        X_resampled, y_resampled = nm.fit_resample(X, y)
        
    elif tecnica == "NearMiss_v2":
        nm = NearMiss(sampling_strategy={'B':285},version=2)
        X_resampled, y_resampled = nm.fit_resample(X, y)
        
    elif tecnica == "ClusterCentroids":
        warnings.filterwarnings("ignore")
            
        cc = ClusterCentroids(sampling_strategy={'B':285}, random_state = 42)
        X_resampled, y_resampled = cc.fit_resample(X, y)
            
        warnings.filterwarnings("default")
    return X_resampled, y_resampled
    
# Funzione per l'oversampling con una percentuale del 50%
def Oversampling_50(tecnica, X, y):
    # ... [Implementazione delle diverse tecniche di oversampling] ...
    if(tecnica == "Random"):
        ros = RandomOverSampler(sampling_strategy={'M':285},random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif(tecnica == "SMOTE"):
        sm = SMOTE(sampling_strategy={'M':285},random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        
    elif(tecnica == "ADASYN"):
        ada = ADASYN(sampling_strategy={'M':285}, random_state=42)
        X_resampled, y_resampled = ada.fit_resample(X, y)
    return X_resampled, y_resampled

# Funzione per la selezione del modello e calcolo dell'accuratezza
def model_selection(model_selection, train_x, train_y, test_x, test_y):
    # ... [Implementazione della selezione del modello e del calcolo dell'accuratezza] ...
    if model_selection == "Albero Decisionale":
        esiti_cancro = decision_tree(train_x, train_y, test_x)
    
    elif model_selection == "SVM":
        esiti_cancro = SVM_classifier(train_x, train_y, test_x)

    elif model_selection == "Rete Neurale":
        esiti_cancro = Artificial_Neural_Network(train_x, train_y, test_x)
    
    elif model_selection == "KNN(Custom)":
        knn = KNN_Classifier_Custom()
        esiti_cancro = knn.fit_predict(train_x, train_y, test_x)
    
    elif model_selection == "Random Forest(Custom)":
        RF = RandomForestCustom()
        esiti_cancro = RF.fit_predict(train_x, train_y, test_x)

    return accuracy_score(test_y,esiti_cancro)

# Funzione per trovare la migliore combinazione di tecniche di undersampling, oversampling e modello
def best_combination():
    tecniche_undersampling = ["Random", "IHT", "NearMiss_v1", "NearMiss_v2", "ClusterCentroids"]
    tecniche_oversampling = ["Random", "SMOTE", "ADASYN"]
    models = ["Albero Decisionale", "SVM", "Rete Neurale", "KNN(Custom)", "Random Forest(Custom)"]
    results_albero = []
    results_SVM = []
    results_ANN = []
    results_KNN = []
    results_RF = []
    X, y = import_dataset()

    # cicli annidati per provare tutte le combinazioni di tecniche di sampling e modelli
    for under in tecniche_undersampling:
        for over in tecniche_oversampling:
            X_resampled, y_resampled = Undersampling_50(under,X,y)
            X_resampled, y_resampled = Oversampling_50(over,X_resampled, y_resampled)

            for model in models:
                # Calcola e memorizza i risultati per ogni combinazione
                # ... (chiamate a model_selection e memorizzazione dei risultati)

                #stratifico il dataset per avere la stessa distribuzione tra il train e il test ed evitare over fitting
                train_x, test_x, train_y, test_y = stratificazione(X_resampled, y_resampled)
                
                if model == "Albero Decisionale":
                   results_albero.append([model, under, over, model_selection(model, train_x, train_y, test_x, test_y),test_y.value_counts().index,test_y.value_counts().values])

                elif model == "SVM":
                    results_SVM.append([model, under, over, model_selection(model, train_x, train_y, test_x, test_y),test_y.value_counts().index,test_y.value_counts().values])

                elif model == "Rete Neurale":
                    results_ANN.append([model, under, over, model_selection(model, train_x, train_y, test_x, test_y),test_y.value_counts().index,test_y.value_counts().values])
                
                elif model == "KNN(Custom)":
                    results_KNN.append([model, under, over, model_selection(model, train_x, train_y, test_x, test_y),test_y.value_counts().index,test_y.value_counts().values])
                
                elif model == "Random Forest(Custom)":
                    results_RF.append([model, under, over, model_selection(model, train_x, train_y, test_x, test_y),test_y.value_counts().index,test_y.value_counts().values])

    # Trova la migliore combinazione per ogni modello               
    best_result_albero = max(results_albero, key=lambda x: x[3])
    best_result_SVM = max(results_SVM, key=lambda x: x[3])
    best_result_ANN = max(results_ANN, key=lambda x: x[3])
    best_result_KNN = max(results_KNN, key=lambda x: x[3])
    best_result_RF =  max(results_RF, key=lambda x: x[3])

    # Stampa i risultati migliori
    print(best_result_albero)
    print(best_result_SVM)
    print(best_result_ANN)
    print(best_result_KNN)
    print(best_result_RF)


# *********************** Attivazione Funzione*********************** #
'''
Questa funzione permette di effettuare una ricerca esaustiva di tutte le possibili combinazioni di oversampling
e undersampling per trovare PER OGNI MODELLO la miglior combinazioni in assoluto, ovvero quella con l'accuratezza 
maggiore.
'''

# Esegui la funzione per trovare la migliore combinazione 
#best_combination()

# ************************************************************************************ #


