from common_imports import split_dataset, accuracy_score
from Knn import KNN_Classifier_Custom
from Pre_Processing import stratificazione, bilanciamento, Feature_Selection, Combinazione_Attributi, unione_UnderSampling_OverSampling, Trasformazione_Attributi
from Multi_Classifier import RandomForestCustom
from ANN import Artificial_Neural_Network
from Decision_Tree import decision_tree
from SVM import SVM_classifier
from Valutazione_Modello import plot_roc_migliori_combinazioni

def best_combination(model):
    # Split del dataset in set di addestramento e test
    train_x, test_x, train_y, test_y = split_dataset()
    
    # Opzioni di modelli e tecniche di pre-elaborazione
    prep = ['Nessuno', 'Campionamento', 'Undersampling', 'Oversampling','Feature Selection', 'Top Combo(Under/Over)', 'Combinazione Attributi', 'Trasformazione Attributi']
    
    # Metodi di campionamento per Undersampling e Oversampling
    sampling_method_under = ['Random', 'IHT', 'NearMiss_v1', 'NearMiss_v2', 'ClusterCentroids']
    sampling_method_over = ['Random', 'SMOTE', 'ADASYN']

    # Metodi di selezione delle caratteristiche
    selection_methods = ['Correlation-Based', 'Variance Threshold', 'Select_K Best', 'Sequential Feature']

    # Metodi di combinazione delle caratteristiche
    combination_methods = ['PCA', 'Sparse Projection', 'Gaussian Projection', 'Feature Agglomeration']

    # Metodi di trasformazione delle caratteristiche
    trasf_methods = ['Standard Scaler', 'MinMaxScaler', 'Normalize']

    results = []
    for preprocessing_selection in prep:
        if preprocessing_selection == "Nessuno":
            # Nessuna pre-elaborazione
            train_x, test_x, train_y, test_y = split_dataset()
            results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), None])

        elif preprocessing_selection == "Campionamento":
            # Stratificazione del dataset
            train_x, test_x, train_y, test_y = stratificazione()
            results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), None])
            
        elif preprocessing_selection == "Undersampling":
            # Undersampling con vari metodi
            for sampling_method in sampling_method_under:
                train_x, test_x, train_y, test_y = bilanciamento(preprocessing_selection, sampling_method)
                results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), sampling_method])

        elif preprocessing_selection == "Oversampling":
            # Oversampling con vari metodi
            for sampling_method in sampling_method_over:
                train_x, test_x, train_y, test_y = bilanciamento(preprocessing_selection, sampling_method)
                results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), sampling_method])

        elif preprocessing_selection == "Feature Selection":
            # Selezione delle caratteristiche con vari metodi
            for selected_method in selection_methods:
                train_x, test_x, train_y, test_y = Feature_Selection(selected_method)
                results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), selected_method])
            
        elif preprocessing_selection == "Top Combo(Under/Over)":
            # Combinazione di Undersampling e Oversampling
            train_x, test_x, train_y, test_y, union_method = unione_UnderSampling_OverSampling(model)
            results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), union_method])

        elif preprocessing_selection == "Combinazione Attributi":
            # Combinazione delle caratteristiche con vari metodi
            for comb_method in combination_methods:
                train_x, test_x, train_y, test_y = Combinazione_Attributi(comb_method)
                results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), comb_method])

        elif preprocessing_selection == "Trasformazione Attributi":
            # Trasformazione delle caratteristiche con vari metodi
            for trsf_method in trasf_methods:
                train_x, test_x, train_y, test_y = Trasformazione_Attributi(trsf_method)
                results.append([model, preprocessing_selection, model_selection(model, train_x, train_y, test_x, test_y), trsf_method])
    
    # Trova la combinazione migliore in base all'accuratezza
    best_result = max(results, key=lambda x: x[2])
    return best_result

def model_selection(model_selection, train_x, train_y, test_x, test_y):
    # Seleziona il modello specificato e restituisce l'accuratezza
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

    return accuracy_score(test_y, esiti_cancro)



def confronta_le_migliori_in_assoluto():
    '''
    questa funzione crea la curva roc con i parametri migliori trovati con la funzione precedente
    '''
    
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = bilanciamento("Undersampling", "IHT")
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm, _ = unione_UnderSampling_OverSampling("SVM")
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann, _ = unione_UnderSampling_OverSampling("Rete Neurale")
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn, _ = unione_UnderSampling_OverSampling("KNN(Custom)")
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf, _ = unione_UnderSampling_OverSampling("Random Forest(Custom)")
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)

 

def confronto_migliori_campionamento():
     
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = stratificazione()
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm = stratificazione()
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann = stratificazione()
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn = stratificazione()
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf = stratificazione()
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)


def confronto_migliori_bilanciamento():
     
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = bilanciamento("Undersampling", "IHT")
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm, _ = unione_UnderSampling_OverSampling("SVM")
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann, _ = unione_UnderSampling_OverSampling("Rete Neurale")
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn, _ = unione_UnderSampling_OverSampling("KNN(Custom)")
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf, _ = unione_UnderSampling_OverSampling("Random Forest(Custom)")
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)

def confronto_migliori_Feature_Selection():
     
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = Feature_Selection("Variance Threshold")
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm = Feature_Selection("Variance Threshold")
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann = Feature_Selection("Variance Threshold")
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn = Feature_Selection("Variance Threshold")
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf = Feature_Selection("Variance Threshold")
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)

def confronto_migliori_Combinazione_Attributi():
     
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = Combinazione_Attributi("Feature Agglomeration")
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm = Combinazione_Attributi("Feature Agglomeration")
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann = Combinazione_Attributi("Feature Agglomeration")
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn = Combinazione_Attributi("Gaussian Projection")
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf = Combinazione_Attributi("Feature Agglomeration")
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)

def confronto_migliori_Trasformazione_Attributi():
     
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = Trasformazione_Attributi("Normalize")
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm = Trasformazione_Attributi("Standard Scaler")
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann = Trasformazione_Attributi("Normalize")
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn = Trasformazione_Attributi("Standard Scaler")
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf = Trasformazione_Attributi("Normalize")
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)

def confronto_Senza_Pre_processing():
     
    #Combinazione per l'Albero Decisionale
    train_x, test_x, train_y, test_y_albero = split_dataset()
    esiti_cancro_albero = decision_tree(train_x, train_y, test_x)

    #combinazione per SVM
    train_x, test_x, train_y, test_y_svm = split_dataset()
    esiti_cancro_svm = SVM_classifier(train_x, train_y, test_x)

    #combinazione per ANN
    train_x, test_x, train_y, test_y_ann = split_dataset()
    esiti_cancro_ann = Artificial_Neural_Network(train_x, train_y, test_x)
    
    #combinazione per KNN
    train_x, test_x, train_y, test_y_knn = split_dataset()
    knn = KNN_Classifier_Custom()
    esiti_cancro_knn = knn.fit_predict(train_x, train_y, test_x)
    
    #combinazione per Random Forest
    train_x, test_x, train_y, test_y_rf = split_dataset()
    RF = RandomForestCustom()
    esiti_cancro_rf = RF.fit_predict(train_x, train_y, test_x)

    plot_roc_migliori_combinazioni(esiti_cancro_albero,test_y_albero, esiti_cancro_svm, test_y_svm, esiti_cancro_ann, test_y_ann, esiti_cancro_knn, test_y_knn, esiti_cancro_rf, test_y_rf)


# *********************** Attivazione Confronti per i vari modelli con vari pre-processing*********************** #
'''
È possibile individuare le combinazioni ottimali facilmente confrontando le aree sotto le curve ROC presenti 
nella tabella della relazione. 
Per attivare la funzione, è sufficiente rimuovere il simbolo # posizionato davanti. 
Al termine dell'esecuzione, il plot mostrerà le curve ROC relative alle migliori 
configurazioni di pre-processing applicate ai diversi modelli.
'''
#CAPIONAMENTO
#confronto_migliori_campionamento()

#BILANCIAMENTO
#confronto_migliori_bilanciamento()

#FEAUTER SELECTION
#confronto_migliori_Feature_Selection()

#COMBINAZIONE DI ATTRIBUTI
#confronto_migliori_Combinazione_Attributi()

#TRASFORMAZIONE ATTRIBUTI
#confronto_migliori_Trasformazione_Attributi()

#NESSUN PRE-PROCESSING
#confronto_Senza_Pre_processing()

# ************************************************************************************ #


